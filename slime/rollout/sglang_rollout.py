import asyncio
import copy

from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.http_utils import get, post
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.types import Sample
import argparse
import logging
import torch
import ray
from slime.utils.ray_utils import Box
from .rm_hub import async_rm, batched_async_rm
import requests
import threading
__all__ = ["generate_rollout"]
import time

import asyncio
import time
from asyncio import Future, Queue
from typing import List, Tuple, Dict, Any


class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args):
        # persistant state for the generation process
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        # FIXME
        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        
        self.sampling_params = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        self.reset()

    def reset(self):
        self.remaining_batch_size = 0
        self.remaining_sample_size = 0
        self.lock = threading.Lock()
        # pendings 用于存储所有未完成的生成任务的 future
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]], actor_model):
        # 这里的提交是以 groups 的形式提交的，每个 group 相当于一个 batch (什么级别？)
        response = requests.get(
            f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}/list_workers"
        )
        
        # print(response)
        urls = response.json()["urls"]
        self.manager = BatchingManager(urls, actor_model = actor_model)
        for index, group in enumerate(samples):

            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    # FIXME do not need actor
                    generate_and_rm_group(
                        self.args,
                        group,
                        actor_model,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                        base_url= urls[index % len(urls)]  # distribute groups across workers
                    )
                )
            )
        self.remaining_batch_size += len(samples)
        self.remaining_sample_size += sum(len(g) for g in samples)

    def check_match_eos(self, last_token_id: int) -> bool:
        matched_eos = False
        if self.tokenizer is not None:
            matched_eos |= last_token_id == self.tokenizer.eos_token_id
        return matched_eos

def sampling_from_probs_torch(probs: torch.Tensor):
    """A sampling implementation with native pytorch operations, without
    top-k, top-p, or min-p filtering."""
    sampled_index = torch.multinomial(probs, num_samples=1)
    batch_next_token_ids = sampled_index.view(-1).to(torch.int32)
    return batch_next_token_ids

def sample_from_the_logits(logits, sampling_params) -> int:
    # just support for top_k = -1, top_p = 1, tempature = 1.0
    # just suppoert single example
    assert (sampling_params["top_k"] == -1 and sampling_params["top_p"] == 1.0 and sampling_params["temperature"] == 1.0)
    batch_logits = logits.unsqueeze(0)
    sample_index = sampling_from_probs_torch(batch_logits)
    return sample_index[0]

def find_inconsistent_positions(log_probs_map_keys, target_order):
    """
    Finds the positions where two lists are inconsistent.

    Args:
        log_probs_map_keys: The keys from the log_probs_map.
        target_order: The target order.

    Returns:
        A list of tuples, where each tuple contains the index and the values
        from both lists at that index.
    """
    inconsistent_positions = []
    # Ensure both lists have the same length
    min_len = min(len(log_probs_map_keys), len(target_order))
    for i in range(min_len):
        if log_probs_map_keys[i] != target_order[i]:
            inconsistent_positions.append((i, log_probs_map_keys[i], target_order[i]))
    
    # Check for length differences
    if len(log_probs_map_keys) != len(target_order):
        print(f"Warning: The lengths of the lists are different. log_probs_map_keys has {len(log_probs_map_keys)} elements, while target_order has {len(target_order)} elements.")

    return inconsistent_positions

def recovery_pros(full_top_logprobs: list[list]):
    # [log_probs, idx, None] -> probs
    log_probs_map = {item[1]: item[0] for item in full_top_logprobs}
    target_order = range(len(full_top_logprobs))
    ori_order = list(log_probs_map.keys())
    ori_order.sort()
    # print(find_inconsistent_positions(ori_order, list(target_order)))
    # print(max(ori_order))
    # print("len: ",  len(full_top_logprobs))
    result_probs = []
    for idx in target_order:
        prob = log_probs_map.get(idx)
        assert(prob is not None), f"idx {idx} not found in log_probs_map, index: {prob}"
        result_probs.append(prob)
    # print(result_probs)
    new_dis = torch.exp(torch.tensor(result_probs, dtype=torch.float32))
    # print("new_sum:", new_dis.sum().item())
    return new_dis

# async def get_full_rollout_probs(url, input_token_ids, vocab_size, sampling_params):

#     sampling_params["max_new_tokens"] = 1
#     payload = {
#             "sampling_params": sampling_params,
#             "return_logprob": True,
#             "top_logprobs_num": vocab_size,   
#     }
#     payload["input_ids"] = input_token_ids
#     output = await post(url, payload, use_http2=False)
#     return recovery_pros(output["meta_info"]["output_top_logprobs"][0][0])
class BatchingManager:
    def __init__(self, urls: List[str], actor_model, max_batch_size: int = 128, batch_timeout: float = 10, recomputation_batch_timeout:float = 10):
        """
        初始化批处理管理器。

        Args:
            urls (List[str]): 提供服务的多个推理服务器 URL 列表。
            max_batch_size (int): 每个批处理的最大请求数。
            batch_timeout (float): 等待填满一个批处理的最大时间（秒）。
        """
        # [MODIFIED] 为不同类型的请求和每个 URL 创建独立的队列字典
        urls = [url + '/generate' for url in urls]
        self.recomputation_queues: Dict[str, Queue] = {url: Queue() for url in urls}
        self.rollout_queues: Dict[str, Queue] = {url: Queue() for url in urls}
        self.actor_queue: Queue = Queue()
        # FIXME
        self.actor_model = actor_model

        #FIXME
        self.rollout_max_batch_size = 256
        self.rollout_batch_timeout = batch_timeout

        self.recomputation_max_batch_size = 32
        self.recomputation_batch_timeout = recomputation_batch_timeout
        
        self.verification_max_batch_size = 256
        self.verification_batch_timeout = batch_timeout

        self.aborted = False

        self.worker_tasks = []

        # [MODIFIED] 为每一个队列创建一个专属的后台 worker 任务
        for url in urls:
            self.worker_tasks.append(
                asyncio.create_task(
                    self._batching_worker(url, self.recomputation_queues[url], 1)
                )
            )
            self.worker_tasks.append(
                asyncio.create_task(
                    self._batching_worker(url, self.rollout_queues[url], 0)
                )
            )
        self.worker_tasks.append (
            asyncio.create_task(
                self._verification_worker(self.actor_queue, self.actor_model)
            )
        )
        self.lock = asyncio.Lock()

    async def submit_request(self, url: str, payload: dict, existing_task_num) -> dict:
        """
        提交单个请求负载到相应的队列，并等待结果。
        """
        async with self.lock:
            future = Future()
            if self.aborted:
                future.set_exception(asyncio.CancelledError("BatchingManager was aborted."))
            else:
                try:
                    if payload.get("sampling_params", {}).get("max_new_tokens") == 1:
                        await self.recomputation_queues[url].put((payload, future))
                        self.recomputation_max_batch_size = min(self.recomputation_max_batch_size, max(existing_task_num/4, 1))
                    else:
                        await self.rollout_queues[url].put((payload, future))
                        self.rollout_max_batch_size = min(self.rollout_max_batch_size, max(existing_task_num,1))
                except (asyncio.CancelledError, IndexError):
                    future.set_exception(asyncio.CancelledError("BatchingManager was aborted."))
        return await future    
    
    async def submit_actor_request(self, temp_data: dict, existing_task_num) -> dict:
        """
        提交单个请求负载到 actor 队列，并等待结果。
        """
        async with self.lock:
            future = Future()
            if self.aborted:
                future.set_exception(asyncio.CancelledError("BatchingManager was aborted."))
            else:
                try:
                    await self.actor_queue.put((temp_data, future))
                    self.verification_max_batch_size = min(self.verification_max_batch_size, max(existing_task_num, 1))
                except asyncio.CancelledError:
                    future.set_exception(asyncio.CancelledError("BatchingManager was aborted."))
        return await future
    
    def _merger_request_batch(self, requests: List[Tuple[dict, Future]]) -> Dict[str, Any]:

        batched_data = {
            "tokens": [],
            "response_lengths": [],
            "sample_indices": [],
            "rollout_log_probs": []
        }

        for data, _ in requests:
            # Assumes each data dict contains lists of size 1
            # FIXME list[list[data]] -> list[data]
            batched_data["tokens"].extend(data["tokens"])
            batched_data["response_lengths"].extend(data["response_lengths"])
            batched_data["sample_indices"].extend(data["sample_indices"])
            batched_data["rollout_log_probs"].extend(data["rollout_log_probs"])
        return batched_data

    def _split_results(self, batched_results: List[Dict[str, Any]], num_requests: int) -> List[Dict[str, Any]]:
        """Splits the batched verification result back into individual results."""
        individual_results = []
        for mini_batch_result in batched_results:
            num = len(mini_batch_result["recompute_index"])
            for i in range(num):
                result = {
                    # Assuming the result keys match this structure
                    "recompute_index": [mini_batch_result["recompute_index"][i]],
                    "logits": [mini_batch_result["logits"][i]]
                }
                individual_results.append(result)
        assert(len(individual_results) == num_requests)
        return individual_results


    async def _verification_worker(self, queue: Queue, actor_model):
        """Background worker to collect, batch, verify, and distribute results."""
        try:
            while not self.aborted:
                batch_size = self.verification_max_batch_size
                batch_timeout = self.verification_batch_timeout
                requests: List[Tuple[Dict[str, Any], Future]] = []
                actor_model = self.actor_model
                # Wait for the first request to start a new batch
                
                first_data, first_future = await queue.get()
                # print("first_data")
                requests.append((first_data, first_future))

                # Collect more requests until the batch is full or timeout occurs
                while len(requests) < batch_size:
                    try:
                        # FIXME 
                        payload, future = await asyncio.wait_for(queue.get(), timeout=batch_timeout)
                        requests.append((payload, future))
                    except (asyncio.TimeoutError):
                        break
                    
                if not requests:
                    continue 

                # print("enough")
                # 1. Merge all requests into a single, large payload
                batched_data = self._merger_request_batch(requests)
                # print(batched_data["response_lengths"])
                print(f"VERIFICATION WORKER: Dispatching a merged batch of {len(requests)} requests.")

                # 2. Send the single, large batch to the Ray actor
                # This is the core logic change
                box_list = ray.get(actor_model.async_verification(0, ray.put(Box(ray.put(batched_data)))))
                # list[Box[Object]], Obj [dict[str, list[Any]]]
                # print("fix--------------")
                # print(box_list)
                batched_results = [ray.get(b.inner) for b in box_list]
                # FIXME more general
                batched_results = [batched_results[0], batched_results[2]]
                # batched_results = ray.get(ray.get(box_list[0]).inner)
                # print("finish here")
                # 3. Split the batched result back into individual results
                individual_results = self._split_results(batched_results, len(requests))

                # 4. Distribute the individual results back to the waiting futures
                for i, result in enumerate(individual_results):
                    original_future = requests[i][1]
                    original_future.set_result(result)
                print(f"VERIFICATION WORKER: completed and Distributed results to {len(requests)} requests.")
        except (asyncio.CancelledError):
            print("VERIFICATION WORKER is cancelled. Cleaning up...")
            # [关键] 为正在处理的请求设置异常，防止调用者无限等待
            for _data, future in requests:
                if not future.done():
                    future.set_exception(asyncio.CancelledError("Worker was cancelled during processing."))
        finally:
            print("VERIFICATION WORKER has shut down.")

    async def _batching_worker(self, url: str, queue: Queue, tag: int):
        """
        核心后台任务：收集请求，然后使用 asyncio.gather 并发调度它们。
        """
        try:
            while not self.aborted:
                if tag == 0:
                    batch_size = self.rollout_max_batch_size
                    batch_timeout = self.rollout_batch_timeout
                else:
                    batch_size = self.recomputation_max_batch_size
                    batch_timeout = self.recomputation_batch_timeout
                requests: List[Tuple[dict, Future]] = []
                
                
                first_payload, first_future = await queue.get()
                requests.append((first_payload, first_future))
               

                while len(requests) < batch_size:
                    try:
                        # FIXME 
                        payload, future = await asyncio.wait_for(queue.get(), timeout=batch_timeout)
                        requests.append((payload, future))
                    except (asyncio.TimeoutError):
                        break
                
                if not requests:
                    continue

                batch_size = len(requests)
                print(f"WORKER tag{tag} ({url[-1]}):  {batch_size} requests. Dispatching concurrently...")

                futures = [req[1] for req in requests]

                # [!!!] 核心修改：为批次中的每个请求创建一个 post 任务
                # 我们不再合并 payload，而是为每个 payload 创建一个独立的 post 调用
                post_tasks = [
                    post(url, req[0], use_http2=False) for req in requests
                ]

                # [!!!] 使用 asyncio.gather 并发执行所有独立的 post 请求
                # return_exceptions=True 确保一个请求失败不会导致整个 gather 崩溃
                concurrent_outputs = await asyncio.gather(*post_tasks, return_exceptions=True)
                
                print(f"WORKER tag{tag} ({url[-1]}): Batch of {batch_size} completed. Distributing results.")
                
                # 将结果分发回等待的 future
                for i, output in enumerate(concurrent_outputs):
                    futures[i].set_result(output)
        except asyncio.CancelledError:
            print(f"WORKER tag{tag} ({url[-1]}) is cancelled. Cleaning up...")
            # [关键] 为正在处理的请求设置异常，防止调用者无限等待
            for _payload, future in requests:
                if not future.done():
                    future.set_exception(asyncio.CancelledError("Worker was cancelled during processing."))
        finally:
            print(f"WORKER tag{tag} ({url[-1]}) has shut down.")


    async def abort(self):
        print("abort the batching manager start")
        """
        中止所有待处理的请求并停止批处理工作者。
        """
        async with self.lock:
            if self.aborted:
                return
            self.aborted = True

            # [核心修改] 创建一个异常，用于通知所有等待的 Future
            cancellation_exception = asyncio.CancelledError("BatchingManager was aborted.")

            # [核心修改] 清理所有队列，并为每个待处理的请求设置异常
            all_queues = list(self.recomputation_queues.values()) + list(self.rollout_queues.values()) + [self.actor_queue]
            for queue in all_queues:
                while not queue.empty():
                    try:
                        # 从队列中取出请求，但不阻塞
                        _payload, future = queue.get_nowait()
                        if not future.done():
                            # 通知等待方，任务已被取消
                            future.set_exception(cancellation_exception)
                    except queue.Empty:
                        break # 队列已空

            # 现在可以安全地取消 worker 任务了
            for task in self.worker_tasks:
                task.cancel()
            
            # 等待所有 worker 任务实际完成取消操作
            if self.worker_tasks:
                await asyncio.gather(*self.worker_tasks, return_exceptions=True)

            self.worker_tasks = []
            # 清空队列引用（里面的 future 已经处理过了）
            self.recomputation_queues.clear()
            self.rollout_queues.clear()
            self.actor_queue = None
            print("abort the batching manager end")


# [Change]
async def spec_generate(args, sample: Sample, actor_model, sampling_params, base_url) -> Sample:
    # 1. deal with initial status
    # logging.info(f"Generating sample {sample.index} with prompt: {sample.prompt}")
    state = GenerateState(args)
    # 2. generate a single sample
    # url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    # print(f"start: {state.remaining_sample_size}")

    # abort all the requests
    # MARK 对于所有的 worker 都发送 abort 的请求
    # FIXME
    url = f"{base_url}/generate"
    # didn't consider the partial rollout here
    prompt_tokens_ids = state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
    sample.tokens = prompt_tokens_ids
    # didn't consider the loss mask here
    round_number = 0
    round_tokens = 100
    # max_round_number = args.rollout_max_response_len // round_tokens + 10
    max_round_number = 1
    max_new_tokens = sampling_params["max_new_tokens"]
    start_time = time.time()
    try:
        while round_number < max_round_number:
            
            # 2.0 deal with the sample status
            # FIXME maybe not strict
            if sample.response_length >= args.rollout_max_response_len:
                break
            # 2.1 deal with data payload & sampling para
            # start_rollout_request_time = time.time()
            sampling_params["max_new_tokens"] = min(round_tokens, max_new_tokens - sample.response_length)
            payload = {
                "sampling_params": sampling_params,
                "return_logprob": True,
                # "seed": 42
            }
            input_token_ids = sample.tokens
            payload["input_ids"] = input_token_ids
            # 2.2 post request
            
            # output = await post(url, payload, use_http2=False)
            output = await state.manager.submit_request (url, payload, state.remaining_sample_size)
            # end_rollout_request_time = time.time()
            # print(f"Latency: Rollout Request took {end_rollout_request_time - start_rollout_request_time:.4f} seconds.")
            # 3. deal with response
            # 3.1 deal with metadata
            # start_verification_time = time.time()
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            
            temp_tokens = sample.tokens + new_response_tokens
            temp_data = {
                "tokens": [temp_tokens],
                # FIXME is this necessary?
                "response_lengths": [len(new_response_tokens)],
                "sample_indices": [sample.index],
                "rollout_log_probs": [[t[0] for t in output["meta_info"]["output_token_logprobs"]]]
                
            }
            # FIXME didn't consider the async
            # verification_res = ray.get(ray.get(actor_model.async_verification(0, Box(ray.put(temp_data)))[0]).inner)
            verification_res = await state.manager.submit_actor_request(temp_data, state.remaining_sample_size)
            # end_verification_time = time.time()
            # print(f"Latency: Verification took {end_verification_time - start_verification_time:.4f} seconds.")

            # transform the index from global to local
            # start_recomputation_time = time.time()
            if verification_res["recompute_index"][0] == -1:
                # if all accepted, and not exceed the max or eos,
                # we can just append the new response tokens
                if sample.response_length + len(new_response_tokens) < args.rollout_max_response_len and \
                not state.check_match_eos(new_response_tokens[-1]):
                    new_distribuation = torch.softmax(torch.tensor(verification_res['logits'][0]), dim = -1)
                    recompute_ids = sample_from_the_logits(new_distribuation, sampling_params).item()
                    accepted_tokens = new_response_tokens + [recompute_ids]
                else:
                    accepted_tokens = new_response_tokens
            else:
                # if not all accepted, we need to reset the new response tokens
                
                sampling_params["max_new_tokens"] = 1
                new_payload = {
                        "sampling_params": sampling_params,
                        "return_logprob": True,
                        "top_logprobs_num": state.config.vocab_size,   
                }
                new_payload["input_ids"] = input_token_ids
                # new_output = await post(url, new_payload, use_http2=False)
                new_output = await state.manager.submit_request(url, new_payload, state.remaining_sample_size)
                # start_recover_top_logprobs = time.time()
                response_recompute_index = verification_res["recompute_index"][0]
                # list[list[[prob,idx,None]]]
                rollout_probs = recovery_pros(new_output["meta_info"]["output_top_logprobs"][0])
                # end_recover_top_logprobs = time.time()
                # print(f"Latency: Rollout Top Logprobs took {end_recover_top_logprobs - start_recover_top_logprobs:.4f} seconds.")
                # FIXME data type
                new_distribuation = rollout_probs - torch.softmax(torch.tensor(verification_res['logits'][0], dtype=torch.float32), dim = -1)
                new_distribuation = torch.clamp(new_distribuation,  min = 0)
                recompute_ids = sample_from_the_logits(new_distribuation, sampling_params).item()
                accepted_tokens = new_response_tokens[:response_recompute_index] + [recompute_ids]
            # end_recomputation_time = time.time()
            # print(f"Latency: Recomputation{verification_res['recompute_index']} took {end_recomputation_time - start_recomputation_time:.4f} seconds.")
            print(f"Round {round_number}, recompute index: {verification_res['recompute_index']}, recompute token id: {recompute_ids}, accepted tokens: {len(accepted_tokens)}, response_length: {sample.response_length}")
            # TEST 
            # accepted_tokens = new_response_tokens
            sample.tokens = sample.tokens + accepted_tokens
            sample.response_length += len(accepted_tokens)
            # logging.info(f"Sample response length: {sample.response_length}")
            sample.response += state.tokenizer.decode(accepted_tokens, skip_special_tokens=False)
            
            if state.check_match_eos(accepted_tokens[-1]):
                sample.status = Sample.Status.COMPLETED
                break
            round_number += 1
        # 3.2 deal with sample status
        # FIXME how to deal with truncated max or truncated spec
        # print(f"Round {round_number}, recompute index: {verification_res['recompute_index']}, recompute token id: {recompute_ids}, accepted tokens: {len(accepted_tokens)}, response_length: {sample.response_length}")
    except asyncio.CancelledError:
        # 这是由 manager.abort() 触发的，是一种预期的“异常”
        logging.warning(f"Generation for sample {sample.index} was cancelled.")
        sample.status = Sample.Status.ABORTED
    
    except (KeyError, IndexError, TypeError) as e:
        # 捕获数据处理相关的错误
        logging.error(f"Data processing error for sample {sample.index}: {e}", exc_info=True)
        sample.status = Sample.Status.ABORTED # 或 FAILED
        
    except Exception as e:
        # 捕获所有其他异常，例如来自 worker 的网络错误、Ray错误等
        # 使用 e.__class__.__name__ 可以获得异常的类型名，如 'ClientConnectorError'
        logging.error(f"An unexpected error '{e.__class__.__name__}' occurred during generation for sample {sample.index}: {e}", exc_info=True)
        sample.status = Sample.Status.ABORTED # 或 FAILED

    finally:
        # [关键] finally 块确保无论成功还是失败，这部分代码都会执行
        end_time = time.time()
        print(f"Finished generation for sample {sample.index} in {end_time - start_time:.4f}s. average round: {round_number}, average round time: {(end_time - start_time)/round_number}, response length: {sample.response_length} ")

        if sample.status != Sample.Status.COMPLETED and sample.status != Sample.Status.ABORTED:
            match output["meta_info"]["finish_reason"]["type"]:
                case "length":
                    # IF the last response is truncated, we should set the status to TRUNCATED
                    sample.status = Sample.Status.TRUNCATED
                case "abort":
                    sample.status = Sample.Status.ABORTED
                case "stop":
                    sample.status = Sample.Status.COMPLETED
    return sample

async def generate(args, sample: Sample, sampling_params) -> Sample:
    # generate a single sample
    state = GenerateState(args)
    # 这里就是基于 server 的含义
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    # Ensure the sample is in a state that allows generation
    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"
    # Adjust max_new_tokens based on existing response length
    if len(sample.response) > 0:
        # FIXME 为设么要减去 prompt 的长度呢？
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(
            state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        )

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload - shared structure
    payload = {
        "sampling_params": sampling_params,
        # MARK
        "return_logprob": args.use_token_output,
    }
    # 特点就是对于已经有 response 的 sample 和没有 response 的 sample 进行不同的处理
    # FIXME 什么是 token-based mode？
    if args.use_token_output:
        # Token-based mode: use tokens directly
        if len(sample.response) > 0:
            input_token_ids = sample.tokens
        else:
            # First turn: initialize with prompt tokens
            prompt_token_ids = state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
            input_token_ids = prompt_token_ids
            # Initialize sample.tokens with prompt for subsequent turns
            if not sample.tokens:  # Only set if empty
                sample.tokens = prompt_token_ids
        payload["input_ids"] = input_token_ids
    else:
        # String-based mode: original implementation
        input_text = sample.prompt + sample.response
        payload["text"] = input_text

    output = await post(url, payload, use_http2=args.use_http2)

    if args.use_token_output:
        # Extract new response tokens
        assert (
            "meta_info" in output and "output_token_logprobs" in output["meta_info"]
        ), "output_token_logprobs is not in the output"
        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]

        # Update sample with tokens directly - avoiding re-tokenization
        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response += state.tokenizer.decode(new_response_tokens, skip_special_tokens=False)
    else:
        # String-based processing
        sample.response += output["text"]
        prompt_tokens_ids = state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        response_token_ids = state.tokenizer(sample.response, add_special_tokens=False)["input_ids"]
        sample.tokens = prompt_tokens_ids + response_token_ids
        sample.response_length = len(response_token_ids)

    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED
    
    return sample


async def generate_and_rm(args, sample: Sample, sampling_params: dict, actor_model = None, evaluation=False, base_url = None) -> Sample:
    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        if args.custom_generate_function_path is not None:
            custom_generate_func = load_function(args.custom_generate_function_path)
            # 相当于给了一个自定义的 generate
            sample = await custom_generate_func(args, sample, actor_model, sampling_params)
        else:
            if evaluation:
                sample = await generate(args, sample, sampling_params)
            else:
                if args.use_verify:
                    sample = await spec_generate(args, sample, actor_model, sampling_params, base_url)
                else:
                    sample = await generate(args, sample, sampling_params)
            

    # [Change] 
    # FIXME how to deal with spec sample?
    # if sample.status == Sample.Status.SPECED:
    #     return sample
    
    # if sample.status == Sample.Status.ABORTED:
    #     return sample

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample
    # 生成结束够后进行 rm
    
    sample.reward = await async_rm(args, sample)

    return sample


async def generate_and_rm_group(args, group: list[Sample], actor_model, sampling_params: dict, evaluation=False, base_url=None) -> list[Sample]:
    # 从名字上来说是生成并且进行奖励, 相比较之前，这里使用 group 的方式进行处理
    state = GenerateState(args)
    
    if state.aborted:
        return group
    # MARK: sampling 是 copy 的
   
    group = await asyncio.gather(
        *[generate_and_rm(args, sample, sampling_params.copy(), actor_model = actor_model, evaluation=evaluation, base_url = base_url) for sample in group]
    )

    # for the rm that need the whole group, we will not do the rm here
    if not state.aborted and args.group_rm:
        # 还有基于 batch 的 rm
        # FIXME batch 和 group 的概念的区别是？
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards):
            sample.reward = reward

    return group


async def abort(args, rollout_id: int):
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True
    response = await get(
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers", use_http2=args.use_http2
    )

    # abort all the requests
    # MARK 对于所有的 worker 都发送 abort 的请求
    for url in response["urls"]:
        print(f"Abort request for {url}", flush=True)
        await post(f"{url}/abort_request", {"abort_all": True}, use_http2=False)

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                # FIXME how to set this data ?
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples += group
            count += len(group)

    if args.partial_rollout:
        print(f"Collected {count} partial samples into the data buffer", flush=True)

    return aborted_samples


async def generate_rollout_async(args, rollout_id: int, actor_model, data_source) -> list[list[Sample]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        list[list[Sample]]: a list of samples generated by the rollout, the length of the list is exactly the same as the `rollout_batch_size`
        FIXME: the meaning of the list
    """
    assert args.rollout_global_dataset

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    over_sampling_filter = (
        load_function(args.over_sampling_filter_path) if args.over_sampling_filter_path is not None else None
    )

    # target_data_size is the total number of valid samples to get
    target_data_size = args.over_sampling_batch_size if over_sampling_filter is not None else args.rollout_batch_size
    # MARK the differen between the over_samping and dynamic samping 
    data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        # meaning of remaining_batch_size
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples, actor_model)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                print(
                    f"First rollout sample: {[group[0].prompt + group[0].response]}, label: {group[0].label}, reward: {group[0].reward}",
                    flush=True,
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            if dynamic_filter is not None and not dynamic_filter(args, group):
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    print(
        f"Finish rollout: {[data[-1][0].prompt + data[-1][0].response]}, label: {data[-1][0].label}, reward: {data[-1][0].reward}",
        flush=True,
    )

    # there are still some unfinished requests, abort them
    await state.manager.abort()
    aborted_samples = await abort(args, rollout_id)
    


    if over_sampling_filter is not None:
        # MARK ovly get the first batch size data
        data = over_sampling_filter(args, data)[: args.rollout_batch_size]

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0].index)

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    return data, aborted_samples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args, rollout_id):
    assert not args.group_rm, "Group RM is not supported for eval rollout"
    results = {}
    for i in range(0, len(args.eval_prompt_data), 2):
        name, path = args.eval_prompt_data[i : i + 2]
        results.update(await eval_rollout_single_dataset(args, rollout_id, name, path))
    return results, []


async def eval_rollout_single_dataset(args, rollout_id, name, path):
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        name: str, the name of the dataset
        path: str, the path of the dataset
    """
    # FIXME why use rollout_id?
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    if name not in EVAL_PROMPT_DATASET:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[name] = Dataset(
            path,
            tokenizer=tokenizer,
            max_length=args.rollout_max_prompt_len,
            prompt_key=args.input_key if args.eval_input_key is None else args.eval_input_key,
            label_key=args.label_key if args.eval_label_key is None else args.eval_label_key,
            metadata_key=args.metadata_key,
            tool_key=args.tool_key if args.eval_tool_key is None else args.eval_tool_key,
            apply_chat_template=args.apply_chat_template,
        )
    dataset = EVAL_PROMPT_DATASET[name]

    sampling_params = dict(
        temperature=args.rollout_temperature if args.eval_temperature is None else args.eval_temperature,
        top_p=args.rollout_top_p if args.eval_top_p is None else args.eval_top_p,
        top_k=args.rollout_top_k if args.eval_top_k is None else args.eval_top_k,
        max_new_tokens=(
            args.rollout_max_response_len if args.eval_max_response_len is None else args.eval_max_response_len
        ),
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for i, prompt_sample in enumerate(dataset.samples):
        for j in range(args.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            tasks.append(
                generate_and_rm(
                    args,
                    sample,
                    sampling_params=sampling_params,
                    actor_model = None,
                    evaluation=True,
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc="Rollout generation", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            print([sample.prompt + sample.response], sample.reward, flush=True)
            do_print = False
        data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.reward_key or args.eval_reward_key
    return {
        name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
        }
    }

# here
# TODO remove this temp function
def generate_rollout(args, rollout_id, data_buffer, actor_model, evaluation=False):
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[list[Sample]]: a list of list of samples generated by the rollout
    """
    completed_samples, aborted_samples = generate_abortable_samples(
        args, rollout_id, data_buffer.get_samples, actor_model, evaluation=evaluation
    )
    # aborted sample 会被添加到 data_buffer 中
    data_buffer.add_samples(aborted_samples)
    return completed_samples


def generate_abortable_samples(args, rollout_id, data_source, actor_model, evaluation=False):
    assert args.rollout_global_dataset
    if evaluation:
        return run(eval_rollout(args, rollout_id))
    return run(generate_rollout_async(args, rollout_id, actor_model, data_source))
