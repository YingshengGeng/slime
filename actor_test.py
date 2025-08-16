from slime.utils.types import Sample
import logging
import torch
import torch.distributed as dist
import ray
from megatron.core import mpu
from megatron.core.utils import get_model_config
from slime.utils.ray_utils import Box
from megatron.core.models.gpt import GPTModel
from slime.ray.placement_group import _create_placement_group, create_actor_group
from slime.backends.megatron_utils.actor import *
from slime.ray.actor_group import RayTrainGroup
from slime.utils.arguments import parse_args
from slime.backends.megatron_utils.data import *
from typing import Optional
from megatron.core.pipeline_parallel import get_forward_backward_func
from functools import partial
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

import asyncio
import time
from asyncio import Future, Queue
from typing import List, Tuple, Dict, Any

# class BatchingManager:
#     def __init__(self, urls: List[str], actor_model, max_batch_size: int = 128, batch_timeout: float = 10, recomputation_batch_timeout:float = 10):
#         """
#         初始化批处理管理器。

#         Args:
#             urls (List[str]): 提供服务的多个推理服务器 URL 列表。
#             max_batch_size (int): 每个批处理的最大请求数。
#             batch_timeout (float): 等待填满一个批处理的最大时间（秒）。
#         """
#         # [MODIFIED] 为不同类型的请求和每个 URL 创建独立的队列字典
#         urls = [url + '/generate' for url in urls]
#         self.recomputation_queues: Dict[str, Queue] = {url: Queue() for url in urls}
#         self.rollout_queues: Dict[str, Queue] = {url: Queue() for url in urls}
#         self.actor_queue: Queue = Queue()
#         # FIXME
#         self.actor_model = actor_model


#         self.rollout_max_batch_size = max_batch_size
#         self.rollout_batch_timeout = batch_timeout

#         self.recomputation_max_batch_size = 32
#         self.recomputation_batch_timeout = recomputation_batch_timeout
        
#         self.verification_max_batch_size = 16
#         self.verification_batch_timeout = batch_timeout


#         self.worker_tasks = []

#         # [MODIFIED] 为每一个队列创建一个专属的后台 worker 任务
#         for url in urls:
#             self.worker_tasks.append(
#                 asyncio.create_task(
#                     self._batching_worker(url, self.recomputation_queues[url], 1)
#                 )
#             )
#             self.worker_tasks.append(
#                 asyncio.create_task(
#                     self._batching_worker(url, self.rollout_queues[url], 0)
#                 )
#             )
#         self.worker_tasks.append (
#             asyncio.create_task(
#                 self._verification_worker(self.actor_queue, self.actor_model)
#             )
#         )

#     async def submit_request(self, url: str, payload: dict, existing_task_num) -> dict:
#         """
#         提交单个请求负载到相应的队列，并等待结果。
#         """
#         future = Future()
        
#         if payload.get("sampling_params", {}).get("max_new_tokens") == 1:
#             await self.recomputation_queues[url].put((payload, future))
#             self.recomputation_max_batch_size = min(self.recomputation_max_batch_size, max(existing_task_num/4, 1))
#         else:
#             await self.rollout_queues[url].put((payload, future))
#             self.rollout_max_batch_size = min(self.rollout_max_batch_size, max(existing_task_num,1))
            
#         return await future
    
#     async def submit_actor_request(self, temp_data: dict, existing_task_num) -> dict:
#         """
#         提交单个请求负载到 actor 队列，并等待结果。
#         """
#         future = Future()
#         await self.actor_queue.put((temp_data, future))
#         self.verification_max_batch_size = min(self.verification_max_batch_size, max(existing_task_num, 1))
#         return await future
    
#     def _merger_request_batch(self, requests: List[Tuple[dict, Future]]) -> Dict[str, Any]:

#         batched_data = {
#             "tokens": [],
#             "response_lengths": [],
#             "sample_indices": [],
#             "rollout_log_probs": []
#         }

#         for data, _ in requests:
#             # Assumes each data dict contains lists of size 1
#             # FIXME list[list[data]] -> list[data]
#             batched_data["tokens"].extend(data["tokens"])
#             batched_data["response_lengths"].extend(data["response_lengths"])
#             batched_data["sample_indices"].extend(data["sample_indices"])
#             batched_data["rollout_log_probs"].extend(data["rollout_log_probs"])
#         return batched_data

#     def _split_results(self, batched_results: List[Dict[str, Any]], num_requests: int) -> List[Dict[str, Any]]:
#         """Splits the batched verification result back into individual results."""
#         individual_results = []
#         for mini_batch_result in batched_results:
#             num = len(mini_batch_result["recompute_index"])
#             for i in range(num):
#                 result = {
#                     # Assuming the result keys match this structure
#                     "recompute_index": [mini_batch_result["recompute_index"][i]],
#                     "logits": [mini_batch_result["logits"][i]]
#                 }
#                 individual_results.append(result)
#         assert(len(individual_results) == num_requests)
#         return individual_results


#     async def _verification_worker(self, queue: Queue, actor_model):
#         """Background worker to collect, batch, verify, and distribute results."""
#         while True:
#             batch_size = self.verification_max_batch_size
#             batch_timeout = self.verification_batch_timeout
#             requests: List[Tuple[Dict[str, Any], Future]] = []
#             actor_model = self.actor_model
#             # Wait for the first request to start a new batch
#             # try:
#             first_data, first_future = await queue.get()
#             print("first_data")
#             requests.append((first_data, first_future))
#             # except asyncio.CancelledError:
#                 # break

#             # Collect more requests until the batch is full or timeout occurs
#             while len(requests) < batch_size:
#                 try:
#                     data, future = await asyncio.wait_for(queue.get(), timeout=1000)
#                     requests.append((data, future))
#                 except (asyncio.TimeoutError, asyncio.CancelledError):
#                     break

#             if not requests:
#                 continue 

#             print("enough")
#             # 1. Merge all requests into a single, large payload
#             batched_data = self._merger_request_batch(requests)
#             print(batched_data["response_lengths"])
#             print(f"VERIFICATION WORKER: Dispatching a merged batch of {len(requests)} requests.")

#             # 2. Send the single, large batch to the Ray actor
#             # This is the core logic change
#             box_list = ray.get(actor_model.async_verification(0, ray.put(Box(ray.put(batched_data)))))
#             # list[Box[Object]], Obj [dict[str, list[Any]]]
#             print("fix--------------")
#             print(box_list)
#             batched_results = [ray.get(b.inner) for b in box_list]
#             # FIXME more general
#             batched_results = [batched_results[0], batched_results[2]]
#             # batched_results = ray.get(ray.get(box_list[0]).inner)
#             print("finish here")
#             # 3. Split the batched result back into individual results
#             individual_results = self._split_results(batched_results, len(requests))

#             # 4. Distribute the individual results back to the waiting futures
#             for i, result in enumerate(individual_results):
#                 original_future = requests[i][1]
#                 original_future.set_result(result)
#             print(f"VERIFICATION WORKER: completed and Distributed results to {len(requests)} requests.")


#     async def _batching_worker(self, url: str, queue: Queue, tag: int):
#         """
#         核心后台任务：收集请求，然后使用 asyncio.gather 并发调度它们。
#         """
        
#         while True:
#             if tag == 0:
#                 batch_size = self.rollout_max_batch_size
#                 batch_timeout = self.rollout_batch_timeout
#             else:
#                 batch_size = self.recomputation_max_batch_size
#                 batch_timeout = self.recomputation_batch_timeout
#             requests: List[Tuple[dict, Future]] = []
            
#             try:
#                 first_payload, first_future = await queue.get()
#                 requests.append((first_payload, first_future))
#             except asyncio.CancelledError:
#                 break

#             while len(requests) < batch_size:
#                 try:
#                     # FIXME 
#                     payload, future = await asyncio.wait_for(queue.get(), timeout=batch_timeout)
#                     requests.append((payload, future))
#                 except (asyncio.TimeoutError, asyncio.CancelledError):
#                     break
            
#             if not requests:
#                 continue

#             batch_size = len(requests)
#             print(f"WORKER tag{tag} ({url[-1]}):  {batch_size} requests. Dispatching concurrently...")

#             futures = [req[1] for req in requests]

#             # [!!!] 核心修改：为批次中的每个请求创建一个 post 任务
#             # 我们不再合并 payload，而是为每个 payload 创建一个独立的 post 调用
#             post_tasks = [
#                 post(url, req[0], use_http2=False) for req in requests
#             ]

#             # [!!!] 使用 asyncio.gather 并发执行所有独立的 post 请求
#             # return_exceptions=True 确保一个请求失败不会导致整个 gather 崩溃
#             concurrent_outputs = await asyncio.gather(*post_tasks, return_exceptions=True)
            
#             print(f"WORKER tag{tag} ({url[-1]}): Batch of {batch_size} completed. Distributing results.")
            
#             # 将结果分发回等待的 future
#             for i, output in enumerate(concurrent_outputs):
#                 futures[i].set_result(output)


#     def abort(self):
#         """
#         Abort all pending requests and stop the batching workers.
#         """
#         for task in self.worker_tasks:
#             task.cancel()
#         self.worker_tasks = []
#         self.recomputation_queues.clear()
#         self.rollout_queues.clear()
#         self.actor_queue = Queue()


class DataIterator:
    def __init__(
        self,
        rollout_data,
        micro_batch_size: Optional[int] = None,
        micro_batch_indices: Optional[list[list[int]]] = None,
    ):
        self.rollout_data = rollout_data
        self.micro_batch_size = micro_batch_size
        self.micro_batch_indices = micro_batch_indices
        assert micro_batch_size is None or micro_batch_indices is None
        self.offset = 0

    def get_next(self, keys):
        batch = {}
        for key in keys:
            vals = self.rollout_data.get(key, None)
            if vals is None:
                batch[key] = None
            else:
                if self.micro_batch_indices is not None:
                    # print(self.offset, self.micro_batch_indices,self.micro_batch_size)
                    indices = self.micro_batch_indices[self.offset]
                    batch[key] = [vals[i] for i in indices]
                else:
                    # FIXME not reguler shape
                    assert self.offset + self.micro_batch_size <= len(
                        vals
                    ), f"offset: {self.offset}, micro_batch_size: {self.micro_batch_size}, len(vals): {len(vals)}, rank: {dist.get_rank()}, key:{key}"
                    batch[key] = vals[self.offset : self.offset + self.micro_batch_size]

        if self.micro_batch_indices is not None:
            self.offset += 1
        else:
            self.offset += self.micro_batch_size
        return batch

    def reset(self):
        self.offset = 0
        return self


def get_ver_data_iterator(args, model, rollout_data):
    """
    Creates data iterators for training and log probability evaluation, supporting both static and dynamic batch sizes,
    with optional virtual pipeline parallelism and sequence length balancing.
    Args:
        args: An object containing configuration parameters, including batch sizes, micro batch sizes,
              dynamic batch size usage, and maximum tokens per GPU et.al.
        model: The model or list of model stages, used to extract configuration for parallelism.
        rollout_data: A dictionary containing rollout data, including 'total_lengths' for each sample.
    Returns:
        tuple: A tuple containing:
            - data_iterator: List of DataIterator objects for log probability evaluation.
            - num_microbatches: Number of microbatches for log probability evaluation.
    """

    num_local_gbs = len(rollout_data["response_lengths"])
    # print("num_local_gbs: ", num_local_gbs)
    # // mpu.get_data_parallel_world_size(with_context_parallel=False)
    # the gradient?
    num_steps_per_rollout = 1

    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
    if vpp_size is None:
        vpp_size = 1

    def _generate_data_iterator(rollout_data, micro_batch_size, micro_batch_indices=None):
        data_iterator = []
        for _ in range(vpp_size):
            data_iterator.append(DataIterator(rollout_data, micro_batch_size, micro_batch_indices))
        return data_iterator
    # if micro = 1, num_local_gbs = num_local_gbs
    num_microbatches = [num_local_gbs // args.micro_batch_size for _ in range(num_steps_per_rollout)]
    data_iterator = _generate_data_iterator(rollout_data, args.micro_batch_size)
    # print("num_microbatches: ", num_microbatches)
    return (
        data_iterator,
        num_microbatches,
    )

    
    
def async_generate(self, rollout_id, evaluation=False):
    return self.data_buffer.generate.remote(rollout_id, self.actor_model, evaluation=evaluation)



def do_verification(self, rollout_id, rollout_data_ref):
    # 1. Get rollout data.
    # print("do_verification start")
    # print(f"do_verification rank{dis/t.get_rank()}")
    rollout_data = self.get_verfication_data(rollout_data_ref,
                        mpu.get_data_parallel_rank(with_context_parallel=False),
                        mpu.get_data_parallel_world_size(with_context_parallel=False)
    )

    # print(rollout_data)
    # Create data iterator for log_probs and train.
    
    data_iterator, num_microbatches = get_ver_data_iterator(self.args, self.model, rollout_data)
    # print("num_microbatches", num_microbatches)
    # print("len", len(rollout_data["response_lengths"]))
    if (len(rollout_data["response_lengths"]) == 0):
        # empty data
        recompute_data = {
            "recompute_index": [],
            "recompute_ids": [],
            "logits": [],
        }
        return Box(ray.put(recompute_data))
    # print(self.weights.keys(), self.weights["actor"].keys())
    # 2. Compute log probabilities.
    # print(num_microbatches)
    rollout_data.update(
        self.compute_log_prob (
            "actor",
            data_iterator,
            num_microbatches,
            store_prefix="",
            is_veri=True,  # FIXME
        )
    )
    # print("rollout_data num: ", len(rollout_data["log_probs"]))
    # 3. use log probs to compute difference
    recompute_index_list = []
    with torch.no_grad():
        data_len = len(rollout_data["response_lengths"])
        for k in range(data_len):
            diff = rollout_data["log_probs"][k] - rollout_data["rollout_log_probs"][k]
            recompute_index = -1
            for i in range(rollout_data["response_lengths"][k]):
                r = torch.rand(1, device=torch.cuda.current_device())
                if r > torch.exp(diff[i]).item():
                    # reject
                    recompute_index = i
                    break
            recompute_index_list.append(recompute_index)
    rollout_data.update(
        {"recompute_index": recompute_index_list}
    )
    # print("recompute_index_list: ", recompute_index_list)
    new_data_iterator, new_num_microbatches = get_ver_data_iterator(self.args, self.model, rollout_data)
    # 3.1 get the full logits for recompute_index
    rollout_data.update(
        self.forward_for_logits(
            self.args,
            "actor",
            new_data_iterator,
            new_num_microbatches,
            store_prefix="",
        )
    )
     

    # 4. return rollout data with log probabilities.
    # FIXME how to return, the structre and data type
    """
    return Box(
        {
            "logits": list[list[float]]
            "recompute_index": list[int],
            "recompute_ids": list[int],
        }
    )
    """
    recompute_data = {
        "recompute_index": recompute_index_list,
        "recompute_ids": [0] * len(recompute_index_list),
        "logits": [data.cpu().tolist() for data in rollout_data["logits"]],
        # "log_probs": [data.cpu().tolist() for data in rollout_data["log_probs"]],
    }
    # if dist.get_rank() == 0:
    #     # print(recompute_data)   
    #     print("vocab_size: ", self.args.vocab_size)
    #     print("log_probs: ", rollout_data["log_probs"][0])
    #     print("rollout_log_probs: ", rollout_data["rollout_log_probs"][0])
    #     print("logits: ", rollout_data["logits"][0].shape)
    return Box(ray.put(recompute_data))


def get_verfication_data(self, rollout_data_ref, dp_rank, dp_size):
    verification_data = {}
    # dp_rank = 1
    # dp_size = 1
    # print("dp_rank, dp_size: ", dp_rank, dp_size)
    # receive data
    rank = dist.get_rank()
    # print("start rank: ", rank)
    if rank == 0:
        data = ray.get(rollout_data_ref.inner)
        # print("len_data_len: ", len(data['response_lengths']))
        dist.broadcast_object_list([data], src=0)
    else:
        data = [None]
        dist.broadcast_object_list(data, src=0)
        data = data[0]
    # print("end rank ", rank, " get data from broadcast")
    # print("data: ", data)
    # FIXME why not check not?
    # assert data is not None
    # FIXME does not deal with rewards, adv
    print(f"{data["response_lengths"]}, rank: {dist.get_rank()}")
    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths
    
    # FIXME dose not deal with balance data
    # FIXME only the first data parallel rank will receive the data
    def get_partition(val):
        return val[dp_rank::dp_size]
    
    for key in [
        "tokens",
        "total_lengths",
        "response_lengths",
        "sample_indices",
        "rollout_log_probs",
    ]:
        if key not in data:
            continue
        val = get_partition(data[key])
        # move tokens to GPU in advance
        if key == "tokens":
            val = [torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device()) for t in val]
        elif key in ["rollout_log_probs"]:
            val = [torch.tensor(t, dtype=torch.float, device=torch.cuda.current_device()) for t in val]
            
        verification_data[key] = val

    return verification_data
        

@torch.no_grad()
def forward_for_logits(self, args, model_tag, data_iterator, num_microbatches, store_prefix="",):
    # 0. update the weights
    self.update_gpu_params_dict(self.weights[model_tag])
    # """Only do the forward pass and calculate the logprob."""
    # 1. prepare model & data & func
    # 1.1 reset data iterator
    # FIXME why reset for all iterators?
    for iterator in data_iterator:
        iterator.reset()
    # 1.2 prepare model
    model = self.model
    # 1.3 prepare func
    # FIXME how this func is used in parallel setting?
    def forward_step(data_iterator, model: GPTModel):
        """Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTModel): The GPT Model
        """

        # Get the batch.
        batch = get_batch(data_iterator, ["tokens", "total_lengths", "response_lengths", "recompute_index"])
        # print(batch)
        unconcat_tokens = batch["unconcat_tokens"]
        tokens = batch["tokens"]
        packed_seq_params = batch["packed_seq_params"]
        total_lengths = batch["total_lengths"]
        response_lengths = batch["response_lengths"]
        recompute_index_list = batch["recompute_index"]
        output_tensor = model(
            input_ids=tokens,
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=packed_seq_params,
        )

        return output_tensor, partial(
            get_full_logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            recompute_index_list = recompute_index_list,
            vocab_size = args.vocab_size 
        )
    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    forward_backward_func = get_forward_backward_func()
    forward_data_store = []
    num_steps_per_rollout = 1
    for step_id in range(num_steps_per_rollout):
        # collect_non_loss_data
        forward_data_store += forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches[step_id],
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=True,
            collect_non_loss_data=True,
        )

    # 3. Move model back to the train mode.
    for model_module in model:
        model_module.train()

    # 4. process output
    rollout_data = {}
    # Store the results on the last stage
    if mpu.is_pipeline_last_stage():
        # FIXME structure of forward_data_store
        keys = forward_data_store[0].keys()
        for key in keys:
            values = []
            for value in forward_data_store:
                assert isinstance(value[key], list)
                values += value[key]

            rollout_data[f"{store_prefix}{key}"] = values
    return rollout_data


def get_full_logits(
    logits: torch.Tensor,
    *,
    args,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    recompute_index_list: list[int],
    vocab_size: int,
    non_loss_data: bool = True,
) -> dict[str, list[torch.Tensor]]:
    # not consider CP
    assert logits.size(0) == 1, f"{logits.shape}"
    assert logits.dtype == torch.float32, f"{logits.dtype}"

    logits = logits.squeeze(0)
    logits.div_(args.rollout_temperature)

    cp_size = mpu.get_context_parallel_world_size()
    assert cp_size == 1, f"{cp_size}"

    logits_list = []
    end = 0
    assert len(unconcat_tokens) == len(total_lengths) == len(response_lengths) == len(recompute_index_list), \
        f"unconcat_tokens: {len(unconcat_tokens)}, total_lengths: {len(total_lengths)}, response_lengths: {len(response_lengths)}, recompute_index_list: {len(recompute_index_list)}, rank: {dist.get_rank()}"  
    for tokens, total_length, response_length, recompute_index in zip(unconcat_tokens, total_lengths, response_lengths, recompute_index_list):
        end += total_length
        start = end - response_length
        # If -1, we compute the next token, else for the index token
        if recompute_index != -1:
            # left shift one
            logits_chunk = logits[start - 1 + recompute_index - 1: start - 1 + recompute_index + 1]
        else:
            # FIXME take two for maintain shape
            logits_chunk = logits[end - 2: end]
        logits = calculate_full_logits(logits_chunk) 
        # FIXME 
        # print("logits shape: ", logits.shape)
        logits_list.append(logits[-1][0][:vocab_size])
        
    return {"logits": logits_list}


def calculate_full_logits(logits):
    if logits.size(0) != 0:
        full_logits = gather_full_logits(logits.clone())
    else:
        full_logits = logits.new_zeros((0,))
    return full_logits


def gather_full_logits(logits: torch.Tensor):
    logits = logits.unsqueeze(1)
    #FIXME how this gather right part of logits
    full_logits = gather_from_tensor_model_parallel_region(logits)
    return full_logits

async def test():
    # 单例模式应用  
    root_logger = logging.getLogger()
    root_logger.handlers = []  # 清空现有处理器
    logging.basicConfig(format='%(asctime)s: %(message)s',level=logging.INFO)
    logging.info("Start")  # 会输出到命令行

    # 0.准备函数
    MegatronTrainRayActor.get_verfication_data = get_verfication_data
    MegatronTrainRayActor.do_verification = do_verification
    # MegatronTrainRayActor.compute_log_prob = compute_log_prob
    # RayTrainGroup.async_verification = async_verification
    MegatronTrainRayActor.forward_for_logits = forward_for_logits

    
    # FIXME the dropout in probs ?
    # 1. 创建参数
    args = parse_args()
    debug_train_only = True
    # args.actor_num_nodes = 1
    # args.actor_num_gpus_per_node = 1
    args.actor_num_microbatches = 8
    args.hf_checkpoint = "/root/Qwen3-4B"
    # args.tensor_model_parallel_size = 1
    # FIXME if single data exceed max_tokens_per_gpu ？
    args.max_tokens_per_gpu = 2048
    # 2. 创建资源
    num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    # FIXME why reorder
    pg, actor_pg_reordered_bundle_indices = _create_placement_group(num_gpus)
    pgs = {
        "actor": (pg, actor_pg_reordered_bundle_indices),
    }
    # 3. 基于资源初始化 actor group (A group of actor models)
    actor_model = create_actor_group(args, pgs["actor"], wandb_run_id=None)
    start_rollout_ids = ray.get(
        actor_model.async_init(args, role="actor", with_ref= False)
    )
    manager = BatchingManager([], actor_model=actor_model)
    # 4. 准备数据 (Only one sample in batch)
    rollout_id = start_rollout_ids[0]
    # """
    # 测试函数，模拟异步调用。
    # """
    # print("Test Start")
    raw_data = {
        "tokens": [[1,2,3,4,5,6,7,8,9,10]],
        "response_lengths": [5 ],
        "sample_indices": [0 ],
        "rollout_log_probs": [[-0.1]*5 ],
    }
    tasks = [manager.submit_actor_request(raw_data, 16) for _ in range(16)]
    batched_results = await asyncio.gather(*tasks)
    batched_results = [batched_results[0], batched_results[2]]
    # data_num = 7
    # raw_data = {
    #     "tokens": [[1,2,3,4,5,6,7,8,9,10] for i in range(data_num)],
    #     "response_lengths": [5 for i in range(data_num)],
    #     "sample_indices": [0 for i in range(data_num)],
    #     "rollout_log_probs": [[-0.1]*5  for i in range(data_num) ],
    # }
    # raw_data = manager._merger_request_batch([(raw_data, Future()) for _ in range(data_num)])
    # rollout_data_ref = ray.put(Box(ray.put(raw_data)))
    # # 5. 调用 forward 计算 log prob
    # box_list = ray.get(actor_model.async_verification(rollout_id, rollout_data_ref))
    # batched_results = [ray.get(box_list[i].inner) for i in range(len(box_list))]
    # batched_results = [batched_results[0], batched_results[2]]
    # FIXME how the logical change here
    print("res: ", len(batched_results), len(batched_results[0]["recompute_index"]), len(batched_results[1]["recompute_index"]))


    

if __name__ == "__main__":
#    
    asyncio.run(test())
   
