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
__all__ = ["generate_rollout"]

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
        for index, group in enumerate(samples):

            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
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

# [Change]
async def spec_generate(args, sample: Sample, actor_model, sampling_params, base_url) -> Sample:
    # 1. deal with initial status
    # logging.info(f"Generating sample {sample.index} with prompt: {sample.prompt}")
    state = GenerateState(args)
    # 2. generate a single sample
    # url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
   

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
    max_round_number = args.rollout_max_response_len // round_tokens + 10
    max_new_tokens = sampling_params["max_new_tokens"]
    while round_number < max_round_number:
        
        # 2.0 deal with the sample status
        # FIXME maybe not strict
        if sample.response_length >= args.rollout_max_response_len:
            break
        # 2.1 deal with data payload & sampling para
        sampling_params["max_new_tokens"] = min(round_tokens, max_new_tokens - sample.response_length)
        # logging.info(f"Round {round_number}: {sampling_params["max_new_tokens"]}")
        # print("vocab_size", state.config.vocab_size )
        payload = {
            "sampling_params": sampling_params,
            "return_logprob": True,
            "top_logprobs_num": state.config.vocab_size,   
            # "top_logprobs_num": 1,
            # "max_new_tokens" : min(10, args.rollout_max_response_len - sample.response_length),
            # "seed": 42
        }
        input_token_ids = sample.tokens
        payload["input_ids"] = input_token_ids
        # 2.2 post request
        
        output = await post(url, payload, use_http2=False)
        # print(output)
        # 3. deal with response
        # 3.1 deal with metadata
        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        
        # logging.info(f"New response tokens: {new_response_tokens}")
        # logging.info(f"New response: {state.tokenizer.decode(new_response_tokens, skip_special_tokens=False)}")
        temp_tokens = sample.tokens + new_response_tokens
        # temp_response_length = sample.response_length + len(new_response_tokens)
        temp_data = {
            "tokens": [temp_tokens],
            # FIXME is this necessary?
            "response_lengths": [len(new_response_tokens)],
            "sample_indices": [sample.index],
            "rollout_log_probs": [[t[0] for t in output["meta_info"]["output_token_logprobs"]]]
            
        }
        # print("rollout_log_probs:", temp_data["rollout_log_probs"][0])

        
        # FIXME didn't consider the async
        # list[Box[ref]]
        # print(output["meta_info"].keys())
        verification_res = ray.get(ray.get(actor_model.async_verification(0, Box(ray.put(temp_data)))[0]).inner)
        temp_data["rollout_full_log_probs"]  = [output["meta_info"]["output_top_logprobs"]]
        # logging.info(f"rollout_full_log_probs: {len(temp_data['rollout_full_log_probs'][0])}")
        # verification_res = {
        #     "recompute_index": min(sampling_params["max_new_tokens"], 7) - 1,
        #     "logits": torch.randn(state.config.vocab_size).tolist()
        # }
        # assert(verification_res["recompute_index"][0] == -1 or verification_res["recompute_index"][0] >= 0)
        # transform the index from global to local
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
            # FIXME data type
            response_recompute_index = verification_res["recompute_index"][0]
            # print(torch.tensor(temp_data["rollout_log_probs"][0][response_recompute_index]).shape)
            # MARK: target - approx
            new_distribuation = torch.softmax(torch.tensor(verification_res['logits'][0], dtype=torch.float32), dim = -1) - recovery_pros(temp_data["rollout_full_log_probs"][0][response_recompute_index])
            new_distribuation = torch.clamp(new_distribuation,  min = 0)
            recompute_ids = sample_from_the_logits(new_distribuation, sampling_params).item()
            response_recompute_index = verification_res["recompute_index"][0]
            accepted_tokens = new_response_tokens[:response_recompute_index] + [recompute_ids]
        print(f"Round {round_number}, recompute index: {verification_res['recompute_index']}, recompute token id: {recompute_ids}, accepted tokens: {len(accepted_tokens)}, response_length: {sample.response_length}")
        # logging.info(f"Recompute index: {verification_res['recompute_index']}")
        # logging.info(f"Recompute token id: {recompute_ids}")
        # logging.info(f"Accepted tokens: {accepted_tokens}")
        
        # print(f"Output: {accepted_tokens}")
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
    # print(sample.response)
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
                sample = sample = await generate(args, sample, sampling_params)
            else:
                sample = sample = await spec_generate(args, sample, actor_model, sampling_params, base_url)
            

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
