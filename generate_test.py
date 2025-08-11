import asyncio
import copy

from tqdm import tqdm
from transformers import AutoTokenizer

from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.http_utils import get, post
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.types import Sample
# import ray
import argparse
import logging
__all__ = ["generate_rollout"]
class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args):
        # persistant state for the generation process
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        # FIXME
        # self.semaphore = asyncio.Semaphore(
        #     args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        # )
        
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

    def submit_generate_tasks(self, samples: list[list[Sample]]):
        # 这里的提交是以 groups 的形式提交的，每个 group 相当于一个 batch (什么级别？)
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)
    def check_match_eos(self, last_token_id: int) -> bool:
        matched_eos = False
        if self.tokenizer is not None:
            matched_eos |= last_token_id == self.tokenizer.eos_token_id
        return matched_eos
# [Change]
async def spec_generate(args, sample: Sample, sampling_params) -> Sample:
    # 1. deal with initial status
    logging.info(f"Generating sample {sample.index} with prompt: {sample.prompt}")
    state = GenerateState(args)
    # 2. generate a single sample
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    # didn't consider the partial rollout here
    prompt_tokens_ids = state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
    sample.tokens = prompt_tokens_ids
    # didn't consider the loss mask here
    round_number = 0
    max_round_number = args.rollout_max_response_len // 10 + 10
    max_new_tokens = sampling_params["max_new_tokens"]
    while round_number < max_round_number:
        
        # 2.0 deal with the sample status
        # FIXME maybe not strict
        if sample.response_length >= args.rollout_max_response_len:
            break
        # 2.1 deal with data payload & sampling para
        sampling_params["max_new_tokens"] = min(10, max_new_tokens - sample.response_length)
        logging.info(f"Round {round_number}: {sampling_params["max_new_tokens"]}")
        payload = {
            "sampling_params": sampling_params,
            "return_logprob": True,
            # "top_logprobs_num": state.tokenizer.vocab_size,   
            "top_logprobs_num": 1,
            # "max_new_tokens" : min(10, args.rollout_max_response_len - sample.response_length),
            "seed": 42
        }
        input_token_ids = sample.tokens
        payload["input_ids"] = input_token_ids
        # 2.2 post request
        output = await post(url, payload, use_http2=args.use_http2)
        
        # 3. deal with response
        # 3.1 deal with metadata
        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        
        logging.info(f"New response tokens: {new_response_tokens}")
        logging.info(f"New response: {state.tokenizer.decode(new_response_tokens, skip_special_tokens=False)}")
        temp_tokens = sample.tokens + new_response_tokens
        temp_response_length = sample.response_length + len(new_response_tokens)
        temp_data = {
            "tokens": [temp_tokens],
            # is this necessary?
            "response_lengths": [temp_response_length],
            "sample_indices": [sample.index],
            "rollout_log_probs": [output["meta_info"]["output_token_logprobs"]]
        }
        # FIXME didn't consider the async
        # verification_res = ray.get(actor.do_verification(rollout_id, temp_data))
        verification_res = {
            "recompute_index": sample.response_length + min(sampling_params["max_new_tokens"], 7) - 1,
            "recompute_ids": 0
        }
        assert(verification_res["recompute_index"] == -1 or verification_res["recompute_index"] >= sample.response_length)
        # transform the index from global to local
        if verification_res["recompute_index"] == -1:
            # if all accepted, and not exceed the max or eos,
            # we can just append the new response tokens
            if sample.response_length + len(new_response_tokens) < args.rollout_max_response_len and \
            not state.check_match_eos(new_response_tokens[-1]):
                accepted_tokens = new_response_tokens + [verification_res["recompute_ids"]]
            else:
                accepted_tokens = new_response_tokens
        else:
            # if not all accepted, we need to reset the new response tokens
            response_recompute_index = verification_res["recompute_index"] - sample.response_length
            response_recompute_id = verification_res["recompute_ids"]
            accepted_tokens = new_response_tokens[:response_recompute_index] + [response_recompute_id]
        logging.info(f"Accepted tokens: {accepted_tokens}")
        # print(f"Output: {accepted_tokens}")
        sample.tokens = sample.tokens + accepted_tokens
        sample.response_length += len(accepted_tokens)
        sample.response += state.tokenizer.decode(accepted_tokens, skip_special_tokens=False)
        
        if state.check_match_eos(accepted_tokens[-1]):
            sample.status = Sample.Status.COMPLETED
            break
        round_number += 1
    # 3.2 deal with sample status
    # FIXME how to deal with truncated max or truncated spec
    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            # IF the last response is truncated, we should set the status to TRUNCATED
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED
    return sample




if __name__ == "__main__":
    # 单例模式应用  
    root_logger = logging.getLogger()
    root_logger.handlers = []  # 清空现有处理器
    logging.basicConfig(format='%(asctime)s: %(message)s',level=logging.INFO)
    logging.info("Start")  # 会输出到命令行

    sglang_router_ip = "127.0.0.1"
    sglang_router_port = 30000
    use_http2 = False
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_checkpoint", type=str, default="/root/Qwen3-4B")
    parser.add_argument("--sglang_router_ip", type=str, default=sglang_router_ip)
    parser.add_argument("--sglang_router_port", type=int, default=sglang_router_port)
    parser.add_argument("--use_http2", type=bool, default=use_http2)
    # parser.add_argument("--rollout_max_response_len", type=int, default=100)
    parser.add_argument("--rollout_max_response_len", type=int, default=104)
    parser.add_argument("--rollout_temperature", type=float, default=1.0)
    parser.add_argument("--rollout_top_p", type=float, default=1.0)
    parser.add_argument("--rollout_top_k", type=int, default=-1)
    parser.add_argument("--rollout_stop", type=str, default=None)
    parser.add_argument("--rollout_stop_token_ids", type=list, default=None)
    # Whether to skip the special tokens during detokenization.
    parser.add_argument("--rollout_skip_special_tokens", type=bool, default=False)

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    raw_prompt = [{'role': 'user', 'content': 'Where the capital of France? Not need think' }]
    prompt = tokenizer.apply_chat_template(raw_prompt, None, tokenize=False, add_generation_prompt=True)
     # Example usage
    sample = Sample(prompt=prompt, index=0)
    sampling_params = {
        "temperature": 1.0,
        "top_p": 1,
        "top_k": -1,
        "max_new_tokens": args.rollout_max_response_len,
    }
    result = run(spec_generate(args, sample, sampling_params))
    print(result)
