
from transformers import AutoTokenizer

from slime.utils.async_utils import run
from slime.utils.types import Sample

import argparse
import logging
import torch






# if __name__ == "__main__":
#     # FIXME 单例模式应用  
#     root_logger = logging.getLogger()
#     root_logger.handlers = []  # 清空现有处理器
#     logging.basicConfig(format='%(asctime)s: %(message)s',level=logging.INFO)
#     logging.info("Start")  # 会输出到命令行

#     sglang_router_ip = "127.0.0.1"
#     sglang_router_port = 30000
#     use_http2 = False
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--hf_checkpoint", type=str, default="/root/Qwen3-4B")
#     parser.add_argument("--sglang_router_ip", type=str, default=sglang_router_ip)
#     parser.add_argument("--sglang_router_port", type=int, default=sglang_router_port)
#     parser.add_argument("--use_http2", type=bool, default=use_http2)
#     # parser.add_argument("--rollout_max_response_len", type=int, default=100)
#     parser.add_argument("--rollout_max_response_len", type=int, default=104)
#     parser.add_argument("--rollout_temperature", type=float, default=1.0)
#     parser.add_argument("--rollout_top_p", type=float, default=1.0)
#     parser.add_argument("--rollout_top_k", type=int, default=-1)
#     parser.add_argument("--rollout_stop", type=str, default=None)
#     parser.add_argument("--rollout_stop_token_ids", type=list, default=None)
#     # Whether to skip the special tokens during detokenization.
#     parser.add_argument("--rollout_skip_special_tokens", type=bool, default=False)

#     args = parser.parse_args()
#     tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
#     raw_prompt = [{'role': 'user', 'content': 'Where the capital of France? Not need think' }]
#     prompt = tokenizer.apply_chat_template(raw_prompt, None, tokenize=False, add_generation_prompt=True)
#      # Example usage
#     sample = Sample(prompt=prompt, index=0)
#     sampling_params = {
#         "temperature": 1.0,
#         "top_p": 1,
#         "top_k": -1,
#         "max_new_tokens": args.rollout_max_response_len,
#     }
#     # result = run(spec_generate(args, sample, sampling_params))
#     print(result)
