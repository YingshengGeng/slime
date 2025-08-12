import asyncio
import copy

from tqdm import tqdm
from transformers import AutoTokenizer

from slime.utils.async_utils import run
from slime.utils.types import Sample
import argparse
import logging
import torch
import torch.distributed as dist
import ray
from megatron.core import mpu
from megatron.core.utils import get_model_config
from slime.utils.ray_utils import Box
from megatron.core.models.gpt import GPTModel
from slime.ray.placement_group import _create_placement_group, create_actor_group
from slime.backends.megatron_utils.actor  import MegatronTrainRayActor
from slime.ray.actor_group import RayTrainGroup
from slime.utils.arguments import parse_args
from slime.backends.megatron_utils.data import *
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
    num_local_samples = 1 # FIXME
    num_local_gbs = 1 # FIXME
    # the gradient?
    num_steps_per_rollout = 1 # FIXME

    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
    if vpp_size is None:
        vpp_size = 1

    def _generate_data_iterator(rollout_data, micro_batch_size, micro_batch_indices=None):
        data_iterator = []
        for _ in range(vpp_size):
            data_iterator.append(DataIterator(rollout_data, micro_batch_size, micro_batch_indices))
        return data_iterator

    if not args.use_dynamic_batch_size:
        num_microbatches = [num_local_gbs // args.micro_batch_size for _ in range(num_steps_per_rollout)]
        data_iterator = _generate_data_iterator(rollout_data, args.micro_batch_size)
    else:
        assert args.max_tokens_per_gpu is not None
        # calculate the number of mirobatches for each step
        cp_size = mpu.get_context_parallel_world_size()
        samples = rollout_data["total_lengths"]
        # assert len(samples) == num_local_samples
        num_microbatches = []
        for i in range(num_steps_per_rollout):
            start, end = i * num_local_gbs, (i + 1) * num_local_gbs
            # MARK in this range, get samples not exceeding max_tokens_per_gpu
            num_microbatches.append(
                get_minimum_num_micro_batch_size(samples[start:end], args.max_tokens_per_gpu, cp_size)
            )
        # FIXME; why use max?
        num_microbatches = torch.tensor(num_microbatches, dtype=torch.int, device=torch.cuda.current_device())
        dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=mpu.get_data_parallel_group())

        # vpp requies the number of microbatches to be divisible by vpp_size
        config = get_model_config(model[0])
        if config.microbatch_group_size_per_vp_stage:
            num_microbatches = torch.clamp(
                num_microbatches
                // config.microbatch_group_size_per_vp_stage
                * config.microbatch_group_size_per_vp_stage,
                min=1,
            )

        num_microbatches = num_microbatches.tolist()

        # balance the each micro batch
        samples = rollout_data["total_lengths"]
        if len(samples) !=  0:
            # balance the number of mirobatches across steps
            micro_batch_indices = []
            for i, num_mbs in enumerate(num_microbatches):
                start, end = i * num_local_gbs, (i + 1) * num_local_gbs
                samples = rollout_data["total_lengths"][start:end]
                partitions = get_seqlen_balanced_partitions(samples, num_mbs, equal_size=False)
                for j in range(num_mbs):
                    for k in range(len(partitions[j])):
                        partitions[j][k] += start
                micro_batch_indices.extend(partitions)

            assert len(set(sum(micro_batch_indices, []))) == num_local_samples
        else:
            micro_batch_indices = []
        data_iterator = _generate_data_iterator(rollout_data, None, micro_batch_indices)

    return (
        data_iterator,
        num_microbatches,
    )


def async_verification(self,rollout_id, rollout_data_ref):
        return [actor.do_verification.remote(rollout_id, rollout_data_ref) for actor in self._actor_handlers]

def do_verification(self, rollout_id, rollout_data_ref):
    # 1. Get rollout data.
    rollout_data = self._get_verfication_data(rollout_data_ref)
    # Create data iterator for log_probs and train.
    data_iterator, num_microbatches = get_ver_data_iterator(self.args, self.model, rollout_data)

    # 2. Compute log probabilities.
    rollout_data.update(
        self.compute_log_prob (
            "actor",
            data_iterator,
            num_microbatches,
            store_prefix="",
        )
    )
    # 3. use log probs to compute difference
    diff = rollout_data["log_probs"] - rollout_data["rollout_log_probs"]
    recompute_index = -1
    with torch.no_grad():
        for k in range(len(rollout_data["total_lengths"])):
            prefix_len = rollout_data["total_lengths"][k] - rollout_data["response_lengths"][k]
            for i in range(rollout_data["response_lengths"][k]):
                r = torch.rand(1, device=torch.cuda.current_device())
                if r > torch.exp(diff[prefix_len + i]).item():
                    # reject
                    recompute_index = prefix_len + i
                    break
    rollout_data.update(
        {
            "recompute_index": [recompute_index],
            # FIXME 
            "recompute_ids": [0]
        }
    )
     

    # 4. return rollout data with log probabilities.
    # FIXME how to return, the structre and data type
    """
    return Box(
        {
            "tokens" list[Tensor],
            "total_lengths": list[int],
            "response_lengths": list[int],
            "sample_indices": list[int],
            "log_probs": list[Tensor],(FIXME)
            "recompute_index": list[int],
            "recompute_ids": list[int],
        }
    )
    """
    recompute_data = {
        "recompute_index": [recompute_index],
        "recompute_ids": [0],
        "log_probs": rollout_data["log_probs"].cpu().tolist(),
    }
    return Box(ray.put(recompute_data))


def _get_verfication_data(self, rollout_data_ref):
    dp_rank = mpu.get_data_parallel_rank()
    dp_size = mpu.get_data_parallel_world_size()
    verification_data = {}
    
    # receive data
    rank = dist.get_rank()
    if rank == 0:
        data = ray.get(rollout_data_ref.inner)
        dist.broadcast_object_list([data], src=0)
    else:
        data = [None]
        dist.broadcast_object_list(data, src=0)
        data = data[0]
    
    # FIXME does not deal with rewards, adv
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
        

def get_full_logits(
    logits: torch.Tensor,
    *,
    args,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
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
    for tokens, total_length, response_length in zip(unconcat_tokens, total_lengths, response_lengths):
        end += total_length
        start = end - response_length
        logits_chunk = logits[start - 1 : end - 1]
        tokens_chunk = tokens[-response_length:]

        logits = calculate_full_logits(logits_chunk, tokens_chunk)  
        logits_list.append(logits)
    return {"logits": logits_list}


# # [Change] 
# @torch.no_grad()
# def verification(args, model, data_iterator, num_microbatches, store_prefix=""):
#     """Only do the forward pass and calculate the logprob."""
#     # 1. prepare model & data & func
#     # 1.1 reset data iterator
#     # FIXME why reset for all iterators?
#     for iterator in data_iterator:
#         iterator.reset()
#     # 1.2 prepare model
#     config = get_model_config(model[0])
#     # Turn on evaluation mode which disables dropout.
#     for model_module in model:
#         model_module.eval()
#     # 1.3 prepare func
#     # FIXME how this func is used in parallel setting?
#     def forward_step(data_iterator, model: GPTModel):
#         """Forward training step.

#         Args:
#             data_iterator : Input data iterator
#             model (GPTModel): The GPT Model
#         """

#         # Get the batch.
#         batch = get_batch(data_iterator, ["tokens", "total_lengths", "response_lengths"])
#         unconcat_tokens = batch["unconcat_tokens"]
#         tokens = batch["tokens"]
#         packed_seq_params = batch["packed_seq_params"]
#         total_lengths = batch["total_lengths"]
#         response_lengths = batch["response_lengths"]
#         output_tensor = model(
#             input_ids=tokens,
#             position_ids=None,
#             attention_mask=None,
#             labels=None,
#             packed_seq_params=packed_seq_params,
#         )

#         return output_tensor, partial(
#             get_full_logits,
#             args=args,
#             unconcat_tokens=unconcat_tokens,
#             total_lengths=total_lengths,
#             response_lengths=response_lengths,
#             with_entropy=args.use_rollout_entropy,
#         )
    
#     # 2. run forward
#     forward_backward_func = get_forward_backward_func()
#     # Don't care about timing during evaluation
#     config.timers = None
#     forward_data_store = []
#     num_steps_per_rollout = args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
#     for step_id in range(num_steps_per_rollout):
#         # collect_non_loss_data
#         forward_data_store += forward_backward_func(
#             forward_step_func=forward_step,
#             data_iterator=data_iterator,
#             model=model,
#             # FIXME structure of num_microbatches
#             num_microbatches=num_microbatches[step_id],
#             seq_length=args.seq_length,
#             micro_batch_size=args.micro_batch_size,
#             forward_only=True,
#             collect_non_loss_data=True,
#         )
        
#     # 3. Move model back to the train mode.
#     for model_module in model:
#         model_module.train()

#     # 4. process output
#     rollout_data = {}
#     # Store the results on the last stage
#     if mpu.is_pipeline_last_stage():
#         # FIXME structure of forward_data_store
#         keys = forward_data_store[0].keys()
#         for key in keys:
#             values = []
#             for value in forward_data_store:
#                 assert isinstance(value[key], list)
#                 values += value[key]

#             rollout_data[f"{store_prefix}{key}"] = values
#     return rollout_data


# def calculate_full_logits(logits, tokens):
#     if logits.size(0) != 0:
#         full_logits = compute_full_logits(logits.clone(), tokens, mpu.get_tensor_model_parallel_group())
#     else:
#         full_logits = logits.new_zeros((0,))
#     return full_logits

# def compute_full_logits(logits: torch.Tensor, tokens: torch.Tensor, tp_process_group: Optional[dist.ProcessGroup]):
#     logits = logits.unsqueeze(1)
#     tokens = tokens.unsqueeze(1)
#     full_logits = tp_process_group.gather_from_tensor_model_parallel_region(logits)
#     return full_logits


if __name__ == "__main__":
    # 0.准备函数
    MegatronTrainRayActor._get_verfication_data = _get_verfication_data
    MegatronTrainRayActor.do_verification = do_verification
    RayTrainGroup.async_verification = async_verification
    # FIXME the dropout in probs ?
    # 1. 创建参数
    args = parse_args()
    debug_train_only = True
    args.actor_num_nodes = 1
    args.actor_num_gpus_per_node = 8
    args.actor_num_microbatches = 1
    args.hf_checkpoint = "/root/Qwen3-4B"
    args.tensor_model_parallel_size = 2
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
    # 4. 准备数据 (Only one sample in batch)
    rollout_id = start_rollout_ids[0]
    raw_data = {
        "tokens": [[1,2,3,4,5,6,7,8,9,10]],
        "response_lengths": [5],
        "sample_indices": [0],
        "rollout_log_probs": [[-0.1]*10],
    }
    rollout_data_ref = ray.put(Box(ray.put(raw_data)))
    # 5. 调用 forward 计算 log prob
    recomp_data_ref = ray.get(actor_model.async_verification(rollout_id, rollout_data_ref))
    print(recomp_data_ref)
