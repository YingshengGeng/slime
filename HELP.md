cp -f ./train_verify.py /apdcephfs/ethangys_test   
cp -f ./verify-test-qwen3-4B.sh /apdcephfs/ethangys_test
cp -f ./verify-test-qwen3-0.6B.sh /apdcephfs/ethangys_test
cp -f ./noverify-test-qwen3-4B.sh /apdcephfs/ethangys_test
cp -f ./noverify-test-qwen3-0.6B.sh /apdcephfs/ethangys_test
cp -f ./actor_test.py /apdcephfs/ethangys_test 
cp -f ./generate_test.py /apdcephfs/ethangys_test 
cp -f ./slime/rollout/sglang_rollout.py /apdcephfs/ethangys_test 
cp -f ./slime/ray/buffer.py /apdcephfs/ethangys_test
cp -f ./slime/ray/rollout.py /apdcephfs/ethangys_test
cp -f ./slime/ray/actor_group.py /apdcephfs/ethangys_test
cp -f ./slime/backends/megatron_utils/model.py /apdcephfs/ethangys_test
cp -f ./slime/backends/megatron_utils/actor.py /apdcephfs/ethangys_test
cp -f ./verify-qwen3-30B-A3B.sh /apdcephfs/ethangys_test
cp -f ./mpi_rack_hostfile /apdcephfs/ethangys_test
cp -f /data/workspace/slime/slime/utils/arguments.py /apdcephfs/ethangys_test




cp -f /apdcephfs/private_ethangeng/train_verify.py ./ && \
cp -f /apdcephfs/private_ethangeng/verify-test-qwen3-4B.sh ./ && \
cp -f /apdcephfs/private_ethangeng/verify-test-qwen3-0.6B.sh ./ && \
cp -f /apdcephfs/private_ethangeng/noverify-test-qwen3-4B.sh ./ && \
cp -f /apdcephfs/private_ethangeng/noverify-test-qwen3-0.6B.sh ./ && \
cp -f /apdcephfs/private_ethangeng/actor_test.py ./ && \
cp -f /apdcephfs/private_ethangeng/generate_test.py ./ && \
cp -f /apdcephfs/private_ethangeng/sglang_rollout.py ./slime/rollout/sglang_rollout.py && \
cp -f /apdcephfs/private_ethangeng/buffer.py ./slime/ray/buffer.py && \
cp -f /apdcephfs/private_ethangeng/rollout.py ./slime/ray/rollout.py && \
cp -f /apdcephfs/private_ethangeng/actor_group.py ./slime/ray/actor_group.py && \
cp -f /apdcephfs/private_ethangeng/model.py ./slime/backends/megatron_utils/model.py && \
cp -f /apdcephfs/private_ethangeng/actor.py ./slime/backends/megatron_utils/actor.py && \
cp -f /apdcephfs/private_ethangeng/verify-qwen3-30B-A3B.sh ./ && \
cp -f /apdcephfs/private_ethangeng/mpi_rack_hostfile ./ && \
cp -f /apdcephfs/private_ethangeng/arguments.py ./slime/utils/
 


bash verify-test-qwen3-0.6B.sh
bash verify-test-qwen3-4B.sh
bash verify-qwen3-30B-A3B.sh






python -m sglang.launch_server --model-path /root/Qwen3-4B --port 30000 --host $LOCAL_IP

cp -r Qwen3-4B_slime/iter_0000084 /apdcephfs/private_ethangeng/Qwen3-4B_slime_ver/

"log_probs"


python com_ben_test_v2.py \
--model_path /root/Qwen3-4B \
--prompt "中国的首都是什么？请介绍一下它的历史。" \
--max_new_tokens 30 \
--num_samples 8 \
--temperature 1.0 



python com_ben_test_v2.py \
--model_path /apdcephfs/private_ethangeng/Qwen3-4B_slime_hf/iter_0000499/ \
--prompt "中国的首都是什么？请介绍一下它的历史。" \
--max_new_tokens 30 \
--num_samples 8 \
--temperature 1.0



python com_ben_test_v2.py \
--model_path /apdcephfs/private_ethangeng/Qwen3-4B_slime_hf_ver/iter_0000074 \
--prompt "中国的首都是什么？请介绍一下它的历史。" \
--max_new_tokens 30 \
--num_samples 8 \
--temperature 1.0


python com_ben_test_v2.py --model_path /apdcephfs/private_ethangeng/Qwen3-4B_slime_hf/iter_0000059/ --prompt "中国的首都是什么？请介绍一下它的历史。" --max_new_tokens 100 --num_samples 1 --temperature 1.0
