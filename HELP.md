cp ./train_verify.py /apdcephfs/ethangys_test   
cp ./verify-test-qwen3-4B.sh /apdcephfs/ethangys_test
cp ./verify-test-qwen3-0.6B.sh /apdcephfs/ethangys_test
cp ./noverify-test-qwen3-4B.sh /apdcephfs/ethangys_test
cp ./noverify-test-qwen3-0.6B.sh /apdcephfs/ethangys_test
cp ./actor_test.py /apdcephfs/ethangys_test 
cp ./generate_test.py /apdcephfs/ethangys_test 
cp ./slime/rollout/sglang_rollout.py /apdcephfs/ethangys_test 
cp ./slime/ray/buffer.py /apdcephfs/ethangys_test
cp ./slime/ray/rollout.py /apdcephfs/ethangys_test
cp ./slime/ray/actor_group.py /apdcephfs/ethangys_test
cp ./slime/backends/megatron_utils/model.py /apdcephfs/ethangys_test
cp ./slime/backends/megatron_utils/actor.py /apdcephfs/ethangys_test
cp ./verify-qwen3-30B-A3B.sh /apdcephfs/ethangys_test
cp ./mpi_rack_hostfile /apdcephfs/ethangys_test
cp /data/workspace/slime/slime/utils/arguments.py /apdcephfs/ethangys_test




cp /apdcephfs/private_ethangeng/train_verify.py ./ && \
cp /apdcephfs/private_ethangeng/verify-test-qwen3-4B.sh ./ && \
cp /apdcephfs/private_ethangeng/verify-test-qwen3-0.6B.sh ./ && \
cp /apdcephfs/private_ethangeng/noverify-test-qwen3-4B.sh ./ && \
cp /apdcephfs/private_ethangeng/noverify-test-qwen3-0.6B.sh ./ && \
cp /apdcephfs/private_ethangeng/actor_test.py ./ && \
cp /apdcephfs/private_ethangeng/generate_test.py ./ && \
cp /apdcephfs/private_ethangeng/sglang_rollout.py ./slime/rollout/sglang_rollout.py && \
cp /apdcephfs/private_ethangeng/buffer.py ./slime/ray/buffer.py && \
cp /apdcephfs/private_ethangeng/rollout.py ./slime/ray/rollout.py && \
cp /apdcephfs/private_ethangeng/actor_group.py ./slime/ray/actor_group.py && \
cp /apdcephfs/private_ethangeng/model.py ./slime/backends/megatron_utils/model.py && \
cp /apdcephfs/private_ethangeng/actor.py ./slime/backends/megatron_utils/actor.py && \
cp /apdcephfs/private_ethangeng/verify-qwen3-30B-A3B.sh ./ && \
cp /apdcephfs/private_ethangeng/mpi_rack_hostfile ./ && \
cp /apdcephfs/private_ethangeng/arguments.py ./slime/utils/
 


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
