wget -O /etc/yum.repos.d/ceph_el7_1.repo http://gaia.repo.oa.com/ceph_el7.repo
yum install -y screen 


export http_proxy=http://9.21.0.122:11113;
export https_proxy=http://9.21.0.122:11113
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/tccl/lib:$LD_LIBRARY_PATH 
export no_proxy="127.0.0.1,localhost,${LOCAL_IP}"
export WANDB_KEY="76103382630a7e59454e9268dd69e25c461f910e"

export BASE_DIR=/root/
cd $BASE_DIR

# sglang
git clone https://github.com/sgl-project/sglang.git --branch v0.4.10.post2 --depth 1
cd $BASE_DIR/sglang/
# vim $BASE_DIR/sglang/python/pyproject.toml
sed -i '/torch==2.7.1\|torchvision\|torchaudio\|flashinfer_python/d' "$BASE_DIR/sglang/python/pyproject.toml"
pip -v install -e "python[all]"
pip install sglang-router --force-reinstall

# apex
cd $BASE_DIR
git clone https://github.com/NVIDIA/apex.git
cd $BASE_DIR/apex
MAX_JOB=32 pip install . -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8"

# flashinfer
MAX_JOBS=32 pip install flashinfer_python==0.2.7.post1  --no-build-isolation

# flash-attn
MAX_JOBS=32 pip install flash-attn==2.7.4.post1  --no-build-isolation

# transformers-engine
NVTE_FRAMEWORK=pytorch pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2

# megatron
cd $BASE_DIR
git clone https://github.com/NVIDIA/Megatron-LM.git
cd $BASE_DIR/Megatron-LM/
MAX_JOBS=32 pip install -e . --no-build-isolation

# other dependencies
pip install git+https://github.com/zhuzilin/cumem_allocator.git --no-build-isolation
pip install git+https://github.com/ISEEKYAN/mbridge.git --no-deps

# slime
cd $BASE_DIR
git clone https://github.com/THUDM/slime.git
cd $BASE_DIR/slime/
pip install -e .


cd $BASE_DIR/sglang
git apply /root/slime/docker/latest/patch/sglang.patch


