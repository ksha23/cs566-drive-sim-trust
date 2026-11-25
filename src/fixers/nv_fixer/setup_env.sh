#!/bin/bash
# Build the NVIDIA Fixer env without Docker (what we ran on sbel; see docs/nv_fixer.md).
set -e
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/lib/env.sh"
FIX=${NV_FIXER_ROOT:-$DATA/nv_fixer}; mkdir -p $FIX/env $FIX/cache
[ -d $FIX/repo ] || git clone https://github.com/nv-tlabs/Fixer $FIX/repo
cd $FIX/env; [ -f pixi.toml ] || { pixi init .; pixi add "python=3.12" pip; }
P=$FIX/env/.pixi/envs/default/bin/python; export PIP_CACHE_DIR=$FIX/cache/pip
$P -m pip download --no-deps -d $FIX/cache/whl cosmos-predict2==1.0.9
$P -m pip install --no-deps $FIX/cache/whl/cosmos_predict2-1.0.9-py3-none-any.whl
$P -m pip install attrs==25.1.0 diffusers==0.33.1 einops==0.8.1 ftfy fvcore==0.1.5.post20221221 huggingface-hub==0.32.4 \
  hydra-core==1.3.2 "imageio[ffmpeg]==2.37.0" iopath==0.1.10 loguru==0.7.2 mediapy megatron-core==0.10.0 numpy==1.26.4 \
  omegaconf==2.3.0 opencv-python-headless==4.10.0.84 pandas peft==0.14.0 pillow==11.1.0 protobuf==4.25.3 pyyaml \
  safetensors==0.5.3 scikit-image sentencepiece termcolor tqdm transformers==4.51.3 webdataset natsort lpips \
  "torchmetrics[image]" bitsandbytes accelerate better-profanity boto3 nvidia-ml-py qwen-vl-utils
# torch last (other packages pull newer torch/triton), then the ABI-matching flash-attn wheel
$P -m pip install torch==2.6.0 torchvision==0.21.0 triton==3.2.0 --index-url https://download.pytorch.org/whl/cu126
$P -m pip uninstall -y pynvml || true
$P -m pip install --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
$P -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/Fixer', local_dir='$FIX/weights')"
