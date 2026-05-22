#!/bin/bash
set -e

mkdir -p third_party

# Patch-NetVLAD
if [ ! -d "third_party/Patch-NetVLAD" ]; then
    git clone https://github.com/QVPR/Patch-NetVLAD.git third_party/Patch-NetVLAD
fi

# SuperPoint
if [ ! -d "third_party/SuperPoint" ]; then
    git clone https://github.com/rpautrat/SuperPoint.git third_party/SuperPoint
fi

# SuperGlue
if [ ! -d "third_party/SuperGluePretrainedNetwork" ]; then
    git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git third_party/SuperGluePretrainedNetwork
fi

# LoFTR
if [ ! -d "third_party/LoFTR" ]; then
    git clone https://github.com/zju3dv/LoFTR.git third_party/LoFTR
fi

# R2D2
if [ ! -d "third_party/r2d2" ]; then
    git clone https://github.com/naver/r2d2.git third_party/r2d2
fi

# D2Net
if [ ! -d "third_party/d2-net" ]; then
    git clone https://github.com/mihaidusmanu/d2-net.git third_party/d2-net
fi

echo "Third party repositories setup complete."