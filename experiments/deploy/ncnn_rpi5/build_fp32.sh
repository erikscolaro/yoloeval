#!/usr/bin/env bash
# benchncnn_fp32 sul Pi: benchncnn con fp16 spento (packed, storage, arithmetic). Il benchncnn normale li accende
# nel codice (benchmark/benchncnn.cpp), quindi con pesi fp32 calcolerebbe comunque in fp16. Si lancia dalla workstation.
set -euo pipefail
ssh rpi5 'set -e
R=~/.cache/yolo-bench/ncnn; V=$(ls $R | head -1); S=$R/$V-fp32
[ -d $S ] || cp -r $R/$V $S
sed -i "s/use_fp16_packed = true/use_fp16_packed = false/; s/use_fp16_storage = true/use_fp16_storage = false/; s/use_fp16_arithmetic = true/use_fp16_arithmetic = false/" $S/benchmark/benchncnn.cpp
grep -n "use_fp16" $S/benchmark/benchncnn.cpp
rm -rf $S/build; cmake -S $S -B $S/build -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_SIMPLEOCV=ON -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_BENCHMARK=ON -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_PYTHON=OFF >/dev/null && cmake --build $S/build -j4 --target benchncnn
install -m755 $S/build/benchmark/benchncnn /home/bench/bench/.venv/bin/benchncnn_fp32'
