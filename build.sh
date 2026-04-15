#!/bin/bash
echo "[build] compiling vec_kernel.cu..."
nvcc -O2 -c vec_kernel.cu -o vec_kernel.o \
  -Wno-deprecated-gpu-targets -gencode arch=compute_61,code=sm_61 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 || exit 1

echo "[build] linking vec..."
nvcc -O2 vec_kernel.o vec.cpp -o vec -lpthread || exit 1

echo "[build] done."
rm -f vec_kernel.o
