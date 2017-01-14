#!/bin/csh

# compile a test file by hand

/Developer/NVIDIA/CUDA-7.5/bin/nvcc -o test.o -c test.cu -arch sm_30 -I/sw/include -I/Developer/NVIDIA/CUDA-7.5/include -I/Developer/NVIDIA/CUDA-7.5/samples/common/inc -I/usr/local/include --machine 64 -I/usr/local/cuda/include -I. -I../.. -I../../include -DHAVE_CONFIG_H -DBUILD_FOR_CUDA

