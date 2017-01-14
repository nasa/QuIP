#!/bin/csh
set echo

#set test_stem=cuda_centroid
#set test_suffix=cu
set test_stem=ff
set test_suffix=c
set lib_stem=fill

gcc -DHAVE_CONFIG_H -I. -I../..  -I../../include -Wall -Wmissing-prototypes -I/sw/include -I/usr/X11/include -I/opt/X11/include -I/sw/include -I/sw/lib/ffmpeg-2.4/include -I/Developer/NVIDIA/CUDA-6.5/include -I/Developer/NVIDIA/CUDA-6.5/samples/common/inc -I/Developer/NVIDIA/CUDA-6.5/samples/common/inc -I/usr/local/include -I/usr/local/cuda/include -I. -I../.. -I../../include -g -MT lib${lib_stem}_a-$test_stem.o -MD -MP -MF .deps/lib${lib_stem}_a-$test_stem.Tpo -E `test -f '$test_stem.$test_suffix' || echo './'`$test_stem.$test_suffix > $test_stem.noindent
cat $test_stem.noindent | indent > $test_stem.debug

