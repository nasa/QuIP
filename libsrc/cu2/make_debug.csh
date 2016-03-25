#!/bin/csh
set echo

#gcc -DHAVE_CONFIG_H -I. -I../..  -I../../include -Wall -Wmissing-prototypes -I/sw/include -I/usr/X11/include -I/sw/include -I/common/inc -I/usr/local/include  -g -MT libcu2_a-cu2_menu.o -MD -MP -MF .deps/libcu2_a-cu2_menu.Tpo -E `test -f 'cu2_menu.c' || echo './'`cu2_menu.c | indent > cu2_menu.debug

#gcc -DHAVE_CONFIG_H -I. -I../..  -I../../include -Wall -Wmissing-prototypes -I/sw/include -I/usr/X11/include -I/sw/include -I/sw/lib/ffmpeg-2.4/include -I/Developer/NVIDIA/CUDA-6.5/include -I/Developer/NVIDIA/CUDA-6.5/samples/common/inc -I/Developer/NVIDIA/CUDA-6.5/samples/common/inc -I/usr/local/include  -g -MT libcu2_a-cu2.o -MD -MP -MF .deps/libcu2_a-cu2.Tpo -E `test -f 'cu2.c' || echo './'`cu2.c | indent > cu2.debug

gcc -DHAVE_CONFIG_H -I. -I../..  -I../../include -Wall -Wmissing-prototypes -I/sw/include -I/usr/X11/include -I/opt/X11/include -I/sw/include -I/sw/lib/ffmpeg-2.4/include -I/Developer/NVIDIA/CUDA-6.5/include -I/Developer/NVIDIA/CUDA-6.5/samples/common/inc -I/Developer/NVIDIA/CUDA-6.5/samples/common/inc -I/usr/local/include -I/usr/local/cuda/include -I. -I../.. -I../../include -g -MT libcu2_a-cu2_veclib.o -MD -MP -MF .deps/libcu2_a-cu2_veclib.Tpo -E `test -f 'cu2_veclib.c' || echo './'`cu2_veclib.c > cu2_veclib.noindent
cat cu2_veclib.noindent | indent > cu2_veclib.debug

