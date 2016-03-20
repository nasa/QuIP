#!/bin/csh
set echo

set stem=glfb

gcc -DHAVE_CONFIG_H -I. -I../..  -I../../include -Wall -Wmissing-prototypes -I/sw/include -I/usr/X11/include -I/sw/include -I/sw/lib/ffmpeg-2.4/include -I/Developer/NVIDIA/CUDA-6.5/include -I/Developer/NVIDIA/CUDA-6.5/samples/common/inc -I/Developer/NVIDIA/CUDA-6.5/samples/common/inc -I/usr/local/include  -g -MT libocl_a-$stem.o -MD -MP -MF .deps/libocl_a-$stem.Tpo -E `test -f '$stem.c' || echo './'`$stem.c > $stem.noindent
cat $stem.noindent | indent > $stem.debug

#gcc -DHAVE_CONFIG_H -I. -I../..  -I../../include -Wall -Wmissing-prototypes -I/sw/include -I/usr/X11/include -I/sw/include -I/sw/lib/ffmpeg-2.4/include -I/Developer/NVIDIA/CUDA-6.5/include -I/Developer/NVIDIA/CUDA-6.5/samples/common/inc -I/Developer/NVIDIA/CUDA-6.5/samples/common/inc -I/usr/local/include  -g -MT libocl_a-ocl_veclib.o -MD -MP -MF .deps/libocl_a-ocl_veclib.Tpo -E `test -f 'ocl_veclib.c' || echo './'`ocl_veclib.c > ocl_veclib.noindent
#cat ocl_veclib.noindent | indent > ocl_veclib.debug

# without CUDA
#gcc -fmacro-backtrace-limit=0 -DHAVE_CONFIG_H -I. -I../..  -I../../include -Wall -Wmissing-prototypes -I/sw/include -I/usr/X11/include -I/sw/include -I/common/inc -I/usr/local/include  -g -MT libocl_a-ocl_veclib.o -MD -MP -MF .deps/libocl_a-ocl_veclib.Tpo -E `test -f 'ocl_veclib.c' || echo './'`ocl_veclib.c | indent > ocl_veclib.debug

#gcc -DHAVE_CONFIG_H -I. -I../..  -I../../include -Wall -Wmissing-prototypes -I/sw/include -I/usr/X11/include -I/sw/include -I/common/inc -I/usr/local/include  -g -MT libocl_a-ocl_menu.o -MD -MP -MF .deps/libocl_a-ocl_menu.Tpo -E `test -f 'ocl_menu.c' || echo './'`ocl_menu.c | indent > ocl_menu.debug

#gcc -DHAVE_CONFIG_H -I. -I../..  -I../../include -Wall -Wmissing-prototypes -I/sw/include -I/usr/X11/include -I/sw/include -I/common/inc -I/usr/local/include  -g -MT libocl_a-ocl.o -MD -MP -MF .deps/libocl_a-ocl.Tpo -E `test -f 'ocl.c' || echo './'`ocl.c | indent > ocl.debug

