#!/bin/csh

# with cuda
gcc -DHAVE_CONFIG_H -I. -I../..  -I../../include -Wall -Wmissing-prototypes -I/sw/include -I/usr/X11/include -I/sw/include -I/sw/lib/ffmpeg-2.4/include -I/Developer/NVIDIA/CUDA-6.5/include -I/Developer/NVIDIA/CUDA-6.5/samples/common/inc -I/Developer/NVIDIA/CUDA-6.5/samples/common/inc -I/usr/local/include  -g -MT libvl2_a-vl2_veclib.o -MD -MP -MF .deps/libvl2_a-vl2_veclib.Tpo -E `test -f 'vl2_veclib.c' || echo './'`vl2_veclib.c > vl2_veclib.noindent 

# no-cuda version
#gcc -DHAVE_CONFIG_H -I. -I../.. -I../../include -Wall -Wmissing-prototypes -I/sw/include -I/usr/X11/include -I/sw/include -I/common/inc -I/usr/local/include -g -MT libvl2_a-vl2_veclib.o -MD -MP -MF .deps/libvl2_a-vl2_veclib.Tpo -E `test -f 'vl2_veclib.c' || echo './'`vl2_veclib.c > vl2_veclib.noindent

indent < vl2_veclib.noindent > vl2_veclib.debug

#gcc -DHAVE_CONFIG_H -I. -I../..  -I../../include -Wall -Wmissing-prototypes -I/sw/include -I/usr/X11/include -I/sw/include -I/common/inc -I/usr/local/include  -g -MT libvl2_a-vl2_utils.o -MD -MP -MF .deps/libvl2_a-vl2_utils.Tpo -E `test -f 'vl2_utils.c' || echo './'`vl2_utils.c > vl2_utils.debug

#gcc -DHAVE_CONFIG_H -I. -I../..  -I../../include -Wall -Wmissing-prototypes -I/sw/include -I/usr/X11/include -I/sw/include -I/common/inc -I/usr/local/include  -g -MT libvl2_a-vl2_menu.o -MD -MP -MF .deps/libvl2_a-vl2_menu.Tpo -E `test -f 'vl2_menu.c' || echo './'`vl2_menu.c > vl2_menu.debug

