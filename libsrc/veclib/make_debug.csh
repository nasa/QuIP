#!/bin/csh
gcc -DHAVE_CONFIG_H -I. -I../..  -I../../include -D_GNU_SOURCE -Wall -Wmissing-prototypes -I/sw/include -I/usr/X11/include -I/sw/include -I/common/inc -I/usr/local/include  -g -MT libvec_a-vectbl.o -MD -MP -MF .deps/libvec_a-vectbl.Tpo -E `test -f 'vectbl.c' || echo './'`vectbl.c | indent > vectbl.debug

