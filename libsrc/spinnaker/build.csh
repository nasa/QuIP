#!/bin/csh

gcc -I/usr/include/spinnaker/spinc -c example.c
gcc -o ./test example.o -lSpinnaker_C

