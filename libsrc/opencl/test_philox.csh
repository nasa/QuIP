#!/bin/csh

gcc -I/Users/jmulliga/src/Random123-1.08/include -E vuni.c > vuni.i
cat vuni.i | indent > foo


