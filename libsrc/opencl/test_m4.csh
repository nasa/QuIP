#!/bin/csh

m4 ocl_veclib.m4 > foo.c
gcc -I../../include -I../.. -c foo_master.c >& errors

