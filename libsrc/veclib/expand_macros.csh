#!/bin/bash

###!/bin/csh

# bash variable set
input_file=spdp_funcs.c

cc -E -I../../include -I../.. $input_file > foo.c

# this works on MBP but non on mac mini???
grep -v '# ' foo.c | sed 's/{/\'$'\n{/g' | sed 's/{/{\'$'\n/g' | sed 's/}/\'$'\n}/g' | sed 's/}/}\'$'\n/g' | sed 's/;/;\'$'\n/g' > foo2.c

# csh-ism
#cc -c foo2.m | & more
# bash-ism
cc -Wall -g -c foo2.c | more 2>&1

