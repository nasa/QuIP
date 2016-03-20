#!/bin/bash

###!/bin/csh

# bash variable set
input_file=vectree.c

cc -E -I../.. -I../../include $input_file > foo.c

# this works on MBP but non on mac mini???
grep -v '# ' foo.c | sed 's/{/\'$'\n{/g' | sed 's/{/{\'$'\n/g' | sed 's/}/\'$'\n}/g' | sed 's/}/}\'$'\n/g' | sed 's/;/;\'$'\n/g' > foo2.c

# csh-ism
#cc -c foo2.c | & more
# bash-ism
cc -Wall -c foo2.c | more 2>&1

