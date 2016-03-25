#!/bin/bash

###!/bin/csh

# bash variable set
suffix=c
input_file=viewer.$suffix

#cc -E -DBUILD_FOR_IOS -I../.. -I../../include $input_file > foo.$suffix
cc -E -I../.. -I../../include $input_file > foo.$suffix

# this works on MBP but non on mac mini???
grep -v '# ' foo.$suffix | sed 's/{/\'$'\n{/g' | sed 's/{/{\'$'\n/g' | sed 's/}/\'$'\n}/g' | sed 's/}/}\'$'\n/g' | sed 's/;/;\'$'\n/g' > foo2.$suffix

# csh-ism
#cc -c foo2.c | & more
# bash-ism
cc -c foo2.$suffix | more 2>&1

