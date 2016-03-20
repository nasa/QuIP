#!/bin/bash

# this works on MBP but non on mac mini???
grep -v '# ' ocl_veclib_expanded.c | sed 's/{/\'$'\n{/g' | sed 's/{/{\'$'\n/g' | sed 's/}/\'$'\n}/g' | sed 's/}/}\'$'\n/g' | sed 's/;/;\'$'\n/g' > ocl_veclib_stripped.c


