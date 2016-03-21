#!/bin/bash

# this works on MBP but non on mac mini???
grep -v '# ' ocl.debug_ | sed 's/{/\'$'\n{/g' | sed 's/{/{\'$'\n/g' | sed 's/}/\'$'\n}/g' | sed 's/}/}\'$'\n/g' | sed 's/;/;\'$'\n/g' > ocl.debug


