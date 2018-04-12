#!/bin/csh

set input=spink_wrappers.c

grep WRAPPER $input | grep -v '//' | sed -e 's/SPINK_WRAPPER_[ONETWHR]*_ARG(//' | \
  sed -e 's/,/		/' | sed -f cut_it.sed | sort > new_index

#head -10 new_index
#more new_index

cat new_index > function_index

cat new_index | dm s2 s1 | sort >> function_index



