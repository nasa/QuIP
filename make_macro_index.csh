#!/bin/csh

grep '^define(' libsrc/veclib2/*.m4 include/veclib/*.m4 | sed -e 's/:/\t/'	| sed -e 's/define(`//' | \
  awk -F "'" '{print $1}' | dm s2 s1 | sort

