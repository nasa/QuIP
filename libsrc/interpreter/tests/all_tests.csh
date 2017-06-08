#!/bin/csh
# run this from the source directory

foreach script ( tests/*.scr )
  echo ' '
  echo Running test script $script
  echo ' '
  quip < $script

  echo ' '
#  echo Type ^D to proceed
#  set x=$<
end

