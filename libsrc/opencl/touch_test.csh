#!/bin/csh

foreach file ( ../../include/veclib/*.m4 )
  echo ' '
  echo TESTING $file
  touch $file
  set n=`make m4_ocl_veclib.c | grep Nothing | wc -l`
  echo n = $n
  if( $n == 0 ) then
    echo File $file triggered a rebuild
  else
    echo File $file did NOT trigger a rebuild
  endif
  sleep 2
end

