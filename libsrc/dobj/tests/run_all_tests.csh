#!/bin/csh

if( $#argv != 1 ) then
  echo usage:  ./run_all_tests.csh program_name
  exit
endif

set prog=$1

foreach file ( *.scr )
  echo RUNNING TEST $file with program $prog
  $prog < $file
end

