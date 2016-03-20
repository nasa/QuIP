#!/bin/csh

cd tests

foreach test ( *.scr )
  cd ..

  set tmp_out=/tmp/output1.$test
  set diff_out=/tmp/diffs.$test
  set expected_out=tests/expected_outputs/output.$test

  echo ' '
  echo Running test $test using coq...
  coq < tests/$test >& $tmp_out
  echo ' '
#  echo Running test $test using quip...
#  quip < tests/$test >& /tmp/output2.$test
#  diff /tmp/output1.$test /tmp/output2.$test
#  echo ' '
  if( -e $expected_out) then
    diff /tmp/output1.$test /tmp/output2.$test > $diff_out
    echo diff status is $status
    /bin/rm $diff_out
  else
    echo Expected output file $expected_out does not exist.
    echo Displaying test results:
    echo ' '
    cat $tmp_out
  endif

  /bin/rm $tmp_out

  echo ' '
  cd tests
end

cd ..

