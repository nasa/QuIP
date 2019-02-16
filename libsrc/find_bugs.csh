#!/bin/csh

set outfile=/tmp/bug_count

if( $#argv > 0 ) then
  set i=1
  while( $i <= $#argv )
#    echo arg $i = $argv[$i]
    set d=$argv[$i]
    pushd $d
    cat < /dev/null > $outfile
    foreach file ( *.[chy] )
      set n=`grep BUG $file | grep -v DEBUG | wc -l`
      if( $n > 0 ) then
        echo $n BUGs in $d/$file >> $outfile
      endif
    end
    cat $outfile | sort -n
    popd
    @ i ++
  end
  exit
endif

cat < /dev/null > $outfile

foreach d ( * )
  if( -d $d ) then
    set n=`grep BUG $d/*.[ch] | grep -v DEBUG | grep -v expanded.c | wc -l`
    if( $n > 0 ) then
      echo $n BUGs in dir $d >> $outfile
    endif
  endif
end

cat $outfile | sort -n

