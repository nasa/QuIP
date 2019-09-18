if( ! $?1 ) then
  echo usage:  source add_file.csh file
  exit
endif

set p=../../macros/$1

if( ! -e $p ) then
    echo Macro file $p not found.
else
  echo "# FILE $1 BEGIN" >> $outfile
  cat $p >> $outfile
endif


