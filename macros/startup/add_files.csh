
if( ! $?suffix ) then
  echo Error:  suffix not defined, setting to default .mac
  set suffix=mac
endif

foreach macro_file ( $file_list )
  set p=../../macros/$subdir/$macro_file.mac
  if( ! -e $p ) then
    echo Macro file $p not found.
  else
#    echo "advise 'BEGIN reading $subdir/$macro_file.mac...'" >> $outfile
    echo "# FILE $subdir/$macro_file.mac BEGIN" >> $outfile
    cat $p >> $outfile
#    echo "advise 'DONE reading $subdir/$macro_file.mac...'" >> $outfile
  endif

#  echo "advise 'startup.scr: file $macro_file.mac loaded'" >> $outfile

end

