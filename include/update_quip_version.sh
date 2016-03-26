#!/bin/sh

# the arguments are determined in Makefile.am

if [ $# -eq 1 ]; then
  srcdir=$1
elif [ $# -eq 0 ]; then
  srcdir=.
else
  echo usage:  $0 '[srcdir]'
  exit
fi

#me=`whoami`
#echo update_quip_version.csh BEGIN, run by $me


# simple script to create a file with the version string as provide by git
# What do all of these sed commands do?  It looks like they are removing some spaces?

#ls -ld $srcdir

outfile=/tmp/current_quip_version.h

# 5 extra spaces before the date string, and a space before the trailing quote
# need --tags if tag is not "annotated" !?
echo '#define' '	' 'QUIP_VERSION_STRING' '	' '"' `git describe --tags` '"' \
	| sed -e 's/ //' \
	| sed -e 's/ //' \
	| sed -e 's/ //' \
	| sed -e 's/ //' \
	| sed -e 's/ //' \
	| sed -e 's/ "/"/' \
	> $outfile

echo '#define' '	' 'QUIP_LOCAL_BUILD_DATE' '	' '"' `date` '"' \
	| sed -e 's/ //' \
	| sed -e 's/ //' \
	| sed -e 's/ //' \
	| sed -e 's/ //' \
	| sed -e 's/ //' \
	| sed -e 's/ "/"/' \
	>> $outfile
echo '#define' '	' 'QUIP_UTC_BUILD_DATE' '	' '"' `date -u` '"' \
	| sed -e 's/ //' \
	| sed -e 's/ //' \
	| sed -e 's/ //' \
	| sed -e 's/ //' \
	| sed -e 's/ //' \
	| sed -e 's/ "/"/' \
	>> $outfile

if test ! -e $srcdir/quip_version.h; then
  echo Creating quip_version.h
  cp $outfile $srcdir/quip_version.h
  exit 0
fi

diff $srcdir/quip_version.h $outfile > /dev/null

if test "$?" != 0; then
  echo Updating quip_version.h ...
  cp $outfile $srcdir/quip_version.h
else
  echo File quip_version.h is already up-to-date.
fi

/bin/rm $outfile

exit 0

