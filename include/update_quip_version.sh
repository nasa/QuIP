#!/bin/sh

me=`whoami`
echo update_quip_version.csh BEGIN, run by $me

# simple script to create a file with the version string as provide by git

#echo '#define' '	' 'QUIP_VERSION_STRING' '	' '"' `git describe` '	' `date +'%D %R'` '"'

echo '#define' '	' 'QUIP_VERSION_STRING' '	' '"' `git describe` '"' \
	| sed -e 's/ //' \
	| sed -e 's/ //' \
	| sed -e 's/ //' \
	| sed -e 's/ //' \
	| sed -e 's/ //' \
	| sed -e 's/ //' \
	| sed -e 's/ //' \
	| sed -e 's/ "/"/' \
	> current_quip_version.h

if test ! -e quip_version.h; then
  cp current_quip_version.h quip_version.h
fi

diff quip_version.h current_quip_version.h > /dev/null

if test "$?" != 0; then
  echo Updating quip_version.h ...
  cp current_quip_version.h quip_version.h
else
  echo File quip_version.h is already up-to-date.
fi

exit 0

