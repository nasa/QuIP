#!/bin/csh
#
# On different machines, we may have the switcher connected to
# a different tty port?

if( $HOSTNAME == brewster.arc.nasa.gov ) then
  set the_tty=/dev/ttyS1
else if( $HOSTNAME == dirac.arc.nasa.gov ) then
  set the_tty=/dev/ttyS0
else if( $HOSTNAME == purkinje.arc.nasa.gov ) then
  set the_tty=/dev/ttyS0
else
  echo root_cmds.csh:  no info about knox serial port for host $HOSTNAME
endif

echo Assuming knox switcher connected to port $the_tty on host $HOSTNAME

if( -e /dev/knox ) then
  echo Removing old knox link.
  /bin/rm /dev/knox
endif

ln -s $the_tty /dev/knox
chmod 666 $the_tty

exit

