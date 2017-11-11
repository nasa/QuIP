#!/bin/csh
#
# On different machines, we may have the switcher connected to
# a different tty port?

# Don't seem to have documentation for galvo system (poisson?)
if( $HOSTNAME == poisson.arc.nasa.gov ) then
  set the_tty=/dev/ttyS1
else if( $HOSTNAME == wheatstone.arc.nasa.gov ) then
  set the_tty=/dev/ttyS1
else
  echo root_cmds.csh:  no info about VISCA serial port for host $HOSTNAME
  set the_tty=/dev/null
endif

echo Assuming VISCA camera network connected to port $the_tty on host $HOSTNAME

if( -e /dev/visca ) then
  echo Removing old visca link.
  /bin/rm /dev/visca
endif

ln -s $the_tty /dev/visca
chmod 666 $the_tty

exit

