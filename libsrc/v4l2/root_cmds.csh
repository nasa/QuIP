#!/bin/csh

if( ! -e /dev/video0 ) then
  echo Error:  /dev/video0 does not exist.
  exit
endif

# We are using four-channel cards, so normally
# the devices will be 0-3
chmod 666 /dev/video?

# OLD:
## commands to create the /dev files
## Apparently the dev files are created
## automatically when the proper driver
## (saa7134) is loaded!
#
#mknod /dev/video0 c 81 0
#chmod 666 /dev/video0

