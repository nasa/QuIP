#!/bin/csh

# commands to create the /dev files
# Apparently the dev files are created
# automatically when the proper driver
# (saa7134) is loaded!

mknod /dev/video0 c 81 0
chmod 666 /dev/video0

