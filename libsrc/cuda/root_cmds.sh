#!/bin/sh

# The devices didn't exist on macbook pro, not sure why not?

# create nvidia device nodes
sudo mknod /dev/nvidia0 c 195 0
sudo mknod /dev/nvidiactl c 195 255

#chown root.video /dev/nvidia0
#chown root.video /dev/nvidiactl

sudo chmod 666 /dev/nvidia0
sudo chmod 666 /dev/nvidiactl

