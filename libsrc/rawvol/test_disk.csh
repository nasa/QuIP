#!/bin/csh
#
# Permissions seem to change on CentOS 6!?

set f=/dev/sdb1
sudo chmod 666 $f
ls -l $f
getfacl $f
dd if=$f of=blk0 bs=1024 count=2
ls -l blk0

echo 'After reading...'
ls -l $f
echo ' '
getfacl $f
echo ' '

# permissions only change after a write!
dd if=blk0 of=$f bs=1024 count=2

echo 'After writing...'
ls -l $f
echo ' '
getfacl $f
echo ' '

sleep 1

echo 'After sleeping...'
ls -l $f
echo ' '
getfacl $f
echo ' '


