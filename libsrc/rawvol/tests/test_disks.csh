#!/bin/csh

# shell script to test speed of striped volume w/ sio
#
# -t flag prints timing
# -b block size
# -s total size

# for 1024x1280, 1k disk block size, and image is 1280 blocks.

set disk1=/dev/sdb1
set disk2=/dev/sdc1
set disk3=/dev/sdd1

set total_seconds=20
set blocks_per_frame=1280
set fps=170
set ntotal=$blocks_per_frame
@ ntotal *= $fps * $total_seconds
set thistotal=$ntotal
set subblocks=$blocks_per_frame

time sio -t -b $blocks_per_frame -s $ntotal $disk1 &
wait
echo done testing 1
exit

@ subblocks = $blocks_per_frame / 2
@ thistotal = $ntotal / 2
time sio -t -b  $subblocks -s $thistotal $disk1 &
time sio -t -b  $subblocks -s $thistotal $disk2 &

wait
echo done testing 2

exit

@ subblocks = $blocks_per_frame / 3
@ thistotal = $ntotal / 3
time sio -t -b  $subblocks -s $thistotal $disk1 &
time sio -t -b  $subblocks -s $thistotal $disk2 &
time sio -t -b  $subblocks -s $thistotal $disk3 &

wait
echo done testing 3

@ subblocks = $blocks_per_frame / 4
@ thistotal = $ntotal / 4
time sio -t -b  $subblocks -s $thistotal $disk1 &
time sio -t -b  $subblocks -s $thistotal $disk2 &
time sio -t -b  $subblocks -s $thistotal $disk3 &
time sio -t -b  $subblocks -s $thistotal $disk4

wait

exit

