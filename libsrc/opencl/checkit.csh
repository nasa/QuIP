#!/bin/csh

set f=vuni.out

#cat $f | dm "if x1 < 0 then INPUT else KILL"
#cat $f | dm "if x1 > 1 then INPUT else KILL"

coq $f < check_hist.scr


