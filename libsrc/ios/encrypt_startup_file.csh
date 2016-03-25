#!/bin/csh

if( ! -e startup.scr ) then
  echo 'File startup.scr does not exist; run ./build_startup_file.csh first.'
  exit
endif

coq startup.scr startup.enc < encrypt_file.scr

