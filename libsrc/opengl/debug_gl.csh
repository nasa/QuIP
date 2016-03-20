#!/bin/csh

# need to "source" this file for it to be effective...

if( $#argv != 1 ) then
  echo usage:  source debug_gl.csh 1\|0
  exit
endif

if( $1 == 1 ) then
  setenv LIBGL_DEBUG		1
  setenv LIBGL_DIAGNOSTIC	1
  setenv LIBGL_DUMP_VISUALID	1
else if( $1 == 0 ) then
  unsetenv LIBGL_DEBUG
  unsetenv LIBGL_DIAGNOSTIC
  unsetenv LIBGL_DUMP_VISUALID
else
  echo usage:  source debug_gl.csh 1|0
  exit
endif


