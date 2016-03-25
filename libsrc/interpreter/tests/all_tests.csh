#!/bin/csh

foreach script ( location.scr )
  coq < $script >& coq_out
  quip < $script >& quip_out

  diff coq_out quip_out
end

