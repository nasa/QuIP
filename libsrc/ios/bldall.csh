#!/bin/csh

foreach p ( csf ezjet iquip )
  ./build_startup_file.csh $p test
end

