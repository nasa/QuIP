#!/bin/csh

set n_tot=`grep API all | wc -l`
grep API all | sed -e 's/,/xyzzy/' | grep -v ',' | \
  sed -e 's/xyzzy/,/g' > f.one

grep API all | sed -e 's/,/xyzzy/' | grep ',' | \
  sed -e 's/,/xyzzy/' | grep -v ',' | \
  sed -e 's/xyzzy/,/g' > f.two

grep API all | sed -e 's/,/xyzzy/' | grep ',' | \
  sed -e 's/,/xyzzy/' | grep ',' | \
  sed -e 's/,/xyzzy/' | grep -v ',' | \
  sed -e 's/xyzzy/,/g' > f.three

grep API all | sed -e 's/,/xyzzy/' | grep ',' | \
  sed -e 's/,/xyzzy/' | grep ',' | \
  sed -e 's/,/xyzzy/' | grep ',' | \
  sed -e 's/,/xyzzy/' | grep -v ',' | \
  sed -e 's/xyzzy/,/g' > f.four

grep API all | sed -e 's/,/xyzzy/' | grep ',' | \
  sed -e 's/,/xyzzy/' | grep ',' | \
  sed -e 's/,/xyzzy/' | grep ',' | \
  sed -e 's/,/xyzzy/' | grep ',' | \
  sed -e 's/,/xyzzy/' | grep -v ',' | \
  sed -e 's/xyzzy/,/g' > f.five

grep API all | sed -e 's/,/xyzzy/' | grep ',' | \
  sed -e 's/,/xyzzy/' | grep ',' | \
  sed -e 's/,/xyzzy/' | grep ',' | \
  sed -e 's/,/xyzzy/' | grep ',' | \
  sed -e 's/,/xyzzy/' | grep ',' | \
  sed -e 's/,/xyzzy/' | grep -v ',' | \
  sed -e 's/xyzzy/,/g' > f.six

echo n_tot = $n_tot
wc f.*

