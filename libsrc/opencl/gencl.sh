#!/bin/sh
# Run the C preprocessor on an OpenCL kernel to generate a C string array
# suitable for clCreateProgramWithSource.  This allows us to create
# standalone OpenCL programs that do not depend on paths to the source
# tree (the programs will still run the OpenCL run-time compiler to
# compile the kernel, but the kernel is a string within the program, with
# no external include dependencies)
# Mark Moraes, D. E. Shaw Research

# indenting the cpp output makes errors from the OpenCL runtime compiler
# much more understandable.  User can override with whatever they want.
# The classic BSD indent (yes, the one that lived in /usr/ucb/indent once)
# defaults to -br, but recent GNU indent versions do not.  Both appear to
# accept -br, fortunately... (BSD indent does not accept -kr or -linux, alas)

# jbm added this to work outside of the Random123 source tree
CPPFLAGS=-I/usr/local/include

PATH=$PATH:/usr/bin
export PATH
if type indent > /dev/null 2>&1; then
	: ${GENCL_INDENT="indent -br"}
else
	: ${GENCL_INDENT=cat}
fi

# We rely on gsub in awk, which exists in everything except classic
# old V7 awk (Solaris!).  If we can find gawk or nawk, we prefer those.
# http://www.shelldorado.com/articles/awkcompat.html
for f in gawk nawk awk; do
	if type "$f" > /dev/null 2>&1; then
		: ${GENCL_AWK="$f"}
		break
	fi
done
case "${GENCL_AWK}" in
'')	echo "$0: could not find awk!">&2; exit 1;;
esac
usage="Usage: $0 inputoclfilename outputfilename"
case $# in
2)	;;
*)	echo "$usage" >&2; exit 1;;
esac
case "$1" in
''|-*)	echo "Invalid or empty inputoclfilename: $1
$usage" >&2; exit 1;;
esac
set -e
echo 'processing file...'
# jbm:  original script put in \\\n\ before each newline to be quoted.
# That seemed to work fine when we were including from the Random123 source
# tree, but after copying the needed files to our own tree, there are stray
# backslashes at the ends of lines that screw things up!?!?

${CC-cc} -xc -E -P -nostdinc -D__OPENCL_VERSION__=1 $CPPFLAGS "$1" | 
	${GENCL_INDENT} | 
	${GENCL_AWK} 'BEGIN {print "const static char *opencl_src = \"\\n\\"}
	{gsub("\"", "\\\"", $0); print $0 "\\n\\"}
	END {print "\";"}' > "$2"

#${CC-cc} -xc -E -P -nostdinc -D__OPENCL_VERSION__=1 $CPPFLAGS "$1" | 
#	${GENCL_INDENT} | 
#	${GENCL_AWK} 'BEGIN {print "const static char *opencl_src = \"\\n\\"}
#	{gsub("\\", "\\\\", $0); gsub("\"", "\\\"", $0); print $0 "\\n\\"}
#	END {print "\";"}' > "$2"

