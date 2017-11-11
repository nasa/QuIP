include(`../../include/veclib/cu2_port.m4')

my_include(`../../include/veclib/gpu_args.m4')
my_include(`../../include/veclib/gpu_call_utils.m4')
my_include(`../../include/veclib/slow_defs.m4')

my_include(`cu2_centroid_defs.m4')
my_include(`../cu2/cu2_kern_call_defs.m4')

my_include(`../../include/veclib/sp_defs.m4')
CK(type_code)			dnl	centroid kernel

my_include(`../../include/veclib/dp_defs.m4')
CK(type_code)			dnl	centroid kernel

my_include(`../../include/veclib/host_typed_call_defs.m4')
my_include(`../cu2/cu2_host_call_defs.m4')


dnl	//#include "host_calls.h"


my_include(`../../include/veclib/sp_defs.m4')
HCF(type_code)

my_include(`../../include/veclib/dp_defs.m4')
HCF(type_code)


