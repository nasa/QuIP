
define(`BUILD_FOR_HOST',`')

my_include(`ocl_host_call_defs.m4')
dnl	First the typed functions...
my_include(`veclib/host_typed_call_defs.m4')
my_include(`veclib/gen_host_calls.m4')

dnl	Where are the untyped calls declared???
dnl	For the tabled functions, there are no untyped functions...
dnl	But for special things like fft2d, we need to declare them

my_include(`ocl_untyped_host_calls.m4')
