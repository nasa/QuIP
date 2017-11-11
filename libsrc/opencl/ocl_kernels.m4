/* ocl_kernels.m4 BEGIN */

dnl	BUG?  include veclib_prot.h instead?
my_include(`veclib/ocl_veclib_prot.m4')
my_include(`veclib/gpu_args.m4')

my_include(`ocl_kernel_src.m4')
my_include(`ocl_kern_call_defs.m4')

// including gen_kernel_calls.m4
my_include(`veclib/gen_kernel_calls.m4')

dnl used to include method_undefs.h here???

/* ocl_kernels.m4 END */

