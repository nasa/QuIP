
include(`../../include/veclib/ocl_port.m4')

suppress_if dnl suppresss output

/* `suppress_if' = suppress_if */
/* ocl_veclib.m4 BEGIN */

define(`BUILD_FOR_GPU',`')

my_include(`../../include/veclib/vecgen.m4')

define(`BUILDING_KERNELS',`')
my_include(`ocl_kernels.m4')
undefine(`BUILDING_KERNELS')

// That declares the kernels - now the host-side functions

void _insure_ocl_device(QSP_ARG_DECL  Data_Obj *dp)
{
	warn("insure_ocl_device:  not implemented!?");
}

my_include(`ocl_host_funcs.m4')

my_include(`ocl_typtbl.m4')

/* ocl_veclib.m4 END */


