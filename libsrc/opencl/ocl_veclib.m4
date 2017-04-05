divert(-1)

define(`suppressing',`')

ifdef(`suppressing',`
/* Suppressing ! */
define(`suppress_if',`divert(-1)')
',`
/* NOT Suppressing ! */
define(`suppress_if',`divert(0)')
')

include(`../../include/veclib/ocl_port.m4')

suppress_if dnl suppresss output

/* `suppress_if' = suppress_if */
/* ocl_veclib.m4 BEGIN */

define(`BUILD_FOR_GPU',`')

include(`../../include/veclib/vecgen.m4')

define(`BUILDING_KERNELS',`')
include(`ocl_kernels.m4')
undefine(`BUILDING_KERNELS')

// That declares the kernels - now the host-side functions

void insure_ocl_device(Data_Obj *dp)
{
	NWARN("insure_ocl_device:  not implemented!?");
}

include(`ocl_host_funcs.m4')

include(`ocl_typtbl.m4')

/* ocl_veclib.m4 END */


