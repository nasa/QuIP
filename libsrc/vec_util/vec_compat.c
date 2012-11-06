#include "quip_config.h"

char VersionId_vec_util_vec_compat[] = QUIP_VERSION_STRING;

/* compatibility between new veclib and old warlib */

#include "vecgen.h"
/* #include "warproto.h" */

void vsum(Vec_Args *argp)
{
	if( for_real ) (*wartbl[FVSUM].f_st1arg) (argp->arg_scalar1,argp->arg_v1,
					&argp->arg_inc1,&argp->arg_n1,
					type_str[ argp->arg_type ],
					prec_str[ argp->arg_prec1 & MACH_PREC_MASK ] );
}

