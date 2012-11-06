#include "quip_config.h"

char VersionId_newvec_bm_funcs[] = QUIP_VERSION_STRING;

#include "debug.h"		/* verbose */
#include "nvf.h"
#include "bm_prot.h"

#define std_type	bitmap_word
#define std_scalar	u_s
#define dest_type	bitmap_word
#define absfunc		abs

#include "new_ops.h"

#define TYP		bm

#include "bit_vec.c"

void bm_obj_rvmov( Vec_Obj_Args *oap )
{
	NWARN("Sorry, bm_obj_rvmov has not been implemented yet (bm_funcs.c).");
}

