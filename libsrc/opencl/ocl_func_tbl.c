
#include "quip_config.h"

#ifdef HAVE_OPENCL

#include <stdlib.h>	/* qsort */
#include "quip_prot.h"
#include "veclib_api.h"
#include "my_ocl.h"

#include "ocl_veclib_prot.h"


static int gf_cmp(const void *gfp1,const void *gfp2)
{
	if( ((const GPU_Func *)gfp1)->gf_code > ((const GPU_Func *)gfp2)->gf_code ) return(1);
	else return(-1);
}

static int gtbl_inited=0;

static void gtbl_init(void)
{
	int i;

	if( gtbl_inited ){
		NWARN("gtbl_init:  already initialized");
		return;
	}

	/* sort the table so that each entry is at the location of its code */

	qsort(ocl_gpu_func_tbl,N_VEC_FUNCS,sizeof(GPU_Func),gf_cmp);

	/* make sure the table is complete */
	for(i=0;i<N_VEC_FUNCS;i++){
		if( ocl_gpu_func_tbl[i].gf_code != i ){
			int j;

			sprintf(DEFAULT_ERROR_STRING,
	"gtbl_init:  GPU_Func table entry %d has code %d!?",i, ocl_gpu_func_tbl[i].gf_code);
			NWARN(DEFAULT_ERROR_STRING);
			sprintf(DEFAULT_ERROR_STRING,"Expected function %s.",VF_NAME(&vec_func_tbl[i]));
			NADVISE(DEFAULT_ERROR_STRING);
			/* Dump all the entries to facilitate locating the missing entry */
			for(j=0;j<N_VEC_FUNCS;j++){
				sprintf(DEFAULT_ERROR_STRING,"\tentry %d holds code %d",j,ocl_gpu_func_tbl[j].gf_code);
				NADVISE(DEFAULT_ERROR_STRING);
			}
			NERROR1("Please correct error in file cuda_func_tbl.cpp");
		}
	}
	gtbl_inited=1;
}

int PF_FUNC_NAME(dispatch)( Vector_Function *vfp, Vec_Obj_Args *oap )
{
	int i;

	if( !gtbl_inited ) gtbl_init();
	
	i = vfp->vf_code;
	if( vec_func_tbl[i].oclfunc == GPU_UNIMP ){
		sprintf(DEFAULT_ERROR_STRING,"Sorry, function %s has not yet been implemented for OpenCL.",VF_NAME(vfp));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	(*vec_func_tbl[i].ocl_func)(oap);
	return(0);
}

#endif /* HAVE_CUDA */

