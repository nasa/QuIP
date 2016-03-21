#include "quip_config.h"

/* this table defines all of the functions.
 * It gives their name, a mask that tells the supported precisions,
 * and an entry point...
 */

#include <stdlib.h>	/* qsort */
//#include "nvf.h"
////#include "item.h"
////#include "version.h"
//#include "rn.h"		/* set_random_seed */
//#include "debug.h"
//#include "warn.h"

//#include "veclib_prot.h"
#ifdef HAVE_CUDA
#include "veclib/cu2_veclib_prot.h"

#include "quip_prot.h"
#include "cu2_func_tbl.h"

/* This used to be an initialized table,
 * but in Objective C we have to do it differently...
 *
 * This table contains information about the functions, but isn't
 * specific to a particular platform.  The elements are the name,
 * the code, a code indicating the argument types, a mask indicating
 * what precisions (and combinations) are allowed, and a mask
 * indicating what types (real/complex/quaternion/mixed/etc)
 * are allowed.
 */

//BEGIN_CU2_VFUNC_DECLS

#define ADD_FUNC_TO_TBL( func, index )	{	index,	h_cu2_##func	},
#define ADD_CPU_FUNC_TO_TBL( func, index )	{	index,	NULL	},

Dispatch_Function cu2_func_tbl[]={

#include "veclib/dispatch_tbl.c"

//END_CU2_VFUNC_DECLS
};


// In the old coding, we created items and copied their values from the table...
// But we don't need to have two copies - we just have to insert the names

#define N_CU2_FUNCS		(sizeof(cu2_func_tbl)/sizeof(Dispatch_Function))

#endif // HAVE_CUDA

#ifdef FOOBAR
static void create_vfs(SINGLE_QSP_ARG_DECL)
{
	u_int i;

	init_vec_funcs(SINGLE_QSP_ARG);	// init item type

	for(i=0;i<N_NVFS;i++){
		add_item(QSP_ARG  vec_func_itp, &vec_func_tbl[i], NO_NODE );
	}
}

static int vf_cmp(CONST void *vfp1,CONST void *vfp2)
{
	if( ((CONST Vector_Function *)vfp1)->vf_code > ((CONST Vector_Function *)vfp2)->vf_code ) return(1);
	else return(-1);
}

static int vfa_cmp(CONST void *vfp1,CONST void *vfp2)
{
	if( ((CONST Vec_Func_Array *)vfp1)->vfa_code > ((CONST Vec_Func_Array *)vfp2)->vfa_code ) return(1);
	else return(-1);
}

int check_vfa_tbl_size(QSP_ARG_DECL  Vec_Func_Array vfa_tbl[], int size)
{
	if( size != N_VEC_FUNCS ){
sprintf(ERROR_STRING,"CAUTIOUS:  %d inititialized vfa_tbl entries, expected %d!?",
			size,N_VEC_FUNCS);
		WARN(ERROR_STRING);
		return -1;
	}
	return 0;
}

void check_vfa_tbl(QSP_ARG_DECL  Vec_Func_Array *vfa_tbl, int size)
{
	int i;
//	int retval=0;

//	if( check_vfa_tbl_size(QSP_ARG  vfa_tbl, size) < 0 )
//		return -1;

	assert( size == N_VEC_FUNCS );

	qsort(vfa_tbl,size,sizeof(Vec_Func_Array),vfa_cmp);

#ifdef CAUTIOUS
	/* make sure the table is complete */
	for(i=0;i<size;i++){

/*
if( verbose ){
sprintf(ERROR_STRING,"check_vfa_tbl:  vfa_tbl[%d] (%s):  code = %d (%s)",
i, VF_NAME(&vec_func_tbl[i]),
vfa_tbl[i].vfa_code,
VF_NAME(&vec_func_tbl[ vfa_tbl[i].vfa_code ])
);
ADVISE(ERROR_STRING);
}
*/

//		if( vfa_tbl[i].vfa_code != i ){
//			sprintf(ERROR_STRING,
//	"CAUTIOUS:  check_vfa_tbl:  Vec_Func_Array table entry %d (%s) has code %d (%s)!?",
//		i,VF_NAME(&vec_func_tbl[i]),
//		vfa_tbl[i].vfa_code,VF_NAME(&vec_func_tbl[ vfa_tbl[i].vfa_code ]) );
//			WARN(ERROR_STRING);
//			retval = (-1);
//		}

		assert( vfa_tbl[i].vfa_code == i );
	}
#endif /* CAUTIOUS */
}

void vl_init(SINGLE_QSP_ARG_DECL)
{
	static int inited=0;
	int i;

	if( inited ){
		// Don't warn, could be called from warmenu & vl_menu...
		/*warn("vl_init:  already initialized"); */
		return;
	}

	if( veclib_debug == 0 )
		veclib_debug = add_debug_module(QSP_ARG  "veclib");

	/* sort the table to insure that each entry is at the location of its code */
//#ifdef CAUTIOUS
//	if( N_VEC_FUNCS != N_NVFS ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  vl_init:  Vector function table is missing %ld entries!?",
//			N_VEC_FUNCS-N_NVFS);
//		ERROR1(ERROR_STRING);
//	}
//#endif // CAUTIOUS

	assert( N_VEC_FUNCS == N_NVFS );

	qsort(vec_func_tbl,N_NVFS,sizeof(Vector_Function),vf_cmp);

#ifdef CAUTIOUS
	/* make sure the table is complete */
	for(i=0;i<N_NVFS;i++){

if( verbose ){
sprintf(ERROR_STRING,"vl_init:  vec_func_tbl[%d] (%s):  code %d (%s)",
i, VF_NAME(&vec_func_tbl[i]),
VF_CODE(&vec_func_tbl[i]), VF_NAME(&vec_func_tbl[ VF_CODE(&vec_func_tbl[i]) ])
);
ADVISE(ERROR_STRING);
}

//		if( VF_CODE(&vec_func_tbl[i]) != i ){
//			sprintf(ERROR_STRING,
//	"CAUTIOUS:  vl_init:  Vec_Func table entry %d (%s) has code %d (%s)!?",
//		i, VF_NAME(&vec_func_tbl[i]),
//		VF_CODE(&vec_func_tbl[i]),
//		VF_NAME(&vec_func_tbl[ VF_CODE(&vec_func_tbl[i]) ]) );
//			ERROR1(ERROR_STRING);
//		}

		assert( VF_CODE(&vec_func_tbl[i]) == i );
	}
#endif /* CAUTIOUS */

	// Initialize the platforms
	//init_all_platforms(SINGLE_QSP_ARG);
	vl2_init_platform(SINGLE_QSP_ARG);

	/* now create some items */

	create_vfs(SINGLE_QSP_ARG);

	declare_vector_functions();

	/* We used to have these in a table, indexed by their code, for fast lookup...
	 * Would that be a good idea still?
	 */

	set_random_seed(SINGLE_QSP_ARG);	/* use low order bits of microsecond clock */

	inited++;
}

#endif // FOOBAR
