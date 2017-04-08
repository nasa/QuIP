#include "quip_config.h"
#include "quip_prot.h"
#include "data_obj.h"

/* vl2_veclib_prot.m4 BEGIN */
/* vl2_port.m4 BEGIN */

/* gen_port.m4 BEGIN */
#include "quip_prot.h"
#include "shape_bits.h"

/* NOT Suppressing ! */


/* gen_port.m4 DONE */




/* Suppressing ! */

/* NOT Suppressing ! */


// vl2_port.m4 - declaring tmp_vec functions
extern void *vl2_tmp_vec (Platform_Device *pdp, size_t size, size_t len, const char *whence);
extern void vl2_free_tmp (void *a, const char *whence);

#include <math.h>	// isinf etc







/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/vecgen.m4
/* vecgen.m4 BEGIN */

/* defns shared by veclib & warlib */







// Why are these called link funcs?  Maybe because they can be chained?
// Kind of a legacy from the old skywarrior library code...
// A vector arg used to just have a length and a stride, but now
// with gpus we have three-dimensional lengths.  But in principle
// there's no reason why we couldn't have full shapes passed...



/* vecgen.m4 DONE */




/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/vecgen.m4




/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/vl2_func_prot.m4


//extern void h_vl2_vuni(const int vf_code,  /*const*/ Vec_Obj_Args *oap);
//extern void h_vl2_sp_vuni(const int vf_code,  /*const*/ Vec_Obj_Args *oap);
//extern void h_vl2_dp_vuni(const int vf_code,  /*const*/ Vec_Obj_Args *oap);

extern void h_vl2_fft2d(const int vf_code,  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_vl2_ift2d(const int vf_code,  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_vl2_fftrows(const int vf_code,  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_vl2_iftrows(const int vf_code,  Data_Obj *dst_dp, Data_Obj *src_dp);

extern void h_vl2_xform_list(const int code, Vec_Obj_Args *oap);
extern void h_vl2_vec_xform(const int code, Vec_Obj_Args *oap);
extern void h_vl2_homog_xform(const int vf_code,  /*const*/ Vec_Obj_Args *oap);
extern void h_vl2_determinant(const int vf_code,  /*const*/ Vec_Obj_Args *oap);

extern int xform_chk(Data_Obj *dpto, Data_Obj *dpfr, Data_Obj *xform );



/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/vl2_func_prot.m4




/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/platform_funcs.m4


	


extern void vl2_init_platform(SINGLE_QSP_ARG_DECL);
extern void vl2_init(SINGLE_QSP_ARG_DECL);
extern void vl2_alloc_data(QSP_ARG_DECL  Data_Obj *dp, dimension_t size);





/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/platform_funcs.m4

/* vl2_veclib_prot.m4 DONE */






/* NOT Suppressing ! */

// BEGIN INCLUDED FILE _vl2_utils.c
//#include "quip_config.h"
//#include "quip_prot.h"
#include "my_vl2.h"
//#include "veclib/vl2_port.h"
//#include "veclib/platform_funcs.h"
#include "veclib_api.h"

//#define MEM_SIZE (16)//suppose we have a vector with 128 elements
#define MAX_SOURCE_SIZE (0x100000)

//In general Intel CPU and NV/AMD's GPU are in different platforms
//But in Mac OSX, all the OpenCL devices are in the platform "Apple"

#define MAX_CL_PLATFORMS	3

#define MAX_PARAM_SIZE	128

#define ERROR_CASE(code,string)	case code: msg = string; break;

#ifdef CAUTIOUS
#define INSURE_CURR_ODP(whence)					\
	if( curr_odp == NULL ){					\
		sprintf(ERROR_STRING,"CAUTIOUS:  %s:  curr_odp is null!?",#whence);	\
		WARN(ERROR_STRING);				\
	}
#else // ! CAUTIOUS
#define INSURE_CURR_ODP(whence)
#endif // ! CAUTIOUS

void vl2_init(SINGLE_QSP_ARG_DECL)
{
	// anything to do?
}

void vl2_alloc_data(QSP_ARG_DECL  Data_Obj *dp, dimension_t size)
{
    	OBJ_DATA_PTR(dp) = getbuf(size);
}

//	put the space after tmpvec_name to prevent parens from being eaten as macro args???

void *vl2_tmp_vec (Platform_Device *pdp, size_t size, size_t len, const char *whence)
{
	return getbuf( size * len );
}

void vl2_free_tmp (void *a, const char *whence)
{
	givbuf(a);
}




/* NOT Suppressing ! */

// END INCLUDED FILE _vl2_utils.c


