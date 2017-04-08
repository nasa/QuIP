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


