/* cu2_veclib_prot.m4 BEGIN */
/* cu2_port.m4 BEGIN */
/* gen_port.m4 BEGIN */
#include "quip_prot.h"
#include "shape_bits.h"

/* NOT Suppressing ! */


/* gen_port.m4 DONE */



#define BUILD_FOR_CUDA
#define BUILD_FOR_GPU


/* Suppressing ! */

/* NOT Suppressing ! */


extern void *cu2_tmp_vec(Platform_Device *pdp, size_t size, size_t len, const char *whence);
extern void cu2_free_tmp(void *a, const char *whence);

/* cu2_port.m4 DONE */





/* NOT Suppressing ! */

// BEGIN INCLUDED FILE veclib/vecgen.m4
/* vecgen.m4 BEGIN */

/* defns shared by veclib & warlib */







// Why are these called link funcs?  Maybe because they can be chained?
// Kind of a legacy from the old skywarrior library code...
// A vector arg used to just have a length and a stride, but now
// with gpus we have three-dimensional lengths.  But in principle
// there's no reason why we couldn't have full shapes passed...



/* vecgen.m4 DONE */




/* NOT Suppressing ! */

// END INCLUDED FILE veclib/vecgen.m4




/* NOT Suppressing ! */

// BEGIN INCLUDED FILE veclib/cu2_func_prot.m4

/* cu2_func_prot.m4 BEGIN */

// these are special cases...
extern void h_cu2_sp_vuni(const int vf_code,  /*const*/ Vec_Obj_Args *oap);
extern void h_cu2_dp_vuni(const int vf_code,  /*const*/ Vec_Obj_Args *oap);
extern void h_cu2_vuni(const int vf_code,  /*const*/ Vec_Obj_Args *oap);

// BUG special case needs to be handled!?
extern void h_cu2_vshl(const int vf_code,  /*const*/ Vec_Obj_Args *oap);
extern void h_cu2_vsshl(const int vf_code,  /*const*/ Vec_Obj_Args *oap);
extern void h_cu2_vsshl2(const int vf_code,  /*const*/ Vec_Obj_Args *oap);

extern void g_cu2_vfft(QSP_ARG_DECL  Data_Obj *dpto, Data_Obj *dpfr);
extern void h_cu2_fft2d(const int vf_code,  /*const*/ Vec_Obj_Args *oap);
extern void h_cu2_ift2d(const int vf_code,  /*const*/ Vec_Obj_Args *oap);
extern void h_cu2_fftrows(const int vf_code,  /*const*/ Vec_Obj_Args *oap);
extern void h_cu2_iftrows(const int vf_code,  /*const*/ Vec_Obj_Args *oap);

/* cu2_func_prot.m4 DONE */



/* NOT Suppressing ! */

// END INCLUDED FILE veclib/cu2_func_prot.m4




/* NOT Suppressing ! */

// BEGIN INCLUDED FILE veclib/platform_funcs.m4



extern COMMAND_FUNC( do_cu2_obj_dnload );
extern COMMAND_FUNC( do_cu2_obj_upload );
extern COMMAND_FUNC( do_cu2_list_devs );
extern COMMAND_FUNC( do_cu2_set_device );

extern void cu2_set_device(QSP_ARG_DECL  Platform_Device *cdp);
extern Platform_Stream * cu2_new_stream(QSP_ARG_DECL  const char *name);

extern void cu2_list_streams(SINGLE_QSP_ARG_DECL);
extern COMMAND_FUNC( do_cu2_new_stream );
extern COMMAND_FUNC( do_cu2_list_streams );
extern COMMAND_FUNC( do_cu2_stream_info );
extern COMMAND_FUNC( do_cu2_sync_stream );
extern COMMAND_FUNC( do_cu2_init_ckpts );
extern COMMAND_FUNC( do_cu2_set_ckpt );
extern COMMAND_FUNC( do_cu2_clear_ckpts );
extern COMMAND_FUNC( do_cu2_show_ckpts );
extern void cu2_shutdown(void);
extern void cu2_sync(SINGLE_QSP_ARG_DECL);
extern void cu2_init_dev_memory(QSP_ARG_DECL  Platform_Device *pdp);
extern void cu2_insure_device(QSP_ARG_DECL  Data_Obj *dp);

	


extern void cu2_init_platform(SINGLE_QSP_ARG_DECL);
extern void cu2_init(SINGLE_QSP_ARG_DECL);
extern void cu2_alloc_data(QSP_ARG_DECL  Data_Obj *dp, dimension_t size);





/* NOT Suppressing ! */

// END INCLUDED FILE veclib/platform_funcs.m4

/* cu2_veclib_prot.m4 DONE */


