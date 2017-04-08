/* ocl_veclib_prot.m4 BEGIN */
/* ocl_port.m4 BEGIN */


/* gen_port.m4 BEGIN */
#include "quip_prot.h"
#include "shape_bits.h"

/* NOT Suppressing ! */


/* gen_port.m4 DONE */




/* Suppressing ! */

/* NOT Suppressing ! */


#include "platform.h"
extern void *ocl_tmp_vec(Platform_Device *pdp, size_t size, size_t len, const char *whence);
extern void ocl_free_tmp(void *a, const char *whence);
extern int get_max_threads_per_block(Data_Obj *odp);
extern int max_threads_per_block;

/* ocl_port.m4 END */





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

// BEGIN INCLUDED FILE ../../include/veclib/ocl_func_prot.m4

extern void h_ocl_vuni(const int vf_code,  /*const*/ Vec_Obj_Args *oap);
extern void h_ocl_sp_vuni(const int vf_code,  /*const*/ Vec_Obj_Args *oap);
extern void h_ocl_dp_vuni(const int vf_code,  /*const*/ Vec_Obj_Args *oap);


extern void h_ocl_fft2d(const int vf_code,  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_ocl_ift2d(const int vf_code,  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_ocl_fftrows(const int vf_code,  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void h_ocl_iftrows(const int vf_code,  Data_Obj *dst_dp, Data_Obj *src_dp);





/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/ocl_func_prot.m4




/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/platform_funcs.m4



extern COMMAND_FUNC( do_ocl_obj_dnload );
extern COMMAND_FUNC( do_ocl_obj_upload );
extern COMMAND_FUNC( do_ocl_list_devs );
extern COMMAND_FUNC( do_ocl_set_device );

extern void ocl_set_device(QSP_ARG_DECL  Platform_Device *cdp);
extern Platform_Stream * ocl_new_stream(QSP_ARG_DECL  const char *name);

extern void ocl_list_streams(SINGLE_QSP_ARG_DECL);
extern COMMAND_FUNC( do_ocl_new_stream );
extern COMMAND_FUNC( do_ocl_list_streams );
extern COMMAND_FUNC( do_ocl_stream_info );
extern COMMAND_FUNC( do_ocl_sync_stream );
extern COMMAND_FUNC( do_ocl_init_ckpts );
extern COMMAND_FUNC( do_ocl_set_ckpt );
extern COMMAND_FUNC( do_ocl_clear_ckpts );
extern COMMAND_FUNC( do_ocl_show_ckpts );
extern void ocl_shutdown(void);
extern void ocl_sync(SINGLE_QSP_ARG_DECL);
extern void ocl_init_dev_memory(QSP_ARG_DECL  Platform_Device *pdp);
extern void ocl_insure_device(QSP_ARG_DECL  Data_Obj *dp);

	

extern void ocl_init_platform(SINGLE_QSP_ARG_DECL);
extern void ocl_init(SINGLE_QSP_ARG_DECL);
extern void ocl_alloc_data(QSP_ARG_DECL  Data_Obj *dp, dimension_t size);





/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/platform_funcs.m4

/* ocl_veclib_prot.m4 DONE */


