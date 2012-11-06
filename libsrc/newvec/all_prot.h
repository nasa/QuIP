#ifndef ALL_PROT_DECLS

#include "vecgen.h"

#define EXT_FFT_DECL( key, func )					\
extern void key##_obj_##func(FFT_Args *);

#define EXT_PROT_DECL( key, func )					\
extern void key##_obj_##func(Vec_Obj_Args *);


/*
#define RAMP_2D_DECL( rkey )						\
extern void rkey##_obj_ramp2d( Vec_Obj_Args * );
*/


#define ODD_DECLS( key )						\
extern void key##_obj_xform_list ( QSP_ARG_DECL  Data_Obj *, Data_Obj *, Data_Obj *);	\
extern void key##_obj_vec_xform ( QSP_ARG_DECL  Data_Obj *, Data_Obj *, Data_Obj *);	\
extern void key##_ramp2d ( Vec_Obj_Args * );				\
extern void key##_obj_c_2dfft( Vec_Obj_Args *, int is_inv );		\
extern void key##_obj_c_rowfft( Vec_Obj_Args *, int is_inv );		\
extern void key##_obj_r_2dfft( Vec_Obj_Args * );			\
extern void key##_obj_r_rowfft( Vec_Obj_Args * );			\
extern void key##_obj_r_2dift( Vec_Obj_Args * );			\
extern void key##_obj_r_rowift( Vec_Obj_Args * );			\
extern void key##_obj_homog_xform( QSP_ARG_DECL  Data_Obj *, Data_Obj *, Data_Obj *);	\
extern double key##_obj_determinant( Data_Obj * );


#define ALL_PROT_DECLS( key )						\
EXT_PROT_DECL( key, vconv_from_bit )					\
EXT_PROT_DECL( key, vconv_to_bit )					\
EXT_PROT_DECL( key, vbnd )						\
EXT_PROT_DECL( key, vclip )						\
EXT_PROT_DECL( key, vsm_lt )						\
EXT_PROT_DECL( key, vsm_gt )						\
EXT_PROT_DECL( key, vsm_le )						\
EXT_PROT_DECL( key, vsm_ge )						\
EXT_PROT_DECL( key, vsm_ne )						\
EXT_PROT_DECL( key, vsm_eq )						\
EXT_PROT_DECL( key, vvm_lt )						\
EXT_PROT_DECL( key, vvm_gt )						\
EXT_PROT_DECL( key, vvm_le )						\
EXT_PROT_DECL( key, vvm_ge )						\
EXT_PROT_DECL( key, vvm_ne )						\
EXT_PROT_DECL( key, vvm_eq )						\
EXT_PROT_DECL( key, viclp )						\
EXT_PROT_DECL( key, vabs )						\
EXT_PROT_DECL( key, rvvv_slct )						\
EXT_PROT_DECL( key, rvvs_slct )						\
EXT_PROT_DECL( key, rvss_slct )						\
EXT_PROT_DECL( key, vmcmp )						\
EXT_PROT_DECL( key, vmcmm )						\
EXT_PROT_DECL( key, rvset )						\
EXT_PROT_DECL( key, rvsum )						\
EXT_PROT_DECL( key, vcmp )						\
EXT_PROT_DECL( key, vibnd )						\
EXT_PROT_DECL( key, vsign )						\
EXT_PROT_DECL( key, vmax )						\
EXT_PROT_DECL( key, vmin )						\
EXT_PROT_DECL( key, vsmax )						\
EXT_PROT_DECL( key, vsmin )						\
EXT_PROT_DECL( key, vsmxm )						\
EXT_PROT_DECL( key, vsmnm )						\
EXT_PROT_DECL( key, vscmp )						\
EXT_PROT_DECL( key, vscmp2 )						\
EXT_PROT_DECL( key, vscmm )						\
EXT_PROT_DECL( key, rvmov )						\
EXT_PROT_DECL( key, rvsqr )						\
EXT_PROT_DECL( key, rvneg )						\
EXT_PROT_DECL( key, rvadd )						\
EXT_PROT_DECL( key, rvsub )						\
EXT_PROT_DECL( key, rvdiv )						\
EXT_PROT_DECL( key, rvmul )						\
EXT_PROT_DECL( key, rvdot )						\
EXT_PROT_DECL( key, rvrand )						\
EXT_PROT_DECL( key, vmaxg )						\
EXT_PROT_DECL( key, vming )						\
EXT_PROT_DECL( key, vmaxi )						\
EXT_PROT_DECL( key, vmini )						\
EXT_PROT_DECL( key, vmxmi )						\
EXT_PROT_DECL( key, vmnmi )						\
EXT_PROT_DECL( key, vmxmg )						\
EXT_PROT_DECL( key, vmnmg )						\
EXT_PROT_DECL( key, vmaxv )						\
EXT_PROT_DECL( key, vminv )						\
EXT_PROT_DECL( key, vmxmv )						\
EXT_PROT_DECL( key, vmnmv )						\
EXT_PROT_DECL( key, vmaxm )						\
EXT_PROT_DECL( key, vminm )						\
									\
EXT_PROT_DECL( key, vv_vv_lt )						\
EXT_PROT_DECL( key, vv_vv_gt )						\
EXT_PROT_DECL( key, vv_vv_le )						\
EXT_PROT_DECL( key, vv_vv_ge )						\
EXT_PROT_DECL( key, vv_vv_eq )						\
EXT_PROT_DECL( key, vv_vv_ne )						\
									\
EXT_PROT_DECL( key, vv_vs_lt )						\
EXT_PROT_DECL( key, vv_vs_gt )						\
EXT_PROT_DECL( key, vv_vs_le )						\
EXT_PROT_DECL( key, vv_vs_ge )						\
EXT_PROT_DECL( key, vv_vs_eq )						\
EXT_PROT_DECL( key, vv_vs_ne )						\
									\
EXT_PROT_DECL( key, vs_vv_lt )						\
EXT_PROT_DECL( key, vs_vv_gt )						\
EXT_PROT_DECL( key, vs_vv_le )						\
EXT_PROT_DECL( key, vs_vv_ge )						\
EXT_PROT_DECL( key, vs_vv_eq )						\
EXT_PROT_DECL( key, vs_vv_ne )						\
									\
EXT_PROT_DECL( key, vs_vs_lt )						\
EXT_PROT_DECL( key, vs_vs_gt )						\
EXT_PROT_DECL( key, vs_vs_le )						\
EXT_PROT_DECL( key, vs_vs_ge )						\
EXT_PROT_DECL( key, vs_vs_eq )						\
EXT_PROT_DECL( key, vs_vs_ne )						\
									\
EXT_PROT_DECL( key, ss_vv_lt )						\
EXT_PROT_DECL( key, ss_vv_gt )						\
EXT_PROT_DECL( key, ss_vv_le )						\
EXT_PROT_DECL( key, ss_vv_ge )						\
EXT_PROT_DECL( key, ss_vv_eq )						\
EXT_PROT_DECL( key, ss_vv_ne )						\
									\
EXT_PROT_DECL( key, ss_vs_lt )						\
EXT_PROT_DECL( key, ss_vs_gt )						\
EXT_PROT_DECL( key, ss_vs_le )						\
EXT_PROT_DECL( key, ss_vs_ge )						\
EXT_PROT_DECL( key, ss_vs_eq )						\
EXT_PROT_DECL( key, ss_vs_ne )						\
									\
EXT_PROT_DECL( key, rvsadd )						\
EXT_PROT_DECL( key, rvssub )						\
EXT_PROT_DECL( key, rvsmul )						\
EXT_PROT_DECL( key, rvsdiv )						\
EXT_PROT_DECL( key, rvsdiv2 )						\
									\
EXT_PROT_DECL( key, vscml )						\
EXT_PROT_DECL( key, vcm )						\
EXT_PROT_DECL( key, vramp1d )						\
EXT_PROT_DECL( key, vramp2d )

/* end ALL_PROT_DECLS */


#define ALL_INT_DECLS( key )						\
EXT_PROT_DECL( key, vcomp )						\
EXT_PROT_DECL( key, vand )						\
EXT_PROT_DECL( key, vnand )						\
EXT_PROT_DECL( key, vsnand )						\
EXT_PROT_DECL( key, vor )						\
EXT_PROT_DECL( key, vxor )						\
EXT_PROT_DECL( key, vsand )						\
EXT_PROT_DECL( key, vshr )						\
EXT_PROT_DECL( key, vsshr )						\
EXT_PROT_DECL( key, vsshr2 )						\
EXT_PROT_DECL( key, vshl )						\
EXT_PROT_DECL( key, vsshl )						\
EXT_PROT_DECL( key, vsshl2 )						\
EXT_PROT_DECL( key, vsor )						\
EXT_PROT_DECL( key, vsxor )						\
EXT_PROT_DECL( key, vnot )						\
EXT_PROT_DECL( key, vmod )						\
EXT_PROT_DECL( key, vsmod )						\
EXT_PROT_DECL( key, vsmod2 )


#define ALL_FP_DECLS( key )						\
EXT_PROT_DECL( key, vj0 )						\
EXT_PROT_DECL( key, vj1 )						\
EXT_PROT_DECL( key, vrint )						\
EXT_PROT_DECL( key, vfloor )						\
EXT_PROT_DECL( key, vround )						\
EXT_PROT_DECL( key, vceil )						\
EXT_PROT_DECL( key, vmgsq )						\
EXT_PROT_DECL( key, vsqrt )						\
EXT_PROT_DECL( key, vuni )						\
EXT_PROT_DECL( key, vlog )						\
EXT_PROT_DECL( key, vlog10 )						\
EXT_PROT_DECL( key, vexp )						\
EXT_PROT_DECL( key, vatan )						\
EXT_PROT_DECL( key, vatn2 )						\
EXT_PROT_DECL( key, vatan2 )						\
EXT_PROT_DECL( key, vsatan2 )						\
EXT_PROT_DECL( key, vsatan22 )						\
EXT_PROT_DECL( key, vspow )						\
EXT_PROT_DECL( key, vspow2 )						\
EXT_PROT_DECL( key, vtan )						\
EXT_PROT_DECL( key, vcos )						\
EXT_PROT_DECL( key, verf )						\
EXT_PROT_DECL( key, vacos )						\
EXT_PROT_DECL( key, vsin )						\
EXT_PROT_DECL( key, vasin )						\
EXT_PROT_DECL( key, rvpow )						\
EXT_PROT_DECL( key, cvmov )						\
EXT_PROT_DECL( key, qvmov )						\
EXT_PROT_DECL( key, vcmul )						\
EXT_PROT_DECL( key, vconj )						\
EXT_PROT_DECL( key, cvneg )						\
EXT_PROT_DECL( key, cvdot )						\
EXT_PROT_DECL( key, cvrand )						\
EXT_PROT_DECL( key, qvneg )						\
EXT_PROT_DECL( key, rvfft )						\
EXT_PROT_DECL( key, rvift )						\
EXT_PROT_DECL( key, cvfft )						\
EXT_PROT_DECL( key, cvift )						\
									\
EXT_PROT_DECL( key, cvpow )						\
EXT_PROT_DECL( key, cvsqr )						\
EXT_PROT_DECL( key, cvvv_slct )						\
EXT_PROT_DECL( key, cvvs_slct )						\
EXT_PROT_DECL( key, cvss_slct )						\
EXT_PROT_DECL( key, cvset )						\
EXT_PROT_DECL( key, cvsum )						\
									\
EXT_PROT_DECL( key, cvadd )						\
EXT_PROT_DECL( key, cvsub )						\
EXT_PROT_DECL( key, cvmul )						\
EXT_PROT_DECL( key, cvdiv )						\
									\
EXT_PROT_DECL( key, qvsqr )						\
EXT_PROT_DECL( key, qvvv_slct )						\
EXT_PROT_DECL( key, qvvs_slct )						\
EXT_PROT_DECL( key, qvss_slct )						\
EXT_PROT_DECL( key, qvset )						\
EXT_PROT_DECL( key, qvsum )						\
									\
EXT_PROT_DECL( key, qvadd )						\
EXT_PROT_DECL( key, qvsub )						\
EXT_PROT_DECL( key, qvmul )						\
EXT_PROT_DECL( key, qvdiv )						\
									\
EXT_PROT_DECL( key, pvadd )						\
EXT_PROT_DECL( key, pvsub )						\
EXT_PROT_DECL( key, pvmul )						\
EXT_PROT_DECL( key, pvdiv )						\
									\
EXT_PROT_DECL( key, mvadd )						\
EXT_PROT_DECL( key, mvsub )						\
EXT_PROT_DECL( key, mvmul )						\
EXT_PROT_DECL( key, mvdiv )						\
									\
EXT_PROT_DECL( key, cvsadd )						\
EXT_PROT_DECL( key, cvssub )						\
EXT_PROT_DECL( key, cvsmul )						\
EXT_PROT_DECL( key, cvsdiv )						\
EXT_PROT_DECL( key, cvsdiv2 )						\
									\
EXT_PROT_DECL( key, mvsadd )						\
EXT_PROT_DECL( key, mvssub )						\
EXT_PROT_DECL( key, mvsmul )						\
EXT_PROT_DECL( key, mvsdiv )						\
EXT_PROT_DECL( key, mvsdiv2 )						\
									\
EXT_PROT_DECL( key, qvsadd )						\
EXT_PROT_DECL( key, qvssub )						\
EXT_PROT_DECL( key, qvsmul )						\
EXT_PROT_DECL( key, qvsdiv )						\
EXT_PROT_DECL( key, qvsdiv2 )						\
									\
EXT_PROT_DECL( key, pvsadd )						\
EXT_PROT_DECL( key, pvssub )						\
EXT_PROT_DECL( key, pvsmul )						\
EXT_PROT_DECL( key, pvsdiv )						\
EXT_PROT_DECL( key, pvsdiv2 )						\
									\
ODD_DECLS( key )							\

#endif /* ! ALL_PROT_DECLS */

