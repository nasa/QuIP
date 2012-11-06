

#define ODD_DECLS( key )						\
extern void key##_ramp2d ( Data_Obj *, double , double, double );	\
extern void key##c_2dfft( Data_Obj *, int is_inv );			\
extern void key##r_2dfft( Data_Obj *,Data_Obj *);			\
extern void key##r_2dift( Data_Obj *,Data_Obj *);

#define FP_PROT_DECL( key, func )						\
extern void key##func( Vec_Args *);

#define FFT_PROT_DECL( key, func )						\
extern void key##func( FFT_Args *);

#define ALL_FP_DECLS( key )						\
FP_PROT_DECL( key, vj0 )				\
FP_PROT_DECL( key, vj1 )				\
FP_PROT_DECL( key, vrint )				\
FP_PROT_DECL( key, vfloor )				\
FP_PROT_DECL( key, vround )				\
FP_PROT_DECL( key, vceil )				\
FP_PROT_DECL( key, rvsqr )				\
FP_PROT_DECL( key, vmgsq )				\
FP_PROT_DECL( key, cvsqr )				\
FP_PROT_DECL( key, vsqrt )				\
FP_PROT_DECL( key, vuni )				\
FP_PROT_DECL( key, vlog )				\
FP_PROT_DECL( key, vlog10 )				\
FP_PROT_DECL( key, vexp )				\
FP_PROT_DECL( key, vatan )				\
FP_PROT_DECL( key, vatn2 )				\
FP_PROT_DECL( key, vatan2 )				\
FP_PROT_DECL( key, vsatan2 )				\
FP_PROT_DECL( key, vsatan22 )				\
FP_PROT_DECL( key, vspow )				\
FP_PROT_DECL( key, vspow2 )				\
FP_PROT_DECL( key, vtan )				\
FP_PROT_DECL( key, vcos )				\
FP_PROT_DECL( key, verf )				\
FP_PROT_DECL( key, vacos )				\
FP_PROT_DECL( key, vsin )				\
FP_PROT_DECL( key, vasin )				\
FP_PROT_DECL( key, rvpow )				\
FP_PROT_DECL( key, cvpow )				\
FP_PROT_DECL( key, rvfft )				\
FP_PROT_DECL( key, rvift )				\
FP_PROT_DECL( key, cvfft )				\
FP_PROT_DECL( key, cvift )				\
ODD_DECLS( key )
