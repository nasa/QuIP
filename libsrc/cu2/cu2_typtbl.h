// There doesn't seem to be anything cuda-specific here, share and move
// to include dir?

// FOOBAR - simd doesn't seem to be used here?
#ifdef USE_SSE
#define SIMD_NAME(stem)		simd_##stem
#else /* ! USE_SSE */
#define SIMD_NAME(stem)		nullobjf
#endif /* ! USE_SSE */

#define NULL_5							\
								\
          nullobjf, nullobjf, nullobjf, nullobjf, nullobjf

#define NULL_4							\
								\
          nullobjf, nullobjf, nullobjf, nullobjf

#define NULL_3							\
								\
          nullobjf, nullobjf, nullobjf

#define NULL_2							\
								\
          nullobjf, nullobjf

#define DUP_4(stem)						\
								\
	stem, stem, stem, stem
	
#define DUP_2(stem)						\
								\
	stem, stem

#define DUP_3(stem)						\
								\
	stem, stem, stem

#define ALL_NULL						\
								\
	  NULL_4,						\
	  NULL_2,						\
	  NULL_4,						\
	  NULL_5

#define DUP_ALL(stem,bmfunc)					\
								\
	DUP_4(stem),						\
	DUP_2(stem),						\
	DUP_4(stem),						\
	DUP_4(stem),						\
	bmfunc

#define FLT_ALL( stem )						\
								\
	NULL_4,							\
	HOST_TYPED_CALL_NAME(stem,sp),				\
	HOST_TYPED_CALL_NAME(stem,dp),				\
	NULL_4,							\
	NULL_3,							\
	HOST_TYPED_CALL_NAME(stem,spdp),			\
	nullobjf

#define RFLT_ALL(stem)						\
								\
	NULL_4,							\
	HOST_TYPED_CALL_NAME_REAL(stem,sp),				\
	HOST_TYPED_CALL_NAME_REAL(stem,dp),				\
	NULL_4,							\
	NULL_3,							\
	HOST_TYPED_CALL_NAME_REAL(stem,spdp),			\
	nullobjf

#define RFLT_NO_MIXED(stem)					\
								\
	NULL_4,							\
	HOST_TYPED_CALL_NAME_REAL(stem,sp),				\
	HOST_TYPED_CALL_NAME_REAL(stem,dp),				\
	NULL_4,							\
	NULL_5

#define CFLT_ALL(stem)						\
								\
	NULL_4,							\
	HOST_TYPED_CALL_NAME_CPX(stem,sp),				\
	HOST_TYPED_CALL_NAME_CPX(stem,dp),				\
	NULL_4,							\
	NULL_3,							\
	HOST_TYPED_CALL_NAME_CPX(stem,spdp),			\
	nullobjf

#define CFLT_NO_MIXED(stem)					\
								\
	NULL_4,							\
	HOST_TYPED_CALL_NAME_CPX(stem,sp),			\
	HOST_TYPED_CALL_NAME_CPX(stem,dp),			\
	NULL_4,							\
	NULL_5

#define MFLT_ALL(stem)						\
								\
	NULL_4,							\
	HOST_TYPED_CALL_NAME_MIXED(stem,sp),			\
	HOST_TYPED_CALL_NAME_MIXED(stem,dp),			\
	NULL_4,							\
	NULL_3,							\
	HOST_TYPED_CALL_NAME_MIXED(stem,spdp)

#define INT_ALL(stem)						\
								\
	HOST_TYPED_CALL_NAME(stem,by),						\
	HOST_TYPED_CALL_NAME(stem,in),						\
	HOST_TYPED_CALL_NAME(stem,di),						\
	HOST_TYPED_CALL_NAME(stem,li),						\
	nullobjf,						\
	nullobjf,						\
	HOST_TYPED_CALL_NAME(stem,uby),					\
	HOST_TYPED_CALL_NAME(stem,uin),					\
	HOST_TYPED_CALL_NAME(stem,udi),					\
	HOST_TYPED_CALL_NAME(stem,uli),					\
	HOST_TYPED_CALL_NAME(stem,ubyin),					\
	HOST_TYPED_CALL_NAME(stem,inby),					\
	HOST_TYPED_CALL_NAME(stem,uindi),					\
	nullobjf,						\
	HOST_TYPED_CALL_NAME(stem,bit)


/* this is for vneg */

#define SIGNED_ALL_REAL(stem)					\
								\
	HOST_TYPED_CALL_NAME_REAL(stem,by),						\
	HOST_TYPED_CALL_NAME_REAL(stem,in),						\
	HOST_TYPED_CALL_NAME_REAL(stem,di),						\
	HOST_TYPED_CALL_NAME_REAL(stem,li),						\
	HOST_TYPED_CALL_NAME_REAL(stem,sp),						\
	HOST_TYPED_CALL_NAME_REAL(stem,dp),						\
	nullobjf,	/* uby */				\
	nullobjf,	/* uin */				\
	nullobjf,	/* udi */				\
	nullobjf,	/* uli */				\
	nullobjf,	/* ubyin */				\
	HOST_TYPED_CALL_NAME_REAL(stem,inby),					\
	nullobjf,	/* uindi */				\
	HOST_TYPED_CALL_NAME_REAL(stem,spdp),					\
	nullobjf	/* bit */

/* 12B means no Bitmap */

#define INT_ALL_NO_BITMAP(stem)					\
								\
	HOST_TYPED_CALL_NAME(stem,by),						\
	HOST_TYPED_CALL_NAME(stem,in),						\
	HOST_TYPED_CALL_NAME(stem,di),						\
	HOST_TYPED_CALL_NAME(stem,li),						\
	nullobjf,						\
	nullobjf,						\
	HOST_TYPED_CALL_NAME(stem,uby),					\
	HOST_TYPED_CALL_NAME(stem,uin),					\
	HOST_TYPED_CALL_NAME(stem,udi),					\
	HOST_TYPED_CALL_NAME(stem,uli),					\
	HOST_TYPED_CALL_NAME(stem,ubyin),					\
	HOST_TYPED_CALL_NAME(stem,inby),					\
	HOST_TYPED_CALL_NAME(stem,uindi),					\
	nullobjf,						\
	nullobjf


#define REAL_INT_ALL(stem)					\
								\
	byr##stem,  inr##stem,  dir##stem,  lir##stem,		\
	nullobjf, nullobjf,					\
	ubyr##stem, uinr##stem, udir##stem, uli##stem,		\
	ubyinr##stem, inbyr##stem, uindir##stem, nullobjf, nullobjf


#define ALL_NO_BITMAP(stem)					\
								\
	HOST_TYPED_CALL_NAME(stem,by),				\
	HOST_TYPED_CALL_NAME(stem,in),				\
	HOST_TYPED_CALL_NAME(stem,di),				\
	HOST_TYPED_CALL_NAME(stem,li),				\
	HOST_TYPED_CALL_NAME(stem,sp),				\
	HOST_TYPED_CALL_NAME(stem,dp),				\
	HOST_TYPED_CALL_NAME(stem,uby),				\
	HOST_TYPED_CALL_NAME(stem,uin),				\
	HOST_TYPED_CALL_NAME(stem,udi),				\
	HOST_TYPED_CALL_NAME(stem,uli),				\
	HOST_TYPED_CALL_NAME(stem,ubyin),			\
	HOST_TYPED_CALL_NAME(stem,inby),			\
	HOST_TYPED_CALL_NAME(stem,uindi),			\
	HOST_TYPED_CALL_NAME(stem,spdp),			\
	nullobjf

#define ALL_REAL_NO_BITMAP_SSE(stem)				\
								\
	HOST_TYPED_CALL_NAME_REAL(stem,by),						\
	HOST_TYPED_CALL_NAME_REAL(stem,in),						\
	HOST_TYPED_CALL_NAME_REAL(stem,di),						\
	HOST_TYPED_CALL_NAME_REAL(stem,li),						\
	stem,							\
	/* SIMD_NAME(stem), */					\
	/* HOST_TYPED_CALL_NAME_REAL(stem,sp), */					\
	HOST_TYPED_CALL_NAME_REAL(stem,dp),						\
	HOST_TYPED_CALL_NAME_REAL(stem,uby),					\
	HOST_TYPED_CALL_NAME_REAL(stem,uin),					\
	HOST_TYPED_CALL_NAME_REAL(stem,udi),					\
	HOST_TYPED_CALL_NAME_REAL(stem,uli),					\
	HOST_TYPED_CALL_NAME_REAL(stem,ubyin),					\
	HOST_TYPED_CALL_NAME_REAL(stem,inby),					\
	HOST_TYPED_CALL_NAME_REAL(stem,uindi),					\
	HOST_TYPED_CALL_NAME_REAL(stem,spdp),					\
	nullobjf

#define ALL_REAL_SAME_PREC_NO_BITMAP(stem)					\
										\
	HOST_TYPED_CALL_NAME_REAL(stem,by),					\
	HOST_TYPED_CALL_NAME_REAL(stem,in),					\
	HOST_TYPED_CALL_NAME_REAL(stem,di),					\
	HOST_TYPED_CALL_NAME_REAL(stem,li),					\
	HOST_TYPED_CALL_NAME_REAL(stem,sp),					\
	HOST_TYPED_CALL_NAME_REAL(stem,dp),					\
	HOST_TYPED_CALL_NAME_REAL(stem,uby),					\
	HOST_TYPED_CALL_NAME_REAL(stem,uin),					\
	HOST_TYPED_CALL_NAME_REAL(stem,udi),					\
	HOST_TYPED_CALL_NAME_REAL(stem,uli),					\
	NULL_4,									\
	nullobjf

#define ALL_REAL_SAME_PREC_NO_BITMAP_SSE(stem)				\
								\
	HOST_TYPED_CALL_NAME_REAL(stem,by),						\
	HOST_TYPED_CALL_NAME_REAL(stem,in),						\
	HOST_TYPED_CALL_NAME_REAL(stem,di),						\
	HOST_TYPED_CALL_NAME_REAL(stem,li),						\
	stem,							\
	/* SIMD_NAME(stem), */					\
	/* HOST_TYPED_CALL_NAME_REAL(stem,sp), */					\
	HOST_TYPED_CALL_NAME_REAL(stem,dp),						\
	HOST_TYPED_CALL_NAME_REAL(stem,uby),					\
	HOST_TYPED_CALL_NAME_REAL(stem,uin),					\
	HOST_TYPED_CALL_NAME_REAL(stem,udi),					\
	HOST_TYPED_CALL_NAME_REAL(stem,uli),					\
	NULL_4,							\
	nullobjf

#define ALL_REAL_SSE(stem)					\
								\
	HOST_TYPED_CALL_NAME_REAL(stem,by),						\
	HOST_TYPED_CALL_NAME_REAL(stem,in),						\
	HOST_TYPED_CALL_NAME_REAL(stem,di),						\
	HOST_TYPED_CALL_NAME_REAL(stem,li),						\
	stem,							\
	/* SIMD_NAME(stem), */					\
	/* HOST_TYPED_CALL_NAME_REAL(stem,sp), */					\
	HOST_TYPED_CALL_NAME_REAL(stem,dp),						\
	HOST_TYPED_CALL_NAME_REAL(stem,uby),					\
	HOST_TYPED_CALL_NAME_REAL(stem,uin),					\
	HOST_TYPED_CALL_NAME_REAL(stem,udi),					\
	HOST_TYPED_CALL_NAME_REAL(stem,uli),					\
	HOST_TYPED_CALL_NAME_REAL(stem,ubyin),					\
	HOST_TYPED_CALL_NAME_REAL(stem,inby),					\
	HOST_TYPED_CALL_NAME_REAL(stem,uindi),					\
	HOST_TYPED_CALL_NAME_REAL(stem,spdp),					\
	HOST_TYPED_CALL_NAME_REAL(stem,bit)

/* really 13 now... */

#define ALL_REAL(stem)						\
								\
	HOST_TYPED_CALL_NAME_REAL(stem,by),						\
	HOST_TYPED_CALL_NAME_REAL(stem,in),						\
	HOST_TYPED_CALL_NAME_REAL(stem,di),						\
	HOST_TYPED_CALL_NAME_REAL(stem,li),						\
	HOST_TYPED_CALL_NAME_REAL(stem,sp),						\
	HOST_TYPED_CALL_NAME_REAL(stem,dp),						\
	HOST_TYPED_CALL_NAME_REAL(stem,uby),					\
	HOST_TYPED_CALL_NAME_REAL(stem,uin),					\
	HOST_TYPED_CALL_NAME_REAL(stem,udi),					\
	HOST_TYPED_CALL_NAME_REAL(stem,uli),					\
	HOST_TYPED_CALL_NAME_REAL(stem,ubyin),					\
	HOST_TYPED_CALL_NAME_REAL(stem,inby),					\
	HOST_TYPED_CALL_NAME_REAL(stem,uindi),					\
	HOST_TYPED_CALL_NAME_REAL(stem,spdp),					\
	HOST_TYPED_CALL_NAME_REAL(stem,bit)

#define ALL_REAL_NO_BITMAP(stem)				\
								\
	HOST_TYPED_CALL_NAME_REAL(stem,by),						\
	HOST_TYPED_CALL_NAME_REAL(stem,in),						\
	HOST_TYPED_CALL_NAME_REAL(stem,di),						\
	HOST_TYPED_CALL_NAME_REAL(stem,li),						\
	HOST_TYPED_CALL_NAME_REAL(stem,sp),						\
	HOST_TYPED_CALL_NAME_REAL(stem,dp),						\
	HOST_TYPED_CALL_NAME_REAL(stem,uby),					\
	HOST_TYPED_CALL_NAME_REAL(stem,uin),					\
	HOST_TYPED_CALL_NAME_REAL(stem,udi),					\
	HOST_TYPED_CALL_NAME_REAL(stem,uli),					\
	HOST_TYPED_CALL_NAME_REAL(stem,ubyin),					\
	HOST_TYPED_CALL_NAME_REAL(stem,inby),					\
	HOST_TYPED_CALL_NAME_REAL(stem,uindi),					\
	HOST_TYPED_CALL_NAME_REAL(stem,spdp),					\
	nullobjf

#define ALL_COMPLEX(stem)						\
								\
	NULL_4,							\
	HOST_TYPED_CALL_NAME_CPX(stem,sp),						\
	HOST_TYPED_CALL_NAME_CPX(stem,dp),						\
	NULL_4,							\
	NULL_3,							\
	HOST_TYPED_CALL_NAME_CPX(stem,spdp),					\
	nullobjf


#define ALL_MIXED(stem)						\
								\
	NULL_4,							\
	HOST_TYPED_CALL_NAME_MIXED(stem,sp),						\
	HOST_TYPED_CALL_NAME_MIXED(stem,dp),						\
	NULL_4,							\
	NULL_3,							\
	HOST_TYPED_CALL_NAME_MIXED(stem,spdp),					\
	nullobjf

#define ALL_QUAT(stem)						\
								\
	NULL_4,							\
	HOST_TYPED_CALL_NAME_QUAT(stem,sp),						\
	HOST_TYPED_CALL_NAME_QUAT(stem,dp),						\
	NULL_4,							\
	NULL_3,							\
	HOST_TYPED_CALL_NAME_QUAT(stem,spdp),					\
	nullobjf

#define ALL_QMIXD(stem)						\
								\
	NULL_4,							\
	HOST_TYPED_CALL_NAME_QMIXD(stem,sp),					\
	HOST_TYPED_CALL_NAME_QMIXD(stem,dp),					\
	NULL_4,							\
	NULL_3,							\
	HOST_TYPED_CALL_NAME_QMIXD(stem,spdp),					\
	nullobjf


#define NULL_ARR( stem, code )						\
	{ code, { ALL_NULL , ALL_NULL, ALL_NULL, ALL_NULL, ALL_NULL } }

#define CONV_ARR( stem, code, bmfunc )					\
	{ code, { DUP_ALL(stem,bmfunc), DUP_ALL(stem,bmfunc), ALL_NULL, DUP_ALL(stem,bmfunc), ALL_NULL } }

#define CFLT_ARR( stem, code )						\
	{ code, { ALL_NULL, FLT_ALL(stem), ALL_NULL, ALL_NULL, ALL_NULL } }

#define RFLT_ARR( stem, code )						\
	{ code, { FLT_ALL(stem), ALL_NULL, ALL_NULL, ALL_NULL, ALL_NULL } }

#define RCFLT_ARR2( stem, code )					\
	{ code, { RFLT_NO_MIXED(stem), CFLT_NO_MIXED(stem), ALL_NULL, ALL_NULL, ALL_NULL } }

#define RCFLT_ARR( stem, code )						\
	{ code, { RFLT_ALL(stem), CFLT_ALL(stem), ALL_NULL, ALL_NULL, ALL_NULL } }

#define FLT_ARR( stem, code )						\
	{ code, { RFLT_ALL(stem), CFLT_ALL(stem), MFLT_ALL(stem), ALL_NULL, ALL_NULL } }

#define ALL_ARR( stem, code )						\
	{ code, { ALL_REAL_NO_BITMAP(stem), ALL_COMPLEX(stem), ALL_MIXED(stem), ALL_NULL, ALL_NULL } }

#define RALL_ARR( stem, code )						\
	{ code, { ALL_NO_BITMAP(stem), ALL_NULL, ALL_NULL, ALL_NULL, ALL_NULL	} }

#define RCALL_ARR( stem, code )						\
	{ code, { ALL_REAL_NO_BITMAP(stem), ALL_COMPLEX(stem), ALL_NULL, ALL_NULL, ALL_NULL } }

#define QALL_ARR( stem, code )						\
	{ code, { ALL_REAL_NO_BITMAP(stem), ALL_COMPLEX(stem), ALL_MIXED(stem), ALL_QUAT(stem), ALL_NULL } }

#define RCQALL_ARR( stem, code )					\
	{ code, { ALL_REAL_NO_BITMAP(stem), ALL_COMPLEX(stem), ALL_NULL, ALL_QUAT(stem), ALL_NULL } }

#define RCQALL_SAME_PREC_ARR_SSE( stem, code )					\
	{ code, { ALL_REAL_SAME_PREC_NO_BITMAP(stem), ALL_COMPLEX(stem), ALL_NULL, ALL_QUAT(stem), ALL_NULL } }

#define RCQPALL_ARR( stem, code )					\
	{ code, { ALL_REAL_NO_BITMAP(stem), ALL_COMPLEX(stem), ALL_NULL, ALL_QUAT(stem), ALL_QMIXD(stem) } }

#define CMALL_ARR( stem, code )						\
	{ code, { ALL_NULL, ALL_COMPLEX(stem), ALL_MIXED(stem), ALL_NULL, ALL_NULL } }

#define QPALL_ARR( stem, code )						\
	{ code, { ALL_NULL, ALL_NULL, ALL_NULL, ALL_QUAT(stem), ALL_QMIXD(stem) } }

#define RCMQPALL_ARR( stem, code )					\
	{ code, { ALL_REAL_NO_BITMAP(stem), ALL_COMPLEX(stem), ALL_MIXED(stem), ALL_QUAT(stem), ALL_QMIXD(stem) } }

#define RC_FIXED_ARR( stem, code, bm_func )				\
	{ code, { DUP_ALL(stem,bm_func), DUP_ALL(stem,nullobjf), ALL_NULL, ALL_NULL, ALL_NULL } }


#ifdef USE_SSE

#define ALL_ARR_SSE( stem, code )					\
	{ code, { ALL_REAL_NO_BITMAP_SSE(stem), ALL_COMPLEX(stem), ALL_MIXED(stem), ALL_NULL, ALL_NULL } }
#ifdef FOOBAR
/* This is not used anywhere??? */
#define RCALL_ARR_SSE( stem, code )					\
	{ code, { /* ALL_REAL_NO_BITMAP_SSE */ALL_REAL_SSE(stem), ALL_COMPLEX(stem), ALL_NULL, ALL_NULL, ALL_NULL } }
#endif // FOOBAR

#define RCQALL_ARR_SSE( stem, code )					\
	{ code, { ALL_REAL_NO_BITMAP_SSE(stem), ALL_COMPLEX(stem), ALL_NULL, ALL_QUAT(stem), ALL_NULL } }

#define RCQALL_SAME_PREC_ARR_SSE( stem, code )					\
	{ code, { ALL_REAL_SAME_PREC_NO_BITMAP_SSE(stem), ALL_COMPLEX(stem), ALL_NULL, ALL_QUAT(stem), ALL_NULL } }

#define QALL_ARR_SSE( stem, code )					\
	{ code, { ALL_REAL_NO_BITMAP_SSE(stem), ALL_COMPLEX(stem), ALL_MIXED(stem), ALL_QUAT(stem), ALL_NULL } }
#define RCMQPALL_ARR_SSE( stem, code )					\
	{ code, { ALL_REAL_NO_BITMAP_SSE(stem), ALL_COMPLEX(stem), ALL_MIXED(stem), ALL_QUAT(stem), ALL_QMIXD(stem) } }

#else /* ! USE_SSE */

#ifdef FOOBAR
#define RCALL_ARR_SSE( stem, code )		RCALL_ARR( stem, code )
#endif // FOOBAR

#define RCQALL_ARR_SSE( stem, code )		RCQALL_ARR( stem, code )
#define ALL_ARR_SSE( stem, code )		ALL_ARR( stem, code ) 
#define QALL_ARR_SSE( stem, code )		QALL_ARR( stem, code )
#define RCMQPALL_ARR_SSE( stem, code )		RCMQPALL_ARR( stem, code )

#endif /* ! USE_SSE */


#define INT_ARR( stem, code )						\
	{ code, { REAL_INT_ALL(stem), ALL_NULL, ALL_NULL, ALL_NULL, ALL_NULL } }

#define REAL_INT_ARR( stem, code )					\
	{ code, { INT_ALL(stem), ALL_NULL, ALL_NULL, ALL_NULL, ALL_NULL } }

#define REAL_INT_ARR_NO_BITMAP( stem, code )				\
	{ code, { INT_ALL_NO_BITMAP(stem), ALL_NULL, ALL_NULL, ALL_NULL, ALL_NULL } }

;

