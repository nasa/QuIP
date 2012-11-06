
#include "calling_args.h"


/* We get the declarations of the fast/slow vector funcs from the METHOD macro decls... */

#define   ONE_VEC_METHOD(          name , statement )		\
	 _ONE_VEC_METHOD( TYP ,    name , statement )
#define  _ONE_VEC_METHOD( prefix , name , statement )		\
	__ONE_VEC_METHOD( prefix , name , statement )
#define __ONE_VEC_METHOD( prefix , name , statement )		\
extern void prefix##_fast_##name( FAST_ARGS_PTR );	\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );	\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   ONE_VEC_SCALAR_METHOD(          name , statement )		\
	 _ONE_VEC_SCALAR_METHOD( TYP ,    name , statement )
#define  _ONE_VEC_SCALAR_METHOD( prefix , name , statement )		\
	__ONE_VEC_SCALAR_METHOD( prefix , name , statement )
#define __ONE_VEC_SCALAR_METHOD( prefix , name , statement )		\
extern void prefix##_fast_##name( FAST_ARGS_PTR );			\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );	\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   ONE_VEC_2SCALAR_METHOD(          name , statement )		\
	 _ONE_VEC_2SCALAR_METHOD( TYP ,    name , statement )
#define  _ONE_VEC_2SCALAR_METHOD( prefix , name , statement )		\
	__ONE_VEC_2SCALAR_METHOD( prefix , name , statement )
#define __ONE_VEC_2SCALAR_METHOD( prefix , name , statement )		\
extern void prefix##_fast_##name( FAST_ARGS_PTR );		\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );	\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   ONE_CPX_VEC_SCALAR_METHOD(          name , statement )		\
	 _ONE_CPX_VEC_SCALAR_METHOD( TYP ,    name , statement )
#define  _ONE_CPX_VEC_SCALAR_METHOD( prefix , name , statement )		\
	__ONE_CPX_VEC_SCALAR_METHOD( prefix , name , statement )
#define __ONE_CPX_VEC_SCALAR_METHOD( prefix , name , statement )			\
extern void prefix##_fast_##name( FAST_ARGS_PTR );			\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );			\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   ONE_QUAT_VEC_SCALAR_METHOD(          name , statement )			\
	 _ONE_QUAT_VEC_SCALAR_METHOD( TYP ,    name , statement )
#define  _ONE_QUAT_VEC_SCALAR_METHOD( prefix , name , statement )		\
	__ONE_QUAT_VEC_SCALAR_METHOD( prefix , name , statement )
#define __ONE_QUAT_VEC_SCALAR_METHOD( prefix , name , statement )			\
extern void prefix##_fast_##name( FAST_ARGS_PTR );		\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );		\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   BITMAP_DST_ONE_VEC_SCALAR_METHOD(          name , statement )			\
	 _BITMAP_DST_ONE_VEC_SCALAR_METHOD( TYP ,    name , statement )
#define  _BITMAP_DST_ONE_VEC_SCALAR_METHOD( prefix , name , statement )		\
	__BITMAP_DST_ONE_VEC_SCALAR_METHOD( prefix , name , statement )
#define __BITMAP_DST_ONE_VEC_SCALAR_METHOD( prefix , name , statement )			\
extern void prefix##_fast_##name( FAST_ARGS_PTR );	\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );	\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );



#define   BITMAP_DST_TWO_VEC_METHOD(          name , statement )			\
	 _BITMAP_DST_TWO_VEC_METHOD( TYP ,    name , statement )
#define  _BITMAP_DST_TWO_VEC_METHOD( prefix , name , statement )		\
	__BITMAP_DST_TWO_VEC_METHOD( prefix , name , statement )
#define __BITMAP_DST_TWO_VEC_METHOD( prefix , name , statement )			\
extern void prefix##_fast_##name( FAST_ARGS_PTR );			\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );	\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );

/* vmov has a custom fast routine */
#define TWO_VEC_MOV_METHOD(n,s)	TWO_VEC_METHOD(n,s)

#define   TWO_VEC_METHOD(          name , statement )		\
	 _TWO_VEC_METHOD( TYP ,    name , statement )
#define  _TWO_VEC_METHOD( prefix , name , statement )		\
	__TWO_VEC_METHOD( prefix , name , statement )
#define __TWO_VEC_METHOD( prefix , name , statement )		\
extern void prefix##_fast_##name( FAST_ARGS_PTR );		\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );		\
extern void prefix##_slow_##name( SLOW_ARGS_PTR  );


#define   TWO_CPX_VEC_METHOD(          name , statement )	\
	 _TWO_CPX_VEC_METHOD( TYP ,    name , statement )
#define  _TWO_CPX_VEC_METHOD( prefix , name , statement )	\
	__TWO_CPX_VEC_METHOD( prefix , name , statement )
#define __TWO_CPX_VEC_METHOD( prefix , name , statement )	\
extern void prefix##_fast_##name( FAST_ARGS_PTR );		\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );		\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   TWO_CPXT_VEC_METHOD(          name , statement )	\
	 _TWO_CPXT_VEC_METHOD( TYP ,    name , statement )
#define  _TWO_CPXT_VEC_METHOD( prefix , name , statement )	\
	__TWO_CPXT_VEC_METHOD( prefix , name , statement )
#define __TWO_CPXT_VEC_METHOD( prefix , name , statement )	\
extern void prefix##_fast_##name( FAST_ARGS_PTR );		\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );		\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );



#define   TWO_QUAT_VEC_METHOD(          name , statement )			\
	 _TWO_QUAT_VEC_METHOD( TYP ,    name , statement )
#define  _TWO_QUAT_VEC_METHOD( prefix , name , statement )			\
	__TWO_QUAT_VEC_METHOD( prefix , name , statement )
#define __TWO_QUAT_VEC_METHOD( prefix , name , statement )			\
extern void prefix##_fast_##name( FAST_ARGS_PTR );	\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );	\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   TWO_VEC_SCALAR_METHOD(         name, statement )			\
	 _TWO_VEC_SCALAR_METHOD( TYP,    name, statement )
#define  _TWO_VEC_SCALAR_METHOD( prefix, name, statement )			\
	__TWO_VEC_SCALAR_METHOD( prefix, name, statement )
#define __TWO_VEC_SCALAR_METHOD( prefix, name, statement )			\
extern void prefix##_fast_##name( FAST_ARGS_PTR);	\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );	\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   TWO_VEC_3SCALAR_METHOD(         name, statement )			\
	 _TWO_VEC_3SCALAR_METHOD( TYP ,   name, statement )
#define  _TWO_VEC_3SCALAR_METHOD( prefix, name, statement )			\
	__TWO_VEC_3SCALAR_METHOD( prefix, name, statement )
#define __TWO_VEC_3SCALAR_METHOD( prefix, name, statement )			\
extern void prefix##_fast_##name( FAST_ARGS_PTR );	\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );	\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   TWO_MIXED_CR_VEC_SCALAR_METHOD(         name, statement )			\
	 _TWO_MIXED_CR_VEC_SCALAR_METHOD( TYP,    name, statement )
#define  _TWO_MIXED_CR_VEC_SCALAR_METHOD( prefix, name, statement )			\
	__TWO_MIXED_CR_VEC_SCALAR_METHOD( prefix, name, statement )
#define __TWO_MIXED_CR_VEC_SCALAR_METHOD( prefix, name, statement )			\
extern void prefix##_fast_##name( FAST_ARGS_PTR );	\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );	\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   TWO_QMIXD_QR_VEC_SCALAR_METHOD(         name, statement )			\
	 _TWO_QMIXD_QR_VEC_SCALAR_METHOD( TYP,    name, statement )
#define  _TWO_QMIXD_QR_VEC_SCALAR_METHOD( prefix, name, statement )			\
	__TWO_QMIXD_QR_VEC_SCALAR_METHOD( prefix, name, statement )
#define __TWO_QMIXD_QR_VEC_SCALAR_METHOD( prefix, name, statement )			\
extern void prefix##_fast_##name( FAST_ARGS_PTR );	\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );	\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#ifdef NOT_NOW
/* Do we need RQ? */
#define   TWO_QMIXD_VEC_SCALAR_METHOD(         name, statement )			\
	 _TWO_QMIXD_VEC_SCALAR_METHOD( TYP,    name, statement )
#define  _TWO_QMIXD_VEC_SCALAR_METHOD( prefix, name, statement )			\
	__TWO_QMIXD_VEC_SCALAR_METHOD( prefix, name, statement )
#define __TWO_QMIXD_VEC_SCALAR_METHOD( prefix, name, statement )			\
extern void prefix##_fast_##name( FAST_ARGS_PTR );	\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );	\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );
#endif /* NOT_NOW */


#define   TWO_CPX_VEC_SCALAR_METHOD(         name, statement )		\
	 _TWO_CPX_VEC_SCALAR_METHOD( TYP,    name, statement )
#define  _TWO_CPX_VEC_SCALAR_METHOD( prefix, name, statement )		\
	__TWO_CPX_VEC_SCALAR_METHOD( prefix, name, statement )
#define __TWO_CPX_VEC_SCALAR_METHOD( prefix, name, statement )		\
extern void prefix##_fast_##name( FAST_ARGS_PTR );			\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );			\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   TWO_CPXT_VEC_SCALAR_METHOD(         name, statement )		\
	 _TWO_CPXT_VEC_SCALAR_METHOD( TYP,    name, statement )
#define  _TWO_CPXT_VEC_SCALAR_METHOD( prefix, name, statement )		\
	__TWO_CPXT_VEC_SCALAR_METHOD( prefix, name, statement )
#define __TWO_CPXT_VEC_SCALAR_METHOD( prefix, name, statement )		\
extern void prefix##_fast_##name( FAST_ARGS_PTR );			\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );			\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   TWO_CPXD_VEC_SCALAR_METHOD(         name, statement )		\
	 _TWO_CPXD_VEC_SCALAR_METHOD( TYP,    name, statement )
#define  _TWO_CPXD_VEC_SCALAR_METHOD( prefix, name, statement )		\
	__TWO_CPXD_VEC_SCALAR_METHOD( prefix, name, statement )
#define __TWO_CPXD_VEC_SCALAR_METHOD( prefix, name, statement )		\
extern void prefix##_fast_##name( FAST_ARGS_PTR );			\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );			\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   TWO_QUAT_VEC_SCALAR_METHOD(         name, statement )			\
	 _TWO_QUAT_VEC_SCALAR_METHOD( TYP,    name, statement )
#define  _TWO_QUAT_VEC_SCALAR_METHOD( prefix, name, statement )			\
	__TWO_QUAT_VEC_SCALAR_METHOD( prefix, name, statement )
#define __TWO_QUAT_VEC_SCALAR_METHOD( prefix, name, statement )			\
extern void prefix##_fast_##name( FAST_ARGS_PTR );	\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );	\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   THREE_VEC_METHOD(         name, statement )			\
	 _THREE_VEC_METHOD( TYP,    name, statement )
#define  _THREE_VEC_METHOD( prefix, name, statement )			\
	__THREE_VEC_METHOD( prefix, name, statement )
#define __THREE_VEC_METHOD( prefix, name, statement )			\
extern void prefix##_fast_##name( FAST_ARGS_PTR );	\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );	\
extern void prefix##_slow_##name( SLOW_ARGS_PTR             );


#define   THREE_MIXED_VEC_METHOD(         name, statement )	\
	 _THREE_MIXED_VEC_METHOD( TYP,    name, statement )
#define  _THREE_MIXED_VEC_METHOD( prefix, name, statement )	\
	__THREE_MIXED_VEC_METHOD( prefix, name, statement )
#define __THREE_MIXED_VEC_METHOD( prefix, name, statement )	\
extern void prefix##_fast_##name( FAST_ARGS_PTR );		\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );		\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   THREE_QMIXD_VEC_METHOD(         name, statement )	\
	 _THREE_QMIXD_VEC_METHOD( TYP,    name, statement )
#define  _THREE_QMIXD_VEC_METHOD( prefix, name, statement )	\
	__THREE_QMIXD_VEC_METHOD( prefix, name, statement )
#define __THREE_QMIXD_VEC_METHOD( prefix, name, statement )	\
extern void prefix##_fast_##name( FAST_ARGS_PTR );		\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );		\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   THREE_CPX_VEC_METHOD(         name, statement )	\
	 _THREE_CPX_VEC_METHOD( TYP,    name, statement )
#define  _THREE_CPX_VEC_METHOD( prefix, name, statement )	\
	__THREE_CPX_VEC_METHOD( prefix, name, statement )
#define __THREE_CPX_VEC_METHOD( prefix, name, statement )	\
extern void prefix##_fast_##name( FAST_ARGS_PTR );		\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );		\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   THREE_CPXT_VEC_METHOD(         name, statement )	\
	 _THREE_CPXT_VEC_METHOD( TYP,    name, statement )
#define  _THREE_CPXT_VEC_METHOD( prefix, name, statement )	\
	__THREE_CPXT_VEC_METHOD( prefix, name, statement )
#define __THREE_CPXT_VEC_METHOD( prefix, name, statement )	\
extern void prefix##_fast_##name( FAST_ARGS_PTR );		\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );		\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   THREE_CPXD_VEC_METHOD(         name, statement )	\
	 _THREE_CPXD_VEC_METHOD( TYP,    name, statement )
#define  _THREE_CPXD_VEC_METHOD( prefix, name, statement )	\
	__THREE_CPXD_VEC_METHOD( prefix, name, statement )
#define __THREE_CPXD_VEC_METHOD( prefix, name, statement )	\
extern void prefix##_fast_##name( FAST_ARGS_PTR );		\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );		\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   THREE_QUAT_VEC_METHOD(         name, statement )			\
	 _THREE_QUAT_VEC_METHOD( TYP,    name, statement )
#define  _THREE_QUAT_VEC_METHOD( prefix, name, statement )			\
	__THREE_QUAT_VEC_METHOD( prefix, name, statement )
#define __THREE_QUAT_VEC_METHOD( prefix, name, statement )			\
extern void prefix##_fast_##name( FAST_ARGS_PTR );	\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );	\
extern void prefix##_slow_##name( SLOW_ARGS_PTR             );


/* vatn2 */
#define   TWO_MIXED_RC_VEC_METHOD(         name, statement )		\
	 _TWO_MIXED_RC_VEC_METHOD( TYP,    name, statement )
#define  _TWO_MIXED_RC_VEC_METHOD( prefix, name, statement )		\
	__TWO_MIXED_RC_VEC_METHOD( prefix, name, statement )
#define __TWO_MIXED_RC_VEC_METHOD( prefix, name, statement )		\
extern void prefix##_fast_##name( FAST_ARGS_PTR );	\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );	\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );



#ifdef FOOBAR
/* mvsadd etc */
#define   TWO_MIXED_CR_VEC_METHOD(         name, statement )		\
	 _TWO_MIXED_CR_VEC_METHOD( TYP,    name, statement )
#define  _TWO_MIXED_CR_VEC_METHOD( prefix, name, statement )		\
	__TWO_MIXED_CR_VEC_METHOD( prefix, name, statement )
#define __TWO_MIXED_CR_VEC_METHOD( prefix, name, statement )		\
extern void prefix##_fast_##name( FAST_ARGS_PTR );	\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );	\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );
#endif



#define   FIVE_VEC_METHOD(         name, statement )			\
	 _FIVE_VEC_METHOD( TYP,    name, statement )
#define  _FIVE_VEC_METHOD( prefix, name, statement )			\
	__FIVE_VEC_METHOD( prefix, name, statement )
#define __FIVE_VEC_METHOD( prefix, name, statement )			\
extern void prefix##_fast_##name( FAST_ARGS_PTR );		\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );		\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   EXTREMA_LOCATIONS_METHOD(          name , c1, c2, statement )			\
	 _EXTREMA_LOCATIONS_METHOD( TYP    , name , c1, c2, statement )
#define  _EXTREMA_LOCATIONS_METHOD( prefix , name , c1, c2, statement )			\
	__EXTREMA_LOCATIONS_METHOD( prefix , name , c1, c2, statement )
#define __EXTREMA_LOCATIONS_METHOD( prefix , name , c1, c2, statement )			\
extern void prefix##_fast_##name( FAST_ARGS_PTR );	\
/*extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );*/	\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   RAMP2D_METHOD(          name  )			\
	 _RAMP2D_METHOD( TYP ,    name   )
#define  _RAMP2D_METHOD( prefix , name   )			\
	__RAMP2D_METHOD( prefix , name   )
#define __RAMP2D_METHOD( prefix , name   )			\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   RAMP1D_METHOD(          name  )			\
	 _RAMP1D_METHOD( TYP ,    name   )
#define  _RAMP1D_METHOD( prefix , name   )			\
	__RAMP1D_METHOD( prefix , name   )
#define __RAMP1D_METHOD( prefix , name   )			\
extern void prefix##_fast_##name( FAST_ARGS_PTR );	\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );	\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


// What are these?
#define   PROJECTION_METHOD_IDX_2( name, s1 , s2  )		\
	 _PROJECTION_METHOD_IDX_2( TYP , name, s1 , s2   )
#define  _PROJECTION_METHOD_IDX_2( prefix , name, s1 , s2   )	\
	__PROJECTION_METHOD_IDX_2( prefix , name, s1 , s2   )
#define __PROJECTION_METHOD_IDX_2( prefix , name, s1 , s2   )	\
extern void prefix##_fast_##name( FAST_ARGS_PTR );	\
/*extern void prefix##_eqsp_##name( EQSP_ARG_PTR );*/	\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define   VV_SELECTION_METHOD(         name, statement1 )	\
	 _VV_SELECTION_METHOD( TYP,    name, statement1 )
#define  _VV_SELECTION_METHOD( prefix, name, statement1 )	\
	__VV_SELECTION_METHOD( prefix, name, statement1 )
#define __VV_SELECTION_METHOD( prefix, name, statement1 )	\
extern void prefix##_fast_##name(FAST_ARGS_PTR ); \
extern void prefix##_eqsp_##name(EQSP_ARGS_PTR ); \
extern void prefix##_slow_##name(SLOW_ARGS_PTR );


#define   VS_SELECTION_METHOD(         name, statement1 )	\
	 _VS_SELECTION_METHOD( TYP,    name, statement1 )
#define  _VS_SELECTION_METHOD( prefix, name, statement1 )	\
	__VS_SELECTION_METHOD( prefix, name, statement1 )
#define __VS_SELECTION_METHOD( prefix, name, statement1 )	\
extern void prefix##_fast_##name(FAST_ARGS_PTR ); \
extern void prefix##_eqsp_##name(EQSP_ARGS_PTR ); \
extern void prefix##_slow_##name(SLOW_ARGS_PTR );


#define   SS_SELECTION_METHOD(         name, statement1 )	\
	 _SS_SELECTION_METHOD( TYP,    name, statement1 )
#define  _SS_SELECTION_METHOD( prefix, name, statement1 )	\
	__SS_SELECTION_METHOD( prefix, name, statement1 )
#define __SS_SELECTION_METHOD( prefix, name, statement1 )	\
extern void prefix##_fast_##name(FAST_ARGS_PTR ); \
extern void prefix##_eqsp_##name(EQSP_ARGS_PTR ); \
extern void prefix##_slow_##name(SLOW_ARGS_PTR );



#define   CPX_VV_SELECTION_METHOD(         name, statement1 )			\
	 _CPX_VV_SELECTION_METHOD( TYP,    name, statement1 )
#define  _CPX_VV_SELECTION_METHOD( prefix, name, statement1 )			\
	__CPX_VV_SELECTION_METHOD( prefix, name, statement1 )
#define __CPX_VV_SELECTION_METHOD( prefix, name, statement1 )			\
extern void prefix##_fast_##name(FAST_ARGS_PTR); \
extern void prefix##_eqsp_##name(EQSP_ARGS_PTR ); \
extern void prefix##_slow_##name(SLOW_ARGS_PTR );


#define   CPX_VS_SELECTION_METHOD(         name, statement1 )	\
	 _CPX_VS_SELECTION_METHOD( TYP,    name, statement1 )
#define  _CPX_VS_SELECTION_METHOD( prefix, name, statement1 )	\
	__CPX_VS_SELECTION_METHOD( prefix, name, statement1 )
#define __CPX_VS_SELECTION_METHOD( prefix, name, statement1 )	\
extern void prefix##_fast_##name(FAST_ARGS_PTR ); \
extern void prefix##_eqsp_##name(EQSP_ARGS_PTR ); \
extern void prefix##_slow_##name(SLOW_ARGS_PTR );


#define   CPX_SS_SELECTION_METHOD(         name, statement1 )	\
	 _CPX_SS_SELECTION_METHOD( TYP,    name, statement1 )
#define  _CPX_SS_SELECTION_METHOD( prefix, name, statement1 )	\
	__CPX_SS_SELECTION_METHOD( prefix, name, statement1 )
#define __CPX_SS_SELECTION_METHOD( prefix, name, statement1 )	\
extern void prefix##_fast_##name(FAST_ARGS_PTR ); \
extern void prefix##_eqsp_##name(EQSP_ARGS_PTR ); \
extern void prefix##_slow_##name(SLOW_ARGS_PTR );


#define   QUAT_VV_SELECTION_METHOD(         name, statement1 )			\
	 _QUAT_VV_SELECTION_METHOD( TYP,    name, statement1 )
#define  _QUAT_VV_SELECTION_METHOD( prefix, name, statement1 )			\
	__QUAT_VV_SELECTION_METHOD( prefix, name, statement1 )
#define __QUAT_VV_SELECTION_METHOD( prefix, name, statement1 )			\
extern void prefix##_fast_##name(FAST_ARGS_PTR ); \
extern void prefix##_eqsp_##name(EQSP_ARGS_PTR ); \
extern void prefix##_slow_##name(SLOW_ARGS_PTR );


#define   QUAT_VS_SELECTION_METHOD(         name, statement1 )	\
	 _QUAT_VS_SELECTION_METHOD( TYP,    name, statement1 )
#define  _QUAT_VS_SELECTION_METHOD( prefix, name, statement1 )	\
	__QUAT_VS_SELECTION_METHOD( prefix, name, statement1 )
#define __QUAT_VS_SELECTION_METHOD( prefix, name, statement1 )	\
extern void prefix##_fast_##name(FAST_ARGS_PTR ); \
extern void prefix##_eqsp_##name(EQSP_ARGS_PTR ); \
extern void prefix##_slow_##name(SLOW_ARGS_PTR );


#define   QUAT_SS_SELECTION_METHOD(         name, statement1 )	\
	 _QUAT_SS_SELECTION_METHOD( TYP,    name, statement1 )
#define  _QUAT_SS_SELECTION_METHOD( prefix, name, statement1 )	\
	__QUAT_SS_SELECTION_METHOD( prefix, name, statement1 )
#define __QUAT_SS_SELECTION_METHOD( prefix, name, statement1 )	\
extern void prefix##_fast_##name(FAST_ARGS_PTR ); \
extern void prefix##_eqsp_##name(EQSP_ARGS_PTR ); \
extern void prefix##_slow_##name(SLOW_ARGS_PTR );


#define   FOUR_VEC_SCALAR_METHOD(         name, statement )			\
	 _FOUR_VEC_SCALAR_METHOD( TYP,    name, statement )
#define  _FOUR_VEC_SCALAR_METHOD( prefix, name, statement )			\
	__FOUR_VEC_SCALAR_METHOD( prefix, name, statement )
#define __FOUR_VEC_SCALAR_METHOD( prefix, name, statement )			\
extern void prefix##_fast_##name(FAST_ARGS_PTR );		\
extern void prefix##_eqsp_##name(EQSP_ARGS_PTR ); \
extern void prefix##_slow_##name(SLOW_ARGS_PTR );



#define   THREE_VEC_2SCALAR_METHOD(         name, statement )			\
	 _THREE_VEC_2SCALAR_METHOD( TYP,    name, statement )
#define  _THREE_VEC_2SCALAR_METHOD( prefix, name, statement )			\
	__THREE_VEC_2SCALAR_METHOD( prefix, name, statement )
#define __THREE_VEC_2SCALAR_METHOD( prefix, name, statement )			\
extern void prefix##_fast_##name(FAST_ARGS_PTR ); \
extern void prefix##_eqsp_##name(EQSP_ARGS_PTR ); \
extern void prefix##_slow_##name(SLOW_ARGS_PTR );

#define   BITMAP_SRC_CONVERSION_METHOD(         name, statement )		\
	 _BITMAP_SRC_CONVERSION_METHOD( TYP,    name, statement )
#define  _BITMAP_SRC_CONVERSION_METHOD( prefix, name, statement )		\
	__BITMAP_SRC_CONVERSION_METHOD( prefix, name, statement )
#define __BITMAP_SRC_CONVERSION_METHOD( prefix, name, statement )		\
extern void prefix##_fast_##name(FAST_ARGS_PTR ); \
extern void prefix##_eqsp_##name(EQSP_ARGS_PTR ); \
extern void prefix##_slow_##name(SLOW_ARGS_PTR );

#define   SCALAR_BIT_METHOD(name,statement)				\
	 _SCALAR_BIT_METHOD(TYP,name,statement)
#define  _SCALAR_BIT_METHOD(prefix,name,statement)			\
	__SCALAR_BIT_METHOD(prefix,name,statement)
#define __SCALAR_BIT_METHOD(prefix,name,statement)			\
extern void prefix##_fast_##name(FAST_ARGS_PTR);		\
extern void prefix##_slow_##name(SLOW_ARGS_PTR);

#define   SCALRET_METHOD(         name, s1 , s2  )			\
	 _SCALRET_METHOD( TYP,    name, s1 , s2  )
#define  _SCALRET_METHOD( prefix, name, s1 , s2  )			\
	__SCALRET_METHOD( prefix, name, s1 , s2  )
#define __SCALRET_METHOD( prefix, name, s1 , s2  )			\
extern void prefix##_fast_##name( FAST_ARGS_PTR );		\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );		\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );

#define   PROJECTION_METHOD_3(         name, statement1, statement2 )		\
	 _PROJECTION_METHOD_3( TYP,    name, statement1, statement2 )
#define  _PROJECTION_METHOD_3( prefix, name, statement1, statement2 )		\
	__PROJECTION_METHOD_3( prefix, name, statement1, statement2 )
#define __PROJECTION_METHOD_3( prefix, name, statement1, statement2 )		\
extern void prefix##_fast_##name( FAST_ARGS_PTR );		\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );		\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );

#define   PROJECTION_METHOD_2(         name, statement1, statement2 )		\
	 _PROJECTION_METHOD_2( TYP,    name, statement1, statement2 )
#define  _PROJECTION_METHOD_2( prefix, name, statement1, statement2 )		\
	__PROJECTION_METHOD_2( prefix, name, statement1, statement2 )
#define __PROJECTION_METHOD_2( prefix, name, statement1, statement2 )		\
extern void prefix##_fast_##name( FAST_ARGS_PTR );		\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );		\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );

#define    CPX_PROJECTION_METHOD_2(         name, statement1, statement2 )	\
	  _CPX_PROJECTION_METHOD_2( TYP,    name, statement1, statement2 )
#define   _CPX_PROJECTION_METHOD_2( prefix, name, statement1, statement2 )	\
	 __CPX_PROJECTION_METHOD_2( prefix, name, statement1, statement2 )
#define  __CPX_PROJECTION_METHOD_2( prefix, name, statement1, statement2 )	\
extern void prefix##_fast_##name( FAST_ARGS_PTR );		\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );		\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define    CPX_PROJECTION_METHOD_3(         name, statement1, statement2 )	\
	  _CPX_PROJECTION_METHOD_3( TYP,    name, statement1, statement2 )
#define   _CPX_PROJECTION_METHOD_3( prefix, name, statement1, statement2 )	\
	 __CPX_PROJECTION_METHOD_3( prefix, name, statement1, statement2 )
#define  __CPX_PROJECTION_METHOD_3( prefix, name, statement1, statement2 )	\
extern void prefix##_fast_##name( FAST_ARGS_PTR );		\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );		\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define    QUAT_PROJECTION_METHOD_2(         name, statement1, statement2 )	\
	  _QUAT_PROJECTION_METHOD_2( TYP,    name, statement1, statement2 )
#define   _QUAT_PROJECTION_METHOD_2( prefix, name, statement1, statement2 )	\
	 __QUAT_PROJECTION_METHOD_2( prefix, name, statement1, statement2 )
#define  __QUAT_PROJECTION_METHOD_2( prefix, name, statement1, statement2 )	\
extern void prefix##_fast_##name( FAST_ARGS_PTR );		\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );		\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );


#define    BITMAP_DST_ONE_VEC_METHOD(         name, statement )			\
	  _BITMAP_DST_ONE_VEC_METHOD( TYP,    name, statement )
#define   _BITMAP_DST_ONE_VEC_METHOD( prefix, name, statement )			\
	 __BITMAP_DST_ONE_VEC_METHOD( prefix, name, statement )
#define  __BITMAP_DST_ONE_VEC_METHOD( prefix, name, statement )			\
extern void prefix##_fast_##name( FAST_ARGS_PTR );	\
extern void prefix##_eqsp_##name( EQSP_ARGS_PTR );	\
extern void prefix##_slow_##name( SLOW_ARGS_PTR );

#define ALL_UNSIGNED_CONVERSIONS(prefix,type)					\
extern void fast_v##prefix##2uby(FAST_ARGS_PTR);	\
extern void fast_v##prefix##2uin(FAST_ARGS_PTR);	\
extern void fast_v##prefix##2udi(FAST_ARGS_PTR);	\
extern void fast_v##prefix##2uli(FAST_ARGS_PTR);	\
extern void slow_v##prefix##2uby(SLOW_ARGS_PTR);	\
extern void slow_v##prefix##2uin(SLOW_ARGS_PTR);	\
extern void slow_v##prefix##2udi(SLOW_ARGS_PTR);	\
extern void slow_v##prefix##2uli(SLOW_ARGS_PTR);

#define ALL_SIGNED_CONVERSIONS(prefix,type)					\
extern void fast_v##prefix##2by(FAST_ARGS_PTR);	\
extern void fast_v##prefix##2in(FAST_ARGS_PTR);	\
extern void fast_v##prefix##2di(FAST_ARGS_PTR);	\
extern void fast_v##prefix##2li(FAST_ARGS_PTR);	\
extern void slow_v##prefix##2by(SLOW_ARGS_PTR);	\
extern void slow_v##prefix##2in(SLOW_ARGS_PTR);	\
extern void slow_v##prefix##2di(SLOW_ARGS_PTR);	\
extern void slow_v##prefix##2li(SLOW_ARGS_PTR);

#define ALL_FLOAT_CONVERSIONS(prefix,type)					\
extern void fast_v##prefix##2sp(FAST_ARGS_PTR);	\
extern void fast_v##prefix##2dp(FAST_ARGS_PTR);	\
extern void slow_v##prefix##2sp(SLOW_ARGS_PTR);	\
extern void slow_v##prefix##2dp(SLOW_ARGS_PTR);

#define REAL_CONVERSION(pref1,typ1,pref2,typ2)					\
extern void fast_v##pref1##2##pref2(FAST_ARGS_PTR);	\
extern void slow_v##pref1##2##pref2( SLOW_ARGS_PTR );

/*	_PROJECTION_METHOD_IDX_2( TYP , name, s1 , s2   ) */
/* These ones are not done yet... */

#define IMPOSSIBLE_METHOD( name )


#define QUAT_PROJECTION_METHOD_3( name, statement1, statement2 )

