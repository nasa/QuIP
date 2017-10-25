
#include "quip_config.h"

#include "vec_util.h"
#include "veclib_api.h"
#include "debug.h"
#include "quip_prot.h"
#include "getbuf.h"


/** wrap subroutine */
/* BUG it would be nice to merge this with wrap3d() */

void wrap(QSP_ARG_DECL  Data_Obj *dst_dp,Data_Obj *src_dp)
{
	int status;
	Vector_Function *vfp;

	vfp=FIND_VEC_FUNC(FVMOV);

	if( (status=old_cksiz(QSP_ARG  VF_FLAGS(vfp),dst_dp,src_dp))==(-1)) return;
#ifdef CAUTIOUS
	if( status!=0){
		sprintf(ERROR_STRING,"CAUTIOUS:  wrap:  old_cksiz() error...");
		WARN(ERROR_STRING);
	}
#endif /* CAUTIOUS */

	if( dp_same_prec(QSP_ARG  dst_dp,src_dp,"wrap") == 0 ) return;
#ifdef FOOBAR
	if( cktype(dst_dp,src_dp)==(-1)) return;
#endif /* FOOBAR */

	dp_scroll(QSP_ARG  dst_dp,src_dp,(incr_t)(OBJ_COLS(dst_dp)/2),(incr_t)(OBJ_ROWS(dst_dp)/2));
}

#ifdef FOOBAR
void wrap3d(QSP_ARG_DECL  dst_dp, src_dp)
Data_Obj *dst_dp, *src_dp;
{
	int i,j,k;	/* to keep track of which octant we're doing */
	u_char *to, *fr, *fr_base, *to_base;
	incr_t f_rowinc, t_rowinc, f_finc, t_finc;
	dimension_t siz;	/* element size in bytes */
	incr_t dst_delta_x, dst_delta_y, dst_delta_t;
	incr_t src_delta_x, src_delta_y, src_delta_t;
	dimension_t rows_to_copy, frms_to_copy;
	int status;

	long finc,tinc;
	Vec_Args args;

	if( (status=old_cksiz(QSP_ARG  vec_func_tbl[FVMOV].vf_flags,dst_dp,src_dp))==(-1)) return;
#ifdef CAUTIOUS
	if( status!=0){
		sprintf(ERROR_STRING,"CAUTIOUS:  wrap:  old_cksiz() error...");
		WARN(ERROR_STRING);
	}
#endif /* CAUTIOUS */
	if( dp_same_prec(QSP_ARG  dst_dp,src_dp,"wrap3d")==0) return;
#ifdef FOOBAR
	if( cktype(dst_dp,src_dp)==(-1)) return;
#endif /* FOOBAR */

	siz = ELEMENT_SIZE(dst_dp);

	dst_delta_x = (incr_t)( OBJ_PXL_INC(dst_dp) * (OBJ_COLS(dst_dp)/2) * siz );
	src_delta_x = (incr_t)( OBJ_PXL_INC(src_dp) * (OBJ_COLS(src_dp)/2) * siz );
	dst_delta_y = (incr_t)( OBJ_ROW_INC(dst_dp) * (OBJ_ROWS(dst_dp)/2) * siz );
	src_delta_y = (incr_t)( OBJ_ROW_INC(src_dp) * (OBJ_ROWS(src_dp)/2) * siz );
	dst_delta_t = (incr_t)( OBJ_FRM_INC(dst_dp) * (OBJ_FRAMES(dst_dp)/2) * siz );
	src_delta_t = (incr_t)( OBJ_FRM_INC(src_dp) * (OBJ_FRAMES(src_dp)/2) * siz );

	f_rowinc = (incr_t)( OBJ_ROW_INC(src_dp) * siz );
	t_rowinc = (incr_t)( OBJ_ROW_INC(dst_dp) * siz );
	f_finc = (incr_t)( OBJ_FRM_INC(src_dp) * siz );
	t_finc = (incr_t)( OBJ_FRM_INC(dst_dp) * siz );

	/* warlib args */

	finc=OBJ_PXL_INC(src_dp);
	tinc=OBJ_PXL_INC(dst_dp);

	/* divide by tdim so correct for complex */
	if( IS_COMPLEX(src_dp) ){
		finc /= 2;
		tinc /= 2;
		if( finc != 1 || tinc != 1 ){
			WARN("Sorry, can only wrap contiguous comples images");
			return;
		}
	} else if( OBJ_COMPS(src_dp) > 1 ){
		WARN("Sorry, can't wrap multidimensional pixels");
		return;
	}

	if( IS_COMPLEX(src_dp) )	args.arg_argstype=COMPLEX_ARGS;
	else				args.arg_argstype=REAL_ARGS;


	args.arg_n1 = OBJ_COLS(src_dp)/2;
	args.arg_n2 = OBJ_COLS(src_dp)/2;
	args.arg_inc1 = finc;
	args.arg_inc2 = tinc;
	args.arg_prec1 = OBJ_PREC(src_dp);
	args.arg_prec2 = OBJ_PREC(src_dp);

	for(i=0;i<2;i++){			/* swap left & right */
		for(j=0;j<2;j++){		/* swap top & bottom */
			for(k=0;k<2;k++){	/* swap beg & end */
				fr_base=OBJ_DATA_PTR(src_dp);
				to_base=OBJ_DATA_PTR(dst_dp);

				if( i ) to_base += dst_delta_x;
				else    fr_base += src_delta_x;
				if( j ) to_base += dst_delta_y;
				else    fr_base += src_delta_y;
				if( k ) to_base += dst_delta_t;
				else    fr_base += src_delta_t;

				/* add 1 so it works for only 1 */
				frms_to_copy = (OBJ_FRAMES(dst_dp)+1)/2;
				while( frms_to_copy-- ){
					rows_to_copy = (OBJ_ROWS(dst_dp)+1)/2;
					fr = fr_base;
					to = to_base;
					while( rows_to_copy-- ){
						args.arg_v1 = fr;
						args.arg_v2 = to;
						vmov(&args);
						fr += f_rowinc;
						to += t_rowinc;
					}
					fr_base += f_finc;
					to_base += t_finc;
				}
			}
		}
	}
}

#endif /* FOOBAR */

/* This calls vmov - but does that call cuda routine???
 * Need to call call_vfunc...
 */

void dp_scroll(QSP_ARG_DECL  Data_Obj *dst_dp,Data_Obj *src_dp,incr_t dx,incr_t dy)
{
	Vec_Obj_Args oa1, *oap=&oa1;
	int status;
	Data_Obj *sub_dst_dp, *sub_src_dp;

	// What is "old" cksiz???
	if( (status=old_cksiz(QSP_ARG  VF_FLAGS( FIND_VEC_FUNC(FVMOV) ), dst_dp,src_dp))==(-1)) return;
#ifdef CAUTIOUS
	if( status!=0){
		sprintf(ERROR_STRING,"CAUTIOUS:  dp_scroll:  old_cksiz() error...");
		WARN(ERROR_STRING);
	}
#endif /* CAUTIOUS */
	if( dp_same_prec(QSP_ARG  dst_dp,src_dp,"dp_scroll")==0) return;

#ifdef FOOBAR
	if( cktype(dst_dp,src_dp)==(-1)) return;
#endif /* FOOBAR */

#ifdef DEBUG
//if( debug & veclib_debug ){
sprintf(ERROR_STRING,"scrolling %s by %d %d into %s",
OBJ_NAME(src_dp),dx,dy,OBJ_NAME(dst_dp));
ADVISE(ERROR_STRING);
sprintf(ERROR_STRING,"destination area:  0x%lx,  source area:  0x%lx",
(int_for_addr)OBJ_DATA_PTR(dst_dp),(int_for_addr)OBJ_DATA_PTR(src_dp));
ADVISE(ERROR_STRING);
//}
#endif /* DEBUG */

	clear_obj_args(oap);

	/* dx is the amount we scroll to the right (yes?) */
	/* dy is the amount we scroll down (yes?) */

	/* make dx & dy modulo image size */

	while( dx < 0 ) dx+=OBJ_COLS(dst_dp);
	while( dy < 0 ) dy+=OBJ_ROWS(dst_dp);
	while( dx >= (incr_t)OBJ_COLS(dst_dp) ) dx-=OBJ_COLS(dst_dp);
	while( dy >= (incr_t)OBJ_ROWS(dst_dp) ) dy-=OBJ_ROWS(dst_dp);

#define DO_BLOCK( xos, yos )								\
											\
/*sprintf(ERROR_STRING,"DO_BLOCK %d %d   dx = %d    dy = %d",xos,yos,dx,dy);		\
ADVISE(ERROR_STRING);									\
sprintf(ERROR_STRING,									\
"src (%s) size:  %ld x %ld",OBJ_NAME(src_dp),OBJ_ROWS(src_dp),OBJ_COLS(src_dp));	\
ADVISE(ERROR_STRING);*/									\
											\
	sub_dst_dp = nmk_subimg(QSP_ARG  dst_dp, (xos)==0?dx:0 , (yos)==0?dy:0 ,	\
					"tmp_dst",					\
					(yos)==0?OBJ_ROWS(src_dp)-(dy):dy,		\
					(xos)==0?OBJ_COLS(src_dp)-(dx):dx,		\
					OBJ_COMPS(dst_dp) );				\
	sub_src_dp = nmk_subimg(QSP_ARG  src_dp, xos, yos, "tmp_src",			\
					(yos)==0?OBJ_ROWS(src_dp)-(dy):dy,		\
					(xos)==0?OBJ_COLS(src_dp)-(dx):dx,		\
					OBJ_COMPS(src_dp) );				\
											\
	setvarg2(oap,sub_dst_dp,sub_src_dp);						\
	if( IS_COMPLEX(dst_dp) ) /* BUG case for QUAT too? */				\
		OA_ARGSTYPE(oap) = COMPLEX_ARGS; 					\
	else										\
		OA_ARGSTYPE(oap) = REAL_ARGS; 						\
	/* vmov(oap); */								\
	call_vfunc( QSP_ARG  FIND_VEC_FUNC(FVMOV), oap );				\
	delvec(sub_dst_dp);							\
	delvec(sub_src_dp);

	if( dx == 0 ){
		if( dy == 0 ){
			DO_BLOCK(0,0)
		} else {	/* shift in y only */
			DO_BLOCK(0,0)
			DO_BLOCK(0,OBJ_ROWS(src_dp)-dy)
		}
	} else if( dy == 0 ){	/* shift in x only */
		DO_BLOCK(0,0)
		DO_BLOCK(OBJ_COLS(src_dp)-dx,0)
	} else {
		/* shift in x and y */
		DO_BLOCK(0,0)
		DO_BLOCK(OBJ_COLS(src_dp)-dx,0)
		DO_BLOCK(OBJ_COLS(src_dp)-dx,OBJ_ROWS(src_dp)-dy)
		DO_BLOCK(0,OBJ_ROWS(src_dp)-dy)
	}


}

