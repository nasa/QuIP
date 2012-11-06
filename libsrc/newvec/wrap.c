
#include "quip_config.h"


char VersionId_newvec_wrap[] = QUIP_VERSION_STRING;

#include "nvf.h"
#include "debug.h"
#include "vec_util.h"	/* old_cksiz() */


/** wrap subroutine */
/* BUG it would be nice to merge this with wrap3d() */

void wrap(QSP_ARG_DECL  Data_Obj *dst_dp,Data_Obj *src_dp)
{
	int status;

	if( (status=old_cksiz(QSP_ARG  vec_func_tbl[FVMOV].vf_flags,dst_dp,src_dp))==(-1)) return;
#ifdef CAUTIOUS
	if( status!=0){
		sprintf(error_string,"CAUTIOUS:  wrap:  old_cksiz() error...");
		WARN(error_string);
	}
#endif /* CAUTIOUS */

	if( dp_same_prec(QSP_ARG  dst_dp,src_dp,"wrap") == 0 ) return;
#ifdef FOOBAR
	if( cktype(dst_dp,src_dp)==(-1)) return;
#endif /* FOOBAR */

	dp_scroll(QSP_ARG  dst_dp,src_dp,(long)(dst_dp->dt_cols/2),(long)(dst_dp->dt_rows/2));
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
		sprintf(error_string,"CAUTIOUS:  wrap:  old_cksiz() error...");
		WARN(error_string);
	}
#endif /* CAUTIOUS */
	if( dp_same_prec(QSP_ARG  dst_dp,src_dp,"wrap3d")==0) return;
#ifdef FOOBAR
	if( cktype(dst_dp,src_dp)==(-1)) return;
#endif /* FOOBAR */

	siz = ELEMENT_SIZE(dst_dp);

	dst_delta_x = (incr_t)( dst_dp->dt_pinc * (dst_dp->dt_cols/2) * siz );
	src_delta_x = (incr_t)( src_dp->dt_pinc * (src_dp->dt_cols/2) * siz );
	dst_delta_y = (incr_t)( dst_dp->dt_rinc * (dst_dp->dt_rows/2) * siz );
	src_delta_y = (incr_t)( src_dp->dt_rinc * (src_dp->dt_rows/2) * siz );
	dst_delta_t = (incr_t)( dst_dp->dt_finc * (dst_dp->dt_frames/2) * siz );
	src_delta_t = (incr_t)( src_dp->dt_finc * (src_dp->dt_frames/2) * siz );

	f_rowinc = (incr_t)( src_dp->dt_rowinc * siz );
	t_rowinc = (incr_t)( dst_dp->dt_rowinc * siz );
	f_finc = (incr_t)( src_dp->dt_finc * siz );
	t_finc = (incr_t)( dst_dp->dt_finc * siz );

	/* warlib args */

	finc=src_dp->dt_pinc;
	tinc=dst_dp->dt_pinc;

	/* divide by tdim so correct for complex */
	if( IS_COMPLEX(src_dp) ){
		finc /= 2;
		tinc /= 2;
		if( finc != 1 || tinc != 1 ){
			WARN("Sorry, can only wrap contiguous comples images");
			return;
		}
	} else if( src_dp->dt_comps > 1 ){
		WARN("Sorry, can't wrap multidimensional pixels");
		return;
	}

	if( IS_COMPLEX(src_dp) )	args.arg_argstype=COMPLEX_ARGS;
	else				args.arg_argstype=REAL_ARGS;


	args.arg_n1 = src_dp->dt_cols/2;
	args.arg_n2 = src_dp->dt_cols/2;
	args.arg_inc1 = finc;
	args.arg_inc2 = tinc;
	args.arg_prec1 = src_dp->dt_prec;
	args.arg_prec2 = src_dp->dt_prec;

	for(i=0;i<2;i++){			/* swap left & right */
		for(j=0;j<2;j++){		/* swap top & bottom */
			for(k=0;k<2;k++){	/* swap beg & end */
				fr_base=src_dp->dt_data;
				to_base=dst_dp->dt_data;

				if( i ) to_base += dst_delta_x;
				else    fr_base += src_delta_x;
				if( j ) to_base += dst_delta_y;
				else    fr_base += src_delta_y;
				if( k ) to_base += dst_delta_t;
				else    fr_base += src_delta_t;

				/* add 1 so it works for only 1 */
				frms_to_copy = (dst_dp->dt_frames+1)/2;
				while( frms_to_copy-- ){
					rows_to_copy = (dst_dp->dt_rows+1)/2;
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

void dp_scroll(QSP_ARG_DECL  Data_Obj *dst_dp,Data_Obj *src_dp,incr_t dx,incr_t dy)
{
	Vec_Obj_Args oargs;
	int status;
	Data_Obj *sub_dst_dp, *sub_src_dp;

	if( (status=old_cksiz(QSP_ARG  vec_func_tbl[FVMOV].vf_flags,dst_dp,src_dp))==(-1)) return;
#ifdef CAUTIOUS
	if( status!=0){
		sprintf(error_string,"CAUTIOUS:  wrap:  old_cksiz() error...");
		WARN(error_string);
	}
#endif /* CAUTIOUS */
	if( dp_same_prec(QSP_ARG  dst_dp,src_dp,"dp_scroll")==0) return;
#ifdef FOOBAR
	if( cktype(dst_dp,src_dp)==(-1)) return;
#endif /* FOOBAR */

#ifdef DEBUG
if( debug & veclib_debug ){
sprintf(error_string,"scrolling %s by %d %d into %s",
src_dp->dt_name,dx,dy,dst_dp->dt_name);
ADVISE(error_string);
sprintf(error_string,"destination area:  0x%lx,  source area:  0x%lx",
(int_for_addr)dst_dp->dt_data,(int_for_addr)src_dp->dt_data);
ADVISE(error_string);
}
#endif /* DEBUG */

	/* dx is the amount we scroll to the right (yes?) */
	/* dy is the amount we scroll down (yes?) */

	/* make dx & dy modulo image size */

	while( dx < 0 ) dx+=dst_dp->dt_cols;
	while( dy < 0 ) dy+=dst_dp->dt_rows;
	while( dx >= (incr_t)dst_dp->dt_cols ) dx-=dst_dp->dt_cols;
	while( dy >= (incr_t)dst_dp->dt_rows ) dy-=dst_dp->dt_rows;

#define DO_BLOCK( xos, yos )									\
												\
/*sprintf(error_string,"DO_BLOCK %d %d   dx = %d    dy = %d",xos,yos,dx,dy);\
ADVISE(error_string);\
sprintf(error_string,"src (%s) size:  %ld x %ld",src_dp->dt_name,src_dp->dt_rows,src_dp->dt_cols);\
ADVISE(error_string);*/\
	sub_dst_dp = nmk_subimg(QSP_ARG  dst_dp, (xos)==0?dx:0 , (yos)==0?dy:0 , "tmp_dst",	\
					(yos)==0?src_dp->dt_rows-(dy):dy,			\
					(xos)==0?src_dp->dt_cols-(dx):dx,			\
					dst_dp->dt_comps );					\
	sub_src_dp = nmk_subimg(QSP_ARG  src_dp, xos, yos, "tmp_src",				\
					(yos)==0?src_dp->dt_rows-(dy):dy,			\
					(xos)==0?src_dp->dt_cols-(dx):dx,			\
					src_dp->dt_comps );					\
												\
	setvarg2(&oargs,sub_dst_dp,sub_src_dp);							\
	if( IS_COMPLEX(dst_dp) )								\
		oargs.oa_argstype = COMPLEX_ARGS; 						\
	else											\
		oargs.oa_argstype = REAL_ARGS; 							\
	vmov(&oargs);										\
	delvec(QSP_ARG  sub_dst_dp);								\
	delvec(QSP_ARG  sub_src_dp);

	if( dx == 0 ){
		if( dy == 0 ){
			DO_BLOCK(0,0)
		} else {	/* shift in y only */
			DO_BLOCK(0,0)
			DO_BLOCK(0,src_dp->dt_rows-dy)
		}
	} else if( dy == 0 ){	/* shift in x only */
		DO_BLOCK(0,0)
		DO_BLOCK(src_dp->dt_cols-dx,0)
	} else {
		/* shift in x and y */
		DO_BLOCK(0,0)
		DO_BLOCK(src_dp->dt_cols-dx,0)
		DO_BLOCK(src_dp->dt_cols-dx,src_dp->dt_rows-dy)
		DO_BLOCK(0,src_dp->dt_rows-dy)
	}


}

