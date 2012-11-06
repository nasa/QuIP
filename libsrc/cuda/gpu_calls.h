#include "gpu_call_utils.h"
/**********************************************************************/

#define _KERN_CALL_VVSLCT(func_name,type_code,statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,SBM_,,,3,)

#define _KERN_CALL_VSSLCT(func_name,type_code,statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,SBM_,,1S_,2,)

#define _KERN_CALL_SSSLCT(func_name,type_code,statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,SBM_,,2S_,1,)

#define _KERN_CALL_SBM_1(func_name,type_code,statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,SBM_,,,1,)

#define _KERN_CALL_SBM_CPX_3V(func_name,type_code,statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,SBM_,CPX_,,3,)

#define _KERN_CALL_SBM_CPX_1S_2V(func_name,type_code,statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,SBM_,CPX_,1S_,2,)

#define _KERN_CALL_SBM_CPX_2S_1V(func_name,type_code,statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,SBM_,CPX_,2S_,1,)

#define _KERN_CALL_3V(func_name, type_code, statement)			\
		GENERIC_KERN_CALL(func_name,type_code,statement,,,,3,)

#define _KERN_CALL_CPX_2V(func_name, type_code, statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,,CPX_,,2,)

#define _KERN_CALL_CPXT_2V(func_name, type_code, statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,,CPX_,,2,T)

#define _KERN_CALL_CPXT_3V(func_name, type_code, statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,,CPX_,,3,T)

#define _KERN_CALL_CPXD_3V(func_name, type_code, statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,,CPX_,,3,D)

#define _KERN_CALL_CPX_1S_2V(func_name, type_code, statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,,CPX_,1S_,2,)

#define _KERN_CALL_CPXT_1S_2V(func_name, type_code, statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,,CPX_,1S_,2,T)

#define _KERN_CALL_CPXD_1S_2V(func_name, type_code, statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,,CPX_,1S_,2,D)

#define _KERN_CALL_CPX_1S_1V(func_name, type_code, statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,,CPX_,1S_,1,)

#define _KERN_CALL_CPX_3V(func_name, type_code, statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,,CPX_,,3,)

#define _KERN_CALL_CCR_3V(func_name, type_code, statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,,CCR_,,3,)

#define _KERN_CALL_CR_1S_2V(func_name, type_code, statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,,CR_,1S_,2,)

#define _KERN_CALL_2V(func_name, type_code, statement)			\
		GENERIC_KERN_CALL(func_name,type_code,statement,,,,2,)

#define _KERN_CALL_2V_MIXED(func_name, type_code, statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,,RC_,,2,)

#define _KERN_CALL_1V_SCAL(func_name, type_code, statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,,,1S_,1,)

#define _KERN_CALL_1V_2SCAL( func_name, type_code , statement )	\
		GENERIC_KERN_CALL(func_name,type_code,statement,,,2S_,1,)

#define _KERN_CALL_2V_SCAL( func_name, type_code , statement )	\
		GENERIC_KERN_CALL(func_name,type_code,statement,,,1S_,2,)

#define _KERN_CALL_5V(func_name, type_code, statement)			\
		GENERIC_KERN_CALL(func_name,type_code,statement,,,,5,)

#define _KERN_CALL_4V_SCAL(func_name, type_code, statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,,,1S_,4,)

#define _KERN_CALL_3V_2SCAL(func_name, type_code, statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,,,2S_,3,)

#define _KERN_CALL_2V_3SCAL(func_name, type_code, statement)		\
		GENERIC_KERN_CALL(func_name,type_code,statement,,,3S_,2,)

#define _KERN_CALL_1V_3SCAL(func_name, type_code, statement)		\
		SLOW_KERN_CALL(func_name,type_code,statement,,,3S_,1)


#define _KERN_CALL_VS_LS( func_name, type_code, dst_arg, src_arg1, src_arg2 )\
	GENERIC_LS_KERN_CALL(func_name,type_code,1S_,2,dst_arg,src_arg1,src_arg2)

#define _KERN_CALL_VV_LS( func_name, type_code, dst_arg, src_arg1, src_arg2 )\
	GENERIC_LS_KERN_CALL(func_name,type_code,,3,dst_arg,src_arg1,src_arg2)

#define _KERN_CALL_VSMAP( fn, tc, op )			\
	GENERIC_KERN_CALL_DBM(fn,tc,SETBIT(src1 op scalar1_val),,1S_,1SRC)

#define _KERN_CALL_VVMAP( fn, tc, op )			\
	GENERIC_KERN_CALL_DBM(fn,tc,SETBIT( src1 op src2 ),,,2SRCS)

// bit_vset

#define _KERN_CALL_DBM_1S(fn, tc, statement)		\
		GENERIC_KERN_CALL_DBM(fn,tc,statement,,1S_,)

#define _KERN_CALL_DBM_1V(fn, tc, statement)		\
		GENERIC_KERN_CALL_DBM(fn,tc,statement,,,1SRC)

#define GENERIC_LS_KERN_CALL(fn,tc,sclrs,vecs,da,sa1,sa2)		\
	GENERIC_KERN_CALL(fn,tc,LSHIFT_SWITCH_32(da,sa1,sa2),,,sclrs,vecs,)



#define GENERIC_KERN_CALL(fn,tc,stat,bm,typ,sclrs,vecs,special)		\
									\
GENERIC_FAST_KERN_FUNC(KERN_FAST_NAME(fn,tc),stat,bm,typ,sclrs,vecs,special)	\
GENERIC_EQSP_KERN_FUNC(KERN_EQSP_NAME(fn,tc),stat,bm,typ,sclrs,vecs,special)	\
GENERIC_FLEN_KERN_FUNC(KERN_FLEN_NAME(fn,tc),stat,bm,typ,sclrs,vecs,special)	\
GENERIC_ELEN_KERN_FUNC(KERN_ELEN_NAME(fn,tc),stat,bm,typ,sclrs,vecs,special)	\
GENERIC_SLEN_KERN_FUNC(KERN_SLEN_NAME(fn,tc),stat,bm,typ,sclrs,vecs,special)


#define SLOW_KERN_CALL(fn,tc,stat,bm,typ,sclrs,vecs)			\
									\
GENERIC_SLEN_KERN_FUNC(KERN_SLEN_NAME(fn,tc),stat,bm,typ,sclrs,vecs,)




#define GENERIC_FAST_KERN_FUNC(name,statement,bm,typ,scalars,vectors,special)	\
									\
__global__ void name( KERN_FAST_ARGS_##bm##typ##scalars##vectors )	\
{									\
	DECL_SPECIAL_##special						\
	INIT_INDICES_##bm##vectors					\
	statement ;							\
}

#define GENERIC_EQSP_KERN_FUNC(name,statement,bm,typ,scalars,vectors,special)	\
									\
__global__ void name( KERN_EQSP_ARGS_##bm##typ##scalars##vectors )	\
{									\
	DECL_SPECIAL_##special						\
	INIT_INDICES_##bm##vectors					\
	SCALE_INDICES_EQSP_##bm##vectors				\
	statement;							\
}

#define GENERIC_FLEN_KERN_FUNC(name,statement,bm,typ,scalars,vectors,special)	\
									\
__global__ void name( KERN_FLEN_ARGS_##bm##typ##scalars##vectors)	\
{									\
	DECL_SPECIAL_##special						\
	INIT_INDICES_##bm##vectors					\
	if( index1.x < len) {					\
		statement ;						\
	}								\
}

#define GENERIC_ELEN_KERN_FUNC(name,statement,bm,typ,scalars,vectors,special)	\
									\
__global__ void name( KERN_ELEN_ARGS_##bm##typ##scalars##vectors)	\
{									\
	DECL_SPECIAL_##special						\
	INIT_INDICES_##bm##vectors					\
	if( index1.x < len ){					\
		SCALE_INDICES_EQSP_##bm##vectors			\
		statement;						\
	}								\
}

#if CUDA_COMP_CAP < 20

#define GENERIC_SLEN_KERN_FUNC(name,statement,bm,typ,scalars,vectors,special)	\
									\
__global__ void name( KERN_SLEN_ARGS_##bm##typ##scalars##vectors)	\
{									\
	DECL_SPECIAL_##special						\
	INIT_INDICES_XY_##bm##vectors					\
	if( index1.x < len.x && index1.y < len.y ){			\
		SCALE_INDICES_XY_##bm##vectors				\
		statement;						\
	}								\
}

#else /* CUDA_COMP_CAP >= 20 */


#define GENERIC_SLEN_KERN_FUNC(name,statement,bm,typ,scalars,vectors,special)	\
									\
__global__ void name( KERN_SLEN_ARGS_##bm##typ##scalars##vectors)	\
{									\
	DECL_SPECIAL_##special						\
	INIT_INDICES_XYZ_##bm##vectors					\
	if( index1.x < len.x && index1.y < len.y ){			\
		SCALE_INDICES_XYZ_##bm##vectors				\
		statement;						\
	}								\
}

#endif /* CUDA_COMP_CAP >= 20 */

/* These are for calls with a destination bitmap (vvm_lt etc)
 *
 * Here we cannot vectorize over all the pixels, because multiple
 * pixels share the same bitmap word.  Each thread has to set all the bits
 * in a given word.
 */

#define GENERIC_KERN_CALL_DBM(fn,tc,stat,typ,sclrs,vecs)		\
									\
GENERIC_FAST_KERN_FUNC_DBM(KERN_FAST_NAME(fn,tc),stat,typ,sclrs,vecs)	\
GENERIC_EQSP_KERN_FUNC_DBM(KERN_EQSP_NAME(fn,tc),stat,typ,sclrs,vecs)	\
GENERIC_FLEN_KERN_FUNC_DBM(KERN_FLEN_NAME(fn,tc),stat,typ,sclrs,vecs)	\
GENERIC_ELEN_KERN_FUNC_DBM(KERN_ELEN_NAME(fn,tc),stat,typ,sclrs,vecs)	\
GENERIC_SLEN_KERN_FUNC_DBM(KERN_SLEN_NAME(fn,tc),stat,typ,sclrs,vecs)


#define FAST_BIT_LOOP( statement, advance )				\
									\
	for(i_bit=0;i_bit<BITS_PER_BITMAP_WORD;i_bit++){		\
		bit = NUMBERED_BIT(i_bit);				\
		statement ;						\
		advance							\
	}


#define FLEN_BIT_LOOP( statement, advance )				\
									\
	for(i_bit=0;i_bit<BITS_PER_BITMAP_WORD;i_bit++){		\
		if( bmi.x >= bit0  && bmi.x < bit0+len ){		\
			bit = NUMBERED_BIT(i_bit);			\
			statement ;					\
		}							\
		advance							\
	}


#define BIT_LOOP( statement, advance )					\
									\
	for(i_bit=0;i_bit<BITS_PER_BITMAP_WORD;i_bit++){		\
		if( bmi.x >= bit0  && bmi.x < bit0+len.x ){		\
			bit = NUMBERED_BIT(i_bit);			\
			statement ;					\
		}							\
		advance							\
	}


#define ADVANCE_FAST_DBM	bmi.x++;
#define ADVANCE_FAST_SRC1	index2.x++;
#define ADVANCE_FAST_SRC2	index3.x++;

#define ADVANCE_FAST_DBM_	ADVANCE_FAST_DBM
#define ADVANCE_FAST_DBM_1SRC	ADVANCE_FAST_DBM ADVANCE_FAST_SRC1
#define ADVANCE_FAST_DBM_2SRCS	ADVANCE_FAST_DBM_1SRC ADVANCE_FAST_SRC2

#define ADVANCE_EQSP_DBM	bmi.x += bm_inc;
#define ADVANCE_EQSP_SRC1	index2.x+=inc2;
#define ADVANCE_EQSP_SRC2	index3.x+=inc3;

#define ADVANCE_EQSP_DBM_	ADVANCE_EQSP_DBM
#define ADVANCE_EQSP_DBM_1SRC	ADVANCE_EQSP_DBM ADVANCE_EQSP_SRC1
#define ADVANCE_EQSP_DBM_2SRCS	ADVANCE_EQSP_DBM_1SRC ADVANCE_EQSP_SRC2

#define ADVANCE_SLOW_DBM	bmi.x += bm_inc.x;
#define ADVANCE_SLOW_SRC1	index2.x+=inc2.x;
#define ADVANCE_SLOW_SRC2	index3.x+=inc3.x;

#define ADVANCE_SLOW_DBM_	ADVANCE_SLOW_DBM
#define ADVANCE_SLOW_DBM_1SRC	ADVANCE_SLOW_DBM ADVANCE_SLOW_SRC1
#define ADVANCE_SLOW_DBM_2SRCS	ADVANCE_SLOW_DBM_1SRC ADVANCE_SLOW_SRC2

#define SET_BITMAP_WORD		i_word=(bmi.x+bmi.y)/BITS_PER_BITMAP_WORD;


#define GENERIC_FAST_KERN_FUNC_DBM(name,statement,typ,scalars,vectors)	\
									\
__global__ void name( KERN_FAST_ARGS_DBM_##typ##scalars##vectors )	\
{									\
	INIT_INDICES_DBM_##vectors					\
	SET_BITMAP_WORD							\
	FAST_BIT_LOOP( statement, ADVANCE_FAST_DBM_##vectors )		\
}

#define GENERIC_EQSP_KERN_FUNC_DBM(name,statement,typ,scalars,vectors)	\
									\
__global__ void name( KERN_EQSP_ARGS_DBM_##typ##scalars##vectors )	\
{									\
	INIT_INDICES_DBM_##vectors					\
	SCALE_INDICES_EQSP_DBM_##vectors				\
	SET_BITMAP_WORD							\
	FAST_BIT_LOOP(statement,ADVANCE_EQSP_DBM_##vectors)		\
}

#define GENERIC_FLEN_KERN_FUNC_DBM(name,statement,typ,scalars,vectors)	\
									\
__global__ void name( KERN_FLEN_ARGS_DBM_##typ##scalars##vectors)	\
{									\
	INIT_INDICES_DBM_##vectors					\
	SET_BITMAP_WORD							\
	/* BUG need to put len test in statement */			\
	FLEN_BIT_LOOP(statement,ADVANCE_FAST_DBM_##vectors)			\
}

#define GENERIC_ELEN_KERN_FUNC_DBM(name,statement,typ,scalars,vectors)	\
									\
__global__ void name( KERN_ELEN_ARGS_DBM_##typ##scalars##vectors)	\
{									\
	INIT_INDICES_DBM_##vectors					\
	SCALE_INDICES_EQSP_DBM_##vectors				\
	/* BUG need to put len test in statement */			\
	SET_BITMAP_WORD							\
	FLEN_BIT_LOOP( statement, ADVANCE_EQSP_DBM_##vectors )		\
}

#if CUDA_COMP_CAP < 20

#define GENERIC_SLEN_KERN_FUNC_DBM(name,statement,typ,scalars,vectors)	\
									\
__global__ void name( KERN_SLEN_ARGS_DBM_##typ##scalars##vectors)	\
{									\
	INIT_INDICES_XY_DBM_##vectors					\
	SCALE_INDICES_XY_DBM_##vectors					\
	/* BUG need to put len test in statement */			\
	SET_BITMAP_WORD							\
	BIT_LOOP( statement , ADVANCE_SLOW_DBM_##vectors )		\
}

#else /* CUDA_COMP_CAP >= 20 */

#define GENERIC_SLEN_KERN_FUNC_DBM(name,statement,typ,scalars,vectors)	\
									\
__global__ void name( KERN_SLEN_ARGS_DBM_##typ##scalars##vectors)	\
{									\
	INIT_INDICES_XYZ_DBM_##vectors					\
	SCALE_INDICES_XYZ_DBM_##vectors					\
	/* BUG need to put len test in statement */			\
	SET_BITMAP_WORD							\
	BIT_LOOP( statement , ADVANCE_SLOW_DBM_##vectors )		\
}

#endif /* CUDA_COMP_CAP >= 20 */



#define _KERN_CALL_MM( func_name, type_code, statement )		\
									\
	__global__ void g_##type_code##_##func_name##_helper		\
	(std_type* a, std_type* b, std_type* c, int len1, int len2)	\
	{								\
		INIT_INDICES_3						\
		if( index3.x < len2 )					\
			statement ;					\
		else if( index1.x < len1 )				\
			dst = src1 ;					\
	}								\
									\

#define _KERN_CALL_MM_IND( func_name, type_code, statement1, statement2 )\
									\
	__global__ void g_##type_code##_##func_name##_index_setup	\
	(index_type* a, std_type* b, std_type* c, u_long len1, u_long len2)\
	{								\
		INIT_INDICES_3						\
		if( index3.x < len2 )					\
			statement1 ;					\
		else if( index1.x < len1 )				\
			dst = index2.x ;				\
	}								\
									\
	__global__ void g_##type_code##_##func_name##_index_helper	\
		(index_type* a, index_type* b, index_type* c,		\
				std_type *orig, int len1, int len2)	\
	{								\
		INIT_INDICES_3						\
		if( index3.x < len2 )					\
			statement2 ;					\
		else if( index1.x < len1 )				\
			dst = src1 ;					\
	}								\
									\

/* For nocc_setup, we index directly into the value and count temp arrays
 * (a and b, respectively), but we have to double the index for the source
 * array c, and the index array d.  Because we are comparing adjacent pairs, 
 */

#define _KERN_CALL_MM_NOCC( func_name, type_code, test1, test2 )	\
									\
	__global__ void g_##type_code##_##func_name##_nocc_setup(	\
		std_type* dst_extrema, index_type* dst_counts,		\
		std_type* src_vals, index_type *dst_indices,		\
		u_long len1, u_long len2)				\
	{								\
		INIT_INDICES_2						\
		index2.x *= 2;						\
		if( index1.x < len2 ){					\
			if( test1 ){					\
				dst_extrema[index1.x] = src_vals[index2.x];	\
				dst_counts[index1.x]=1;			\
				dst_indices[index2.x]=index2.x;		\
			} else if( test2 ){				\
				dst_extrema[index1.x] = src_vals[index2.x+1];\
				dst_counts[index1.x]=1;			\
				dst_indices[index2.x]=index2.x+1;		\
			} else {					\
				dst_extrema[index1.x] = src_vals[index2.x];	\
				dst_counts[index1.x]=2;			\
				dst_indices[index2.x]=index2.x;		\
				dst_indices[index2.x+1]=index2.x+1;		\
			}						\
		} else {						\
			/* Nothing to compare */			\
			dst_extrema[index1.x] = src_vals[index2.x];		\
			dst_counts[index1.x]=1;				\
			dst_indices[index2.x]=index2.x;			\
		}							\
	}								\
									\
	__global__ void g_##type_code##_##func_name##_nocc_helper(	\
		std_type* dst_extrema, index_type* dst_counts,		\
		std_type* src_vals, index_type *src_counts,		\
		index_type *dst_indices,				\
		int len1, int len2, int stride)				\
	{								\
		int i;							\
		INIT_INDICES_2						\
		index2.x *= 2;						\
		if( index1.x < len2 ){					\
			if( test1 ){					\
				dst_extrema[index1.x]=src_vals[index2.x];	\
				dst_counts[index1.x]=src_counts[index2.x];	\
				/* No copy necessary */			\
			} else if( test2 ){				\
				dst_extrema[index1.x]=src_vals[index2.x+1];	\
				dst_counts[index1.x]=src_counts[index2.x+1];\
				/* Now copy the indices down */		\
				for(i=0;i<dst_counts[index1.x];i++){	\
					dst_indices[index1.x*stride+i] =	\
			dst_indices[index1.x*stride+stride/2+i];		\
				}					\
			} else {					\
				dst_extrema[index1.x]=src_vals[index2.x];	\
				dst_counts[index1.x] = src_counts[index2.x] + \
					src_counts[index2.x+1];		\
				/* Now copy the second half of the indices */\
				for(i=0;i<src_counts[index2.x+1];i++){	\
		dst_indices[index1.x*stride+src_counts[index2.x]+i] =	\
			dst_indices[index1.x*stride+stride/2+i];		\
				}					\
			}						\
		} else {						\
			dst_extrema[index1.x]=src_vals[index2.x];		\
			dst_counts[index1.x]=src_counts[index2.x];		\
			/* No copy necessary */				\
		}							\
	}

// vsum, vdot, etc
// BUG this is hard-coded for vsum!?
//
// The idea is that , because all threads cannot access the destination simultaneously,
// we have to make the source smaller recursively...  But when we project
// to a vector instead of a scalar, we can to the elements of the vector in parallel...
// This is quite tricky.
//
// Example:  col=sum(m)
//
// m = | 1 2 3 4 |
//     | 5 6 7 8 |
//
// tmp = | 4  6  |
//       | 12 14 |
//
// col = | 10 |
//       | 26 |
     

// BUG - we need to make this do vmaxv and vminv as well.
// It's the same except for the sum line, which would be replaced with
//

#define psrc1	s1[index1.x]
#define psrc2	s2[index1.x]

// for vsum:   psrc1 + psrc2
// for vmaxv:  psrc1 > psrc2 ? psrc1 : psrc2

#define _KERN_CALL_2V_PROJ( func_name, type_code, expr )		\
									\
	__global__ void g_##type_code##_##func_name(			\
		std_type* dest, std_type* s1,				\
		int len1, int len2 )					\
	{								\
		INIT_INDICES_1						\
									\
		if( index1.x < len2 ){					\
			std_type *s2;					\
			s2 = s1 + len1;					\
			dest[index1.x] = expr ;				\
		} else if( index1.x < len1 ){				\
			dest[index1.x] = s1[index1.x];			\
		}							\
	}


#define _KERN_CALL_3V_PROJ( func_name, type_code )			\
									\
	__global__ void g_##type_code##_##func_name(			\
		std_type* dest, std_type* s1, std_type* s2,		\
		int len1, int len2 )					\
	{								\
		INIT_INDICES_1						\
		if( index1.x < len2 ){					\
			std_type *s1b, *s2b;				\
			s1b = s1 + len1;				\
			s2b = s2 + len1;				\
			dest[index1.x] = s1[index1.x] * s2[index1.x] +	\
				s1b[index1.x] * s2b[index1.x] ;		\
		} else if( index1.x < len1 ){				\
			dest[index1.x] = s1[index1.x];			\
		}							\
	}



#define LSHIFT_SWITCH_32( ls_dst , arg1 , arg2 )			\
	switch(arg2) {							\
		case 0:  ls_dst = arg1;     break;			\
		case 1:  ls_dst = arg1<<1;  break;			\
		case 2:  ls_dst = arg1<<2;  break;			\
		case 3:  ls_dst = arg1<<3;  break;			\
		case 4:  ls_dst = arg1<<4;  break;			\
		case 5:  ls_dst = arg1<<5;  break;			\
		case 6:  ls_dst = arg1<<6;  break;			\
		case 7:  ls_dst = arg1<<7;  break;			\
		case 8:  ls_dst = arg1<<8;  break;			\
		case 9:  ls_dst = arg1<<9;  break;			\
		case 10: ls_dst = arg1<<10; break;			\
		case 11: ls_dst = arg1<<11; break;			\
		case 12: ls_dst = arg1<<12; break;			\
		case 13: ls_dst = arg1<<13; break;			\
		case 14: ls_dst = arg1<<14; break;			\
		case 15: ls_dst = arg1<<15; break;			\
		case 16: ls_dst = arg1<<16; break;			\
		case 17: ls_dst = arg1<<17; break;			\
		case 18: ls_dst = arg1<<18; break;			\
		case 19: ls_dst = arg1<<19; break;			\
		case 20: ls_dst = arg1<<20; break;			\
		case 21: ls_dst = arg1<<21; break;			\
		case 22: ls_dst = arg1<<22; break;			\
		case 23: ls_dst = arg1<<23; break;			\
		case 24: ls_dst = arg1<<24; break;			\
		case 25: ls_dst = arg1<<25; break;			\
		case 26: ls_dst = arg1<<26; break;			\
		case 27: ls_dst = arg1<<27; break;			\
		case 28: ls_dst = arg1<<28; break;			\
		case 29: ls_dst = arg1<<29; break;			\
		case 30: ls_dst = arg1<<30; break;			\
		case 31: ls_dst = arg1<<31; break;			\
	}

