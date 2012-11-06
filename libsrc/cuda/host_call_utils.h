#ifndef _HOST_CALL_UTILS_H_
#define _HOST_CALL_UTILS_H_ 

// this is 80 columns
//345678901234567890123456789012345678901234567890123456789012345678901234567890


#define NN_GPU		n_blocks, n_threads_per_block

#define BLOCK_VARS_DECLS						\
									\
	cudaError_t e;							\
	dim3 n_blocks, n_threads_per_block;				\
	dim3 extra;

#define GET_MAX_THREADS( dp )						\
									\
	insure_cuda_device( dp );					\
	max_threads_per_block =	curr_cdp->cudev_prop.maxThreadsPerBlock;

// We should have an intelligent way of designing blocks...
// On wheatstone, the card can have a maximum of 512 threads/block.
// But this doesn't divide typical video image dimensions
// of 640x480.  But 640 is 128x5, so we could have blocks
// that are 128x4 and not have any leftover.  Using our dumb
// strategy, we have blocks that are 16x32... 480/32=15,
// so we are OK here.
//
// We found a bug doing interlaces of 640/480, the 240 line half-frames
// came out to 7.5 32 row blocks in height.  In that case it's better
// to have the blocks wider and less tall.
//
// It might be nice to adjust things like this automatically, but
// it's kind of tricky!?

// This is kind of kludgy, but let's try it
#define MAX_THREADS_PER_ROW	32

// This assumes we are dealing with an image

#define GET_THREADS_PER_ARRAY( dp )				\
								\
	len.x = dp->dt_cols;					\
	len.y = dp->dt_rows;					\
								\
	SETUP_BLOCKS_XY


// SETUP_BLOCKS_XY - set up blocks & threads using len as input

#define SETUP_BLOCKS_XY_	SETUP_BLOCKS_XY
#define SETUP_BLOCKS_XY_SBM_	SETUP_BLOCKS_XY

#define SETUP_BLOCKS_XY						\
								\
	SETUP_BLOCKS_X(len.x)					\
	SETUP_BLOCKS_Y



#define SETUP_BLOCKS_XYZ_	SETUP_BLOCKS_XYZ
#define SETUP_BLOCKS_XYZ_SBM_	SETUP_BLOCKS_XYZ

#define SETUP_BLOCKS_XYZ					\
								\
	SETUP_BLOCKS_X(len.x)					\
	SETUP_BLOCKS_Y						\
	SETUP_BLOCKS_Z


/* If we have a destination bitmap, we handle all the bits in one word
 * in a single thread.
 *
 * BUG - here we ignore bit0 ???
 *
 * MAX_THREADS_PER_ROW is 32...  for a 512 pixel wide image, the nmber
 * of bitmap words is either 8 (64 bit words) or 16 (32 bit words).
 * So we need to 
 */

#define SETUP_BLOCKS_XY_DBM_					\
								\
	SETUP_BLOCKS_X( N_BITMAP_WORDS(len.x) )			\
	SETUP_BLOCKS_Y


#define SETUP_BLOCKS_XYZ_DBM_					\
								\
	SETUP_BLOCKS_X( N_BITMAP_WORDS(len.x) )			\
	SETUP_BLOCKS_Y						\
	SETUP_BLOCKS_Z


#define SETUP_BLOCKS_X(w)					\
								\
/*sprintf(DEFAULT_ERROR_STRING,"SETUP_BLOCKS_X:  len = %d",w);\
advise(DEFAULT_ERROR_STRING);*/\
	if( (w) < MAX_THREADS_PER_ROW ) {			\
		n_threads_per_block.x = w;			\
		n_blocks.x = 1;					\
		extra.x = 0;					\
	} else {						\
		n_blocks.x = (w) / MAX_THREADS_PER_ROW;		\
		n_threads_per_block.x = MAX_THREADS_PER_ROW;	\
		extra.x = (w) % MAX_THREADS_PER_ROW;		\
	}


#define SETUP_BLOCKS_Y						\
								\
	n_threads_per_block.y = max_threads_per_block /		\
					n_threads_per_block.x;	\
	if( len.y < n_threads_per_block.y ){			\
		n_threads_per_block.y = len.y;			\
		n_blocks.y = 1;					\
		extra.y = 0;					\
	} else {						\
		n_blocks.y = len.y / n_threads_per_block.y;	\
		extra.y = len.y % n_threads_per_block.y;	\
	}							\
	if( extra.x > 0 ) n_blocks.x++;				\
	if( extra.y > 0 ) n_blocks.y++;

#define SETUP_BLOCKS_Z						\
								\
	n_threads_per_block.z = max_threads_per_block /		\
		(n_threads_per_block.x*n_threads_per_block.y);	\
	if( len.z < n_threads_per_block.z ){			\
		n_threads_per_block.z = len.z;			\
		n_blocks.z = 1;					\
		extra.z = 0;					\
	} else {						\
		n_blocks.z = len.z / n_threads_per_block.z;	\
		extra.z = len.z % n_threads_per_block.z;	\
	}							\
	if( extra.z > 0 ) n_blocks.z++;

#ifdef MORE_DEBUG

#define REPORT_THREAD_INFO					\
								\
sprintf(DEFAULT_ERROR_STRING,"Blocks:  %d x %d x %d    Threads:  %d x %d x %d",	\
n_blocks.x,n_blocks.y,n_blocks.z,	\
n_threads_per_block.x,n_threads_per_block.y,n_threads_per_block.z);\
advise(DEFAULT_ERROR_STRING);						\
sprintf(DEFAULT_ERROR_STRING,"Length:  %d x %d x %d    Extra:  %d x %d x %d",	\
len.x,len.y,len.z,extra.x,extra.y,extra.z);				\
advise(DEFAULT_ERROR_STRING);

#define REPORT_THREAD_INFO2					\
								\
sprintf(DEFAULT_ERROR_STRING,"Blocks:  %d x %d    Threads:  %d x %d",	\
n_blocks.x,n_blocks.y,n_threads_per_block.x,n_threads_per_block.y);\
advise(DEFAULT_ERROR_STRING);						\
sprintf(DEFAULT_ERROR_STRING,"Len1:  %d   Len2:  %d   Extra:  %d x %d",	\
len1,len2,extra.x,extra.y);					\
advise(DEFAULT_ERROR_STRING);

#else /* ! MORE_DEBUG */

#define REPORT_THREAD_INFO
#define REPORT_THREAD_INFO2

#endif /* ! MORE_DEBUG */


#define DEFAULT_YZ						\
								\
	n_threads_per_block.y = n_threads_per_block.z =		\
	n_blocks.y = n_blocks.z = 1;				\
	extra.y = extra.z = 0;

#define GET_THREADS_PER_BLOCK					\
								\
	DEFAULT_YZ						\
	SET_BLOCKS_FROM_LEN(len.x)

#define SET_BLOCKS_FROM_LEN( n_tot )				\
								\
	if( n_tot < max_threads_per_block ) {			\
		n_threads_per_block.x = n_tot;			\
		n_blocks.x = 1;					\
		extra.x = 0;					\
	} else {						\
		n_blocks.x = n_tot / max_threads_per_block;	\
		n_threads_per_block.x = max_threads_per_block;	\
		extra.x = n_tot % max_threads_per_block;	\
	}

#define GET_THREADS_PER_SBM_BLOCK	GET_THREADS_PER_BLOCK

// This used to be called GET_THREADS_PER_BITMAP_BLOCK

#define GET_THREADS_PER_DBM_BLOCK				\
								\
	DEFAULT_YZ						\
	if( (vap->va_bit0+len.x) < BITS_PER_BITMAP_WORD ) {	\
		n_threads_per_block.x = 1;			\
		n_blocks.x = 1;					\
	} else {						\
		int nw;						\
		nw = N_BITMAP_WORDS(vap->va_bit0+len.x);	\
		SET_BLOCKS_FROM_LEN(nw)				\
	}


// SETUP_BLOCKS_XY - set up blocks & threads using len as input

#define SETUP_BLOCKS_XY_TEMP					\
								\
	if( len.x < MAX_THREADS_PER_ROW ) {			\
		n_threads_per_block.x = len.x;			\
		n_blocks.x = 1;					\
		extra.x = 0;					\
	} else {						\
		n_blocks.x = len.x / MAX_THREADS_PER_ROW;	\
		n_threads_per_block.x = MAX_THREADS_PER_ROW;	\
		extra.x = len.x % MAX_THREADS_PER_ROW;		\
	}							\
	n_threads_per_block.y = max_threads_per_block /		\
					n_threads_per_block.x;	\
	if( len.y < n_threads_per_block.y ){			\
		n_threads_per_block.y = len.y;			\
		n_blocks.y = 1;					\
		extra.y = 0;					\
	} else {						\
		n_blocks.y = len.y / n_threads_per_block.y;	\
		extra.y = len.y % n_threads_per_block.y;	\
	}							\
	if( extra.x > 0 ) n_blocks.x++;				\
	if( extra.y > 0 ) n_blocks.y++;


#define MAX_THREADS_PER_BITMAP_ROW	MIN(MAX_THREADS_PER_ROW,BITS_PER_BITMAP_WORD)

// Because we iterate over bits, we have to divide up one of the dimensions.
// For now, we always do the x dimension.  Not very smart, but hopefully
// good enough most of the time.
//


#define SETUP_BLOCKS_BITMAP_XY					\
								\
	SETUP_BLOCKS_XY						\
								\
	{							\
	int tot_len;						\
	tot_len = len.x * len.y * len.z;			\
	/* BUG this assumes bit0 == 0 */			\
	if(tot_len<BITS_PER_BITMAP_WORD)			\
		n_bits = tot_len;				\
	else	n_bits = BITS_PER_BITMAP_WORD;			\
	}							\
	{							\
	int len_per_bit;					\
	len_per_bit = len.x/BITS_PER_BITMAP_WORD;		\
	n_extra_bits = len.x % BITS_PER_BITMAP_WORD;		\
	max_len = len_per_bit + (n_extra_bits>0?1:0);		\
								\
	if( max_len < MAX_THREADS_PER_ROW ){			\
		n_threads_per_block.x = max_len;		\
		n_blocks.x = 1;					\
	} else {						\
		n_blocks.x = max_len / MAX_THREADS_PER_ROW;	\
		n_threads_per_block.x = MAX_THREADS_PER_ROW;	\
		extra.x = max_len % MAX_THREADS_PER_ROW;	\
	}							\
	}


								\


#define INSIST_CONTIG( dp , msg )					\
									\
	if( ! is_contiguous( dp ) ){					\
		sprintf(DEFAULT_ERROR_STRING,					\
	"Sorry, object %s must be contiguous for GPU %s.",		\
			dp->dt_name,msg);				\
		NWARN(DEFAULT_ERROR_STRING);					\
		return;							\
	}

#define INSIST_LENGTH( n , msg , name )					\
									\
		if( (n) == 1 ){						\
			sprintf(DEFAULT_ERROR_STRING,				\
	"Oops, kind of silly to do %s of 1-len vector %s!?",msg,name);	\
			NWARN(DEFAULT_ERROR_STRING);				\
			return;						\
		}

/* BUG - should these defns go away?? */

#define DEST_DECL( type )				\
							\
	type *arg1;					\
	dim3 len;					\
	dim3 inc1;

#define DECLS_2V( type )				\
							\
	DEST_DECL( type )				\
	type *arg2;					\
	dim3 inc2;

#define DECLS_2V_MIXED( type, cpx_type )		\
							\
	DEST_DECL( type )				\
	cpx_type *arg2;					\
	dim3 inc2;

#define DECLS_3V( type )				\
							\
	DECLS_2V( type )				\
	type *arg3;					\
	dim3 inc3;

#define DECLS_4V( type )				\
							\
	DECLS_3V( type )				\
	type *arg4;					\
	dim3 inc4;

#define DECLS_5V( type )				\
							\
	DECLS_4V( type )				\
	type *arg5;					\
	dim3 inc5;

#define DECLS_2SRC( type )				\
							\
	DECLS_1SRC( type )				\
	type *arg3;					\
	dim3 inc3;

#define DECLS_1SRC( type )				\
							\
	type *arg2;					\
	dim3 len, inc2;

#define BITMAP_DECL					\
							\
	bitmap_word *bmp, which_bit;

#define GET_DEST(type)					\
							\
	arg1 = (type *)oap->oa_dest->dt_data;		\
	len.x = oap->oa_dest->dt_n_type_elts;		\
	len.y = len.z = 1;				\
	GET_MAX_THREADS( oap->oa_dest )

#define GET_2V(type)					\
							\
	GET_DEST(type)					\
	arg2 = (type *)oap->oa_dp[0]->dt_data;

#define GET_2V_MIXED(type,cpx_type)					\
							\
	GET_DEST(type)					\
	arg2 = (cpx_type *)oap->oa_dp[0]->dt_data;

#define GET_3V(type)					\
							\
	GET_2V(type)					\
	arg3 = (type *)oap->oa_dp[1]->dt_data;

#define GET_4V(type)					\
							\
	GET_3V(type)					\
	arg4 = (type *)oap->oa_dp[2]->dt_data;

#define GET_5V(type)					\
							\
	GET_4V(type)					\
	arg5 = (type *)oap->oa_dp[3]->dt_data;

#define GET_2SRC( type )				\
							\
	GET_1SRC( type )				\
	arg3 = (type *)oap->oa_dp[1]->dt_data;

#define GET_1SRC( type )				\
							\
	arg2 = (type *)oap->oa_dp[0]->dt_data;		\
	len.x = oap->oa_dp[0]->dt_n_type_elts;		\
	len.y = len.z = 1;

#ifdef MORE_DEBUG

#define REPORT_VECTORIZATION1( host_func_name )				\
									\
	sprintf(DEFAULT_ERROR_STRING,						\
"%s:  ready to vectorize:\tlen.x = %ld, inc1.x = %ld, inc1.y = %ld",	\
		#host_func_name,len.x,inc1.x,inc1.y);			\
	advise(DEFAULT_ERROR_STRING);

#define REPORT_VECTORIZATION2( host_func_name )				\
									\
	REPORT_VECTORIZATION1( host_func_name )				\
	sprintf(DEFAULT_ERROR_STRING, "\t\t\t\tinc2.x = %ld, inc2.y = %ld",	\
		inc2.x,inc2.y);						\
	advise(DEFAULT_ERROR_STRING);

#define REPORT_VECTORIZATION3( host_func_name )				\
									\
	REPORT_VECTORIZATION2( host_func_name )				\
	sprintf(DEFAULT_ERROR_STRING, "\t\t\t\tinc3.x = %ld, inc3.y = %ld",	\
		inc3.x,inc3.y);						\
	advise(DEFAULT_ERROR_STRING);

#define REPORT_VECTORIZATION4( host_func_name )				\
									\
	REPORT_VECTORIZATION3( host_func_name )				\
	sprintf(DEFAULT_ERROR_STRING, "\t\t\t\tinc4.x = %ld, inc4.y = %ld",	\
		inc4.x,inc4.y);						\
	advise(DEFAULT_ERROR_STRING);

#define REPORT_VECTORIZATION5( host_func_name )				\
									\
	REPORT_VECTORIZATION4( host_func_name )				\
	sprintf(DEFAULT_ERROR_STRING, "\t\t\t\tinc5.x = %ld, inc5.y = %ld",	\
		inc5.x,inc5.y);						\
	advise(DEFAULT_ERROR_STRING);

#else /* ! MORE_DEBUG */

#define REPORT_VECTORIZATION1( host_func_name )
#define REPORT_VECTORIZATION2( host_func_name )
#define REPORT_VECTORIZATION3( host_func_name )
#define REPORT_VECTORIZATION4( host_func_name )
#define REPORT_VECTORIZATION5( host_func_name )

#endif /* ! MORE_DEBUG */


#define DEFAULT_INCS1							\
									\
	len.z=1;							\
	inc1.z=0;
	
#define DEFAULT_INCS2							\
									\
	DEFAULT_INCS1							\
	inc2.z=0;
	
#define DEFAULT_INCS3							\
									\
	DEFAULT_INCS2							\
	inc3.z=0;
	
#define DEFAULT_INCS4							\
									\
	DEFAULT_INCS3							\
	inc4.z=0;
	
#define DEFAULT_INCS5							\
									\
	DEFAULT_INCS4							\
	inc5.z=0;
	

#define SETUP_5_INCS( host_func_name )					\
									\
		int i;							\
		int level;						\
									\
		DEFAULT_INCS5						\
		i=oap->oa_dest->dt_mindim;				\
		len.x = oap->oa_dest->dt_type_dim[i];			\
		inc1.x = oap->oa_dest->dt_type_inc[i];			\
		inc2.x = oap->oa_dp[0]->dt_type_inc[i];		\
		inc3.x = oap->oa_dp[1]->dt_type_inc[i];		\
		inc4.x = oap->oa_dp[2]->dt_type_inc[i];		\
		inc5.x = oap->oa_dp[3]->dt_type_inc[i];		\
		i++;							\
		level = 1;						\
		/* Find the longest evenly-spaced run */		\
		while( i <= oap->oa_dest->dt_maxdim && level == 1 ){	\
	if( oap->oa_dest->dt_type_inc[i]  == ( len.x * inc1.x ) &&	\
	    oap->oa_dp[0]->dt_type_inc[i] == ( len.x * inc2.x ) &&	\
	    oap->oa_dp[1]->dt_type_inc[i] == ( len.x * inc3.x ) &&	\
	    oap->oa_dp[2]->dt_type_inc[i] == ( len.x * inc4.x ) &&	\
	    oap->oa_dp[3]->dt_type_inc[i] == ( len.x * inc5.x ) ){	\
				len.x *= oap->oa_dest->dt_type_dim[i];	\
			} else {					\
				len.y = oap->oa_dest->dt_type_dim[i];	\
				inc1.y = oap->oa_dest->dt_type_inc[i];	\
				inc2.y = oap->oa_dp[0]->dt_type_inc[i];\
				inc3.y = oap->oa_dp[1]->dt_type_inc[i];\
				inc4.y = oap->oa_dp[2]->dt_type_inc[i];\
				inc5.y = oap->oa_dp[3]->dt_type_inc[i];\
				level = 2;				\
			}						\
			i++;						\
		}							\
		/* Make sure we can do the rest with just one more inc */ \
		while( i <= oap->oa_dest->dt_maxdim ){			\
	if( oap->oa_dest->dt_type_inc[i]  == ( len.y * inc1.y ) &&	\
	    oap->oa_dp[0]->dt_type_inc[i] == ( len.y * inc2.y ) &&	\
	    oap->oa_dp[1]->dt_type_inc[i] == ( len.y * inc3.y ) &&	\
	    oap->oa_dp[2]->dt_type_inc[i] == ( len.y * inc4.y ) &&	\
	    oap->oa_dp[3]->dt_type_inc[i] == ( len.y & inc5.y ) ){	\
				len.y *= oap->oa_dest->dt_type_dim[i];	\
			} else {					\
				sprintf(DEFAULT_ERROR_STRING,			\
"%s:  More than two increments required for non-contiguous object(s)",	\
						#host_func_name);	\
				NWARN(DEFAULT_ERROR_STRING);			\
				return;					\
			}						\
			i++;						\
		}							\
		SETUP_BLOCKS_XY						\
		/* report for debugging */				\
		REPORT_VECTORIZATION5( host_func_name )



#define SETUP_4_INCS( host_func_name )					\
									\
		int i;							\
		int level;						\
									\
		DEFAULT_INCS4						\
		i=oap->oa_dest->dt_mindim;				\
		len.x = oap->oa_dest->dt_type_dim[i];			\
		inc1.x = oap->oa_dest->dt_type_inc[i];			\
		inc2.x = oap->oa_dp[0]->dt_type_inc[i];		\
		inc3.x = oap->oa_dp[1]->dt_type_inc[i];		\
		inc4.x = oap->oa_dp[2]->dt_type_inc[i];		\
		i++;							\
		level = 1;						\
		/* Find the longest evenly-spaced run */		\
		while( i <= oap->oa_dest->dt_maxdim && level == 1 ){	\
	if( oap->oa_dest->dt_type_inc[i]  == ( len.x * inc1.x ) &&	\
	    oap->oa_dp[0]->dt_type_inc[i] == ( len.x * inc2.x ) &&	\
	    oap->oa_dp[1]->dt_type_inc[i] == ( len.x * inc3.x ) &&	\
	    oap->oa_dp[2]->dt_type_inc[i] == ( len.x * inc4.x ) ){	\
				len.x *= oap->oa_dest->dt_type_dim[i];	\
			} else {					\
				len.y = oap->oa_dest->dt_type_dim[i];	\
				inc1.y = oap->oa_dest->dt_type_inc[i];	\
				inc2.y = oap->oa_dp[0]->dt_type_inc[i];\
				inc3.y = oap->oa_dp[1]->dt_type_inc[i];\
				inc4.y = oap->oa_dp[2]->dt_type_inc[i];\
				level = 2;				\
			}						\
			i++;						\
		}							\
		/* Make sure we can do the rest with just one more inc */ \
		while( i <= oap->oa_dest->dt_maxdim ){			\
	if( oap->oa_dest->dt_type_inc[i]  == ( len.y * inc1.y ) &&	\
	    oap->oa_dp[0]->dt_type_inc[i] == ( len.y * inc2.y ) &&	\
	    oap->oa_dp[1]->dt_type_inc[i] == ( len.y * inc3.y ) &&	\
	    oap->oa_dp[2]->dt_type_inc[i] == ( len.y & inc4.y ) ){	\
				len.y *= oap->oa_dest->dt_type_dim[i];	\
			} else {					\
				sprintf(DEFAULT_ERROR_STRING,			\
"%s:  More than two increments required for non-contiguous object(s)",	\
						#host_func_name);	\
				NWARN(DEFAULT_ERROR_STRING);			\
				return;					\
			}						\
			i++;						\
		}							\
		SETUP_BLOCKS_XY						\
		/* report for debugging */				\
		REPORT_VECTORIZATION4( host_func_name )



#define SETUP_3_INCS( host_func_name )					\
									\
		int i;							\
		int level;						\
									\
		DEFAULT_INCS3						\
		i=oap->oa_dest->dt_mindim;				\
		len.x = oap->oa_dest->dt_type_dim[i];			\
		inc1.x = oap->oa_dest->dt_type_inc[i];			\
		inc2.x = oap->oa_dp[0]->dt_type_inc[i];		\
		inc3.x = oap->oa_dp[1]->dt_type_inc[i];		\
		i++;							\
		level = 1;						\
		/* Find the longest evenly-spaced run */		\
		while( i <= oap->oa_dest->dt_maxdim && level == 1 ){	\
	if( oap->oa_dest->dt_type_inc[i]  == ( len.x * inc1.x ) &&	\
	    oap->oa_dp[0]->dt_type_inc[i] == ( len.x * inc2.x ) &&	\
	    oap->oa_dp[1]->dt_type_inc[i] == ( len.x * inc3.x ) ){	\
				len.x *= oap->oa_dest->dt_type_dim[i];	\
			} else {					\
				len.y = oap->oa_dest->dt_type_dim[i];	\
				inc1.y = oap->oa_dest->dt_type_inc[i];	\
				inc2.y = oap->oa_dp[0]->dt_type_inc[i];\
				inc3.y = oap->oa_dp[1]->dt_type_inc[i];\
				level = 2;				\
			}						\
			i++;						\
		}							\
		/* Make sure we can do the rest with just one more inc */ \
		while( i <= oap->oa_dest->dt_maxdim ){			\
	if( oap->oa_dest->dt_type_inc[i]  == ( len.y * inc1.y ) &&	\
	    oap->oa_dp[0]->dt_type_inc[i] == ( len.y * inc2.y ) &&	\
	    oap->oa_dp[1]->dt_type_inc[i] == ( len.y * inc3.y ) ){	\
				len.y *= oap->oa_dest->dt_type_dim[i];	\
			} else {					\
				sprintf(DEFAULT_ERROR_STRING,			\
"%s:  More than two increments required for non-contiguous object(s)",	\
						#host_func_name);	\
				NWARN(DEFAULT_ERROR_STRING);			\
				return;					\
			}						\
			i++;						\
		}							\
		SETUP_BLOCKS_XY						\
		/* report for debugging */				\
		REPORT_VECTORIZATION3( host_func_name )



#define SETUP_2_INCS( host_func_name )					\
									\
		int i;							\
		int level;						\
									\
		DEFAULT_INCS2						\
		i=oap->oa_dest->dt_mindim;				\
		len.x = oap->oa_dest->dt_type_dim[i];			\
		inc1.x = oap->oa_dest->dt_type_inc[i];			\
		inc2.x = oap->oa_dp[0]->dt_type_inc[i];		\
		i++;							\
		level = 1;						\
		/* Find the longest evenly-spaced run */		\
		while( i <= oap->oa_dest->dt_maxdim && level == 1 ){	\
	if( oap->oa_dest->dt_type_inc[i]  == ( len.x * inc1.x ) &&	\
	    oap->oa_dp[0]->dt_type_inc[i] == ( len.x * inc2.x ) ){	\
				len.x *= oap->oa_dest->dt_type_dim[i];	\
			} else {					\
				len.y = oap->oa_dest->dt_type_dim[i];	\
				inc1.y = oap->oa_dest->dt_type_inc[i];	\
				inc2.y = oap->oa_dp[0]->dt_type_inc[i];\
				level = 2;				\
			}						\
			i++;						\
		}							\
		/* Make sure we can do the rest with just one more inc */ \
		while( i <= oap->oa_dest->dt_maxdim ){			\
	if( oap->oa_dest->dt_type_inc[i]  == ( len.y * inc1.y ) &&	\
	    oap->oa_dp[0]->dt_type_inc[i] == ( len.y * inc2.y ) ){	\
				len.y *= oap->oa_dest->dt_type_dim[i];	\
			} else {					\
				sprintf(DEFAULT_ERROR_STRING,			\
"%s:  More than two increments required for non-contiguous object(s)",	\
						#host_func_name);	\
				NWARN(DEFAULT_ERROR_STRING);			\
				return;					\
			}						\
			i++;						\
		}							\
		/* Now set up blocks & threads XY */			\
		SETUP_BLOCKS_XY						\
		/* report for debugging */				\
		REPORT_VECTORIZATION2( host_func_name )






#define SETUP_1_INCS( host_func_name )					\
									\
		int i;							\
		int level;						\
									\
		DEFAULT_INCS1						\
		i=oap->oa_dest->dt_mindim;				\
		len.x = oap->oa_dest->dt_type_dim[i];			\
		inc1.x = oap->oa_dest->dt_type_inc[i];			\
		i++;							\
		level = 1;						\
		/* Find the longest evenly-spaced run */		\
		while( i <= oap->oa_dest->dt_maxdim && level == 1 ){	\
	if( oap->oa_dest->dt_type_inc[i]  == ( len.x * inc1.x ) ){	\
				len.x *= oap->oa_dest->dt_type_dim[i];	\
			} else {					\
				len.y = oap->oa_dest->dt_type_dim[i];	\
				inc1.y = oap->oa_dest->dt_type_inc[i];	\
				level = 2;				\
			}						\
			i++;						\
		}							\
		/* Make sure we can do the rest with just one more inc */ \
		while( i <= oap->oa_dest->dt_maxdim ){			\
	if( oap->oa_dest->dt_type_inc[i]  == ( len.y * inc1.y ) ){	\
				len.y *= oap->oa_dest->dt_type_dim[i];	\
			} else {					\
				sprintf(DEFAULT_ERROR_STRING,			\
"%s:  More than two increments required for non-contiguous object(s)",	\
						#host_func_name);	\
				NWARN(DEFAULT_ERROR_STRING);			\
				return;					\
			}						\
			i++;						\
		}							\
		SETUP_BLOCKS_XY						\
		/* report for debugging */				\
		REPORT_VECTORIZATION1( host_func_name )


// kludgy hack to deal with complex...
// Maybe we need dt_mach_mindim and dt_type_mindim???

#define SET_SIMPLE_INC( incvar , dp )					\
									\
	incvar.x = dp->dt_type_inc[ dp->dt_mindim ];			\
	if( incvar.x == 0 ) incvar.x = dp->dt_type_inc[ dp->dt_mindim + 1 ];


#define SETUP_SIMPLE_INCS1						\
									\
	SET_SIMPLE_INC(inc1,oap->oa_dest)


#define SETUP_SIMPLE_INCS2						\
									\
	SETUP_SIMPLE_INCS1						\
	SET_SIMPLE_INC(inc2,oap->oa_dp[0])


#define SETUP_SIMPLE_INCS3						\
									\
	SETUP_SIMPLE_INCS2						\
	SET_SIMPLE_INC(inc3,oap->oa_dp[1])


#define SETUP_SIMPLE_INCS4						\
									\
	SETUP_SIMPLE_INCS3						\
	SET_SIMPLE_INC(inc4,oap->oa_dp[2])


#define SETUP_SIMPLE_INCS5						\
									\
	SETUP_SIMPLE_INCS4						\
	SET_SIMPLE_INC(inc5,oap->oa_dp[3])


#define CLEAR_CUDA_ERROR(name)	_CLEAR_CUDA_ERROR(name)

#define _CLEAR_CUDA_ERROR(name)					\
	e = cudaGetLastError();						\
	if( e != cudaSuccess ){						\
		describe_cuda_error(#name,e);			\
	}

#define CLEAR_CUDA_ERROR2(hfunc,gfunc)					\
	e = cudaGetLastError();						\
	if( e != cudaSuccess ){						\
		describe_cuda_error2(#hfunc,#gfunc,e);			\
	}

#define CHECK_CUDA_ERROR(hfunc,gfunc)					\
	e = cudaPeekAtLastError();					\
	if( e != cudaSuccess ){						\
		describe_cuda_error2(#hfunc,#gfunc,e);			\
	}

#endif /* _HOST_CALL_UTILS_H_ */

