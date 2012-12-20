#include "quip_config.h"

char VersionId_dataf_makedobj[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_CUDA
#include "cuda_supp.h"
#endif /* HAVE_CUDA */

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>	/* strcpy() */
#endif

/* a surprising amt of work is required to get a prototype
 * for memalign() !?
 */

#ifdef HAVE_STDLIB_H
#include <stdlib.h>		/* memalign() */
#endif

#include "data_obj.h"
#include "items.h"
#include "debug.h"
#include "img_file.h"
#include "getbuf.h"
#include "savestr.h"
#include "query.h"		/* current_input_stack() */

static int default_align=(-1);

static int get_data_space(QSP_ARG_DECL  Data_Obj *, dimension_t, int min_align);

void set_dp_alignment(int aval)
{
	if( aval == 1 ) default_align=(-1);
	else default_align=aval;
}

/* Allocate the memory for a new object
 *
 * getbuf (and malloc) generally return aligned blocks...
 * In the case of getbuf, it is because each block returned
 * to the user is preceded by a word that tells the size
 * of the following block (used to free).
 *
 * Data areas that use getspace aren't guaranteed to be aligned in any way.
 * The original scheme is kind of inefficient, we round up the requested size
 * to the smallest size that will guarantee an aligned block of the requested
 * size.
 *
 * But this means we almost always waste an extra word.
 */


static int get_data_space(QSP_ARG_DECL  Data_Obj *dp,dimension_t size, int min_align)
{
	Data_Area *ap;
	int align=0;
#ifdef HAVE_CUDA
	cudaError_t e;
#endif /* HAVE_CUDA */


#ifdef DEBUG
if( debug & debug_data ){
sprintf(error_string,"get_data_space:  requesting %d (0x%x) bytes for object %s",
size,size,dp->dt_name);
advise(error_string);
}
#endif
	ap = dp->dt_ap;

	if( default_align > 0 )
		align = default_align;
	if( min_align > 1 && min_align > align )
		align = min_align;

	if( ap->da_ma_p != NO_MEMORY_AREA ){
		long start=0;

#ifdef CAUTIOUS
		if( ap->da_freelist.fl_blockp == NO_FREEBLK )
			NERROR1("CAUTIOUS:  data area has a null freelist!?");
#endif /* CAUTIOUS */

		if( align > 0 ){
			size += align - 1;	/* guarantee we will be able to align */
		}

		start=getspace(&ap->da_freelist,size);
		if( start == -1 ) {
			sprintf(error_string,
		"Out of data memory in area %s, %d bytes requested",
				ap->da_name,size);
			WARN(error_string);
			return(-1);
		}
		if( align > 0 ){
			long aligned_start;
			long unused_before, unused_after;

			aligned_start = (start+align-1) & ~(align-1);
			unused_before = aligned_start - start;
			if( unused_before > 0 )
				givspace(&ap->da_freelist,unused_before,start);
			unused_after = (align-1)-unused_before;
			if( unused_after > 0 )
				givspace(&ap->da_freelist,unused_after,aligned_start+size-(align-1));
			start = aligned_start;
		}

		dp->dt_data = ((char *)ap->da_base) + start;
		ap->da_memfree -= size;
	} else {
		unsigned char *st;

		switch( dp->dt_ap->da_flags & DA_TYPE_MASK ){
			case DA_RAM:
				/* if getbuf is not aliased to malloc but instead uses getspace,
				 * then this is not correct!?  BUG
				 */

				/* malloc always aligns to an 8 byte boundary, but for SSE we need 16 */
				if( align > 8 ){
					size += align - 1;	/* guarantee we will be able to align */
				}
				st=(unsigned char *)getbuf(size);
				if( st == (unsigned char *)NULL ){
					mem_err("no more RAM for data objects");
					return(-1);
				}
				dp->dt_data = st;
				dp->dt_unaligned_data = dp->dt_data;

				if( align > 0 ){
#ifdef DEBUG
if( debug & debug_data ){
sprintf(error_string,"get_data_space:  aligning area provided by getbuf (align = %d)",align);
advise(error_string);
}
#endif
					/* remember the original address of the data for freeing! */
					dp->dt_data = (void *)((((u_long)dp->dt_data)+align-1) & ~(align-1));
				}
				break;

#ifdef HAVE_CUDA
			case DA_CUDA_GLOBAL:
				e = cudaMalloc( &dp->dt_data, size);
				if( e != cudaSuccess ){
					describe_cuda_error2("get_data_space","cudaMalloc",e);
					sprintf(ERROR_STRING,"Attempting to allocate %d bytes.",size);
					advise(ERROR_STRING);
					return(-1);
				}
				break;

			case DA_CUDA_HOST:
				/* This returns cudaErrorSetOnActiveProcess
				 * if called here... */
				/*
				e = cudaSetDeviceFlags( cudaDeviceMapHost );
				if( e != cudaSuccess ){
					describe_cuda_error2("get_data_space",
						"cudaSetDeviceFlags",e);
				}
				*/
				e = cudaHostAlloc( &dp->dt_data, size, 0 
					| cudaHostAllocPortable
					/* When we pass this flag, we get
					 * an invalid parameter error,
					 * but the documentation suggests
					 * it should work!?
					 * Answer:  set flag as above,
					 * but in device init...
					 */
					| cudaHostAllocMapped );
				if( e != cudaSuccess ){
					describe_cuda_error2("get_data_space","cudaHostAlloc",e);
					return(-1);
				}
				break;
#endif /* HAVE_CUDA */

			default:
				sprintf(error_string,"Oops, memory allocator not implemented for area %s",
					dp->dt_ap->da_name);
				WARN(error_string);
				break;
		}
	}

	return(0);
} /* end get_data_space() */

/* stuff shared with sub_obj initialization */

static Data_Obj *setup_dp_with_shape(QSP_ARG_DECL  Data_Obj *dp,prec_t prec,uint32_t type_flag)
{
	dp->dt_prec = prec;
	dp->dt_refcount = 0;
	dp->dt_declfile = savestr( current_input_stack(SINGLE_QSP_ARG) );
	dp->dt_bit0=0;

	/* set_shape_flags is where mindim gets set */
	if( set_shape_flags(&dp->dt_shape,dp,type_flag) < 0 )
		return(NO_OBJ);

	/* check_contiguity used to be called from set_obj_flags,
	 * but Shape_Info structs don't have increments, so we
	 * can't do it in set_shape_flags()
	 */
	check_contiguity(dp);

	return(dp);
}

Data_Obj *setup_dp(QSP_ARG_DECL  Data_Obj *dp,prec_t prec)
{
	return setup_dp_with_shape(QSP_ARG  dp,prec,AUTO_SHAPE);
}


#ifdef HAVE_CUDA
static void make_device_alias( QSP_ARG_DECL  Data_Obj *dp, uint32_t type_flag )
{
	char name[LLEN];
	Data_Obj *new_dp;
	Data_Area *ap;
	cudaError_t e;
	int i;

#ifdef CAUTIOUS
	if( dp->dt_ap->da_flags != DA_CUDA_HOST ){
		sprintf(error_string,
"CAUTIOUS:  make_device_alaias:  object %s is not host-mapped!?",dp->dt_name);
		NERROR1(error_string);
	}
#endif /* CAUTIOUS */

	/* Find the pseudo-area for the device mapping */
	sprintf(name,"%s_mapped",dp->dt_ap->da_name);
	ap = get_data_area(QSP_ARG  name);
	if( ap == NO_AREA ){
		WARN("Failed to make device alias");
		return;
	}

	/* BUG check name length to make sure no buffer overrun */
	sprintf(name,"dev_%s",dp->dt_name);
	new_dp = new_dobj(QSP_ARG  name);
	if( new_dp==NO_OBJ )
		NERROR1("make_device_alias:  error creating alias object");
	if( set_obj_dimensions(QSP_ARG  new_dp,&dp->dt_type_dimset,dp->dt_prec) < 0 )
		NERROR1("make_device_alias:  error setting alias dimensions");
	parent_relationship(dp,new_dp);
	for(i=0;i<N_DIMENSIONS;i++){
		new_dp->dt_mach_inc[i] = dp->dt_mach_inc[i];
		new_dp->dt_type_inc[i] = dp->dt_type_inc[i];
	}
	new_dp = setup_dp_with_shape(QSP_ARG  new_dp,dp->dt_prec,type_flag);
	if( new_dp==NO_OBJ )
		NERROR1("make_device_alias:  failure in setup_dp");

	new_dp->dt_ap = ap;

	/* Now the magic:  get the address on the device! */

	e = cudaHostGetDevicePointer( &new_dp->dt_data, dp->dt_data, 0 );
	if( e != cudaSuccess ){
		describe_cuda_error2("make_device_alias",
					"cudaHostGetDevicePointer",e);
		/* BUG should clean up and destroy object here... */
	}
}
#endif /* HAVE_CUDA */

/* fix_bitmap_increments
 */

static void fix_bitmap_increments(Data_Obj *dp)
{
	int i_dim;
	int minc,tinc,n,nw,n_bits;

	i_dim=dp->dt_mindim;

	n_bits = dp->dt_type_dim[i_dim];
	if( n_bits > 1 ) dp->dt_type_inc[i_dim] = 1; 
	else             dp->dt_type_inc[i_dim] = 0; 

	nw = (n_bits + BITS_PER_BITMAP_WORD - 1)/BITS_PER_BITMAP_WORD;
	if( nw > 1 ) dp->dt_mach_inc[i_dim] = 1; 
	else         dp->dt_mach_inc[i_dim] = 0; 

	minc = nw;		/* number of words at mindim */
	tinc = minc * BITS_PER_BITMAP_WORD;	/* total bits */
	for(i_dim=dp->dt_mindim+1;i_dim<N_DIMENSIONS;i_dim++){
		if( (n=dp->dt_type_dim[i_dim]) == 1 ){
			dp->dt_type_inc[i_dim]=0;
			dp->dt_mach_inc[i_dim]=0;
		} else {
			dp->dt_type_inc[i_dim]=tinc;
			dp->dt_mach_inc[i_dim]=minc;
			tinc *= n;
			minc *= n;
		}
	}
	/* We used to clear the contig & evenly-spaced flags here,
	 * but that is wrong in the case that there's only one word...
	 */
	dp->dt_flags &= ~DT_CHECKED;
	check_contiguity(dp);

	/*
	dp->dt_flags &= ~DT_CONTIG;
	dp->dt_flags &= ~DT_EVENLY;
	*/
}

/*
 * Initialize an existing header structure
 */

Data_Obj *init_dp_with_shape(QSP_ARG_DECL  Data_Obj *dp,
			Dimension_Set *dsp,prec_t prec,uint32_t type_flag)
{
	if( dp == NO_OBJ )	/* name already used */
		return(dp);

	/* these four fields are initialized for sub_obj's in
	 * parent_relationship()
	 */
	dp->dt_ap = curr_ap;
	dp->dt_parent = NO_OBJ;
	dp->dt_children = NO_LIST;
	dp->dt_flags=0;
	/* We make sure that these pointers are set so that we can know when not to free them... */
	dp->dt_declfile = NULL;
	dp->dt_data = NULL;
	dp->dt_unaligned_data = NULL;

	dp->dt_extra = NULL;
	dp->dt_offset = 0;

	/* A common error is to specify a dimension with
	 * regard to a nonexistent object:  i.e. ncols(i1)
	 * where i1 has not been created.  The expression
	 * then returns 0 (or -1... this has changed over
	 * various revisions...), and setup_dp returns NO_OBJ
	 */

	if( set_obj_dimensions(QSP_ARG  dp,dsp,prec) < 0 ){
		WARN("init_dp_with_shape:  error setting dimensions");
		return(NO_OBJ);
		/* BUG might want to clean up */
	}

	make_contiguous(dp);

	if( setup_dp_with_shape(QSP_ARG  dp,prec,type_flag) == NO_OBJ ){
		/* set this flag so delvec doesn't free nonexistent mem */
		dp->dt_flags |= DT_NO_DATA;
		delvec(QSP_ARG  dp);
		return(NO_OBJ);
	}

	return(dp);
}

Data_Obj *init_dp(QSP_ARG_DECL  Data_Obj *dp,Dimension_Set *dsp,prec_t prec)
{
	return init_dp_with_shape(QSP_ARG  dp,dsp,prec,AUTO_SHAPE);
}

/*
 *  Set up a new header, but don't allocate the data.
 *
 *  This routine creates a new header structure and initializes it.
 *
 *  Returns a pointer to the new header structure, or NO_OBJ if the name is
 *  already in use, or if the name contains illegal characters.
 */

static Data_Obj * _make_dp_with_shape(QSP_ARG_DECL  const char *name,Dimension_Set *dsp,prec_t prec, uint32_t type_flag)
{
	Data_Obj *dp;

	if( curr_ap == NO_AREA ){
		if( ram_area == NO_AREA ) dataobj_init(SINGLE_QSP_ARG);
		curr_ap = ram_area;
	}

	/* make sure that the new name contains only legal chars */
	if( !is_valid_dname(QSP_ARG  name) ){
		sprintf(error_string,"invalid data object name \"%s\"",name);
		WARN(error_string);
		return(NO_OBJ);
	}

	/* Check if we are using contexts...
	 */
	dp = new_dobj(QSP_ARG  name);

	if( dp == NO_OBJ ){
		dp = dobj_of(QSP_ARG  name);
		if( dp != NO_OBJ ){
			sprintf(error_string,
		"Object \"%s\" created w/ filestack:  %s",
				dp->dt_name,dp->dt_declfile);
			advise(error_string);
			sprintf(error_string,
		"Ignoring redeclaration w/ filestack: %s",
				current_input_stack(SINGLE_QSP_ARG));
			advise(error_string);
		}
		return(NO_OBJ);
	}

	if( init_dp_with_shape(QSP_ARG  dp,dsp,prec,type_flag) == NO_OBJ ){
		delvec(QSP_ARG   dp );
		return(NO_OBJ);
	}
		
	return(dp);
} /* end _make_dp_with_shape */

Data_Obj * _make_dp(QSP_ARG_DECL  const char *name,Dimension_Set *dsp,prec_t prec)
{
	return _make_dp_with_shape(QSP_ARG  name,dsp,prec, AUTO_SHAPE);
}

/*
 * Create a new data object.  This routine calls _make_dp() to create a new
 * header structure and then allocates space for the data.
 * Returns a pointer to the newly created header.  Prints a warning and
 * returns NO_OBJ if the name is already in use, or if space cannot be allocated
 * in the data area,.
 */

Data_Obj *
make_dobj_with_shape(QSP_ARG_DECL  const char *name,
			Dimension_Set *dsp,prec_t prec, uint32_t type_flag)
{
	Data_Obj *dp;
	dimension_t size;

	dp = _make_dp_with_shape(QSP_ARG  name,dsp,prec,type_flag);
	if( dp == NO_OBJ ) return(dp);

	if( dp->dt_n_type_elts == 0 ){	/* maybe an unknown size obj */
		dp->dt_data=NULL;
	} else {
		if( IS_BITMAP(dp) ){
			int n_bits, n_words, i_dim;

			/* size is in elements (bits), convert to # words */
			/* round n bits up to an even number of words */

			/* we used to just slosh all the bits together,
			 * but, for reasons relating to CUDA implementation,
			 * we now don't want to have words crossing
			 * dimension boundaries.  For example, if we
			 * have a one component image, then we want
			 * to pad each row so that an integral number
			 * of words is allocated for each row.
			 * In general, we have to have an integral
			 * number of words for mindim.
			 *
			 * In the case that there is padding, we have
			 * to fix the increments.
			 */
			n_bits = dp->dt_type_dim[dp->dt_mindim];
			// n_words is generally the row length, but it doesn't have to be. TEST
			n_words = (n_bits+BITS_PER_BITMAP_WORD-1)/
						BITS_PER_BITMAP_WORD;
			dp->dt_mach_dim[dp->dt_mindim] = n_words;

			if( n_words * BITS_PER_BITMAP_WORD != n_bits ){
				fix_bitmap_increments(dp);
			}

			size = n_words;
			dp->dt_n_mach_elts = n_words;
			for(i_dim=dp->dt_mindim+1;i_dim<N_DIMENSIONS;i_dim++){
				size *= dp->dt_type_dim[i_dim];
				dp->dt_n_mach_elts *= dp->dt_mach_dim[i_dim];
			}

			/* size is now in words */
		} else {
			size = dp->dt_n_mach_elts;
		}
		size *= ELEMENT_SIZE(dp);

#ifdef DEBUG
if( debug & debug_data ){
sprintf(error_string,"make_dobj %s, requesting %d data bytes",name,size);
advise(error_string);
}
#endif /* DEBUG */
		/* Now that we pass the element size as an alignment requirement,
		 * maybe the request should not be in terms of bytes??
		 */
		if( get_data_space(QSP_ARG  dp,size,ELEMENT_SIZE(dp) ) < 0 ){
			dp->dt_data = dp->dt_unaligned_data = NULL;
			delvec(QSP_ARG  dp);
			return(NO_OBJ);
		}
	}
#ifdef HAVE_CUDA
	/* If this object is using mapped host memory, create
	 * another "alias" object that is usable on the device
	 */
	if( dp->dt_ap->da_flags & DA_CUDA_HOST )
		make_device_alias(QSP_ARG  dp,type_flag);
#endif /* HAVE_CUDA */

	return(dp);
} /* end make_dobj_with_shape */

Data_Obj *
make_dobj(QSP_ARG_DECL  const char *name,Dimension_Set *dsp,prec_t prec)
{
	return make_dobj_with_shape(QSP_ARG  name,dsp,prec,AUTO_SHAPE);
}


/*
 * Set the objects dimensions from a user-supplied array
 * of dimensions.
 * This function assumes that the data is contiguous.
 * (why?  it doesn't set increments...)
 */

int set_obj_dimensions(QSP_ARG_DECL  Data_Obj *dp,Dimension_Set *dsp,prec_t prec)
{
	int retval=0;

	if( set_shape_dimensions(QSP_ARG  &dp->dt_shape,dsp,prec) < 0 ){
		sprintf(error_string,
			"set_obj_dimensions:  error setting shape dimensions for object %s",
			dp->dt_name);
		WARN(error_string);
		retval=(-1);
	}
	return(retval);
}

/*
 * Set up a shape_info struct from a user-supplied array of dimensions.
 * BUG? maybe this function should completely initialize the shape struct?
 *
 * If they are ALL 0 , then it flags it as unknown and doesn't squawk.
 */

int set_shape_dimensions(QSP_ARG_DECL  Shape_Info *shpp,Dimension_Set *dsp,prec_t prec)
{
	int i;
	int retval=0;
	int nzero=0;

	/* all dimensions 0 is a special case of an unknown object... */

	shpp->si_prec = prec;

	/* check that all dimensions have positive values */
	/* BUT if ALL the dimensions are 0, then this is an unknown obj... */
	for(i=0;i<N_DIMENSIONS;i++)
		if( dsp->ds_dimension[i] == 0 ){
			nzero++;
		}

	if( nzero==N_DIMENSIONS ){	/* all zero!? */
		shpp->si_flags |= DT_UNKNOWN_SHAPE;
		shpp->si_mach_dimset = *dsp;
		shpp->si_type_dimset = *dsp;
		shpp->si_n_mach_elts = 0;
		shpp->si_n_type_elts = 0;
		return(0);
	}

	if( COMPLEX_PRECISION(prec) ){
#ifdef CAUTIOUS
		if( dsp->ds_dimension[0] != 1 ){
			sprintf(error_string,
"CAUTIOUS:  set_shape_dimensions:  Sorry, multi-component (%d) not allowed for complex",
				dsp->ds_dimension[0]);
			WARN(error_string);
		}
#endif /* CAUTIOUS */
		dsp->ds_dimension[0]=1;
		shpp->si_n_mach_elts = 2;
	} else if( QUAT_PRECISION(prec) ){
#ifdef CAUTIOUS
		if( dsp->ds_dimension[0] != 1 ){
			sprintf(error_string,
"CAUTIOUS:  set_shape_dimensions:  Sorry, multiple (%d) components not allowed for quaternion",
				dsp->ds_dimension[0]);
			WARN(error_string);
		}
#endif /* CAUTIOUS */
		dsp->ds_dimension[0]=1;
		shpp->si_n_mach_elts = 4;
	} else {
		shpp->si_n_mach_elts = 1;
	}
	shpp->si_n_type_elts = 1;


	for(i=0;i<N_DIMENSIONS;i++){
		if( dsp->ds_dimension[i] <= 0 ){
			sprintf(error_string,
	"set_shape_dimensions:  Bad %s dimension (%d) specified",
				dimension_name[i],dsp->ds_dimension[i]);
			WARN(error_string);
			dsp->ds_dimension[i]=1;
			retval=(-1);
		}
		if( i == 0 ){
			/* BUG?  could we handle bitmaps here too? */
			if( NORMAL_PRECISION(prec) ){
				shpp->si_type_dim[i] = dsp->ds_dimension[i];
				shpp->si_mach_dim[i] = dsp->ds_dimension[i];
				shpp->si_n_mach_elts *= dsp->ds_dimension[i];
				shpp->si_n_type_elts *= dsp->ds_dimension[i];
			} else {
				/* complex or quaternion */
				shpp->si_type_dim[i] = 1;
				shpp->si_mach_dim[i] = shpp->si_n_mach_elts;
				shpp->si_n_type_elts *= dsp->ds_dimension[i];
			}
		} else {
			shpp->si_type_dim[i] =
			shpp->si_mach_dim[i] = dsp->ds_dimension[i];
			shpp->si_n_mach_elts *= dsp->ds_dimension[i];
			shpp->si_n_type_elts *= dsp->ds_dimension[i];
		}
	}
	return(retval);
}


/* Make a copy of the given object, but with n components */

Data_Obj *
comp_replicate(QSP_ARG_DECL  Data_Obj *dp,int n,int allocate_data)
{
	char str[256],*s;
	Data_Obj *dp2;
	Dimension_Set dimset;

#ifdef HAVE_CUDA
	push_data_area(dp->dt_ap);
#endif /* HAVE_CUDA */
	dimset = dp->dt_type_dimset;
	dimset.ds_dimension[0] = n;

	/* BUG if the original image is subscripted, there will be
	 * illegal chars in the name...
	 */

	strcpy(str,dp->dt_name);
	/* make sure not an array name... */
	s=str;
	while( *s && *s != '[' && *s != '{' )
		s++;
	sprintf(s,".%d",n);

	if( allocate_data )
		dp2=make_dobj(QSP_ARG  str,&dimset,dp->dt_prec);
	else {
		/* We call this from xsupp when we want to point to an XImage */
		dp2 = _make_dp(QSP_ARG  str,&dimset,dp->dt_prec);
		dp2->dt_flags |= DT_NO_DATA;
	}
#ifdef HAVE_CUDA
	pop_data_area();
#endif /* HAVE_CUDA */

	return(dp2);
}

