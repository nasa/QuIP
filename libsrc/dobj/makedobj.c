#include "quip_config.h"

#include <stdio.h>

//#ifdef HAVE_OPENCL
//#include "my_ocl.h"		// why was this needed?  FOOBAR
//#endif /* HAVE_OPENCL */

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

#include "quip_prot.h"
#include "data_obj.h"
#include "debug.h"
//#include "img_file.h"


static int default_align=(-1);

void set_dp_alignment(int aval)
{
	if( aval == 1 ) default_align=(-1);
	else default_align=aval;
}

void *cpu_mem_alloc(QSP_ARG_DECL  Platform_Device *pdp, dimension_t size, int align )
{
	void *ptr;

	/* if getbuf is not aliased to malloc but instead uses getspace,
	 * then this is not correct!?  BUG
	 */

	// BUG?  Why not use posix_memalign here?
	// ans:  because we don't always compile to use
	// malloc as our memory allocator.  Given the capabilities
	// of new malloc_debug, however, it is not clear that
	// there is still a need to support the old getbuf interface...

	/* malloc always aligns to an 8 byte boundary, but for SSE we need 16 */
	if( align > 8 ){
		size += align - 1;	/* guarantee we will be able to align */
	}

	ptr=(unsigned char *)getbuf(size);
	if( ptr == (unsigned char *)NULL ){
		mem_err("no more RAM for data objects");
	}
	return ptr;
}

// allocate memory for a new object in ram

int cpu_obj_alloc(QSP_ARG_DECL  Data_Obj *dp, dimension_t size, int align )
{
	unsigned char *st;
	st = cpu_mem_alloc(QSP_ARG  NULL, size, align );

	/* remember the original address of the data for freeing! */
	SET_OBJ_UNALIGNED_PTR(dp,st);

	if( align > 0 ){
#ifdef QUIP_DEBUG
if( debug & debug_data ){
sprintf(ERROR_STRING,"get_data_space:  aligning area provided by getbuf (align = %d)",align);
advise(ERROR_STRING);
}
#endif
		st = (u_char *)((((u_long)st)+align-1) & ~(align-1));
	}

	SET_OBJ_DATA_PTR(dp,st);

	return 0;
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
 *
 * Perhaps we ought to use memalign (posix_memalign) here instead of getbuf?
 */


static int get_data_space(QSP_ARG_DECL  Data_Obj *dp,dimension_t size, int min_align)
{
	int align=0;

#ifdef QUIP_DEBUG
if( debug & debug_data ){
sprintf(ERROR_STRING,"get_data_space:  requesting %d (0x%x) bytes for object %s",
size,size,OBJ_NAME(dp));
advise(ERROR_STRING);
}
#endif

	if( default_align > 0 )
		align = default_align;
	if( min_align > 1 && min_align > align )
		align = min_align;

	assert( PF_OBJ_ALLOC_FN( OBJ_PLATFORM(dp) ) != NULL );

	return (* PF_OBJ_ALLOC_FN( OBJ_PLATFORM(dp) ) )( QSP_ARG  dp, size, align );

} /* end get_data_space() */

/* stuff shared with sub_obj initialization */

static Data_Obj *setup_dp_with_shape(QSP_ARG_DECL  Data_Obj *dp,Precision * prec_p,uint32_t type_flag)
{
	SET_OBJ_PREC_PTR(dp,prec_p);
	SET_OBJ_REFCOUNT(dp,0);
	SET_OBJ_DECLFILE(dp, savestr( CURRENT_FILENAME ) );
// current_input_stack(SINGLE_QSP_ARG)
	SET_OBJ_BIT0(dp,0);

	/* set_shape_flags is where mindim gets set */
	if( set_shape_flags( OBJ_SHAPE(dp), dp, type_flag) < 0 )
		return(NULL);

	/* check_contiguity used to be called from set_obj_flags,
	 * but Shape_Info structs don't have increments, so we
	 * can't do it in set_shape_flags()
	 */

	/* We don't want to call check_contiguity if the object
	 * has unknown shape...
	 */
	if( ! UNKNOWN_SHAPE( OBJ_SHAPE(dp) ) ){
		check_contiguity(dp);
	}

	return(dp);
}

Data_Obj *setup_dp(QSP_ARG_DECL  Data_Obj *dp,Precision * prec_p)
{
	return setup_dp_with_shape(QSP_ARG  dp,prec_p,AUTO_SHAPE);
}

// THIS NEEDS TO BE MOVED TO A CUDA LIBRARY!?
#ifdef HAVE_CUDA
#ifdef NOT_YET
static void make_device_alias( QSP_ARG_DECL  Data_Obj *dp, uint32_t type_flag )
{
	char name[LLEN];
	Data_Obj *new_dp;
	Data_Area *ap;
	cudaError_t e;
	int i;

	assert( OBJ_AREA(dp)->da_flags == DA_CUDA_HOST ){

	/* Find the pseudo-area for the device mapping */
	sprintf(name,"%s_mapped",OBJ_AREA(dp)->da_name);
	//ap = data_area_of(QSP_ARG  name);
	ap = get_data_area(QSP_ARG  name);
	if( ap == NULL ){
		WARN("Failed to find mapped data area");
		return;
	}

	/* BUG check name length to make sure no buffer overrun */
	sprintf(name,"dev_%s",OBJ_NAME(dp));

	new_dp = new_dobj(QSP_ARG  name);
	if( new_dp==NULL )
		NERROR1("make_device_alias:  error creating alias object");

	// Need to allocate dimensions and increments...
	SET_OBJ_SHAPE(new_dp, ALLOC_SHAPE );

	if( set_obj_dimensions(QSP_ARG  new_dp,OBJ_TYPE_DIMS(dp),OBJ_PREC_PTR(dp)) < 0 )
		NERROR1("make_device_alias:  error setting alias dimensions");
	parent_relationship(dp,new_dp);
	for(i=0;i<N_DIMENSIONS;i++){
		SET_OBJ_MACH_INC(new_dp,i,OBJ_MACH_INC(dp,i));
		SET_OBJ_TYPE_INC(new_dp,i,OBJ_TYPE_INC(dp,i));
	}
	new_dp = setup_dp_with_shape(QSP_ARG  new_dp,OBJ_PREC_PTR(dp),type_flag);
	if( new_dp==NULL )
		NERROR1("make_device_alias:  failure in setup_dp");

	SET_OBJ_AREA(new_dp, ap);

	/* Now the magic:  get the address on the device! */

	e = cudaHostGetDevicePointer( &OBJ_DATA_PTR(new_dp), OBJ_DATA_PTR(dp), 0 );
	if( e != cudaSuccess ){
		describe_cuda_driver_error2("make_device_alias",
					"cudaHostGetDevicePointer",e);
		/* BUG should clean up and destroy object here... */
	}
}
#else // ! NOT_YET
static void make_device_alias( QSP_ARG_DECL  Data_Obj *dp, uint32_t type_flag )
{
	ERROR1("make_device_alias:  not implemented, check makedobj.c!?");
}
#endif // ! NOT_YET
#endif /* HAVE_CUDA */

#ifdef PAD_MINDIM
/* fix_bitmap_increments
 */

static void fix_bitmap_increments(Data_Obj *dp)
{
	int i_dim;
	int minc,tinc,n,nw,n_bits;

	i_dim=OBJ_MINDIM(dp);

	n_bits = OBJ_TYPE_DIM(dp,i_dim);
	if( n_bits > 1 ) SET_OBJ_TYPE_INC(dp,i_dim,1);
	else             SET_OBJ_TYPE_INC(dp,i_dim,0);

	nw = (n_bits + BITS_PER_BITMAP_WORD - 1)/BITS_PER_BITMAP_WORD;
	if( nw > 1 ) SET_OBJ_MACH_INC(dp,i_dim,1);
	else         SET_OBJ_MACH_INC(dp,i_dim,0);

	minc = nw;		/* number of words at mindim */
	tinc = minc * BITS_PER_BITMAP_WORD;	/* total bits */
	for(i_dim=OBJ_MINDIM(dp)+1;i_dim<N_DIMENSIONS;i_dim++){
		if( (n=OBJ_TYPE_DIM(dp,i_dim)) == 1 ){
			SET_OBJ_TYPE_INC(dp,i_dim,0);
			SET_OBJ_MACH_INC(dp,i_dim,0);
		} else {
			SET_OBJ_TYPE_INC(dp,i_dim,tinc);
			SET_OBJ_MACH_INC(dp,i_dim,minc);
			tinc *= n;
			minc *= n;
		}
	}
	/* We used to clear the contig & evenly-spaced flags here,
	 * but that is wrong in the case that there's only one word...
	 */
	CLEAR_OBJ_FLAG_BITS(dp,DT_CHECKED);
	check_contiguity(dp);
}
#endif // PAD_MINDIM

/*
 * Initialize an existing header structure
 */

static Data_Obj *init_dp_with_shape(QSP_ARG_DECL  Data_Obj *dp,
			Dimension_Set *dsp,Precision * prec_p,uint32_t type_flag)
{
	if( dp == NULL )	/* name already used */
		return(dp);

	/* these four fields are initialized for sub_obj's in
	 * parent_relationship()
	 */
	SET_OBJ_AREA(dp, curr_ap);

	// BUG?  this code could be simplified if new_item called bzero on the struct...
	SET_OBJ_PARENT(dp,NULL);
	SET_OBJ_CHILDREN(dp,NULL);
	/* We make sure that these pointers are set so that
	 * we can know when not to free them... */
	SET_OBJ_DECLFILE(dp,NULL);
	SET_OBJ_DATA_PTR(dp,NULL);
	SET_OBJ_UNALIGNED_PTR(dp,NULL);

	SET_OBJ_EXTRA(dp,NULL);
	SET_OBJ_OFFSET(dp,0);

	// This must be done before touching the flags,
	// because the flags are really part of the shape struct...
	SET_OBJ_SHAPE(dp, ALLOC_SHAPE );

	SET_OBJ_FLAGS(dp,0);

	/* A common error is to specify a dimension with
	 * regard to a nonexistent object:  i.e. ncols(i1)
	 * where i1 has not been created.  The expression
	 * then returns 0 (or -1... this has changed over
	 * various revisions...), and setup_dp returns NULL
	 */

	if( set_obj_dimensions(QSP_ARG  dp,dsp,prec_p) < 0 ){
		WARN("init_dp_with_shape:  error setting dimensions");
		return(NULL);
		/* BUG might want to clean up */
	}

	make_contiguous(dp);

	if( setup_dp_with_shape(QSP_ARG  dp,prec_p,type_flag) == NULL ){
		/* set this flag so delvec doesn't free nonexistent mem */
		SET_OBJ_FLAG_BITS(dp,DT_NO_DATA);
		delvec(QSP_ARG  dp);
		return(NULL);
	}

#ifdef HAVE_ANY_GPU
	SET_BITMAP_OBJ_GPU_INFO_HOST_PTR(dp,NULL);
#endif // HAVE_ANY_GPU

	return(dp);
} // end init_dp_with_shape

Data_Obj *init_dp(QSP_ARG_DECL  Data_Obj *dp,Dimension_Set *dsp,Precision * prec_p)
{
	return init_dp_with_shape(QSP_ARG  dp,dsp,prec_p,AUTO_SHAPE);
}


/*
 *  Set up a new header, but don't allocate the data.
 *
 *  This routine creates a new header structure and initializes it.
 *
 *  Returns a pointer to the new header structure, or NULL if the name is
 *  already in use, or if the name contains illegal characters.
 */

static Data_Obj * _make_dp_with_shape(QSP_ARG_DECL  const char *name,
			Dimension_Set *dsp,Precision * prec_p, uint32_t type_flag)
{
	Data_Obj *dp;

	if( curr_ap == NULL ){
		if( ram_area_p == NULL ) dataobj_init(SINGLE_QSP_ARG);
		curr_ap = ram_area_p;
	}

	/* make sure that the new name contains only legal chars */
	if( !is_valid_dname(QSP_ARG  name) ){
		sprintf(ERROR_STRING,"invalid data object name \"%s\"",name);
		WARN(ERROR_STRING);
		return(NULL);
	}

	/* Check if we are using contexts...
	 */
	dp = new_dobj(QSP_ARG  name);

	if( dp == NULL ){
		dp = dobj_of(QSP_ARG  name);
		if( dp != NULL ){

	// BUG the declfile is ok for the expression
	// language, but for classic scripts, we get told
	// that the declarations are in macro Vector,
	// which is not very helpful...
	// We really need to save the whole decl_stack!

			sprintf(ERROR_STRING,
		"Object \"%s\" created w/ filestack:  %s",
				OBJ_NAME(dp),OBJ_DECLFILE(dp));
			advise(ERROR_STRING);
			sprintf(ERROR_STRING,
		"Ignoring redeclaration w/ filestack: %s",
				CURRENT_FILENAME );
				//current_input_stack(SINGLE_QSP_ARG)
			advise(ERROR_STRING);
		}
		return(NULL);
	}

	if( init_dp_with_shape(QSP_ARG  dp,dsp,prec_p,type_flag) == NULL ){
		delvec(QSP_ARG   dp );
		return(NULL);
	}

	return(dp);
} /* end _make_dp_with_shape */

Data_Obj * _make_dp(QSP_ARG_DECL  const char *name,Dimension_Set *dsp,Precision * prec_p)
{
	return _make_dp_with_shape(QSP_ARG  name,dsp,prec_p, AUTO_SHAPE);
}

/*
 * Create a new data object.  This routine calls _make_dp() to create a new
 * header structure and then allocates space for the data.
 * Returns a pointer to the newly created header.  Prints a warning and
 * returns NULL if the name is already in use, or if space cannot be allocated
 * in the data area,.
 */

Data_Obj *
make_dobj_with_shape(QSP_ARG_DECL  const char *name,
			Dimension_Set *dsp,Precision * prec_p, uint32_t type_flag)
{
	Data_Obj *dp;
	dimension_t size;

	dp = _make_dp_with_shape(QSP_ARG  name,dsp,prec_p,type_flag);
	if( dp == NULL ) return(dp);

	// area should be set here

	if( OBJ_N_TYPE_ELTS(dp) == 0 ){	/* maybe an unknown size obj */
		SET_OBJ_DATA_PTR(dp,NULL);
	} else {
		if( IS_BITMAP(dp) ){
			int n_bits, n_words;
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
			 *
			 * BUG now we have redone the GPU implementation, and this
			 * doesn't seem to be an important consideration any more -
			 * so let's go back and try it the old way, with all the bits
			 * contiguous...  We should try to do this with macros/functions
			 * so that it can be changed easily...
			 */

#ifdef PAD_MINDIM
			int i_dim;

			n_bits = OBJ_TYPE_DIM(dp,OBJ_MINDIM(dp));
			// n_words is generally the row length, but it doesn't have to be. TEST
			n_words = (n_bits+BITS_PER_BITMAP_WORD-1)/
						BITS_PER_BITMAP_WORD;
			SET_OBJ_MACH_DIM(dp,OBJ_MINDIM(dp),n_words);

			// We used to only call this when there were
			// bits left over, but we need it all the time!
			fix_bitmap_increments(dp);

			size = n_words;
			SET_OBJ_N_MACH_ELTS(dp,n_words);
			for(i_dim=OBJ_MINDIM(dp)+1;i_dim<N_DIMENSIONS;i_dim++){
				size *= OBJ_TYPE_DIM(dp,i_dim);
				SET_OBJ_N_MACH_ELTS(dp,
					OBJ_N_MACH_ELTS(dp) * OBJ_MACH_DIM(dp,i_dim) );
			}
#else // ! PAD_MINDIM
			n_bits = OBJ_N_TYPE_ELTS(dp);	// total bits
			n_words = (n_bits+BITS_PER_BITMAP_WORD-1)/
						BITS_PER_BITMAP_WORD;
			size = n_words;
			SET_OBJ_N_MACH_ELTS(dp,n_words);
			SET_OBJ_MACH_DIM(dp,0,1);
			SET_OBJ_MACH_DIM(dp,1,n_words);
			SET_OBJ_MACH_DIM(dp,2,1);
			SET_OBJ_MACH_DIM(dp,3,1);
			SET_OBJ_MACH_DIM(dp,4,1);
			/* What about the machine dimensions??? */
#endif // ! PAD_MINDIM

		} else {
			size = OBJ_N_MACH_ELTS(dp);
		}
		size *= ELEMENT_SIZE(dp);

#ifdef QUIP_DEBUG
if( debug & debug_data ){
sprintf(ERROR_STRING,"make_dobj %s, requesting %d data bytes",name,size);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		/* Now that we pass the element size as an alignment requirement,
		 * maybe the request should not be in terms of bytes??
		 */
		if( get_data_space(QSP_ARG  dp,size,ELEMENT_SIZE(dp) ) < 0 ){
			SET_OBJ_DATA_PTR(dp,NULL);
			SET_OBJ_UNALIGNED_PTR(dp,NULL);
			delvec(QSP_ARG  dp);
			return(NULL);
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
make_dobj(QSP_ARG_DECL  const char *name,Dimension_Set *dsp,Precision * prec_p)
{
	return make_dobj_with_shape(QSP_ARG  name,dsp,prec_p,AUTO_SHAPE);
}

/*
 * Set the objects dimensions from a user-supplied array
 * of dimensions.
 * This function assumes that the data is contiguous.
 * (why?  it doesn't set increments...)
 */

int set_obj_dimensions(QSP_ARG_DECL  Data_Obj *dp,Dimension_Set *dsp,Precision * prec_p)
{
//	int retval=0;

	if( set_shape_dimensions(QSP_ARG  OBJ_SHAPE(dp),dsp,prec_p) < 0 ){
//		sprintf(ERROR_STRING,
//			"set_obj_dimensions:  error setting shape dimensions for object %s",
//			OBJ_NAME(dp));
//		WARN(ERROR_STRING);
//		retval=(-1);
		return -1;
	}
//	return(retval);
	return 0;
}

/*
 * Set up a shape_info struct from a user-supplied array of dimensions.
 * BUG? maybe this function should completely initialize the shape struct?
 *
 * If they are ALL 0 , then it flags it as unknown and doesn't squawk.
 */

int set_shape_dimensions(QSP_ARG_DECL  Shape_Info *shpp,Dimension_Set *dsp,Precision * prec_p)
{
	int i;
	int retval=0;
	int nzero=0;

	/* all dimensions 0 is a special case of an unknown object... */
//fprintf(stderr,"set_shape_dimensions:  shpp = 0x%lx, dsp = 0x%lx, prec_p = 0x%lx\n",
//(long)shpp,(long)dsp,(long)prec_p);

	SET_SHP_PREC_PTR(shpp,prec_p);
//advise("prec ptr set");
	/* check that all dimensions have positive values */
	/* BUT if ALL the dimensions are 0, then this is an unknown obj... */
	for(i=0;i<N_DIMENSIONS;i++)
		if( DIMENSION(dsp,i) == 0 ){
			nzero++;
		}

//advise("zero dims counted");
	if( nzero==N_DIMENSIONS ){	/* all zero!? */
//advise("all dims zero!");
		SET_SHP_FLAG_BITS(shpp,DT_UNKNOWN_SHAPE);
		DIMSET_COPY(SHP_MACH_DIMS(shpp),dsp);
		DIMSET_COPY(SHP_TYPE_DIMS(shpp),dsp);
		SET_SHP_N_MACH_ELTS(shpp,0);
		SET_SHP_N_TYPE_ELTS(shpp,0);
		return 0;
	}

	if( COMPLEX_PRECISION(PREC_CODE(prec_p)) ){
		if( DIMENSION(dsp,0) != 1 ){
			sprintf(ERROR_STRING,
		"Sorry, multi-component (%d) not allowed for complex",
				DIMENSION(dsp,0));
			WARN(ERROR_STRING);
			return -1;
		}

		SET_DIMENSION(dsp,0,1);
		SET_SHP_N_MACH_ELTS(shpp,2);
	} else if( QUAT_PRECISION(PREC_CODE(prec_p)) ){
//advise("quaternion precision...");
		assert( DIMENSION(dsp,0) == 1 );
		SET_DIMENSION(dsp,0,1);
		SET_SHP_N_MACH_ELTS(shpp,4);
	} else {
		SET_SHP_N_MACH_ELTS(shpp,1);
	}
	SET_SHP_N_TYPE_ELTS(shpp,1);
//advise("n_elts set...");


	for(i=0;i<N_DIMENSIONS;i++){
//fprintf(stderr,"set_shape_dimensions:  setting dimension %d\n",i);
		if( DIMENSION(dsp,i) <= 0 ){
			sprintf(ERROR_STRING,
	"set_shape_dimensions:  Bad %s dimension (%d) specified",
				dimension_name[i],DIMENSION(dsp,i));
			WARN(ERROR_STRING);
			SET_DIMENSION(dsp,i,1);
			retval=(-1);
		}
		if( i == 0 ){
			/* BUG?  could we handle bitmaps here too? */
			if( NORMAL_PRECISION(PREC_CODE(prec_p)) ){
				SET_SHP_TYPE_DIM(shpp,i,DIMENSION(dsp,i));
				SET_SHP_MACH_DIM(shpp,i,DIMENSION(dsp,i));
				SET_SHP_N_MACH_ELTS(shpp, SHP_N_MACH_ELTS(shpp) * DIMENSION(dsp,i) );
				SET_SHP_N_TYPE_ELTS(shpp, SHP_N_TYPE_ELTS(shpp) * DIMENSION(dsp,i) );
			} else {
				/* complex or quaternion */
				SET_SHP_TYPE_DIM(shpp,i,1);
				SET_SHP_MACH_DIM(shpp,i,SHP_N_MACH_ELTS(shpp));
				SET_SHP_N_TYPE_ELTS(shpp,
					SHP_N_TYPE_ELTS(shpp) * DIMENSION(dsp,i) );
			}
		} else {
			SET_SHP_TYPE_DIM(shpp,i, DIMENSION(dsp,i) );
			SET_SHP_MACH_DIM(shpp,i,DIMENSION(dsp,i) );
			SET_SHP_N_MACH_ELTS(shpp, SHP_N_MACH_ELTS(shpp) * DIMENSION(dsp,i) );
			SET_SHP_N_TYPE_ELTS(shpp, SHP_N_TYPE_ELTS(shpp) * DIMENSION(dsp,i) );
		}
	}
//fprintf(stderr,"set_shape_dimensions:  returning %d\n",retval);
	return(retval);
}


/* Make a copy of the given object, but with n components */

Data_Obj *
comp_replicate(QSP_ARG_DECL  Data_Obj *dp,int n,int allocate_data)
{
	char str[256],*s;
	Data_Obj *dp2;
	Dimension_Set ds1, *dsp=(&ds1);

#ifdef HAVE_ANY_GPU
	push_data_area(OBJ_AREA(dp));
#endif /* HAVE_ANY_GPU */

	DIMSET_COPY(dsp , OBJ_TYPE_DIMS(dp) );

	SET_DIMENSION(dsp,0,n);

	/* BUG if the original image is subscripted, there will be
	 * illegal chars in the name...
	 */

	strcpy(str,OBJ_NAME(dp));
	/* make sure not an array name... */
	s=str;
	while( *s && *s != '[' && *s != '{' )
		s++;
	sprintf(s,".%d",n);

	if( allocate_data )
		dp2=make_dobj(QSP_ARG  str,dsp,OBJ_PREC_PTR(dp));
	else {
		/* We call this from xsupp when we want to point to an XImage */
		dp2 = _make_dp(QSP_ARG  str,dsp,OBJ_PREC_PTR(dp));
		SET_OBJ_FLAG_BITS(dp2,DT_NO_DATA);
	}
#ifdef HAVE_ANY_GPU
	pop_data_area();
#endif /* HAVE_ANY_GPU */

	return(dp2);
} /* end comp_replicate */

Shape_Info *alloc_shape(void)
{
	Shape_Info *shpp;
	shpp = (Shape_Info *)getbuf(sizeof(Shape_Info));
	// I'm not sure I see the point of allocating these
	// dynamically, because we always need them???
	SET_SHP_MACH_DIMS(shpp,((Dimension_Set *)getbuf(sizeof(Dimension_Set))));
	SET_SHP_TYPE_DIMS(shpp,((Dimension_Set *)getbuf(sizeof(Dimension_Set))));
	SET_SHP_MACH_INCS(shpp,((Increment_Set *)getbuf(sizeof(Increment_Set))));
	SET_SHP_TYPE_INCS(shpp,((Increment_Set *)getbuf(sizeof(Increment_Set))));
	return shpp;
}

void rls_shape( Shape_Info *shpp )
{
	givbuf( SHP_TYPE_DIMS(shpp) );
	givbuf( SHP_MACH_DIMS(shpp) );
	givbuf( SHP_TYPE_INCS(shpp) );
	givbuf( SHP_MACH_INCS(shpp) );
	givbuf(shpp);
}

void copy_shape(Shape_Info *dst_shpp, Shape_Info *src_shpp)
{
	SET_SHP_PREC_PTR(dst_shpp, SHP_PREC_PTR(src_shpp) );
	SET_SHP_MAXDIM(dst_shpp, SHP_MAXDIM(src_shpp) );
	SET_SHP_MINDIM(dst_shpp, SHP_MINDIM(src_shpp) );
	SET_SHP_RANGE_MAXDIM(dst_shpp, SHP_RANGE_MAXDIM(src_shpp) );
	SET_SHP_RANGE_MINDIM(dst_shpp, SHP_RANGE_MINDIM(src_shpp) );
	SET_SHP_FLAGS(dst_shpp, SHP_FLAGS(src_shpp) );
	/*SET_SHP_LAST_SUBI(dst_shpp, SHP_LAST_SUBI(src_shpp) ); */

	COPY_DIMS( (SHP_TYPE_DIMS(dst_shpp)) , (SHP_TYPE_DIMS(src_shpp)) );
	COPY_DIMS( (SHP_MACH_DIMS(dst_shpp)) , (SHP_MACH_DIMS(src_shpp)) );
	COPY_INCS( (SHP_TYPE_INCS(dst_shpp)) , (SHP_TYPE_INCS(src_shpp)) );
	COPY_INCS( (SHP_MACH_INCS(dst_shpp)) , (SHP_MACH_INCS(src_shpp)) );
}


