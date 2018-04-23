#include "quip_config.h"

#include <stdio.h>
#include <strings.h>	// bzero

#include "quip_prot.h"
#include "data_obj.h"
#include "dobj_private.h"
#include "platform.h"
#ifdef HAVE_OPENCL
#include "ocl_platform.h"
#endif // HAVE_OPENCL

typedef struct {
	const char *	eqd_name;
	Precision *	eqd_prec_p;
	Data_Obj *	eqd_parent;
	Data_Obj *	eqd_child;
	Dimension_Set *	eqd_type_dsp;
	Dimension_Set *	eqd_mach_dsp;
	int		eqd_n_per_parent;
	int		eqd_n_per_child;
	int		eqd_bytes_per_parent_elt;
	int		eqd_bytes_per_child_elt;
	dimension_t	eqd_total_child_bytes;
	dimension_t	eqd_total_parent_bytes;
	dimension_t	eqd_n_child_bytes;
	dimension_t	eqd_n_parent_bytes;
	dimension_t	eqd_parent_contig_bytes;
	dimension_t	eqd_prev_parent_mach_elts;
	Increment_Set *	eqd_child_type_incs;
} Equivalence_Data;

#define EQ_NAME(eqd_p)		(eqd_p)->eqd_name
#define EQ_PREC_PTR(eqd_p)	(eqd_p)->eqd_prec_p
#define EQ_PREC_CODE(eqd_p)	PREC_CODE(EQ_PREC_PTR(eqd_p))
#define EQ_PARENT(eqd_p)	(eqd_p)->eqd_parent
#define EQ_CHILD(eqd_p)		(eqd_p)->eqd_child
#define EQ_TYPE_DIMS(eqd_p)		(eqd_p)->eqd_type_dsp
#define EQ_MACH_DIMS(eqd_p)		(eqd_p)->eqd_mach_dsp
#define EQ_N_PER_PARENT(eqd_p)	(eqd_p)->eqd_n_per_parent
#define EQ_N_PER_CHILD(eqd_p)	(eqd_p)->eqd_n_per_child
#define EQ_BYTES_PER_PARENT_ELT(eqd_p)	(eqd_p)->eqd_bytes_per_parent_elt
#define EQ_BYTES_PER_CHILD_ELT(eqd_p)	(eqd_p)->eqd_bytes_per_child_elt
#define EQ_TOTAL_CHILD_BYTES(eqd_p)	(eqd_p)->eqd_total_child_bytes
#define EQ_TOTAL_PARENT_BYTES(eqd_p)	(eqd_p)->eqd_total_parent_bytes
#define EQ_CHILD_TYPE_INCS(eqd_p)	(eqd_p)->eqd_child_type_incs
#define EQ_PREV_PARENT_MACH_ELTS(eqd_p)	(eqd_p)->eqd_prev_parent_mach_elts
#define EQ_N_CHILD_BYTES(eqd_p)		(eqd_p)->eqd_n_child_bytes
#define EQ_N_PARENT_BYTES(eqd_p)	(eqd_p)->eqd_n_parent_bytes
#define EQ_PARENT_CONTIG_BYTES(eqd_p)	(eqd_p)->eqd_parent_contig_bytes

#define SET_EQ_N_PER_PARENT(eqd_p,v)	(eqd_p)->eqd_n_per_parent = v
#define SET_EQ_N_PER_CHILD(eqd_p,v)	(eqd_p)->eqd_n_per_child = v
#define SET_EQ_BYTES_PER_PARENT_ELT(eqd_p,v)	(eqd_p)->eqd_bytes_per_parent_elt = v
#define SET_EQ_BYTES_PER_CHILD_ELT(eqd_p,v)	(eqd_p)->eqd_bytes_per_child_elt = v

#define n_bytes_per_child_elt	EQ_BYTES_PER_CHILD_ELT(eqd_p)
#define n_bytes_per_parent_elt	EQ_BYTES_PER_PARENT_ELT(eqd_p)
#define n_per_parent		EQ_N_PER_PARENT(eqd_p)
#define n_per_child		EQ_N_PER_CHILD(eqd_p)
#define total_child_bytes	EQ_TOTAL_CHILD_BYTES(eqd_p)
#define total_parent_bytes	EQ_TOTAL_PARENT_BYTES(eqd_p)
#define n_child_bytes		EQ_N_CHILD_BYTES(eqd_p)
#define n_parent_bytes		EQ_N_PARENT_BYTES(eqd_p)
#define prev_parent_mach_elts	EQ_PREV_PARENT_MACH_ELTS(eqd_p)
#define parent_contig_bytes	EQ_PARENT_CONTIG_BYTES(eqd_p)

#define PARENT_INC(idx)		OBJ_MACH_INC(EQ_PARENT(eqd_p),idx)
#define PARENT_DIM(idx)		OBJ_MACH_DIM(EQ_PARENT(eqd_p),idx)
#define PARENT_MACH_SIZE	OBJ_PREC_MACH_SIZE(EQ_PARENT(eqd_p))
#define CHILD_ELT_SIZE		PREC_SIZE( EQ_PREC_PTR(eqd_p) )
#define PARENT_ELT_SIZE		PREC_SIZE(OBJ_PREC_PTR(EQ_PARENT(eqd_p)))

// Make sure that only one component is specified for certain
// pseudo-precisions

#define check_n_comps(eqd_p) _check_n_comps(QSP_ARG  eqd_p)

static int _check_n_comps(QSP_ARG_DECL  Equivalence_Data *eqd_p)
{
	DIMSET_COPY( EQ_MACH_DIMS(eqd_p), EQ_TYPE_DIMS(eqd_p));

	if(		BITMAP_PRECISION( EQ_PREC_CODE(eqd_p) ) ||
			COLOR_PRECISION( EQ_PREC_CODE(eqd_p)) ||
			COMPLEX_PRECISION( EQ_PREC_CODE(eqd_p)) ||
			QUAT_PRECISION(EQ_PREC_CODE(eqd_p)) ){

		if( DIMENSION(EQ_TYPE_DIMS(eqd_p),0) != 1 ){
			sprintf(ERROR_STRING,
		"Sorry, %s objects must have component dimension (%d) equal to 1",
				PREC_NAME(EQ_PREC_PTR(eqd_p)),
				DIMENSION(EQ_TYPE_DIMS(eqd_p),0));
			warn(ERROR_STRING);
			return -1;
		}
		SET_DIMENSION(EQ_MACH_DIMS(eqd_p),0,PREC_N_COMPS(EQ_PREC_PTR(eqd_p)));
	}
	return 0;
}

// check_parent_contig computes the size of the largest contiguous block of data
// in the parent, and makes sure that it is at least as large as a single child
// element.

#define check_parent_contig(eqd_p) _check_parent_contig(QSP_ARG  eqd_p )

static int _check_parent_contig(QSP_ARG_DECL  Equivalence_Data *eqd_p)
{
	incr_t n_contig;
	int parent_dim;

	/* Find the largest number of contiguous elements in the parent */
	n_contig=1;
	for(parent_dim=0;parent_dim<N_DIMENSIONS;parent_dim++){
		if( PARENT_INC(parent_dim) == n_contig )
			n_contig *= PARENT_DIM(parent_dim);
	}

	if( n_contig < EQ_N_PER_CHILD(eqd_p) ){
		sprintf(ERROR_STRING,
	"get_n_per_child:  parent object %s n_contig = %d < %d, can't cast to %s",
			OBJ_NAME(EQ_PARENT(eqd_p)),n_contig,EQ_N_PER_CHILD(eqd_p),PREC_NAME(EQ_PREC_PTR(eqd_p)));
		warn(ERROR_STRING);
		return -1;
	}

	parent_contig_bytes = n_contig * PARENT_MACH_SIZE;

	return 0;
}

// Figure out the relation between the element sizes

static void compare_element_sizes(Equivalence_Data *eqd_p)
{
	n_bytes_per_child_elt = CHILD_ELT_SIZE;
	n_bytes_per_parent_elt = PARENT_ELT_SIZE;

	/*
	 * Bitmaps are a special case... - ?
	 */

	// Set default values
	n_per_parent = 1;
	n_per_child = 1;

	if( n_bytes_per_child_elt > n_bytes_per_parent_elt ) {
		assert(n_bytes_per_child_elt % n_bytes_per_parent_elt == 0 );
		n_per_child = n_bytes_per_child_elt / n_bytes_per_parent_elt ;
	} else if( n_bytes_per_child_elt < n_bytes_per_parent_elt ) {
		assert(n_bytes_per_parent_elt % n_bytes_per_child_elt == 0 );
		n_per_parent = n_bytes_per_parent_elt / n_bytes_per_child_elt ;
	}
	// otherwise same size, default values of 1 are correct
}

#define check_eq_size_match(eqd_p) _check_eq_size_match(QSP_ARG  eqd_p )

static int _check_eq_size_match(QSP_ARG_DECL  Equivalence_Data *eqd_p)
{
	int child_dim;

	total_child_bytes = 1;
	assert( EQ_TYPE_DIMS(eqd_p) != NULL );
	for(child_dim=0;child_dim<N_DIMENSIONS;child_dim++)
		total_child_bytes *= DIMENSION( EQ_TYPE_DIMS(eqd_p),child_dim);

	if( EQ_PREC_CODE(eqd_p) == PREC_BIT ){
		/* convert number of bits to number of words */
		total_child_bytes += BITS_PER_BITMAP_WORD - 1;
		total_child_bytes /= BITS_PER_BITMAP_WORD;
		total_child_bytes *= PREC_MACH_SIZE( EQ_PREC_PTR(eqd_p) );
	} else {
		total_child_bytes *= CHILD_ELT_SIZE;
	}


	total_parent_bytes = ELEMENT_SIZE( EQ_PARENT(eqd_p) ) *
				OBJ_N_MACH_ELTS( EQ_PARENT(eqd_p) );

	if( total_child_bytes != total_parent_bytes){
		sprintf(ERROR_STRING,
	"make_equivalence %s:  total requested size (%d bytes) does not match parent %s (%d bytes)",
			EQ_NAME(eqd_p),total_child_bytes,
			OBJ_NAME( EQ_PARENT(eqd_p) ),total_parent_bytes);
		warn(ERROR_STRING);
		return -1;
	}
	return 0;
}

#ifdef FOOBAR
static incr_t get_child_type_inc(Equivalence_Data *eqd_p)
{
	incr_t inc;

fprintf(stderr,"n_per_child = %d,  n_per_parent = %d\n",n_per_child,n_per_parent);
	if( n_per_parent > 1 ){
		// If we have multiple child elements per single
		// parent element, they have to be contiguous
		return 1;
	} else {
		/* The size of a child element is greater than or equal to
		 * the size of a parent element.
		 * Set child_type_inc to the smallest non-zero increment of the parent.
		 * (What is the sense of that?)
		 * The parent has to have contiguous elements to fill a child
		 * element, but could then skip...
		 */
		int i;

		inc=0;
		i=0;
		while( inc==0 && i < N_DIMENSIONS ){
			inc = PARENT_INC(i);
			i++;
		}
		if( inc == 0 ){
			if( total_child_bytes == total_parent_bytes &&
					total_child_bytes == CHILD_ELT_SIZE )
				inc=1;
		}

		assert( inc != 0 );

		/* Is this correct in all cases?  If multiple parent elements make up one child
		 * element, the contiguity check should have been performed above.  If they are the same
		 * size then we are ok.
		 */
	}
	return inc;
}
#endif // FOOBAR

/* Now we need to see if we can come up with
 * a set of increments for the child that will work.
 *
 * If the parent is contiguous, there is no problem,
 * we simply use the standard increments.
 * 
 * Otherwise, we have two kinds of problems:
 * If we are coalescing rows in a subimage, then we won't be able
 * to do this with a single increment and should fail.
 *
 * But if the parent image is an evenly spaced component image,
 * or a column, then we shouldn't have a problem...
 *
 * "n_per_child" is the number of elements from the parent object
 * that are combined to make up one of the child elements, if we
 * are casting up in size, one otherwise.
 *	ppPPppPPppPP
 *	ccccCCCCcccc	two parent elts for each child elt
 *
 * "n_per_parent" is the number of elements of the
 * child that are formed from a single element of the parent, if
 * we are casting down in size, or 1 otherwise.
 *	ppppPPPPpppp
 *	ccCCccCCccCC	two child elts for each parent elt
 *
 * We maintain a count of how many elements we have for each, and add
 * the next dimension to the one which is smaller.
 *
 * We have a special problem with bitmaps, because the machine unit
 * is a word, but we count the dimensions in bits.  We no longer pad
 * rows to be an integral number of words...
 *
 */

/* When we bring in more parent elements
 * to fill in a child dimension, we have to insure that
 * the spacing remains even.
 *
 * This test is like the contiguity test - but we only care
 * if we already were using more than 1 element from the parent...
 *
 * We only check for non-zero parent increments.
 * But we probably should remember the last non-zero increment?
 */

// BUG - this test checks for a non-contiguous parent jump, but doesn't make
// sure that it is a problem for the child???

// We call check_parent_spacing each time we set a new child increment - we need to
// insure that everything up to the increment is in good data...
//
// In order to use this increment, the parent has to be evenly-spaced over the length of
// the child dimension...

#define check_parent_spacing(eqd_p) _check_parent_spacing(QSP_ARG  eqd_p)

static int _check_parent_spacing(QSP_ARG_DECL  Equivalence_Data *eqd_p )
{
	dimension_t child_contig_size;
	int i_dim;

	// We begin by computing the largest block of contiguous space
	// for the parent and child, and checking for a match...

fprintf(stderr,"check_parent_spacing:  parent_contig_size = %d\n",parent_contig_bytes);

#define CHILD_INC(idx)	INCREMENT(EQ_CHILD_TYPE_INCS(eqd_p),idx)
#define CHILD_DIM(idx)	DIMENSION(EQ_TYPE_DIMS(eqd_p),idx)

	child_contig_size = CHILD_ELT_SIZE;
	if( CHILD_INC(0) == 1 ){
		child_contig_size *= CHILD_DIM(0);
		for(i_dim=1;i_dim<N_DIMENSIONS;i_dim++){
			if( CHILD_INC(i_dim-1)*CHILD_DIM(i_dim-1) == CHILD_INC(i_dim) ){
				child_contig_size *= CHILD_DIM(i_dim);
			} else {
				i_dim = N_DIMENSIONS;	// break out of loop
			}
		}
	}

	if( child_contig_size != parent_contig_bytes ){
		sprintf(ERROR_STRING,"check_parent_spacing:  contiguous block size mismatch!?");
		warn(ERROR_STRING);
		return -1;
	}

#ifdef FOOBAR
fprintf(stderr,"check_parent_spacing:  parent_dim = %d, child_dim = %d\n",parent_dim,child_dim);
fprintf(stderr,"check_parent_spacing:  prev_parent_mach_elts = %d\n",prev_parent_mach_elts);
fprintf(stderr,"check_parent_spacing:  parent_inc[%d] = %d, parent_inc[%d] = %d\n",parent_dim,
PARENT_INC(parent_dim),
parent_dim-1,
PARENT_INC(parent_dim-1)
);
	if(		prev_parent_mach_elts > 1
		&&	PARENT_INC(parent_dim) != 0
		&&	PARENT_INC(parent_dim-1) != 0
		&&	PARENT_INC(parent_dim) !=
				(incr_t)(PARENT_INC(parent_dim-1)
					* OBJ_MACH_DIM(EQ_PARENT(eqd_p),parent_dim-1)) ){
		sprintf(ERROR_STRING,
			"check_parent_spacing:  problem with unevenly spaced parent object %s",
			OBJ_NAME(EQ_PARENT(eqd_p)));
		warn(ERROR_STRING);
		sprintf(ERROR_STRING,"%s inc[%d] (%d) != inc[%d] (%d) * dim[%d] (%d)",
			OBJ_NAME(EQ_PARENT(eqd_p)),parent_dim,PARENT_INC(parent_dim),
			parent_dim-1,PARENT_INC(parent_dim-1),
			parent_dim-1,OBJ_MACH_DIM(EQ_PARENT(eqd_p),parent_dim-1));
		advise(ERROR_STRING);

		sprintf(ERROR_STRING,"prev_parent_mach_elts = %d, n_parent_bytes = %d, n_child_bytes = %d",
			prev_parent_mach_elts,n_parent_bytes,n_child_bytes);
		advise(ERROR_STRING);

		return -1;
	}
#endif // FOOBAR
	return 0;
}

#ifdef FOOBAR
			parent_dim++;
			prev_parent_mach_elts = total_parent_bytes;
			if( parent_dim < N_DIMENSIONS ){
				total_parent_bytes *= OBJ_MACH_DIM(EQ_PARENT(eqd_p),parent_dim);
			} else if( PREC_CODE(EQ_PREC_PTR(eqd_p)) == PREC_BIT ){
fprintf(stderr,"child_dim = %d\n",child_dim);
				total_child_bytes = (total_child_bytes + BITS_PER_BITMAP_WORD -1 ) / BITS_PER_BITMAP_WORD;
fprintf(stderr,"total_child_bytes = %d\n",total_child_bytes);
			} else {
				sprintf(ERROR_STRING,"check_eq_fit:  can't fit objects");
				warn(ERROR_STRING);
				sprintf(ERROR_STRING,"total_parent_bytes = %d   total_child_bytes = %d",
					total_parent_bytes,total_child_bytes);
				advise(ERROR_STRING);
				return -1;
			}
#endif // FOOBAR

#define ADVANCE_PARENT									\
	prev_parent_mach_elts *= OBJ_MACH_DIM(EQ_PARENT(eqd_p),parent_dim);		\
	n_parent_bytes *= OBJ_MACH_DIM(EQ_PARENT(eqd_p),parent_dim);			\
	parent_dim++;									\
	while( parent_dim < N_DIMENSIONS &&						\
			PARENT_INC(parent_dim) == 0 ){		\
		parent_dim++;								\
	}										\
	if( parent_dim < N_DIMENSIONS ){						\
		curr_parent_inc = PARENT_INC(parent_dim);		\
fprintf(stderr,"ADVANCE_PARENT updated curr_parent_inc to %d, parent_dim = %d\n",curr_parent_inc,parent_dim);\
	}

// ADVANCE_CHILD - increase child_dim until we get some more elements

	//total_child_bytes *= DIMENSION(EQ_TYPE_DIMS(eqd_p),0);
#define ADVANCE_CHILD								\
	while( child_dim < N_DIMENSIONS &&					\
			DIMENSION(EQ_TYPE_DIMS(eqd_p),child_dim) == 1 ){	\
		SET_INCREMENT(EQ_CHILD_TYPE_INCS(eqd_p),child_dim,0);		\
fprintf(stderr,"set increment %d to zero\n",child_dim);\
		child_dim++;							\
	}									\
	if( child_dim < N_DIMENSIONS ){						\
		if( n_child_bytes < parent_size ){				\
fprintf(stderr,"setting inc %d to 1 (curr_parent_inc = %d)\n",child_dim,curr_parent_inc);\
			SET_INCREMENT(EQ_CHILD_TYPE_INCS(eqd_p),child_dim,1);	\
		} else {							\
fprintf(stderr,"setting inc %d to %d (curr_parent_inc = %d)\n",child_dim,(curr_parent_inc*parent_size)/child_size,curr_parent_inc);\
			SET_INCREMENT(EQ_CHILD_TYPE_INCS(eqd_p),child_dim,(curr_parent_inc*parent_size)/child_size);	\
fprintf(stderr,"before scaling, n_child_bytes = %d, n_parent_bytes = %d\n",n_child_bytes,n_parent_bytes);\
		}								\
		n_child_bytes *= DIMENSION(EQ_MACH_DIMS(eqd_p),child_dim);	\
		child_dim++;							\
	}

/*
 * check_eq_fit - this routine decides whether this is going to work,
 * for a non-contiguous parent.  We have several cases to consider:
 * 1) subimages
 * 2) subsamples (like the real part of a complex image)
 * 3) combinations of 1 & 2
 *
 * Subimages:  Here is a case where the row increment is larger
 * than the row dimension:
 *
 *	p p p x x
 *	p p p x x
 *	p p p x x
 *
 * In this case, in order to be able to construct a child, each row
 * has to be a complete dimension, so that the child increment can
 * be adjusted.
 *
 * When we are done, we should have computed the child increment set.
 */


#define check_eq_fit(eqd_p) _check_eq_fit(QSP_ARG  eqd_p)

static int _check_eq_fit(QSP_ARG_DECL  Equivalence_Data *eqd_p)
{
	int parent_dim,child_dim;
	incr_t curr_parent_inc;
	int parent_size;
	int child_size;

	if( IS_CONTIGUOUS( EQ_PARENT(eqd_p) ) ){
		// use standard increments
		// call make_contiguous() AFTER we have created the object...
		// (requires mach_dims...)
		return 0;
	}

	//child_type_inc = get_child_type_inc(eqd_p);

	child_dim=0;	/* index of current child dimension/increment */

	// If the parent is a bitmap, this is NOT the number of machine elts!?
	child_size = CHILD_ELT_SIZE;
	parent_size = PARENT_MACH_SIZE;

	n_child_bytes = child_size;
	n_parent_bytes = parent_size;
	parent_dim=(-1);	/* index of current parent dimension/increment */
	do {
		parent_dim++;
		curr_parent_inc = PARENT_INC(parent_dim);
	} while( /* curr_parent_inc*parent_size < child_size */ curr_parent_inc <= 0 );
	prev_parent_mach_elts = 1;
	// curr_parent_inc holds the increment to get to the next parent elt.

	/* n_parent_bytes, n_child_bytes count the number of elements of the parent
	 * that have been used at the current level of the dimension counters.
	 */

	while( parent_dim < N_DIMENSIONS || child_dim < N_DIMENSIONS ){
fprintf(stderr,"top of loop:  parent_dim = %d, child_dim = %d, n_parent_bytes = %d, n_child_bytes = %d, curr_parent_inc = %d\n",
parent_dim,child_dim,n_parent_bytes,n_child_bytes,curr_parent_inc);
		if( n_parent_bytes == n_child_bytes ){
fprintf(stderr,"advancing both...\n");
			if( child_dim < N_DIMENSIONS ){
				ADVANCE_CHILD /* set increments, increase the child dimension */
			}
			if( parent_dim < N_DIMENSIONS ){
				ADVANCE_PARENT /* increase the parent dimension, read increment */
			}
		} else if( n_parent_bytes > n_child_bytes ){
fprintf(stderr,"advancing child...\n");
			ADVANCE_CHILD
		} else { /* n_parent_bytes < n_child_bytes */
fprintf(stderr,"advancing parent...\n");
			ADVANCE_PARENT
		}
	}
	// Now make sure the increments are good
	return check_parent_spacing(eqd_p);
}


/* We insist that the parent is evenly spaced...
 *
 * We need to figure out what the increment should be...
 * we need to keep track of which dimension of the parent we are in...
 *
 * One case:  cast a column to an image.
 *	in the child, the column increment is the rowinc of the parent
 *	the rowinc of the child is the column inc * the number of columns.
 *	Once we find the lowest increment, we are good to go.
 *
 * 2nd case:  cast an image to a row or column...
 *	the increment is the column increment of the parent.
 */

static void set_eqsp_incs(Equivalence_Data *eqd_p)
{
	incr_t	parent_mach_inc=0,
		parent_type_inc=0;
		/* autoinit to elim compiler warning */
	dimension_t pdim;
	int parent_dim;
	int child_dim;

	pdim=1;
	/* find the basic parent increment */
	for(parent_dim=0;parent_dim<N_DIMENSIONS;parent_dim++){
		//pdim *= OBJ_MACH_DIM(parent,parent_dim);	/* how many elements we are up to */
		pdim *= OBJ_TYPE_DIM(EQ_PARENT(eqd_p),parent_dim);	/* how many elements we are up to */
		if( pdim > 1 ){
			parent_mach_inc = PARENT_INC(parent_dim);
			parent_type_inc = OBJ_TYPE_INC(EQ_PARENT(eqd_p),parent_dim);
			parent_dim=N_DIMENSIONS;			/* break out of loop */
		}
	}

	SET_OBJ_MACH_INC(EQ_CHILD(eqd_p),0,parent_mach_inc);
	SET_OBJ_TYPE_INC(EQ_CHILD(eqd_p),0,parent_type_inc);
	for(child_dim=1;child_dim<N_DIMENSIONS;child_dim++){
		SET_OBJ_MACH_INC(EQ_CHILD(eqd_p),child_dim,OBJ_MACH_INC(EQ_CHILD(eqd_p),child_dim-1) * OBJ_MACH_DIM(EQ_CHILD(eqd_p),child_dim-1));
		SET_OBJ_TYPE_INC(EQ_CHILD(eqd_p),child_dim,OBJ_TYPE_INC(EQ_CHILD(eqd_p),child_dim-1) * OBJ_TYPE_DIM(EQ_CHILD(eqd_p),child_dim-1));
	}
}

static void update_type_increments(Equivalence_Data *eqd_p)
{
	int divisor;
	int multiplier;
	int child_dim;

	if( COMPLEX_PRECISION(PREC_CODE(EQ_PREC_PTR(eqd_p))) )
		divisor=2;
	else if( QUAT_PRECISION(PREC_CODE(EQ_PREC_PTR(eqd_p))) )
		divisor=4;
	else	divisor=1;

	if( BITMAP_PRECISION(PREC_CODE(EQ_PREC_PTR(eqd_p))) )
		multiplier = BITS_PER_BITMAP_WORD;
	else	multiplier = 1;

	for(child_dim=0;child_dim<N_DIMENSIONS;child_dim++){
		SET_OBJ_TYPE_INC(EQ_CHILD(eqd_p),child_dim,multiplier * OBJ_MACH_INC(EQ_CHILD(eqd_p),child_dim) / divisor);
	}
}


#define create_child_object(eqd_p) _create_child_object(QSP_ARG  eqd_p)

static int _create_child_object(QSP_ARG_DECL  Equivalence_Data *eqd_p)
{
	Data_Obj *newdp;
	const char *s;
	int child_dim;

	// Why not call make_dp???
	newdp=new_dobj( EQ_NAME(eqd_p) );

	if( newdp == NULL ){
		warn("couldn't create equivalence");
		return -1;
	}
	eqd_p->eqd_child = newdp;

	s = OBJ_NAME(newdp);	/* save, while we overwrite temporarily */
	//*newdp = *parent;	/* copy all of the fields... */
	OBJ_COPY_FROM(newdp, EQ_PARENT(eqd_p));	// newdp points to parent's shape...
	// after the copy, the shape pointer is the same as the parent's...
	DUP_OBJ_SHAPE(newdp,EQ_PARENT(eqd_p));

	SET_OBJ_NAME(newdp,s);
	SET_OBJ_OFFSET(newdp,0);

	if( set_obj_dimensions(newdp,EQ_TYPE_DIMS(eqd_p),EQ_PREC_PTR(eqd_p)) ){
		rls_shape( OBJ_SHAPE(newdp) );
		del_dobj(newdp);
		return -1;
	}

	if( IS_CONTIGUOUS(EQ_PARENT(eqd_p)) )
		make_contiguous(newdp);
	else if( IS_EVENLY_SPACED(EQ_PARENT(eqd_p)) ){
		set_eqsp_incs(eqd_p);
	} else {	/* Not evenly spaced or contiguous... */
		//
		for(child_dim=0;child_dim<N_DIMENSIONS;child_dim++){
			SET_OBJ_MACH_INC(newdp,child_dim,INCREMENT(EQ_CHILD_TYPE_INCS(eqd_p),child_dim));
		}
	}

	/* Now we have the machine increments.
	 * Copy them to the type increments, and adjust if necessary.
	 */
	update_type_increments(eqd_p);

	/* Where should we adjust the row increment when the parent is a bitmap? */


	/* this must be called before setup_dp,
	 * because flags are first copied from parent
	 */
	parent_relationship(EQ_PARENT(eqd_p),newdp);

	newdp=setup_dp(newdp,EQ_PREC_PTR(eqd_p));
	assert( newdp != NULL );


	/* If the parent is a bitmap, but the equivalence is not, then
	 * we need to clear the bitmap flag...
	 * We also need to adjust the row increment...
	 */

	if( IS_BITMAP(newdp) && PREC_CODE(EQ_PREC_PTR(eqd_p)) != PREC_BIT ){
		CLEAR_OBJ_FLAG_BITS(newdp,DT_BIT);
	}

	return 0;
}

/* make_equivalence
 *
 * Make an object of arbirary shape, which points to the data area
 * of an existing object.  It should not be necessary that the
 * parent object be contiguous as long as the dimensions
 * and increments of the new object can be set appropriately.
 *
 * A common use is to cast between a long integer to 4 bytes,
 * and vice-versa.  For example, we can copy byte images faster
 * if we do long word transfers.  (Note:  that particular
 * example has been implemented inside of vmov, so is no longer
 * very relevant.)  As long as the bytes in a row
 * are contiguous, and the number of bytes is a multiple of 4,
 * then it doesn't matter if the rows are contiguous.  Calculation
 * of the increments is tricky, however...
 *
 * Correct calculation of the increments has not yet been
 * implemented.  BUG
 *
 * It is not clear if the above comment is accurate and up-to-date;
 * We need to develope a comprehensive test suite!
 *
 * Here is an outline of the general strategy:  if parent and child
 * are different types, we compute n_per_child and n_per_parent,
 * which represent the number of elements of the other object which
 * make up one element of the given object.  For example, if we
 * are casting a byte image (the parent) to a long image (the child)
 * then n_per_parent is 1 (actually 1/4!?), and n_per_child is 4.
 * Then, starting at the bottom, we count elements up to the next
 * dimension boundary (parent or child).  When we reach a dimension
 * boundary of the child, we have to set the child increment.
 * If this boundary coincides with a dimension boundary of the parent,
 * then we can use the parent increment (which doesn't have to be
 * the same dimension).  If we reach a parent dimension boundary
 * in the middle of a child dimension, then the parent has to be
 * contiguous or there will be trouble.
 *
 * There is clearly a problem here with bitmaps...
 * And what about complex?
 * We are trying to fix this with typ instead of mach...
 *
 * Type/Mach dimensions and increments...
 *
 * For complex:  type dimension and inc are number of cpx numbers:
 *	r1 i1 r2 i2 r3 i3	type_dim = 3 mach_dim = 6
 *	R1 I1 R2 I2 R3 I3	type_inc = 1 mach_inc = 2 (col)
 *
 * But for bitmaps, type dimension boundaries may not align
 * with word boundaries; example uses 4 bit words:
 *
 *	bbbb bb		type_dim = 6  mach_dim = 2 ?
 *	bb bbbb		type_inc = 1 (col), 6 (row)  mach_inc = 1 (col), 2 (row)
 *
 * Because bitmaps can include don't-care bits at the end, the dimensions
 * may not be an exact match...
 */

Data_Obj *_make_equivalence( QSP_ARG_DECL  const char *name, Data_Obj *parent, Dimension_Set *dsp, Precision * prec_p )
{
	Dimension_Set ds1;
	Increment_Set is1;
	Equivalence_Data eqd1, *eqd_p=(&eqd1);

	bzero(eqd_p,sizeof(*eqd_p));

	// passed args
	eqd1.eqd_name = name;
	eqd1.eqd_parent = parent;
	eqd1.eqd_prec_p = prec_p;
	eqd1.eqd_type_dsp = dsp;

	// to be filled in
	eqd1.eqd_mach_dsp = (&ds1);
	eqd1.eqd_child_type_incs = (&is1);

	/* If we are casting to a larger machine type (e.g. byte to long)
	 * We have to have at least 4 bytes contiguous.
	 */

	/* Step 1
	 *
	 * Figure out how the elements match up.
	 */

	compare_element_sizes(eqd_p);	// sets n_per_child and n_per_parent

	// Make sure the parent can fit one child elt
	// BUG?  this check is only necessary when the parent is
	// not contiguous...
	if( check_parent_contig(eqd_p) < 0 ) return NULL;

	// Make sure we've only requested one component for bit, or
	// multi-component pseudo-precisions
	if( check_n_comps(eqd_p) < 0 )
		return NULL;

	/* make sure the total size matches */
	/* We need the machine dimensions of the new object */
	if( check_eq_size_match(eqd_p) < 0 )
		return NULL;

	if( check_eq_fit(eqd_p) < 0 )
		return NULL;

	if( create_child_object(eqd_p) < 0 )
		return(NULL);

	return EQ_CHILD(eqd_p);
} /* end make_equivalence */


