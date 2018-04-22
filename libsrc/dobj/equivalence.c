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
	Increment_Set *	eqd_child_mach_incs;
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
#define EQ_CHILD_MACH_INCS(eqd_p)	(eqd_p)->eqd_child_mach_incs

#define SET_EQ_N_PER_PARENT(eqd_p,v)	(eqd_p)->eqd_n_per_parent = v
#define SET_EQ_N_PER_CHILD(eqd_p,v)	(eqd_p)->eqd_n_per_child = v
#define SET_EQ_BYTES_PER_PARENT_ELT(eqd_p,v)	(eqd_p)->eqd_bytes_per_parent_elt = v
#define SET_EQ_BYTES_PER_CHILD_ELT(eqd_p,v)	(eqd_p)->eqd_bytes_per_child_elt = v

#define n_bytes_per_child_elt	EQ_BYTES_PER_CHILD_ELT(eqd_p)
#define n_bytes_per_parent_elt	EQ_BYTES_PER_PARENT_ELT(eqd_p)
#define n_per_parent	EQ_N_PER_PARENT(eqd_p)
#define n_per_child	EQ_N_PER_CHILD(eqd_p)
#define total_child_bytes	EQ_TOTAL_CHILD_BYTES(eqd_p)
#define total_parent_bytes	EQ_TOTAL_PARENT_BYTES(eqd_p)

#define get_machine_dimensions(eqd_p) _get_machine_dimensions(QSP_ARG  eqd_p)

static void _get_machine_dimensions(QSP_ARG_DECL  Equivalence_Data *eqd_p)
{
	DIMSET_COPY( EQ_MACH_DIMS(eqd_p), EQ_TYPE_DIMS(eqd_p));

	if( BITMAP_PRECISION( EQ_PREC_CODE(eqd_p) ) ){
		if( DIMENSION(EQ_TYPE_DIMS(eqd_p),0) != 1 )
			error1("get_machine_dimensions:  Sorry, don't handle multi-component bitmaps");
		SET_DIMENSION(EQ_MACH_DIMS(eqd_p),0,1);

		// round number of columns up
		// BUG for a bitmap, we should represent it as a row vector?
		// But that might not be OK of we have a rare legal
		// non-contiguous bitmap...
		SET_DIMENSION(EQ_MACH_DIMS(eqd_p),1,N_BITMAP_WORDS(DIMENSION(EQ_TYPE_DIMS(eqd_p),1)));
	} else if( COMPLEX_PRECISION( EQ_PREC_CODE(eqd_p)) ){
		// complex can't have a component dimension
		if( DIMENSION(EQ_TYPE_DIMS(eqd_p),0) != 1 ){
			sprintf(ERROR_STRING,
		"Sorry, complex images must have component dimension (%d) equal to 1",
				DIMENSION(EQ_TYPE_DIMS(eqd_p),0));
			error1(ERROR_STRING);
		}
		SET_DIMENSION(EQ_MACH_DIMS(eqd_p),0,2);
	} else if( QUAT_PRECISION(EQ_PREC_CODE(eqd_p)) ){
		if( DIMENSION(EQ_TYPE_DIMS(eqd_p),0) != 1 )
			error1("Sorry, complex quaternion images must have component dimension equal to 1");
		SET_DIMENSION(EQ_MACH_DIMS(eqd_p),0,4);
	}
}

#define get_n_per_child(eqd_p) _get_n_per_child(QSP_ARG  eqd_p )

static int _get_n_per_child(QSP_ARG_DECL  Equivalence_Data *eqd_p)
{
	incr_t n_contig;
	int parent_dim;

	SET_EQ_N_PER_CHILD(eqd_p, n_bytes_per_child_elt / n_bytes_per_parent_elt);
	/* new size is larger, first increment must be 1 */

	/* Find the largest number of contiguous elements in the parent */
	n_contig=1;
	for(parent_dim=0;parent_dim<N_DIMENSIONS;parent_dim++)
		if( OBJ_MACH_INC(EQ_PARENT(eqd_p),parent_dim) == n_contig )
			n_contig *= OBJ_MACH_DIM(EQ_PARENT(eqd_p),parent_dim);
	/* n_contig is the number of contiguous machine elements in the parent... */

	if( n_contig < EQ_N_PER_CHILD(eqd_p) ){
		sprintf(ERROR_STRING,
	"get_n_per_child:  parent object %s n_contig = %d < %d, can't cast to %s",
			OBJ_NAME(EQ_PARENT(eqd_p)),n_contig,EQ_N_PER_CHILD(eqd_p),PREC_NAME(EQ_PREC_PTR(eqd_p)));
		warn(ERROR_STRING);
		return -1;
	}
	return 0;
}

// Figure out the relation between the element sizes

#define compare_element_sizes(eqd_p) _compare_element_sizes(QSP_ARG  eqd_p)

static int _compare_element_sizes(QSP_ARG_DECL  Equivalence_Data *eqd_p)
{
	n_bytes_per_child_elt = PREC_SIZE(EQ_PREC_PTR(eqd_p));
	n_bytes_per_parent_elt = PREC_SIZE(OBJ_PREC_PTR(EQ_PARENT(eqd_p)));

	/* Now we know how many bits in each basic element.
	 * Figure out how many elements of one makes up an element of the other.
	 * The results end up in n_per_parent and n_per_child.
	 *
	 * Bitmaps are a special case...
	 */
	SET_EQ_N_PER_PARENT(eqd_p,1);
	SET_EQ_N_PER_CHILD(eqd_p,1);

	/* Case 1:  child element size is greater than parent element size - casting up */
	if( n_bytes_per_child_elt > n_bytes_per_parent_elt ) {
		assert(n_bytes_per_child_elt % n_bytes_per_parent_elt == 0 );
		if( get_n_per_child(eqd_p) < 0 )
			return -1;
	} else if( n_bytes_per_child_elt < n_bytes_per_parent_elt ) {
		/* Case 2:  child element size is less than parent element size - casting down */
		//n_per_parent = OBJ_PREC_MACH_SIZE( parent ) / PREC_SIZE( prec & MACH_PREC_MASK ) ;
		assert(n_bytes_per_parent_elt % n_bytes_per_child_elt == 0 );
		n_per_parent = n_bytes_per_parent_elt / n_bytes_per_child_elt ;
	}
	return 0;
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
		total_child_bytes *= PREC_SIZE( EQ_PREC_PTR(eqd_p) );
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

static incr_t get_child_mach_inc(Equivalence_Data *eqd_p)
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
		 * Set child_mach_inc to the smallest non-zero increment of the parent.
		 * (What is the sense of that?)
		 * The parent has to have contiguous elements to fill a child
		 * element, but could then skip...
		 */
		int i;

		inc=0;
		i=0;
		while( inc==0 && i < N_DIMENSIONS ){
			inc = OBJ_MACH_INC(EQ_PARENT(eqd_p),i);
			i++;
		}
		if( inc == 0 ){
			if( total_child_bytes == total_parent_bytes &&
				total_child_bytes == PREC_SIZE(EQ_PREC_PTR(eqd_p)) )
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


/* Now we need to see if we can come up with
 * a new set of increments that works.
 * 
 * We have two kinds of problems:
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
 * is a word, but we count the dimensions in bits.  Remember we pad
 * rows to be an integral number of words...
 *
 * 
 */

#define check_eq_fit(eqd_p) _check_eq_fit(QSP_ARG  eqd_p)

static int _check_eq_fit(QSP_ARG_DECL  Equivalence_Data *eqd_p)
{
	int parent_dim,child_dim;
	incr_t child_mach_inc;
	dimension_t	prev_parent_mach_elts;

	child_mach_inc = get_child_mach_inc(eqd_p);

	child_dim=0;	/* index of current child dimension/increment */
	parent_dim=0;	/* index of current parent dimension/increment */

	// Set the total number of bytes equal to the bits
	// in the first element.

	// If the parent is a bitmap, this is NOT the number of machine elts!?
	total_child_bytes = PREC_MACH_SIZE( EQ_PREC_PTR(eqd_p) );
	total_child_bytes *= DIMENSION(EQ_MACH_DIMS(eqd_p),0);

	total_parent_bytes = OBJ_PREC_MACH_SIZE( EQ_PARENT(eqd_p) );
	total_parent_bytes *= OBJ_MACH_DIM(EQ_PARENT(eqd_p),0);


	/* total_parent_bytes, total_child_bytes count the number of elements of the parent
	 * that have been used at the current level of the dimension counters.
	 *
	 * child_mach_inc is the current increment to get to the next child element...
	 * it is the parent increment unless we are casting to a smaller type.
	 *
	 * But what about bitmaps???  BUGGY!?
	 * the dsp arg refers to type dimension, not machine dimension...
	 */

	while( parent_dim < N_DIMENSIONS || child_dim < N_DIMENSIONS ){
//fprintf(stderr,"make_equivalence:  parent_dim = %d (%d), total_parent_bytes = %d, child_dim = %d (%d), total_child_bytes = %d\n",
//parent_dim,OBJ_MACH_DIM(parent,parent_dim),total_parent_bytes,child_dim,DIMENSION(new_dsp,child_dim),total_child_bytes);
		if( total_parent_bytes == total_child_bytes ){
			/* increase the child dimension */
			SET_INCREMENT(EQ_CHILD_MACH_INCS(eqd_p),child_dim,child_mach_inc);
			child_dim++;
			if( child_dim < N_DIMENSIONS )
				total_child_bytes *= DIMENSION(EQ_MACH_DIMS(eqd_p),child_dim);
			/* increase the parent dimension */
			parent_dim++;
			if( parent_dim < N_DIMENSIONS ){
				total_parent_bytes *= OBJ_MACH_DIM(EQ_PARENT(eqd_p),parent_dim);

				/* BUG - need to make sure that the math comes out even, no fractions or truncations */
				if( OBJ_MACH_INC(EQ_PARENT(eqd_p),parent_dim) != 0 ){
					child_mach_inc = n_per_parent * OBJ_MACH_INC(EQ_PARENT(eqd_p),parent_dim) / n_per_child;
				}
			}
		} else if( total_parent_bytes > total_child_bytes ){
			/* Increment the child dimension only.
			 */
			if( child_dim >= N_DIMENSIONS ){
				warn("make_equivalence:  giving up!?");
				return -1;
			}
			/* increase the child dimension */
			SET_INCREMENT(EQ_CHILD_MACH_INCS(eqd_p),child_dim,child_mach_inc);
			child_dim++;
			if( child_dim < N_DIMENSIONS )
				total_child_bytes *= DIMENSION(EQ_MACH_DIMS(eqd_p),child_dim);
			/* Increasing the increment assumes that the spacing
			 * within the parent is even at this larger size.
			 * This is guaranteed it the new child size is LTE
			 * the parent size.  Otherwise, we will have to increase
			 * the parent dim next time round, and that is when
			 * we will check it.
			 */
			/* assumes child is evenly-spaced between the new dimension and the old */ 
			if( INCREMENT(EQ_CHILD_MACH_INCS(eqd_p),child_dim-1) > 0 )
				child_mach_inc = DIMENSION(EQ_MACH_DIMS(eqd_p),child_dim-1) * INCREMENT(EQ_CHILD_MACH_INCS(eqd_p),child_dim-1);
		} else { /* total_parent_bytes < total_child_bytes */
			/* Increment the parent dimension WITHOUT incrementing the child.
			 *
			 * The parent MUST be even-spaced across the dimensions
			 */
			/* increase the parent dimension */
			parent_dim++;
			prev_parent_mach_elts = total_parent_bytes;
			if( parent_dim < N_DIMENSIONS ){
				total_parent_bytes *= OBJ_MACH_DIM(EQ_PARENT(eqd_p),parent_dim);
			} else if( PREC_CODE(EQ_PREC_PTR(eqd_p)) == PREC_BIT ){
fprintf(stderr,"child_dim = %d\n",child_dim);
				total_child_bytes = (total_child_bytes + BITS_PER_BITMAP_WORD -1 ) / BITS_PER_BITMAP_WORD;
fprintf(stderr,"total_child_bytes = %d\n",total_child_bytes);
			} else {
				sprintf(ERROR_STRING,"make_equivalence:  can't fit objects");
				warn(ERROR_STRING);
				sprintf(ERROR_STRING,"total_parent_bytes = %d   total_child_bytes = %d",
					total_parent_bytes,total_child_bytes);
				advise(ERROR_STRING);
				return -1;
			}
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

			if( prev_parent_mach_elts > 1 && OBJ_MACH_INC(EQ_PARENT(eqd_p),parent_dim) != 0 &&
							OBJ_MACH_INC(EQ_PARENT(eqd_p),parent_dim-1) != 0 &&
					OBJ_MACH_INC(EQ_PARENT(eqd_p),parent_dim) !=
					(incr_t)(OBJ_MACH_INC(EQ_PARENT(eqd_p),parent_dim-1)
					* OBJ_MACH_DIM(EQ_PARENT(eqd_p),parent_dim-1)) ){
				sprintf(ERROR_STRING,
						"make_equivalence:  problem with unevenly spaced parent object %s",OBJ_NAME(EQ_PARENT(eqd_p)));
				warn(ERROR_STRING);
				sprintf(ERROR_STRING,"%s inc[%d] (%d) != inc[%d] (%d) * dim[%d] (%d)",
					OBJ_NAME(EQ_PARENT(eqd_p)),parent_dim,OBJ_MACH_INC(EQ_PARENT(eqd_p),parent_dim),
					parent_dim-1,OBJ_MACH_INC(EQ_PARENT(eqd_p),parent_dim-1),
					parent_dim-1,OBJ_MACH_DIM(EQ_PARENT(eqd_p),parent_dim-1));
				advise(ERROR_STRING);

				sprintf(ERROR_STRING,"prev_parent_mach_elts = %d, total_parent_bytes = %d, total_child_bytes = %d",
					prev_parent_mach_elts,total_parent_bytes,total_child_bytes);
				advise(ERROR_STRING);

				return -1;
			}

		}
	}
	return 0;
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
			parent_mach_inc = OBJ_MACH_INC(EQ_PARENT(eqd_p),parent_dim);
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
			SET_OBJ_MACH_INC(newdp,child_dim,INCREMENT(EQ_CHILD_MACH_INCS(eqd_p),child_dim));
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
	eqd1.eqd_name = name;
	eqd1.eqd_parent = parent;
	eqd1.eqd_prec_p = prec_p;
	eqd1.eqd_type_dsp = dsp;
	eqd1.eqd_mach_dsp = (&ds1);
	eqd1.eqd_child_mach_incs = (&is1);

	/* If we are casting to a larger machine type (e.g. byte to long)
	 * We have to have at least 4 bytes contiguous.
	 */

	/* Step 1
	 *
	 * Figure out how the elements match up.
	 */

	if( compare_element_sizes(eqd_p) < 0 )
		return NULL;

	/* first make sure the total size matches */

	/* We need the machine dimensions of the new object */
	// They should only differ for complex, quaternion, color and bit?
	get_machine_dimensions(eqd_p);

	if( check_eq_size_match(eqd_p) < 0 )
		return NULL;

	if( check_eq_fit(eqd_p) < 0 )
		return NULL;

	if( create_child_object(eqd_p) < 0 )
		return(NULL);

	return EQ_CHILD(eqd_p);
} /* end make_equivalence */


