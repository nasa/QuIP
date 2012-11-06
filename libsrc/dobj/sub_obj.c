#include "quip_config.h"

char VersionId_dataf_sub_obj[] = QUIP_VERSION_STRING;

#include <stdio.h>

#include "data_obj.h"
#include "items.h"
#include "debug.h"
#include "getbuf.h"
#include "savestr.h"

#ifdef HAVE_CUDA
#include "cuda_supp.h"
#endif

/* local prototypes */

static int check_posn(QSP_ARG_DECL  Data_Obj *parent,index_t *offsets,
			Dimension_Set *dsp,const char *name);
static int check_inset(QSP_ARG_DECL  Data_Obj *parent,index_t *offsets,Dimension_Set *dsp,
			incr_t *incrs,const char *name);
static void relocate_children(Data_Obj *);
static int is_inside(QSP_ARG_DECL  index_t index,int which_dim,const char *sub_name,Data_Obj *parent);
static void set_child_increments(Data_Obj *dp, incr_t *type_incrs, incr_t *mach_incrs);

/* This used to be declared withing check_posn(),
 * but old sun compiler says "no automatic aggregate initialization"
 */

static incr_t ones[N_DIMENSIONS]={1,1,1,1,1};	/* dummy increments */

static void set_child_increments(Data_Obj *dp, incr_t *type_incrs, incr_t *mach_incrs)
{
	int i;

	/* Here we copy the increments from the parent -
	 * But if a subscripted dimension is now 1,
	 * then the corresponding increment must
	 * be set to 0...
	 */

	for(i=0;i<N_DIMENSIONS;i++){
		if( dp->dt_mach_dim[i] == 1 ){
			dp->dt_mach_inc[i] = 0;
			dp->dt_type_inc[i] = 0;
		} else {
			dp->dt_type_inc[i] = type_incrs[i];
			dp->dt_mach_inc[i] = mach_incrs[i];
		}
	}
}

static index_t get_pix_base_offset(index_t *offsets, Data_Obj *parent)
{
	index_t pix_offset;
	int i;

	pix_offset=0L;
	for(i=0;i<N_DIMENSIONS;i++)
		pix_offset += offsets[i] * parent->dt_mach_inc[i];

	return(pix_offset);
}

static int check_posn( QSP_ARG_DECL  Data_Obj *parent, index_t *offsets, Dimension_Set *dsp, const char *name )
{
	return( check_inset(QSP_ARG  parent,offsets,dsp,ones,name) );
}

/* check one dimension */

static int is_inside( QSP_ARG_DECL  index_t index, int which_dim, const char *sub_name, Data_Obj *parent )
{
	dimension_t pd;		/* parent dim */

	/* if indices are unsigned, the negative test is meaningless... */
	pd = parent->dt_type_dim[which_dim];

	if( /* index < 0 || */ index >= pd){
		sprintf(error_string,
"%s offset %d for subobject \"%s\" falls outside of parent \"%s\"",
			dimension_name[which_dim],
			index,sub_name,parent->dt_name);
		WARN(error_string);
		sprintf(error_string,
			"dim index %d:  parent size = %u",
			which_dim, parent->dt_type_dim[which_dim]);
		advise(error_string);
		return(0);
	}
	return(1);
}

/* make sure that a requested subobject fits within the parent */

static int check_inset( QSP_ARG_DECL  Data_Obj *parent, index_t *offsets, Dimension_Set *dsp, incr_t *incrs, const char *name )
{
	int i;
	int retval=0;

#ifdef CAUTIOUS
	/* make sure that all the sizes are valid */

	for(i=0;i<N_DIMENSIONS;i++){
		if( dsp->ds_dimension[i] == 0 ){
			sprintf(error_string,
	"CAUTIOUS:  subobject %s dimension %d has value 0!??",
				name,i);
			WARN(error_string);
			retval=(-1);
		}
	}
	if( retval < 0 ) return(retval);
#endif /* CAUTIOUS */

	for(i=0;i<N_DIMENSIONS;i++){
		index_t extreme_index;

		extreme_index = offsets[i];
		if( ! is_inside(QSP_ARG  extreme_index,i,name,parent) )
			retval=(-1);

		/* This test is needed if indices are unsigned... */
		if( incrs[i] < 0 && (-incrs[i]*(dsp->ds_dimension[i]-1)) > offsets[i] )
			/* BUG print a warning here */
			retval=(-1);

		extreme_index = offsets[i]+incrs[i]*(dsp->ds_dimension[i]-1);
		if( ! is_inside(QSP_ARG  extreme_index,i,name,parent) )
			retval=(-1);

	}
	return(retval);
} /* end check_inset() */

/* Set family fields for both child and parent
 */

void parent_relationship( Data_Obj *parent, Data_Obj *child )
{
	child->dt_ap = parent->dt_ap;
	child->dt_parent = parent;

	/* the child is always a new object, so we're not
	 * losing anything here
	 */
	child->dt_children = NO_LIST;

	if( parent->dt_children == NO_LIST )
		parent->dt_children = new_list();

	addHead(parent->dt_children,mk_node(child));

	child->dt_flags = parent->dt_flags;
	child->dt_flags |= DT_NO_DATA;

	/* Clear the TEMP flag in case
	 * parent is a subscripted object
	 * (not in hash table).
	 *
	 * It is the responsibility of the caller to explicitly
	 * set this flag if needed (see array.c)
	 */

	child->dt_flags &= ~DT_TEMP;

#ifdef HAVE_CUDA
	if( IS_GL_BUFFER(parent) ){
		child->dt_gl_info_p = parent->dt_gl_info_p;
		xfer_cuda_flag(child,parent,DT_GL_BUF);
		xfer_cuda_flag(child,parent,DT_BUF_MAPPED);
	}
#endif

}

Data_Obj *
mk_subseq( QSP_ARG_DECL  const char *name, Data_Obj *parent, index_t *offsets, Dimension_Set *dsp )
{
	Data_Obj *dp;
	index_t pix_offset;
	int i;

	if( check_posn(QSP_ARG  parent,offsets,dsp,name) < 0 ) return(NO_OBJ);

	dp = new_dobj(QSP_ARG  name);
	if( dp==NO_OBJ ) return(NO_OBJ);

	if( set_obj_dimensions(QSP_ARG  dp,dsp,parent->dt_prec) < 0 ){
		del_dobj(QSP_ARG  name);
		return(NO_OBJ);
	}

	/* this must be called before setup_dp, because
	 * flags are first copied from parent
	 */
	parent_relationship(parent,dp);

	set_child_increments(dp,parent->dt_type_inc,parent->dt_mach_inc);
	pix_offset = get_pix_base_offset(offsets,parent);

	dp = setup_dp(QSP_ARG  dp,parent->dt_prec);
	if( dp==NO_OBJ ){
		/* BUG? where does the cleanup happen? */
		return(dp);
	}

	if( IS_BITMAP(parent) ){
		dp->dt_data = ((long *)parent->dt_data) +
			((parent->dt_bit0+pix_offset)>>LOG2_BITS_PER_BITMAP_WORD);
	} else {
		pix_offset *= ELEMENT_SIZE(dp);
		dp->dt_data = ((char *)parent->dt_data) + pix_offset;
	}
	dp->dt_offset = pix_offset;		/* offset is in bytes */
	if( IS_BITMAP(parent) ){
		dp->dt_bit0 = parent->dt_bit0;
		for(i=0;i<N_DIMENSIONS;i++)
			dp->dt_bit0 += offsets[i] * parent->dt_type_inc[i];
		/* We used to mask bit0 here, but now the bit offset can span words */
	}

	return(dp);
} /* end mk_subseq() */

Data_Obj *
make_subsamp( QSP_ARG_DECL  const char *name, Data_Obj *parent,
		Dimension_Set *dsp, index_t *offsets, incr_t *incrs )
{
	Data_Obj *dp;
	index_t pix_offset;	/* can be neg if image runs backwards... */
	incr_t new_mach_incrs[N_DIMENSIONS];
	incr_t new_type_incrs[N_DIMENSIONS];
	int i;

	if( check_inset(QSP_ARG  parent,offsets,dsp,incrs,name) < 0 )
		return(NO_OBJ);

	dp = new_dobj(QSP_ARG  name);
	if( dp==NO_OBJ )
		return(NO_OBJ);

	if( set_obj_dimensions(QSP_ARG  dp,dsp,parent->dt_prec) < 0 ){
		del_dobj(QSP_ARG  name);
		return(NO_OBJ);
	}

	/* this must be called before setup_dp, because
	 * flags are first copied from parent
	 */
	parent_relationship(parent,dp);

	/* setup dp sets the increments as if the object were contiguous */
	for(i=0;i<N_DIMENSIONS;i++){
		new_mach_incrs[i] = parent->dt_mach_inc[i] * incrs[i];
		new_type_incrs[i] = parent->dt_type_inc[i] * incrs[i];
	}
	set_child_increments(dp,new_type_incrs, new_mach_incrs);


	dp = setup_dp(QSP_ARG  dp,parent->dt_prec);
	if( dp==NO_OBJ )
		return(dp);

	/* pix_offset can be negative if the parent image is a
	 * reversed image...
	 */

	pix_offset = get_pix_base_offset(offsets,parent);
	if( IS_BITMAP(dp) ){
		dp->dt_data = parent->dt_data;
		dp->dt_bit0 = parent->dt_bit0;
		for(i=0;i<N_DIMENSIONS;i++)
			dp->dt_bit0 += offsets[i]*parent->dt_type_inc[i];
	} else {
		pix_offset *= ELEMENT_SIZE(dp);		/* offset in bytes */
		dp->dt_data = ((char *)parent->dt_data) + pix_offset;
	}

	return(dp);
} /* end mk_subsamp */

Data_Obj *
mk_ilace( QSP_ARG_DECL  Data_Obj *parent, const char *name, int parity )
{
	Data_Obj *dp;
	Dimension_Set dimset;
	int i;
	index_t offset;

	dp=new_dobj(QSP_ARG  name);
	if( dp==NO_OBJ ) return(NO_OBJ);

	dimset = parent->dt_type_dimset;
	dimset.ds_dimension[2] /= 2;

	if( set_obj_dimensions(QSP_ARG  dp,&dimset,parent->dt_prec) < 0 ){
		del_dobj(QSP_ARG  name);
		return(NO_OBJ);
	}

	/* this must be called before setup_dp, because
	 * flags are first copied from parent
	 */
	parent_relationship(parent,dp);

	for(i=0;i<N_DIMENSIONS;i++){
		dp->dt_type_inc[i] = parent->dt_type_inc[i];
		dp->dt_mach_inc[i] = parent->dt_mach_inc[i];
	}
	dp->dt_rinc *= 2;

	dp = setup_dp(QSP_ARG  dp,parent->dt_prec);
#ifdef CAUTIOUS
	if( dp==NO_OBJ ){
		WARN("CAUTIOUS:  mk_ilace error");
		return(dp);
	}
#endif /* CAUTIOUS */

	/* BUG?  even parity gets us the first set of lines, but by convention
	 * in video terminology line numbering starts with 1, and the first set
	 * of lines is referred to as the "odd" field.  So the scripts have to
	 * reverse this, it is kind of ugly and would be nice to hide it.
	 * However, historical inertia prevents us from doing it!?
	 */

	if( parity & 1 )
		offset = parent->dt_rowinc * ELEMENT_SIZE(parent);
	else
		offset = 0;

	dp->dt_data = ((char *)parent->dt_data) + offset;
	dp->dt_offset = offset;

	return(dp);
}

/* When we relocate a subimage, we also have to relocate any subimages of
 * the subimage!!!  This is why each object has to remember its offset
 * in dt_offset...
 */

static void relocate_children( Data_Obj *dp )
{
	Node *np;
	Data_Obj *child;

	np=dp->dt_children->l_head;
	while(np!=NO_NODE){
		child = (Data_Obj *)np->n_data;
		child->dt_data = ((char *)dp->dt_data) + child->dt_offset;
		if( child->dt_children != NO_LIST )
			relocate_children(child);
		np = np->n_next;
	}
}

/* Relocate a subimage within the parent - assumes same data type.
 * Checks for valid offsets.
 */

int __relocate( QSP_ARG_DECL  Data_Obj *dp, index_t *offsets )
{
	index_t os;

	if( dp->dt_parent == NO_OBJ ){
		sprintf(error_string,
	"__relocate:  object \"%s\" is not a subimage",
			dp->dt_name);
		WARN(error_string);
		return(-1);
	}
		
	if( check_posn(QSP_ARG  dp->dt_parent,offsets,
		&dp->dt_type_dimset,dp->dt_name) < 0 ){

		sprintf(error_string,
			"bad relocation info for %s",dp->dt_name);
		WARN(error_string);
		return(-1);
	}
	os = get_pix_base_offset(offsets,dp->dt_parent);

	os *= ELEMENT_SIZE(dp);	/* offset in bytes */
	dp->dt_data = ((char *)dp->dt_parent->dt_data) + os;
	dp->dt_offset = os;

	/*
	 * Need to recompute the data pointers of any children
	 */

	if( dp->dt_children != NO_LIST )
		relocate_children(dp);

	return(0);
}

/* relocate position of the subimage */
int _relocate( QSP_ARG_DECL  Data_Obj *dp, index_t xos, index_t yos,index_t tos )
{
	index_t offsets[N_DIMENSIONS];

	offsets[0]=0L;
	offsets[1]=xos;
	offsets[2]=yos;
	offsets[3]=tos;
	offsets[4]=0L;

	return( __relocate(QSP_ARG  dp,offsets) );
}


Data_Obj *
mk_subimg( QSP_ARG_DECL  Data_Obj *parent, index_t xos,index_t yos, const char *name, dimension_t rows,dimension_t cols )
{
	index_t offsets[N_DIMENSIONS];
	Dimension_Set dimset;

	offsets[0]=0L;	dimset.ds_dimension[0]=parent->dt_type_dim[0];
	offsets[1]=xos;	dimset.ds_dimension[1]=cols;
	offsets[2]=yos;	dimset.ds_dimension[2]=rows;
	offsets[3]=0L;	dimset.ds_dimension[3]=parent->dt_type_dim[3];
	offsets[4]=0L;	dimset.ds_dimension[4]=parent->dt_type_dim[4];

	return(mk_subseq(QSP_ARG  name,parent,offsets,&dimset));
}

Data_Obj *
nmk_subimg( QSP_ARG_DECL  Data_Obj *parent, index_t xos,index_t yos, const char *name, dimension_t rows,dimension_t cols,dimension_t tdim )
{
	index_t offsets[N_DIMENSIONS];
	Dimension_Set dimset;

	offsets[0]=0L;	dimset.ds_dimension[0]=tdim;
	offsets[1]=xos;	dimset.ds_dimension[1]=cols;
	offsets[2]=yos;	dimset.ds_dimension[2]=rows;
	offsets[3]=0L;	dimset.ds_dimension[3]=parent->dt_type_dim[3];
	offsets[4]=0L;	dimset.ds_dimension[4]=parent->dt_type_dim[4];

	return(mk_subseq(QSP_ARG  name,parent,offsets,&dimset));
}

/* get_machine_dimensions - utility function to support make_equivalence
 */

static void get_machine_dimensions(Dimension_Set *dst_dsp, Dimension_Set *src_dsp, prec_t prec)
{
	*dst_dsp = *src_dsp;	/* Default - they are the same */

	if( BITMAP_PRECISION(prec) ){
		if( src_dsp->ds_dimension[0] != 1 )
			NERROR1("get_machine_dimensions:  Sorry, don't handle multi-component bitmaps");
		dst_dsp->ds_dimension[0] = 1;

		// round number of columns up
		dst_dsp->ds_dimension[1] = N_BITMAP_WORDS(src_dsp->ds_dimension[1]);
	} else if( COMPLEX_PRECISION(prec) ){
		// complex can't have a component dimension
		if( src_dsp->ds_dimension[0] != 1 ){
			sprintf(DEFAULT_ERROR_STRING,
		"Sorry, complex images must have component dimension (%d) equal to 1",
				src_dsp->ds_dimension[0]);
			NERROR1(DEFAULT_ERROR_STRING);
		}
		dst_dsp->ds_dimension[0]=2;
	} else if( QUAT_PRECISION(prec) ){
		if( src_dsp->ds_dimension[0] != 1 )
			NERROR1("Sorry, complex quaternion images must have component dimension equal to 1");
		dst_dsp->ds_dimension[0]=4;
	}
}

/* Make an object of arbirary shape, which points to the data area
 * of an existing object.  It should not be necessary that the
 * parent object be contiguous as long as the dimensions of the
 * new object are such that it can be evenly spaced;
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
 * But for bitmap, type count is larger:  (example uses 4 bit words)
 *
 *	bbbb bbxx	type_dim = 6  mach_dim = 2
 *	bbbb bbxx	type_inc = 1 (col), 8 (row)  mach_inc = 1 (col), 2 (row)
 *
 * When we equivalence bitmaps, the xx bits are as good as any others - so
 * we only care about machine dimensions and increments...
 *
 */

Data_Obj *make_equivalence( QSP_ARG_DECL  const char *name, Data_Obj *parent, Dimension_Set *dsp, prec_t prec )
{
	Data_Obj *newdp;
	const char *s;
	dimension_t n_child_bytes_per_elt,n_parent_bytes_per_elt;
	int parent_dim,child_dim;
	int n_per_parent, n_per_child;
	dimension_t	total_child_bytes,
			total_parent_bytes,
			prev_parent_mach_elts;
	incr_t child_mach_inc;
	incr_t new_mach_inc[N_DIMENSIONS];
	int multiplier, divisor;
	Dimension_Set new_mach_dim;

	/* If we are casting to a larger machine type (e.g. byte to long)
	 * We have to have at least 4 bytes contiguous.
	 */

	/* Step 1
	 *
	 * Figure out how the elements match up.
	 */

	n_child_bytes_per_elt = siztbl[ prec & MACH_PREC_MASK ];
	if( COMPLEX_PRECISION(prec) )
		n_child_bytes_per_elt *= 2;
	else if( QUAT_PRECISION(prec) )
		n_child_bytes_per_elt *= 4;

	n_parent_bytes_per_elt = siztbl[ MACHINE_PREC(parent) ];
	if( IS_COMPLEX(parent) )
		n_parent_bytes_per_elt *= 2;
	else if( IS_QUAT(parent) )
		n_parent_bytes_per_elt *= 4;

	/* Now we know how many bits in each basic element.
	 * Figure out how many elements of one makes up an element of the other.
	 * The results end up in n_per_parent and n_per_child.
	 */
	n_per_parent = n_per_child = 1;

	/* Case 1:  child element size is greater than parent element size - casting up */
	if( n_child_bytes_per_elt > n_parent_bytes_per_elt ) {
		incr_t n_contig;

		//n_per_child = siztbl[ prec & MACH_PREC_MASK ] / siztbl[ MACHINE_PREC(parent) ];
		n_per_child = n_child_bytes_per_elt / n_parent_bytes_per_elt;

		/* new size is larger, first increment must be 1 */

		/* Find the largest number of contiguous elements in the parent */
		n_contig=1;
		for(parent_dim=0;parent_dim<N_DIMENSIONS;parent_dim++)
			if( parent->dt_mach_inc[parent_dim] == n_contig )
				n_contig *= parent->dt_mach_dim[parent_dim];
		/* n_contig is the number of contiguous machine elements in the parent... */

		if( n_contig < n_per_child ){
			sprintf(error_string,
	"make_equivalence:  parent object %s n_contig = %d < %d, can't case to %s",
				parent->dt_name,n_contig,n_per_child,name_for_prec(prec));
			WARN(error_string);
			return(NO_OBJ);
		}
	} else if( n_child_bytes_per_elt < n_parent_bytes_per_elt ) {
		/* Case 2:  child element size is less than parent element size - casting down */
		//n_per_parent = siztbl[ MACHINE_PREC(parent) ] / siztbl[ prec & MACH_PREC_MASK ] ;
		n_per_parent = n_parent_bytes_per_elt / n_child_bytes_per_elt ;
	}

	/* first make sure the total size matches */

	/* We need the machine dimensions of the new object */
	get_machine_dimensions(&new_mach_dim,dsp,prec);

	total_child_bytes = 1;
	for(child_dim=0;child_dim<N_DIMENSIONS;child_dim++)
		total_child_bytes *= new_mach_dim.ds_dimension[child_dim];
	total_child_bytes *= siztbl[ prec & MACH_PREC_MASK ];

	total_parent_bytes = ELEMENT_SIZE(parent)*parent->dt_n_mach_elts;

	if( total_child_bytes != total_parent_bytes){
		sprintf(error_string,
	"make_equivalence %s:  total requested size (%d bytes) does not match parent %s (%d bytes)",
			name,total_child_bytes,parent->dt_name,total_parent_bytes);
		WARN(error_string);
		return(NO_OBJ);
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

	if( n_per_parent > 1 ){
		// If we have multiple child elements per single
		// parent element, they have to be contiguous
		child_mach_inc = 1;
	} else {
		/* The size of a child element is greater than or equal to
		 * the size of a parent element.
		 * Set child_mach_inc to the smallest non-zero increment of the parent.
		 * (What is the sense of that?)
		 * The parent has to have contiguous elements to fill a child
		 * element, but could then skip...
		 */
		int i;

		child_mach_inc=0;
		i=0;
		while( child_mach_inc==0 && i < N_DIMENSIONS ){
			child_mach_inc = parent->dt_mach_inc[i];
			i++;
		}
#ifdef CAUTIOUS
		if( child_mach_inc == 0 ) ERROR1("CAUTIOUS:  make_equivalence:  could not determine child_mach_inc!?");
#endif /* CAUTIOUS */

		/* Is this correct in all cases?  If multiple parent elements make up one child
		 * element, the contiguity check should have been performed above.  If they are the same
		 * size then we are ok.
		 */
	}

	child_dim=0;	/* index of current child dimension/increment */
	parent_dim=0;	/* index of current parent dimension/increment */

	// Set the total number of bytes equal to the bits
	// in the first element.

	// If the parent is a bitmap, this is NOT the number of machine elts!?
	total_child_bytes = siztbl[ prec & MACH_PREC_MASK ];
	total_child_bytes *= new_mach_dim.ds_dimension[0];

	total_parent_bytes = siztbl[ MACHINE_PREC(parent) ];
	total_parent_bytes *= parent->dt_mach_dim[0];


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
		if( total_parent_bytes == total_child_bytes ){
			/* increase the child dimension */
			new_mach_inc[child_dim] = child_mach_inc;
			child_dim++;
			if( child_dim < N_DIMENSIONS )
				total_child_bytes *= new_mach_dim.ds_dimension[child_dim];
			/* increase the parent dimension */
			parent_dim++;
			if( parent_dim < N_DIMENSIONS ){
				total_parent_bytes *= parent->dt_mach_dim[parent_dim];

				/* BUG - need to make sure that the math comes out even, no fractions or truncations */
				if( parent->dt_mach_inc[parent_dim] != 0 ){
					child_mach_inc = n_per_parent * parent->dt_mach_inc[parent_dim] / n_per_child;
				}
			}
		} else if( total_parent_bytes > total_child_bytes ){
			/* Increment the child dimension only.
			 */
			if( child_dim >= N_DIMENSIONS ){
				WARN("make_equivalence:  giving up!?");
				return(NO_OBJ);
			}
			/* increase the child dimension */
			new_mach_inc[child_dim] = child_mach_inc;
			child_dim++;
			if( child_dim < N_DIMENSIONS )
				total_child_bytes *= new_mach_dim.ds_dimension[child_dim];
			/* Increasing the increment assumes that the spacing
			 * within the parent is even at this larger size.
			 * This is guaranteed it the new child size is LTE
			 * the parent size.  Otherwise, we will have to increase
			 * the parent dim next time round, and that is when
			 * we will check it.
			 */
			/* assumes child is evenly-spaced between the new dimension and the old */ 
			if( new_mach_inc[child_dim-1] > 0 )
				child_mach_inc = new_mach_dim.ds_dimension[child_dim-1] * new_mach_inc[child_dim-1];
		} else { /* total_parent_bytes < total_child_bytes */
			/* Increment the parent dimension WITHOUT incrementing the child.
			 *
			 * The parent MUST be even-spaced across the dimensions
			 */
			/* increase the parent dimension */
			parent_dim++;
			prev_parent_mach_elts = total_parent_bytes;
			if( parent_dim < N_DIMENSIONS )
				total_parent_bytes *= parent->dt_mach_dim[parent_dim];
			else {
				sprintf(error_string,"make_equivalence:  can't fit objects");
				WARN(error_string);
				sprintf(error_string,"total_parent_bytes = %d   total_child_bytes = %d",
					total_parent_bytes,total_child_bytes);
				advise(error_string);
				return(NO_OBJ);
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

			if( prev_parent_mach_elts > 1 && parent->dt_mach_inc[parent_dim] != 0 &&
							parent->dt_mach_inc[parent_dim-1] != 0 &&
					parent->dt_mach_inc[parent_dim] !=
					(incr_t)(parent->dt_mach_inc[parent_dim-1]
					* parent->dt_mach_dim[parent_dim-1]) ){
				sprintf(error_string,
						"make_equivalence:  problem with unevenly spaced parent object %s",parent->dt_name);
				WARN(error_string);
				sprintf(error_string,"%s inc[%d] (%d) != inc[%d] (%d) * dim[%d] (%d)",
					parent->dt_name,parent_dim,parent->dt_mach_inc[parent_dim],
					parent_dim-1,parent->dt_mach_inc[parent_dim-1],
					parent_dim-1,parent->dt_mach_dim[parent_dim-1]);
				advise(error_string);

				sprintf(error_string,"prev_parent_mach_elts = %d, total_parent_bytes = %d, total_child_bytes = %d",
					prev_parent_mach_elts,total_parent_bytes,total_child_bytes);
				advise(error_string);

				return(NO_OBJ);
			}

		}
	}

	newdp=new_dobj(QSP_ARG  name);

	if( newdp == NO_OBJ ){
		WARN("couldn't create equivalence");
		return(NO_OBJ);
	}

	s = newdp->dt_name;	/* save, while we overwrite temporarily */
	*newdp = *parent;	/* copy all of the fields... */

	newdp->dt_name = s;
	newdp->dt_offset = 0;

	if( set_obj_dimensions(QSP_ARG  newdp,dsp,prec) ){
		del_dobj(QSP_ARG  newdp->dt_name);
		return(NO_OBJ);
	}

	if( IS_CONTIGUOUS(parent) )
		make_contiguous(newdp);
	else if( IS_EVENLY_SPACED(parent) ){
		incr_t	parent_mach_inc=0,
			parent_type_inc=0;
			/* autoinit to elim compiler warning */
		dimension_t pdim;

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
		pdim=1;
		/* find the basic parent increment */
		for(parent_dim=0;parent_dim<N_DIMENSIONS;parent_dim++){
			//pdim *= parent->dt_mach_dim[parent_dim];	/* how many elements we are up to */
			pdim *= parent->dt_type_dim[parent_dim];	/* how many elements we are up to */
			if( pdim > 1 ){
				parent_mach_inc = parent->dt_mach_inc[parent_dim];
				parent_type_inc = parent->dt_type_inc[parent_dim];
				parent_dim=N_DIMENSIONS;			/* break out of loop */
			}
		}

		newdp->dt_mach_inc[0]=parent_mach_inc;
		newdp->dt_type_inc[0]=parent_type_inc;
		for(child_dim=1;child_dim<N_DIMENSIONS;child_dim++){
			newdp->dt_mach_inc[child_dim] = newdp->dt_mach_inc[child_dim-1] * newdp->dt_mach_dim[child_dim-1];
			newdp->dt_type_inc[child_dim] = newdp->dt_type_inc[child_dim-1] * newdp->dt_type_dim[child_dim-1];
		}
	} else {	/* Not evenly spaced or contiguous... */
		//
		for(child_dim=0;child_dim<N_DIMENSIONS;child_dim++){
			newdp->dt_mach_inc[child_dim] = new_mach_inc[child_dim];
		}
	}

	/* Now we have the machine increments.
	 * Copy them to the type increments, and adjust if necessary.
	 */
	if( COMPLEX_PRECISION(prec) )
		divisor=2;
	else if( QUAT_PRECISION(prec) )
		divisor=4;
	else	divisor=1;

	if( BITMAP_PRECISION(prec) )
		multiplier = BITS_PER_BITMAP_WORD;
	else	multiplier = 1;

	for(child_dim=0;child_dim<N_DIMENSIONS;child_dim++){
		newdp->dt_type_inc[child_dim] = multiplier * newdp->dt_mach_inc[child_dim] / divisor;
	}

	/* Where should we adjust the row increment when the parent is a bitmap? */


	/* this must be called before setup_dp,
	 * because flags are first copied from parent
	 */
	parent_relationship(parent,newdp);

	newdp=setup_dp(QSP_ARG  newdp,prec);

#ifdef CAUTIOUS
	if( newdp == NO_OBJ ){
		/* BUG do something??? */
		ERROR1("CAUTIOUS:  make_equivalence:  unable to setup equivalance object!?");
	}
#endif /* CAUTIOUS */


	/* If the parent is a bitmap, but the equivalence is not, then
	 * we need to clear the bitmap flag...
	 * We also need to adjust the row increment...
	 */

	if( IS_BITMAP(newdp) && prec != PREC_BIT ){
		newdp->dt_flags &= ~DT_BIT;
	}

	return(newdp);
} /* end make_equivalence */

void propagate_flag_to_children(Data_Obj *dp, uint32_t flags_to_set )
{
	Data_Obj *child;
	Node *np;

	if( dp->dt_children == NO_LIST ) return;

	np = dp->dt_children->l_head;
	while( np != NO_NODE ){
		child = (Data_Obj *)np->n_data;

		child->dt_flags |= flags_to_set;

		if( dp->dt_children != NO_LIST ) propagate_flag_to_children(child,flags_to_set);

		np = np->n_next;
	}
}

