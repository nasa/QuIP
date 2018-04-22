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
	Dimension_Set *	eqd_type_dsp;
	Dimension_Set *	eqd_mach_dsp;
	int		eqd_n_per_parent;
	int		eqd_n_per_child;
	int		eqd_bytes_per_parent_elt;
	int		eqd_bytes_per_child_elt;
	dimension_t	eqd_total_child_bytes;
	dimension_t	eqd_total_parent_bytes;
	incr_t		eqd_child_mach_inc;
	Increment_Set	eqd_child_mach_incs;
} Equivalence_Data;

#define EQ_NAME(eqd_p)		(eqd_p)->eqd_name
#define EQ_PREC_PTR(eqd_p)	(eqd_p)->eqd_prec_p
#define EQ_PREC_CODE(eqd_p)	PREC_CODE(EQ_PREC_PTR(eqd_p))
#define EQ_PARENT(eqd_p)	(eqd_p)->eqd_parent
#define EQ_TYPE_DIMS(eqd_p)		(eqd_p)->eqd_type_dsp
#define EQ_MACH_DIMS(eqd_p)		(eqd_p)->eqd_mach_dsp
#define EQ_N_PER_PARENT(eqd_p)	(eqd_p)->eqd_n_per_parent
#define EQ_N_PER_CHILD(eqd_p)	(eqd_p)->eqd_n_per_child
#define EQ_BYTES_PER_PARENT_ELT(eqd_p)	(eqd_p)->eqd_bytes_per_parent_elt
#define EQ_BYTES_PER_CHILD_ELT(eqd_p)	(eqd_p)->eqd_bytes_per_child_elt
#define EQ_TOTAL_CHILD_BYTES(eqd_p)	(eqd_p)->eqd_total_child_bytes
#define EQ_TOTAL_PARENT_BYTES(eqd_p)	(eqd_p)->eqd_total_parent_bytes
#define EQ_CHILD_MACH_INC(eqd_p)	(eqd_p)->eqd_child_mach_inc
#define EQ_CHILD_MACH_INC(eqd_p)	(eqd_p)->eqd_child_mach_inc

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
#define child_mach_inc	EQ_CHILD_MACH_INC(eqd_p)

/* This used to be declared withing check_posn(),
 * but old sun compiler says "no automatic aggregate initialization"
 */

static incr_t ones[N_DIMENSIONS]={1,1,1,1,1};	/* dummy increments */

static void set_child_increments(Data_Obj *dp, Increment_Set *type_incrs, Increment_Set *mach_incrs)
{
	int i;

	/* Here we copy the increments from the parent -
	 * But if a subscripted dimension is now 1,
	 * then the corresponding increment must
	 * be set to 0...
	 */

	for(i=0;i<N_DIMENSIONS;i++){
		if( OBJ_MACH_DIM(dp,i) == 1 ){
			SET_OBJ_MACH_INC(dp,i,0);
			SET_OBJ_TYPE_INC(dp,i,0);
		} else {
			SET_OBJ_TYPE_INC(dp,i,INCREMENT(type_incrs,i));
			SET_OBJ_MACH_INC(dp,i,INCREMENT(mach_incrs,i));
		}
	}
}

// This was OBJ_MACH_INC but perhaps it should be OBJ_TYPE_INC?
// It probably makes more sense to use type units, but for the sake
// of minimal breakage, we leave it in machine units, and change
// the OpenCL data offset function to make adjustments for complex and quat.

static index_t get_pix_base_offset(index_t *offsets, Data_Obj *parent)
{
	index_t pix_offset;
	int i;

	pix_offset=0L;
	for(i=0;i<N_DIMENSIONS;i++)
		pix_offset += offsets[i] * OBJ_MACH_INC(parent,i);
		//pix_offset += offsets[i] * OBJ_TYPE_INC(parent,i);

	return(pix_offset);
}

/* check one dimension */

#define is_inside(index,which_dim,sub_name,parent)	_is_inside(QSP_ARG  index,which_dim,sub_name,parent)

static int _is_inside( QSP_ARG_DECL  index_t index, int which_dim, const char *sub_name, Data_Obj *parent )
{
	dimension_t pd;		/* parent dim */

	/* if indices are unsigned, the negative test is meaningless... */
	pd = OBJ_TYPE_DIM(parent,which_dim);

	if( /* index < 0 || */ index >= pd){
		sprintf(ERROR_STRING,
"%s offset %d for subobject \"%s\" falls outside of parent \"%s\" (%s count = %d)",
			dimension_name[which_dim],
			index,sub_name,OBJ_NAME(parent),
			dimension_name[which_dim],pd);
		warn(ERROR_STRING);
		sprintf(ERROR_STRING,
			"dim index %d:  parent size = %u",
			which_dim, OBJ_TYPE_DIM(parent,which_dim));
		advise(ERROR_STRING);
		return(0);
	}
	return(1);
}

/* make sure that a requested subobject fits within the parent */

#define check_inset(parent,offsets,dsp,incrs,name)	_check_inset(QSP_ARG  parent,offsets,dsp,incrs,name)

static int _check_inset( QSP_ARG_DECL  Data_Obj *parent, index_t *offsets, Dimension_Set *dsp, incr_t *incrs, const char *name )
{
	int i;
	int retval=0;

	/* make sure that all the sizes are valid */

	for(i=0;i<N_DIMENSIONS;i++){
		assert( DIMENSION(dsp,i) > 0 );
	}

	for(i=0;i<N_DIMENSIONS;i++){
		index_t extreme_index;

		extreme_index = offsets[i];
		if( ! is_inside(extreme_index,i,name,parent) )
			retval=(-1);

		extreme_index = offsets[i]+incrs[i]*(DIMENSION(dsp,i)-1);
		if( ! is_inside(extreme_index,i,name,parent) )
			retval=(-1);

	}
	return retval;
} /* end check_inset() */

#define check_posn(parent,offsets,dsp,name)	_check_posn(QSP_ARG  parent,offsets,dsp,name)

static int _check_posn( QSP_ARG_DECL  Data_Obj *parent, index_t *offsets, Dimension_Set *dsp, const char *name )
{
	return( check_inset(parent,offsets,dsp,ones,name) );
}

void point_obj_to_ext_data( Data_Obj *dp, void *ptr )
{
	SET_OBJ_DATA_PTR(dp,ptr);
	SET_OBJ_FLAG_BITS(dp,DT_NO_DATA);
}

/* Set family fields for both child and parent
 */

void parent_relationship( Data_Obj *parent, Data_Obj *child )
{
	SET_OBJ_AREA(child,OBJ_AREA(parent));
	SET_OBJ_PARENT(child,parent);

	/* the child is always a new object, so we're not
	 * losing anything here
	 */
	SET_OBJ_CHILDREN(child, NULL);

	if( OBJ_CHILDREN(parent) == NULL )
		SET_OBJ_CHILDREN(parent,new_list());

	addHead(OBJ_CHILDREN(parent),mk_node(child));

	SET_OBJ_FLAGS(child,OBJ_FLAGS(parent));
	SET_OBJ_FLAG_BITS(child,DT_NO_DATA);

	/* Clear the TEMP flag in case
	 * parent is a subscripted object
	 * (not in hash table).
	 *
	 * It is the responsibility of the caller to explicitly
	 * set this flag if needed (see array.c)
	 */

	/* Clear the EXPORTED flag - subobjects not
	 * automatically exported, even if the parent is.
	 */

	CLEAR_OBJ_FLAG_BITS(child,DT_TEMP|DT_EXPORTED);

	// We would like to remove dependencies on OpenGL
	// and cuda...  One solution is to make the flag
	// copying an object method?  Then, if an object
	// is created as a gl buffer, the function pointer
	// can be set...

// this section used to be ifdef'd HAVE_OPENGL?

	if( IS_GL_BUFFER(parent) ){
		// Unaligned_data is an overloaded field here... ?
		SET_OBJ_GL_INFO(child,OBJ_GL_INFO(parent) );
		// These two flag transfer commands
		// used to be ifdef'd HAVE_CUDA...
		xfer_dobj_flag(child,parent,DT_GL_BUF);
		xfer_dobj_flag(child,parent,DT_BUF_MAPPED);
	}

}

// Is pix_offset in machine or type units?  (Important for float/complex)
// Originally it was in machine units, but this broke things for the OpenCL
// implementation.
//

// update_offset should be here too, instead of in veclib2, even though
// it is a platform function...

void default_offset_data_func(QSP_ARG_DECL  Data_Obj *dp, index_t pix_offset )
{
	Data_Obj *parent;

//fprintf(stderr,"default_offset_data_func:  %s, pix_offset = %d\n",
//OBJ_NAME(dp),pix_offset);
	parent = OBJ_PARENT(dp);
	if( IS_BITMAP(parent) ){
		/*
		point_obj_to_ext_data(dp, ((long *)OBJ_DATA_PTR(parent)) +
			((OBJ_BIT0(parent)+pix_offset)>>LOG2_BITS_PER_BITMAP_WORD) );
		*/
		// bitmap offsets are handled by bit0
		point_obj_to_ext_data(dp, ((long *)OBJ_DATA_PTR(parent)) );
		pix_offset=0;
	} else {
		pix_offset *= ELEMENT_SIZE(dp);
		point_obj_to_ext_data(dp, ((char *)OBJ_DATA_PTR(parent)) + pix_offset);
	}
	SET_OBJ_OFFSET(dp,pix_offset);
}

static void set_child_bit0(Data_Obj *dp, index_t *offsets)
{
	int i;
	Data_Obj *parent;

	parent = OBJ_PARENT(dp);

	SET_OBJ_BIT0(dp, OBJ_BIT0(parent) );
	for(i=0;i<N_DIMENSIONS;i++)
		SET_OBJ_BIT0(dp, OBJ_BIT0(dp)
			 + offsets[i] * OBJ_TYPE_INC(parent,i) );
}

Data_Obj *
_mk_subseq( QSP_ARG_DECL  const char *name, Data_Obj *parent, index_t *offsets, Dimension_Set *dsp )
{
	Data_Obj *dp;
	index_t pix_offset;

	if( check_posn(parent,offsets,dsp,name) < 0 ) return(NULL);

	dp = new_dobj(name);
	if( dp==NULL ) return(NULL);

	SET_OBJ_SHAPE(dp,ALLOC_SHAPE);

	if( set_obj_dimensions(dp,dsp,OBJ_PREC_PTR(parent)) < 0 ){
		rls_shape( OBJ_SHAPE(dp) );
		del_dobj(dp);
		return(NULL);
	}

	/* this must be called before setup_dp, because
	 * flags are first copied from parent
	 */
	parent_relationship(parent,dp);

	set_child_increments(dp,OBJ_TYPE_INCS(parent),OBJ_MACH_INCS(parent));
	pix_offset = get_pix_base_offset(offsets,parent);

	dp = setup_dp(dp,OBJ_PREC_PTR(parent));
	if( dp==NULL ){
		/* BUG? where does the cleanup happen? */
		return(dp);
	}

	// Clear the volatile flag if set...
	// This is necessary because volatile objects are deleted in ascii_menu.c,
	// to deal with platform copies - probably better to introduce a new flag.
	CLEAR_OBJ_FLAG_BITS(dp,DT_VOLATILE);

	// In openCL, the "data_ptr" is actually a pointer to the memory buffer,
	// object, not the address of the data itself.  (This appears to be a private
	// structure, so we don't have access to the innards.)
	// Instead, we have to call clCreateSubBuffer to get a sub-buffer with
	// the required offset.  Unfortunately, we can't call this multiple times,
	// so if we are creating a sub-object of a sub-object, we have to go back
	// and get the buffer from the ultimate owner of the data.
	//
	// We address this by using adding an offset parameter to the data object,
	// which is passed to all OpenCL kernel calls.  A bit wasteful when not
	// needed, but guaranteed to work.  The offset is passed to the kernel
	// and is in units of the basic type (complex, not float!)
	// 
	( * PF_OFFSET_DATA_FN(OBJ_PLATFORM(parent)) ) (QSP_ARG  dp, pix_offset );

#ifdef REMOVE_BAD
	// This has to be part of the platform-specific function !!!
	SET_OBJ_OFFSET(dp,pix_offset);		/* offset was in bytes */
						/* but now is in elements! */
#endif // REMOVE_BAD

	if( IS_BITMAP(parent) ) set_child_bit0(dp,offsets);

	return(dp);
} /* end mk_subseq() */

Data_Obj * _make_subsamp( QSP_ARG_DECL  const char *name, Data_Obj *parent,
		Dimension_Set *dsp, index_t *offsets, incr_t *incrs )
{
	Data_Obj *dp;
	index_t pix_offset;	/* can be neg if image runs backwards... */
	Increment_Set mis, *new_mach_isp=(&mis);
	Increment_Set tis, *new_type_isp=(&tis);
	int i;

	if( check_inset(parent,offsets,dsp,incrs,name) < 0 )
		return(NULL);

	dp = new_dobj(name);
	if( dp==NULL )
		return(NULL);

	// init_dp uses AUTO_SHAPE...
	if( init_dp(dp,dsp,OBJ_PREC_PTR(parent) ) == NULL ){
		del_dobj(dp);
		return(NULL);
	}

	/* this must be called before setup_dp, because
	 * flags are first copied from parent
	 */
	parent_relationship(parent,dp);

	/* setup dp sets the increments as if the object were contiguous */
	for(i=0;i<N_DIMENSIONS;i++){
		SET_INCREMENT(new_mach_isp,i, OBJ_MACH_INC(parent,i) * incrs[i] );
		SET_INCREMENT(new_type_isp,i, OBJ_TYPE_INC(parent,i) * incrs[i] );
	}
	set_child_increments(dp,new_type_isp, new_mach_isp);


	// did init_dp already call setup_dp?? YES
	// BUT we need to reset the flags!?
	// dp = setup_dp(dp,OBJ_PREC_PTR(parent));

	// We might want to not use AUTO_SHAPE - for example, if we subsample
	// a column vector that we want to treat as an image?
	if( auto_shape_flags( OBJ_SHAPE(dp) ) < 0 )
		return(NULL);

	check_contiguity(dp);		// almost sure not contiguous if subsample!

	if( dp==NULL )
		return(dp);

	/* pix_offset can be negative if the parent image is a
	 * reversed image...
	 */

	pix_offset = get_pix_base_offset(offsets,parent);

	( * PF_OFFSET_DATA_FN(OBJ_PLATFORM(parent)) ) (QSP_ARG  dp, pix_offset );

	//SET_OBJ_OFFSET(dp,pix_offset);

	if( IS_BITMAP(dp) ) set_child_bit0(dp,offsets);

	return(dp);
} /* end mk_subsamp */

Data_Obj * _mk_ilace( QSP_ARG_DECL  Data_Obj *parent, const char *name, int parity )
{
	Data_Obj *dp;
	Dimension_Set ds1, *dsp=(&ds1);
	int i;
	index_t offset;

	dp=new_dobj(name);
	if( dp==NULL ) return(NULL);

	SET_OBJ_SHAPE(dp,ALLOC_SHAPE);

	DIMSET_COPY(dsp , OBJ_TYPE_DIMS(parent) );
	SET_DIMENSION(dsp,2,DIMENSION(dsp,2) / 2);

	if( set_obj_dimensions(dp,dsp,OBJ_PREC_PTR(parent)) < 0 ){
		rls_shape( OBJ_SHAPE(dp) );
		del_dobj(dp);
		return(NULL);
	}

	/* this must be called before setup_dp, because
	 * flags are first copied from parent
	 */
	parent_relationship(parent,dp);

	for(i=0;i<N_DIMENSIONS;i++){
		SET_OBJ_TYPE_INC(dp,i,OBJ_TYPE_INC(parent,i));
		SET_OBJ_MACH_INC(dp,i,OBJ_MACH_INC(parent,i));
	}
	SET_OBJ_ROW_INC(dp,OBJ_ROW_INC(dp) * 2);

	/* Even parity gets us the first set of lines, but by convention
	 * in video terminology line numbering starts with 1, and the first set
	 * of lines is referred to as the "odd" field.  So the scripts have to
	 * reverse this, it is kind of ugly and would be nice to hide it.
	 * However, historical inertia prevents us from doing it!?
	 */

	if( parity & 1 )
		offset = OBJ_ROW_INC(parent);
	else
		offset = 0;

	( * PF_OFFSET_DATA_FN(OBJ_PLATFORM(parent)) ) (QSP_ARG  dp, offset );

	return(dp);
} // mk_ilace

/* When we relocate a subimage, we also have to relocate any subimages of
 * the subimage!!!  This is why each object has to remember its offset
 * in dt_offset...  We used to keep this in bytes, but as OpenCL needs
 * to do something different, we now keep the offset in elements.
 *
 * The update_offset function should update the offset of a child object
 * relative to its parent...
 */

#define relocate_children(dp)	_relocate_children(QSP_ARG  dp)

static void _relocate_children(QSP_ARG_DECL  Data_Obj *dp )
{
	Node *np;
	Data_Obj *child;

	np=QLIST_HEAD( OBJ_CHILDREN(dp) );
	while(np!=NULL){
		child = (Data_Obj *)NODE_DATA(np);
		( * PF_UPDATE_OFFSET_FN(OBJ_PLATFORM(dp)) ) (QSP_ARG  child );

		if( OBJ_CHILDREN(child) != NULL )
			relocate_children(child);
		np = NODE_NEXT(np);
	}
}

/* Relocate a subimage within the parent - assumes same data type.
 * Checks for valid offsets.
 */

int _relocate_with_offsets( QSP_ARG_DECL  Data_Obj *dp, index_t *offsets )
{
	index_t os;

	if( OBJ_PARENT(dp) == NULL ){
		sprintf(ERROR_STRING,
	"__relocate:  object \"%s\" is not a subimage",
			OBJ_NAME(dp));
		warn(ERROR_STRING);
		return(-1);
	}
		
	if( check_posn(OBJ_PARENT(dp),offsets,
		OBJ_TYPE_DIMS(dp),OBJ_NAME(dp)) < 0 ){

		sprintf(ERROR_STRING,
			"bad relocation info for %s",OBJ_NAME(dp));
		warn(ERROR_STRING);
		return(-1);
	}
	os = get_pix_base_offset(offsets,OBJ_PARENT(dp));

	( * PF_OFFSET_DATA_FN(OBJ_PLATFORM(OBJ_PARENT(dp))) ) (QSP_ARG  dp, os );
	//SET_OBJ_OFFSET(dp,os);

	/*
	 * Need to recompute the data pointers of any children
	 */

	if( OBJ_CHILDREN(dp) != NULL )
		relocate_children(dp);

	return(0);
} /* __relocate */

/* relocate position of the subimage */
int _relocate( QSP_ARG_DECL  Data_Obj *dp, index_t xos, index_t yos,index_t tos )
{
	index_t offsets[N_DIMENSIONS];

	offsets[0]=0L;
	offsets[1]=xos;
	offsets[2]=yos;
	offsets[3]=tos;
	offsets[4]=0L;

	return( relocate_with_offsets(dp,offsets) );
}


Data_Obj *
_mk_subimg( QSP_ARG_DECL  Data_Obj *parent, index_t xos,index_t yos, const char *name, dimension_t rows,dimension_t cols )
{
	index_t offsets[N_DIMENSIONS];
	Dimension_Set ds1, *dsp=(&ds1);

	offsets[0]=0L;	SET_DIMENSION(dsp,0,OBJ_TYPE_DIM(parent,0));
	offsets[1]=xos;	SET_DIMENSION(dsp,1,cols);
	offsets[2]=yos;	SET_DIMENSION(dsp,2,rows);
	offsets[3]=0L;	SET_DIMENSION(dsp,3,OBJ_TYPE_DIM(parent,3));
	offsets[4]=0L;	SET_DIMENSION(dsp,4,OBJ_TYPE_DIM(parent,4));

	return(mk_subseq(name,parent,offsets,dsp));
}


Data_Obj *
_mk_substring( QSP_ARG_DECL  Data_Obj *parent, index_t sos,const char *name, dimension_t len )
{
	index_t offsets[N_DIMENSIONS];
	Dimension_Set ds1, *dsp=(&ds1);

	offsets[0]=sos;	SET_DIMENSION(dsp,0,len);
	offsets[1]=0L;	SET_DIMENSION(dsp,1,OBJ_TYPE_DIM(parent,1));
	offsets[2]=0L;	SET_DIMENSION(dsp,2,OBJ_TYPE_DIM(parent,2));
	offsets[3]=0L;	SET_DIMENSION(dsp,3,OBJ_TYPE_DIM(parent,3));
	offsets[4]=0L;	SET_DIMENSION(dsp,4,OBJ_TYPE_DIM(parent,4));

	return(mk_subseq(name,parent,offsets,dsp));
}

Data_Obj * _nmk_subimg( QSP_ARG_DECL  Data_Obj *parent, index_t xos,index_t yos, const char *name, dimension_t rows,dimension_t cols,dimension_t tdim )
{
	index_t offsets[N_DIMENSIONS];
	Dimension_Set ds1, *dsp=(&ds1);

	offsets[0]=0L;	SET_DIMENSION(dsp,0,tdim);
	offsets[1]=xos;	SET_DIMENSION(dsp,1,cols);
	offsets[2]=yos;	SET_DIMENSION(dsp,2,rows);
	offsets[3]=0L;	SET_DIMENSION(dsp,3,OBJ_TYPE_DIM(parent,3));
	offsets[4]=0L;	SET_DIMENSION(dsp,4,OBJ_TYPE_DIM(parent,4));

	return(mk_subseq(name,parent,offsets,dsp));
}

void propagate_flag_to_children(Data_Obj *dp, uint32_t flags_to_set )
{
	Data_Obj *child;
	Node *np;

	if( OBJ_CHILDREN(dp) == NULL ) return;

	np = QLIST_HEAD( OBJ_CHILDREN(dp) );
	while( np != NULL ){
		child = (Data_Obj *)NODE_DATA(np);

		SET_OBJ_FLAG_BITS(child,flags_to_set);

		if( OBJ_CHILDREN(dp) != NULL ) propagate_flag_to_children(child,flags_to_set);

		np = NODE_NEXT(np);
	}
}

