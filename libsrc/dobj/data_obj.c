/**		data_obj.c	general data objects	*/

#include "quip_config.h"

char VersionId_dataf_data_obj[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif /* HAVE_STRING_H */

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	/* free() */
#endif /* HAVE_STDLIB_H */

#include "data_obj.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include "cuda_supp.h"
#endif /* HAVE_CUDA */

#include "items.h"
#include "debug.h"
#include "function.h"	/* prototype for add_sizable() */
#include "nports_api.h"

/* this stuff added to support pick_obj() */
/* take this out when not needed! */
#include "query.h"		/* prototype for intractive() */


/* local prototypes */
static void release_data(Data_Obj *);

static void set_minmaxdim(Shape_Info *,uint32_t);

static void make_zombie(QSP_ARG_DECL  Data_Obj *dp);
static void del_subs(QSP_ARG_DECL  Data_Obj *dp);

/* declare globals here */
const char *prec_name[N_NAMED_PRECS];		/* user friendly descriptions */

uint32_t debug_data=0;
short siztbl[N_MACHINE_PRECS];			/* sizes in bytes */

ITEM_INTERFACE_DECLARATIONS(Data_Obj,dobj)

List *dobj_list(SINGLE_QSP_ARG_DECL)
{
	if( dobj_itp == NO_ITEM_TYPE ) return(NO_LIST);
	return(item_list(QSP_ARG  dobj_itp));
}

Data_Obj *pick_obj(QSP_ARG_DECL  const char *pmpt)
{
	const char *s;

	if( pmpt==NULL || *pmpt==0 )
		pmpt="data object";		/* default prompt */

#ifdef HAVE_HISTORY
	/* pick_item() won't accept names with appended subscripts;
	 * therefore use nameof and initialize the choices manually
	 */

	/* We might accidentally call this before dataobj_init()... */
	if( dobj_itp == NO_ITEM_TYPE ) dataobj_init(SINGLE_QSP_ARG);

	if( intractive(SINGLE_QSP_ARG) ) init_item_hist(QSP_ARG  dobj_itp,pmpt);
#endif /* HAVE_HISTORY */

	s=NAMEOF(pmpt);
	return( get_obj(QSP_ARG  s) );
}

static void release_data( Data_Obj *dp )
{
#ifdef HAVE_CUDA
	cudaError_t e;
#endif /* HAVE_CUDA */

	if( dp->dt_data != (unsigned char *)NULL ){
#ifdef CAUTIOUS
		if( dp->dt_ap==NO_AREA )
			NERROR1("CAUTIOUS:  release_data:  no data area!?");
#endif /* CAUTIOUS */

		if( dp->dt_ap->da_ma_p != NO_MEMORY_AREA ){
			int_for_addr s;

			/* BUG what about aligned data from a data area? */
			s = dp->dt_n_mach_elts * ELEMENT_SIZE( dp);
			s += ( ((int_for_addr)dp->dt_data) - ((int_for_addr)dp->dt_unaligned_data) );

			givspace(&dp->dt_ap->da_freelist, (int32_t)s,
		((char *)dp->dt_unaligned_data) - ((char *)dp->dt_ap->da_base));
		} else {	/* ram area */
			switch(AREA_TYPE(dp)){
				case DA_RAM:
					givbuf(dp->dt_unaligned_data);
					break;
#ifdef HAVE_CUDA
				case DA_CUDA_GLOBAL:
					e = cudaFree(dp->dt_data);
					if( e != cudaSuccess ){
			describe_cuda_error2("release_data","cudaFree",e);
					}
					break;
				case DA_CUDA_HOST:
					e = cudaFreeHost(dp->dt_data);
					if( e != cudaSuccess ){
			describe_cuda_error2("release_data","cudaFreeHost",e);
					}
					break;
#ifdef CAUTIOUS
				case DA_CUDA_HOST_MAPPED:
	NERROR1("CAUTIOUS:  release_data:  cuda host-mapped data should not be owned");
					break;
#endif /* CAUTIOUS */

#endif /* HAVE_CUDA */

#ifdef CAUTIOUS
				default:
					sprintf(DEFAULT_ERROR_STRING,
"CAUTIOUS:  release_data:  don't know how to release data from area %s!?",
						dp->dt_ap->da_name);
					NWARN(DEFAULT_ERROR_STRING);
					break;
#endif /* CAUTIOUS */
			}
		}
	}

#ifdef CAUTIOUS
	  else {
	  	sprintf(DEFAULT_ERROR_STRING,
"CAUTIOUS:  release_data:  Object %s owns data but has a null data pointer!?",
			dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
	}
#endif /* CAUTIOUS */

}

/*
 *	Remove a child object from parent's list
 */

void disown_child( Data_Obj *dp )
{
	Node *np;

	np=remData(dp->dt_parent->dt_children,dp);

#ifdef CAUTIOUS
	if( np==NO_NODE ){
		NWARN("CAUTIOUS:  disown_child:  couldn't find child node");
		if( dp->dt_parent == NO_OBJ )
			NERROR1("object has no parent!?");
		if( dp->dt_parent->dt_children == NO_LIST )
			NERROR1("parent object has no children!?");
		else {
			sprintf(DEFAULT_ERROR_STRING,"Children of %s:",dp->dt_parent->dt_name);
			advise(DEFAULT_ERROR_STRING);
			np=dp->dt_parent->dt_children->l_head;
			while(np!=NO_NODE){
				sprintf(DEFAULT_ERROR_STRING,"\t0x%lx",
					((int_for_addr)np->n_data));
				advise(DEFAULT_ERROR_STRING);
				np=np->n_next;
			}
		}
		NERROR1("giving up");
	} else
#endif /* CAUTIOUS */
		rls_node(np);

	if( eltcount(dp->dt_parent->dt_children) == 0 ){
		rls_list(dp->dt_parent->dt_children);	/* free list */
		dp->dt_parent->dt_children = NO_LIST;
	}
}

static void del_subs(QSP_ARG_DECL  Data_Obj *dp)			/** delete all subimages */
{
	Node *np;

	/*
	 * delvec() will remove the parent's child list
	 * after deleting last child.
	 *
	 * List elements are de-allocated by delvec()
	 */

	while( dp->dt_children != NO_LIST ){
		np=dp->dt_children->l_head;
		delvec( QSP_ARG  (Data_Obj *) np->n_data );
	}
}

/* give this object a new (hopefully unique) name */

static void make_zombie(QSP_ARG_DECL  Data_Obj *dp)
{
	static int n_zombie=1;
	char zname[LLEN];

	/* this function removes it from the hash table and
	 * active item list, doesn't add the structure to the
	 * item free list.  This is why we release the node
	 * and later free the object...
	 *
	 * I think the idea here is that there might be a dangling reference
	 * to an image - for instance, if we display an image, then delete it,
	 * and then later want to refresh the window...
	 */
	zombie_item(QSP_ARG  dobj_itp,(Item *)dp);

	sprintf(zname,"Z.%s.%d",dp->dt_name,n_zombie++);
	rls_str( (char *) dp->dt_name );		/* unsave old name */
	dp->dt_name = savestr(zname);

	dp->dt_flags |= DT_ZOMBIE;
}

/*
 * Delete a data object.
 *
 * If this object has a non-zero reference count, (a viewer
 * needs it for refresh), then we rename it and make it a zombie.
 * It will be the responsibility of the zombie owner to decrement
 * the reference count and delete it (with another call to delvec)
 * when it is no longer needed.
 *
 * Frees both the data and header structure for the pointed to data object.
 * If it has children, the children are also deleted.  If it is a temporary
 * object, a warning is printed and no action is taken.
 */

void delvec(QSP_ARG_DECL  Data_Obj *dp)
{

//sprintf(error_string,"delvec %s",dp->dt_name);
//advise(error_string);

	if( dp->dt_flags & DT_STATIC && OWNS_DATA(dp) ){
//sprintf(error_string,"delvec:  static object %s will be made a zombie",dp->dt_name);
//advise(error_string);
		make_zombie(QSP_ARG  dp);
		return;
	}


	if( dp->dt_refcount > 0 ){
		/* This zombie business was introduced at a time
		 * when a displayed image had to be kept around
		 * to refresh its window...  with the current
		 * X windows implementation of viewers that is
		 * no longer the case, so this may be unecessary...
		 */
//sprintf(error_string,"delvec:  object %s (refcount = %d) will be made a zombie",dp->dt_name,dp->dt_refcount);
//advise(error_string);
		make_zombie(QSP_ARG  dp);
		return;
	}
	if( dp->dt_children != NO_LIST ){
		del_subs(QSP_ARG  dp);
	}
	if( dp->dt_parent != NO_OBJ ){
		disown_child(dp);
	}

	if( IS_TEMP(dp) ){
		release_tmp_obj(dp);
		/*
		 * Most likely called when parent is deleted.
		 * Temp objects are not hashed, and are not dynamically
		 * allocated.
		 *
		 * Simply mark as free by clearing name field.
		 */
		return;
	}

	if( OWNS_DATA(dp) ){
		release_data(dp);
	}

	/* BUG cntext code assumes that this is really deleted... */
	if( ! IS_ZOMBIE(dp) ){
		del_item(QSP_ARG  dobj_itp, dp );
	}

	/* The name might be null if we had an error creating the object... */
	if( dp->dt_declfile != NULL ){
		rls_str( dp->dt_declfile );
	}

	rls_str( (char *) dp->dt_name );		/* unsave stored name */

	if( IS_ZOMBIE(dp) ){
		/* NOTE:  we used to release the struct here with guvbuf, but in the current
		 * implementation of the item package, objects aren't allocated with getbuf!
		 */
		/* put this back on the free list... */
		add_to_item_freelist(dobj_itp,dp);
	}
}

/* Mindim and maxdim are used to determine which dimensions should be used for
 * indexing.  Here, we use mach_dim instead of type_dim, so that we can index
 * the real and imaginary parts of a complex number.
 *
 * This is the main reason that we need two sets of dimensions - kind of UGLY.
 */

static void set_minmaxdim(Shape_Info *shpp,uint32_t shape_flag)
{
	int i;

	/* set maxdim */
	if( shape_flag == AUTO_SHAPE ){
		shpp->si_maxdim = 0;
		for(i=0;i<N_DIMENSIONS;i++){
			if( shpp->si_mach_dim[i] > 1 ) shpp->si_maxdim = i;
		}
	} else {
		if( shape_flag == DT_SCALAR )
			shpp->si_maxdim = 0;
		else if( shape_flag == DT_ROWVEC )
			shpp->si_maxdim = 1;
		else if( shape_flag == DT_COLVEC )
			shpp->si_maxdim = 2;
		else if( shape_flag == DT_IMAGE )
			shpp->si_maxdim = 2;
		else if( shape_flag == DT_SEQUENCE )
			shpp->si_maxdim = 3;
		else if( shape_flag == DT_HYPER_SEQ )
			shpp->si_maxdim = 4;
#ifdef CAUTIOUS
		else {
			NWARN("CAUTIOUS:  set_minmaxdim:  unexpected type flag!?");
		}
#endif /* CAUTIOUS */
	}
				

	/* set mindim */
	shpp->si_mindim = N_DIMENSIONS-1;
	for(i=N_DIMENSIONS-1;i>=0;i--){
		if( shpp->si_mach_dim[i] > 1 ) shpp->si_mindim = i;
	}

	shpp->si_last_subi = shpp->si_maxdim + 1;
}

int same_shape(Shape_Info *shpp1,Shape_Info *shpp2)
{
	int i;

	/* first check that the flags are the same */

	if( (shpp1->si_flags & SHAPE_MASK) != (shpp2->si_flags & SHAPE_MASK) )
		return(0);
/*
{
sprintf(error_string,"same_shape:  masked flags are 0x%lx and 0x%lx",
shpp1->si_flags&SHAPE_MASK,shpp2->si_flags&SHAPE_MASK);
advise(error_string);
		return(0);
}
*/

	/* Use type_dim here instead of mach_dim? */
	for(i=0;i<N_DIMENSIONS;i++){
		if( shpp1->si_type_dim[i] != shpp2->si_type_dim[i] ){
			return(0);
		}
	}

	return(1);
}


/* Set the flags in a shape_info struct based on the values
 * in the dimension array.  The object pointer dp may be null,
 * its only use is to provide a name when printing an error msg.
 *
 * This routine determines the type (real/complex) from dt_prec...
 */

int set_shape_flags(Shape_Info *shpp,Data_Obj *dp,uint32_t shape_flag)
{
	int i;

	/* if shape is unknown, the flag should already be set */
	if( UNKNOWN_SHAPE(shpp) ){
		shpp->si_maxdim = 0;
		shpp->si_mindim = N_DIMENSIONS-1;
		return(0);
	}

	shpp->si_flags &= ~SHAPE_DIM_MASK;

#ifdef CAUTIOUS
	for(i=0;i<N_DIMENSIONS;i++){
		if( shpp->si_type_dim[i] <= 0 ){
			if( dp != NO_OBJ && dp->dt_name != NULL )
				sprintf(DEFAULT_ERROR_STRING,
	"CAUTIOUS:  set_shape_flags:  Object \"%s\", zero dimension[%d] = %d",
				dp->dt_name,i,shpp->si_type_dim[i]);
			else
				sprintf(DEFAULT_ERROR_STRING,
	"CAUTIOUS:  set_shape_flags:  zero dimension[%d] = %d",
				i,shpp->si_type_dim[i]);
			NWARN(DEFAULT_ERROR_STRING);
			return(-1);
		}
	}
#endif /* CAUTIOUS */

	/* BUG?  here we set the shape type based
	 * on dimension length, which makes it impossible
	 * to have a one-length vector.  This causes problems
	 * when we index a variable-length vector which
	 * can have length one.  An example of this is a camera
	 * array, when we sometimes only have one camers...
	 *
	 * Perhaps the solution is to set the shape flag when
	 * the object is created, and use that to set maxdim?
	 */

	set_minmaxdim(shpp,shape_flag);

	if( shape_flag == AUTO_SHAPE ){
		if( shpp->si_seqs > 1 )
			shpp->si_flags |= DT_HYPER_SEQ;
		else if( shpp->si_frames > 1 )
			shpp->si_flags |= DT_SEQUENCE;
		else if( shpp->si_rows > 1 ){
			if( shpp->si_type_dim[1]==1 )
				shpp->si_flags |= DT_COLVEC;
			else
				shpp->si_flags |= DT_IMAGE;
		}
		else {
			dimension_t nc;

			nc=shpp->si_cols;

			if( nc > 1 )
				shpp->si_flags |= DT_ROWVEC;
			else	shpp->si_flags |= DT_SCALAR;
		}
	} else {
sprintf(DEFAULT_ERROR_STRING,"setting shape flag bit to 0x%x",shape_flag);
advise(DEFAULT_ERROR_STRING);
		shpp->si_flags |= shape_flag;
	}

	shpp->si_flags &= ~ SHAPE_TYPE_MASK;
	if( COMPLEX_PRECISION(shpp->si_prec) ){
		shpp->si_flags |= DT_COMPLEX;
	} else if( QUAT_PRECISION(shpp->si_prec) ){
		shpp->si_flags |= DT_QUAT;
	} else {
		if( shpp->si_comps != 1 ){
			shpp->si_flags |= DT_MULTIDIM;
		}
	}

	/* BUG?  should the string bit be part of shape dim mask??? */
	if( STRING_PRECISION(shpp->si_prec) ){
		shpp->si_flags |= DT_STRING;
	}

	if( CHAR_PRECISION(shpp->si_prec) ){
		shpp->si_flags |= DT_CHAR;
	}

	if( BITMAP_PRECISION(shpp->si_prec) )
		shpp->si_flags |= DT_BIT;

	return(0);
} /* end set_shape_flags() */

/*
 * Call a function on all the elements of an object.
 *
 * This is an inefficient but easy to use way of performing an operation
 * on all the elements of a data object.  The user supplied function func()
 * is called with the object pointer dp and a long component offset as
 * arguments.  The offset is in units of the basic component type
 * (not necessarily bytes).  Dobj_iterate() increments the offset properly
 * for non-contiguous data objects such as subimages and interleaved
 * components, and frees the programmer from worrying about the details,
 * the disadvantage being that there is the extra overhead of one function
 * call for each element processed.
 */

void dobj_iterate(Data_Obj *dp,void (*func)(Data_Obj *,index_t))
{
	dimension_t comp,col,row,frm,seq;

	/* offsets for sequence, frame, row, pixel, component */
	dimension_t s_os, f_os, r_os, p_os, c_os;

	s_os=0;
	for(seq=0;seq<dp->dt_seqs;seq++){
		f_os = s_os;
		for(frm=0;frm<dp->dt_frames;frm++){
			r_os = f_os;
			for(row=0;row<dp->dt_rows;row++){
				p_os = r_os;
				for(col=0;col<dp->dt_cols;col++){
					c_os = p_os;
					for(comp=0;comp<dp->dt_comps;comp++){
						(*func)(dp,c_os);
						c_os += dp->dt_cinc;
					}
					p_os += dp->dt_pinc;
				}
				r_os += dp->dt_rinc;
			}
			f_os += dp->dt_finc;
		}
		s_os += dp->dt_sinc;
	}
}


void dpair_iterate(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,void (*func)(QSP_ARG_DECL  Data_Obj *,index_t,Data_Obj *,index_t))
{
	dimension_t comp,col,row,frm,seq;

	/* offsets for sequence, frame, row, pixel, component */
	dimension_t s_os1, f_os1, r_os1, p_os1, c_os1;
	dimension_t s_os2, f_os2, r_os2, p_os2, c_os2;

	s_os1=0;
	s_os2=0;
	for(seq=0;seq<dp1->dt_seqs;seq++){
		f_os1 = s_os1;
		f_os2 = s_os2;
		for(frm=0;frm<dp1->dt_frames;frm++){
			r_os1 = f_os1;
			r_os2 = f_os2;
			for(row=0;row<dp1->dt_rows;row++){
				p_os1 = r_os1;
				p_os2 = r_os2;
				for(col=0;col<dp1->dt_cols;col++){
					c_os1 = p_os1;
					c_os2 = p_os2;
					for(comp=0;comp<dp1->dt_comps;comp++){
						(*func)(QSP_ARG  dp1,c_os1,dp2,c_os2);
						c_os1 += dp1->dt_cinc;
						c_os2 += dp2->dt_cinc;
					}
					p_os1 += dp1->dt_pinc;
					p_os2 += dp2->dt_pinc;
				}
				r_os1 += dp1->dt_rinc;
				r_os2 += dp2->dt_rinc;
			}
			f_os1 += dp1->dt_finc;
			f_os2 += dp2->dt_finc;
		}
		s_os1 += dp1->dt_sinc;
		s_os2 += dp2->dt_sinc;
	}
}

/* generalized transpose, in-place */

/*
 * Example:  make an interleaved color image appear as a sequence of frames
 *
 * this example was worked out to convince myself that there we nothing
 * extra-tricky to be done...  jbm 9-10-94
 *
 *		dim		inc
 *  before:
 *		3		1
 *		dx		3
 *		dy		3*dx
 *		1		3*dx*dy
 *
 * after:
 *		1		3*dx*dy
 *		dx		3
 *		dy		3*dx
 *		3		1
 */

/*
 * Generalized transpose (no data movement)
 *
 * Changes the header information to effect a transpose between the
 * dimensions indexed by dim1 and dim2.  The dimension indexed by 0
 * is the number of components, 1 number of columns, 2 number of rows,
 * 3 number of frames, 4 number of sequences.  For example, to do a normal
 * row/column transpose of an image (matrix), call gen_xpose(dp,1,2); to
 * transform an interleaved RGB image to a sequence of 3 single component
 * frames, call gen_xpose(dp,0,3); to transform an interleaved RGB sequence
 * to a hypersequence of 3-frame sequences (a la HIPS2), two calls are needed:
 * gen_xpose(dp,0,4); gen_xpose(dp,3,4);
 */

#define EXCHANGE_DIMS(array,i1,i2)	EXCHANGE_ELTS(tmp_dim,array,i1,i2)
#define EXCHANGE_INCS(array,i1,i2)	EXCHANGE_ELTS(tmp_inc,array,i1,i2)

#define EXCHANGE_ELTS(tmpvar,array,i1,i2)		\
							\
	tmpvar = array[i1];				\
	array[i1] = array[i2];				\
	array[i2] = tmpvar;

void gen_xpose(Data_Obj *dp,int dim1,int dim2)
{
	dimension_t	tmp_dim;
	incr_t		tmp_inc;
#ifdef CAUTIOUS
	if( dim1 < 0 || dim1 >= N_DIMENSIONS ||
	    dim2 < 0 || dim2 >= N_DIMENSIONS ){
		NWARN("CAUTIOUS:  gen_xpose:  bad dimension index");
		return;
	}
#endif /* CAUTIOUS */
	EXCHANGE_DIMS(dp->dt_type_dim,dim1,dim2)
	EXCHANGE_DIMS(dp->dt_mach_dim,dim1,dim2)

	EXCHANGE_INCS(dp->dt_type_inc,dim1,dim2)
	EXCHANGE_INCS(dp->dt_mach_inc,dim1,dim2)

	/* should this be CAUTIOUS??? */ 
	if( set_shape_flags(&dp->dt_shape,dp,AUTO_SHAPE) < 0 )
		NWARN("gen_xpose:  RATS!?");

	check_contiguity(dp);
}

double get_dobj_il_flg(Item *ip)
{
	Data_Obj *dp;
	dp = (Data_Obj *) ip;

	if( INTERLACED_SHAPE(&dp->dt_shape) ) return(1.0);
	else return(0.0);
}

double get_dobj_size(Item *ip,int index)
{
	Data_Obj *dp;
	dp = (Data_Obj *)ip;
#ifdef CAUTIOUS
	if( dp == NO_OBJ )
		NERROR1("CAUTIOUS:  null dp in get_dobj_size()");
	if( index < 0 || index > N_DIMENSIONS )
		NERROR1("CAUTIOUS:  dimension index out of range");
#endif /* CAUTIOUS */


#ifdef FOOBAR
	if( index == 1 && BITMAP_SHAPE(&dp->dt_shape) )
		return( N_BITMAP_COLS(&dp->dt_shape) );
	else
#endif /* FOOBAR */
		return( (double) dp->dt_type_dim[index] );
}

static Item* i_d_subscript(Item *ip, index_t idx)
{
	return (Item *) d_subscript( DEFAULT_QSP_ARG   (Data_Obj *) ip, idx );
}

static Item* i_c_subscript(Item *ip, index_t idx)
{
	return (Item *) c_subscript( DEFAULT_QSP_ARG   (Data_Obj *) ip, idx );
}

static Size_Functions dobj_sf={
	/*(double (*)(Item *,int))*/	get_dobj_size,
	/*(Item * (*)(Item *,index_t))*/	i_d_subscript,
	/*(Item * (*)(Item *,index_t))*/	i_c_subscript,
	/*(double (*)(Item *))*/		get_dobj_il_flg
};

void dataobj_init(SINGLE_QSP_ARG_DECL)
{
	static int dobj_inited=0;

	if( dobj_inited ){
#ifdef CAUTIOUS
		/* We need to call this from other menus
		 * like fileio and rawvol
		 */
		/* warn("CAUTIOUS:  dataobj_init called more than once!?"); */
#endif /* CAUTIOUS */
		return;
	}

	debug_data = add_debug_module(QSP_ARG  "data");

	dobj_init(SINGLE_QSP_ARG);		/* initialize items */

	ram_area=area_init(QSP_ARG  "ram",NULL,0L,MAX_RAM_OBJS,DA_RAM);
	init_tmp_dps();

	set_del_method(QSP_ARG  dobj_itp,(void (*)(TMP_QSP_ARG_DECL  Item *))delvec);

	init_dfuncs();
	set_obj_funcs(get_obj,dobj_of,d_subscript,c_subscript);
	/* BUG need to make context items sizables too!? */
	add_sizable(QSP_ARG  dobj_itp,&dobj_sf,
		(Item * (*)(QSP_ARG_DECL  const char *))hunt_obj);

	/* set up additional port data type */
	define_port_data_type(QSP_ARG  P_DATA,"data","name of data object",
		(const char *(*)(QSP_ARG_DECL  Port *)) recv_obj,	// recv_obj returns Data_Obj *
		null_proc,
		(const char *(*)(QSP_ARG_DECL  const char *))pick_obj,
		(void (*)(QSP_ARG_DECL Port *,const void *,int)) xmit_obj
		);

	/* Version control */
	verdata(SINGLE_QSP_ARG);

	dobj_inited =1;
}

