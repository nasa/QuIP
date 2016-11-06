/**		data_obj.c	general data objects	*/

#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif /* HAVE_STRING_H */

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	/* free() */
#endif /* HAVE_STDLIB_H */

#include "quip_prot.h"
#include "data_obj.h"
#include "vectree.h"
#include "nexpr.h"		// set_obj_funcs
#include "nports_api.h"		// define_port_data_type
#include "debug.h"
#include "query_stack.h"	// like to eliminate this dependency BUG?

// Originally zombies were introduced to be able to refresh
// an X11 canvas displaying an image after the image had been
// "deleted."  Then it was ifdef'd out...  But now we discovered
// another need for deferring deletion, because we can have
// references to the object within expression language code.
#define ZOMBIE_SUPPORT


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

// free function for ram data area

void cpu_mem_free(QSP_ARG_DECL  Data_Obj *dp)
{
	givbuf(dp->dt_unaligned_ptr);
}

static void release_data(QSP_ARG_DECL  Data_Obj *dp )
{
	if( OBJ_DATA_PTR(dp) != (unsigned char *)NULL ){
//#ifdef CAUTIOUS
//		if( PF_FREE_FN( OBJ_PLATFORM(dp) ) == NULL ){
//			sprintf(ERROR_STRING,
//"CAUTIOUS:  release_data:  platform %s (object %s) has a null free function!?",
//				PLATFORM_NAME(OBJ_PLATFORM(dp)),OBJ_NAME(dp));
//			ERROR1(ERROR_STRING);
//			IOS_RETURN;
//		}
//#endif // CAUTIOUS
		assert( PF_FREE_FN( OBJ_PLATFORM(dp) ) != NULL );

		(* PF_FREE_FN( OBJ_PLATFORM(dp) ) )(QSP_ARG  dp);
	}

//#ifdef CAUTIOUS
	  else {
//	  	sprintf(ERROR_STRING,
//"CAUTIOUS:  release_data:  Object %s owns data but has a null data pointer!?",
//			OBJ_NAME(dp) );
//		WARN(ERROR_STRING);
//		assert( AERROR("Object owns data but has a null data ptr!?") );

		// This is normally an error, but this case can occur if we
		// are deleting an partially created object that had some other
		// initialization error...
	}
//#endif /* CAUTIOUS */

}	// release_data


/*
 *	Remove a child object from parent's list
 */

void disown_child( QSP_ARG_DECL  Data_Obj *dp )
{
	Node *np;

	np=remData(OBJ_CHILDREN( OBJ_PARENT(dp) ),dp);
	assert( np != NO_NODE );

#ifdef FOOBAR
//#ifdef CAUTIOUS
	if( np==NO_NODE ){
		NWARN("CAUTIOUS:  disown_child:  couldn't find child node");
		if( OBJ_PARENT(dp) == NO_OBJ ){
			ERROR1("object has no parent!?");
			IOS_RETURN
		}
		if( OBJ_CHILDREN( OBJ_PARENT(dp) ) == NO_LIST ){
			ERROR1("parent object has no children!?");
			IOS_RETURN
		} else {
			sprintf(ERROR_STRING,"Children of %s:",OBJ_NAME( OBJ_PARENT(dp) ) );
			advise(ERROR_STRING);
			np=QLIST_HEAD( OBJ_CHILDREN( OBJ_PARENT(dp) ) );
			while(np!=NO_NODE){
				sprintf(ERROR_STRING,"\t0x%lx",
					((int_for_addr)NODE_DATA(np) ));
				advise(ERROR_STRING);
				np=NODE_NEXT(np);
			}
		}
		ERROR1("giving up");
		IOS_RETURN
	} else
//#endif /* CAUTIOUS */
#endif // FOOBAR

	rls_node(np);

	if( eltcount(OBJ_CHILDREN( OBJ_PARENT(dp) )) == 0 ){
		rls_list(OBJ_CHILDREN( OBJ_PARENT(dp) ));	/* free list */
		OBJ_CHILDREN( OBJ_PARENT(dp) ) = NO_LIST;
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

	while( OBJ_CHILDREN( dp ) != NO_LIST ){
		np=QLIST_HEAD( OBJ_CHILDREN( dp ) );
		delvec( QSP_ARG  (Data_Obj *) NODE_DATA(np) );
	}
}

#ifdef ZOMBIE_SUPPORT
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

	sprintf(zname,"Z.%s.%d",OBJ_NAME(dp),n_zombie++);
	rls_str( (char *) OBJ_NAME(dp) );	/* unsave old name, make_zombie */
	SET_OBJ_NAME(dp,savestr(zname));

	SET_OBJ_FLAG_BITS(dp,DT_ZOMBIE);
} // make_zombie
#endif /* ZOMBIE_SUPPORT */

/*
 * Delete a data object.
 *
 * If this object has a non-zero reference count, (a viewer
 * needs it for refresh, or it's referenced in the expression
 * language), then we rename it and make it a zombie.
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

#ifdef ZOMBIE_SUPPORT
	// This should go back in eventually!
	if( OBJ_FLAGS(dp) & DT_STATIC && OWNS_DATA(dp) ){
//sprintf(ERROR_STRING,"delvec:  static object %s will be made a zombie",OBJ_NAME(dp));
//advise(ERROR_STRING);
		make_zombie(QSP_ARG  dp);
		return;
	}


	if( OBJ_REFCOUNT(dp) > 0 ){
		/* This zombie business was introduced at a time
		 * when a displayed image had to be kept around
		 * to refresh its window...  with the current
		 * X windows implementation of viewers that is
		 * no longer the case, so this may be unecessary...
		 *
		 * But another case arises when we have a static
		 * object in the expression language, that gets
		 * deleted outside of the expression language.
		 * This shouldn't be done, but we don't want to
		 * be able to crash the program either...
		 */

//sprintf(ERROR_STRING,"delvec:  object %s (refcount = %d) will be made a zombie",OBJ_NAME(dp),dp->dt_refcount);
//advise(ERROR_STRING);
		make_zombie(QSP_ARG  dp);
		return;
	}
#endif /* ZOMBIE_SUPPORT */

	// If the object has been exported, we need to delete
	// the associated identifier...
	//
	// BUT if it was exported, then it may be referenced!?
	// So it should be static...
	//
	// If we have references, and are therefore keeping the
	// object as a zombie, then we don't want to delete the
	// identifier, and we probably don't want to change the
	// object's name...

	if( IS_EXPORTED(dp) ){
		Identifier *idp;

		idp = ID_OF(OBJ_NAME(dp));
//#ifdef CAUTIOUS
//		if( idp == NO_IDENTIFIER ){
//			sprintf(ERROR_STRING,
//	"CAUTIOUS:  delvec:  No associated identifier found for exported object %s!?",
//				OBJ_NAME(dp));
//			ERROR1(ERROR_STRING);
//		}
//#endif // CAUTIOUS
		assert( idp != NO_IDENTIFIER );
		delete_id(QSP_ARG  (Item *)idp);
	}

	if( OBJ_CHILDREN( dp ) != NO_LIST ){
		del_subs(QSP_ARG  dp);
	}
	if( OBJ_PARENT(dp) != NO_OBJ ){
		disown_child(QSP_ARG  dp);
	}

	if( IS_TEMP(dp) ){

if( OBJ_DECLFILE(dp) != NULL ){
sprintf(ERROR_STRING,"delvec %s:  temp object has declfile %s!?\n",
OBJ_NAME(dp),OBJ_DECLFILE(dp));
advise(ERROR_STRING);
}
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

	// We might clean this up by making the release
	// function a platform member...

	if( OWNS_DATA(dp) ){
		if( ! UNKNOWN_SHAPE(OBJ_SHAPE(dp)) ){
			release_data(QSP_ARG  dp);
		}
	}
	// In the first OpenCL implementation, we used subbuffers, which had
	// to be released here even for subimages.  But now we handle subobjects
	// ourselves, managing offsets, so non-data-owners don't need to release.

	rls_shape( OBJ_SHAPE(dp) );

	// We don't need to do this if we have garbage collection?
	/* The name might be null if we had an error creating the object... */
	if( OBJ_DECLFILE(dp) != NULL ){
		rls_str( OBJ_DECLFILE(dp) );
	}

#ifdef ZOMBIE_SUPPORT
	/* BUG context code assumes that this is really deleted... */
	// not sure I understand the above comment?
	if( IS_ZOMBIE(dp) ){
		// The object is a zombie that is no longer referenced...

		/* NOTE:  we used to release the struct here with givbuf,
		 * but in the current implementation of the item package,
		 * objects aren't allocated with getbuf!
		 */

		// The object has already been removed from the dictionary,
		// so we don't need to call del_item...

		/* put this back on the free list... */
		recycle_item(dobj_itp,dp);
	} else {
		del_item(QSP_ARG  dobj_itp, dp );
	}
#else /* ! ZOMBIE_SUPPORT */

	//del_item(QSP_ARG  dobj_itp, dp );
	DELETE_OBJ_ITEM(dp);	// del_dobj - item function

#endif /* ! ZOMBIE_SUPPORT */

	// when we call this, bad things seem to happen, even
	// though it seems to be a leak!?
	rls_str( (char *) OBJ_NAME(dp) );		/* unsave stored name */
	SET_OBJ_NAME(dp,NULL);
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
		SET_SHP_MAXDIM(shpp,0);
		for(i=0;i<N_DIMENSIONS;i++){
			if( BITMAP_SHAPE(shpp) ){
				if( SHP_TYPE_DIM(shpp,i) > 1 ) SET_SHP_MAXDIM(shpp,i);
			} else {
				if( SHP_MACH_DIM(shpp,i) > 1 ) SET_SHP_MAXDIM(shpp,i);
			}
		}
	} else {
		if( shape_flag == DT_SCALAR )
			SET_SHP_MAXDIM(shpp, 0);
		else if( shape_flag == DT_ROWVEC )
			SET_SHP_MAXDIM(shpp, 1);
		else if( shape_flag == DT_COLVEC )
			SET_SHP_MAXDIM(shpp, 2);
		else if( shape_flag == DT_IMAGE )
			SET_SHP_MAXDIM(shpp, 2);
		else if( shape_flag == DT_SEQUENCE )
			SET_SHP_MAXDIM(shpp, 3);
		else if( shape_flag == DT_HYPER_SEQ )
			SET_SHP_MAXDIM(shpp, 4);
//#ifdef CAUTIOUS
		else {
			//NWARN("CAUTIOUS:  set_minmaxdim:  unexpected type flag!?");
			assert( AERROR("set_minmaxdim:  unexpected type flag!?") );
		}
//#endif /* CAUTIOUS */
	}

	/* set mindim */
	/* We used to use mach_dim here...  but for bitmaps we must use type dim */
	SET_SHP_MINDIM(shpp,N_DIMENSIONS-1);
	for(i=N_DIMENSIONS-1;i>=0;i--){
		if( BITMAP_SHAPE(shpp) ){
			if( SHP_TYPE_DIM(shpp,i) > 1 ) SET_SHP_MINDIM(shpp,i);
		} else {
			if( SHP_MACH_DIM(shpp,i) > 1 ) SET_SHP_MINDIM(shpp,i);
		}
	}

	// What is last_subi???
	/*SET_SHP_LAST_SUBI(shpp, SHP_MAXDIM(shpp) + 1 ); */
	SET_SHP_RANGE_MAXDIM(shpp,SHP_MAXDIM(shpp));
	SET_SHP_RANGE_MINDIM(shpp,SHP_MINDIM(shpp));
} // set_minmaxdim


int same_shape(Shape_Info *shpp1,Shape_Info *shpp2)
{
	int i;

	/* first check that the flags are the same */

	if( (SHP_FLAGS(shpp1) & SHAPE_MASK) != (SHP_FLAGS(shpp2) & SHAPE_MASK) )
		return(0);
/*
{
sprintf(ERROR_STRING,"same_shape:  masked flags are 0x%lx and 0x%lx",
SHP_FLAGS(shpp1)&SHAPE_MASK,SHP_FLAGS(shpp2)&SHAPE_MASK);
advise(ERROR_STRING);
		return(0);
}
*/

	/* Use type_dim here instead of mach_dim? */
	for(i=0;i<N_DIMENSIONS;i++){
		if( SHP_TYPE_DIM(shpp1,i) != SHP_TYPE_DIM(shpp2,i) ){
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
		SET_SHP_MAXDIM(shpp,0);
		SET_SHP_MINDIM(shpp,N_DIMENSIONS-1);

		// Should we initialize OBJ_RANGE_MAXDIM etc?
		SET_SHP_RANGE_MAXDIM(shpp,SHP_MAXDIM(shpp));
		SET_SHP_RANGE_MINDIM(shpp,SHP_MINDIM(shpp));

		return(0);
	}

	CLEAR_SHP_FLAG_BITS(shpp,SHAPE_DIM_MASK);

#ifdef CAUTIOUS
	for(i=0;i<N_DIMENSIONS;i++){
		assert( SHP_TYPE_DIM(shpp,i) > 0 );
//		if( SHP_TYPE_DIM(shpp,i) <= 0 ){
//			if( dp != NO_OBJ && OBJ_NAME(dp) != NULL )
//				sprintf(DEFAULT_ERROR_STRING,
//	"CAUTIOUS:  set_shape_flags:  Object \"%s\", zero dimension[%d] = %d",
//				OBJ_NAME(dp),i,SHP_TYPE_DIM(shpp,i));
//			else
//				sprintf(DEFAULT_ERROR_STRING,
//	"CAUTIOUS:  set_shape_flags:  zero dimension[%d] = %d",
//				i,SHP_TYPE_DIM(shpp,i));
//			NWARN(DEFAULT_ERROR_STRING);
//			return(-1);
//		}
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

	set_minmaxdim(shpp,shape_flag);		// set_shape_flags

	if( shape_flag == AUTO_SHAPE ){
		if( SHP_SEQS(shpp) > 1 )
			SET_SHP_FLAG_BITS(shpp,DT_HYPER_SEQ);
		else if( SHP_FRAMES(shpp) > 1 )
			SET_SHP_FLAG_BITS(shpp, DT_SEQUENCE);
		else if( SHP_ROWS(shpp) > 1 ){
			if( SHP_TYPE_DIM(shpp,1)==1 )
				SET_SHP_FLAG_BITS(shpp, DT_COLVEC);
			else
				SET_SHP_FLAG_BITS(shpp, DT_IMAGE);
		}
		else {
			dimension_t nc;

			nc=SHP_COLS(shpp);

			if( nc > 1 )
				SET_SHP_FLAG_BITS(shpp, DT_ROWVEC);
			else	SET_SHP_FLAG_BITS(shpp, DT_SCALAR);
		}
	} else {
//sprintf(ERROR_STRING,"setting shape flag bit to 0x%x",shape_flag);
//advise(ERROR_STRING);
		SET_SHP_FLAG_BITS(shpp,shape_flag);
	}


	CLEAR_SHP_FLAG_BITS(shpp, SHAPE_TYPE_MASK);
	if( COMPLEX_PRECISION(SHP_PREC(shpp)) ){
		SET_SHP_FLAG_BITS(shpp, DT_COMPLEX);
	} else if( QUAT_PRECISION(SHP_PREC(shpp)) ){
		SET_SHP_FLAG_BITS(shpp, DT_QUAT);
	} else {
		if( SHP_COMPS(shpp) != 1 ){
			SET_SHP_FLAG_BITS(shpp, DT_MULTIDIM);
		}
	}

	/* BUG?  should the string bit be part of shape dim mask??? */
	if( STRING_PRECISION(SHP_PREC(shpp)) ){
		SET_SHP_FLAG_BITS(shpp, DT_STRING);
	}

	if( CHAR_PRECISION(SHP_PREC(shpp)) ){
		SET_SHP_FLAG_BITS(shpp, DT_CHAR);
	}

	if( BITMAP_PRECISION(SHP_PREC(shpp)) )
		SET_SHP_FLAG_BITS(shpp, DT_BIT);

	return(0);
} /* end set_shape_flags() */

int auto_shape_flags(Shape_Info *shpp,Data_Obj *dp)
{
	return set_shape_flags(shpp,dp,AUTO_SHAPE);
}

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
	for(seq=0;seq<OBJ_SEQS(dp);seq++){
		f_os = s_os;
		for(frm=0;frm<OBJ_FRAMES(dp);frm++){
			r_os = f_os;
			for(row=0;row<OBJ_ROWS(dp);row++){
				p_os = r_os;
				for(col=0;col<OBJ_COLS(dp);col++){
					c_os = p_os;
					for(comp=0;comp<OBJ_COMPS(dp);comp++){
						(*func)(dp,c_os);
						c_os += OBJ_COMP_INC(dp);
					}
					p_os += OBJ_PXL_INC(dp);
				}
				r_os += OBJ_ROW_INC(dp);
			}
			f_os += OBJ_FRM_INC(dp);
		}
		s_os += OBJ_SEQ_INC(dp);
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
	for(seq=0;seq<OBJ_SEQS(dp1);seq++){
		f_os1 = s_os1;
		f_os2 = s_os2;
		for(frm=0;frm<OBJ_FRAMES(dp1);frm++){
			r_os1 = f_os1;
			r_os2 = f_os2;
			for(row=0;row<OBJ_ROWS(dp1);row++){
				p_os1 = r_os1;
				p_os2 = r_os2;
				for(col=0;col<OBJ_COLS(dp1);col++){
					c_os1 = p_os1;
					c_os2 = p_os2;
					for(comp=0;comp<OBJ_COMPS(dp1);comp++){
						(*func)(QSP_ARG  dp1,c_os1,dp2,c_os2);
						c_os1 += OBJ_COMP_INC(dp1);
						c_os2 += OBJ_COMP_INC(dp2);
					}
					p_os1 += OBJ_PXL_INC(dp1);
					p_os2 += OBJ_PXL_INC(dp2);
				}
				r_os1 += OBJ_ROW_INC(dp1);
				r_os2 += OBJ_ROW_INC(dp2);
			}
			f_os1 += OBJ_FRM_INC(dp1);
			f_os2 += OBJ_FRM_INC(dp2);
		}
		s_os1 += OBJ_SEQ_INC(dp1);
		s_os2 += OBJ_SEQ_INC(dp2);
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


#define EXCHANGE_DIMS(dsp,i1,i2)			\
	tmp_dim = DIMENSION(dsp,i1);			\
	SET_DIMENSION(dsp,i1,DIMENSION(dsp,i2));	\
	SET_DIMENSION(dsp,i2,tmp_dim);

#define EXCHANGE_INCS(isp,i1,i2)			\
	tmp_inc = INCREMENT(isp,i1);			\
	SET_INCREMENT(isp,i1,INCREMENT(isp,i2) );	\
	SET_INCREMENT(isp,i2,tmp_inc);


void gen_xpose(Data_Obj *dp,int dim1,int dim2)
{
	dimension_t	tmp_dim;
	incr_t		tmp_inc;
//#ifdef CAUTIOUS
//	if( dim1 < 0 || dim1 >= N_DIMENSIONS ||
//	    dim2 < 0 || dim2 >= N_DIMENSIONS ){
//		NWARN("CAUTIOUS:  gen_xpose:  bad dimension index");
//		return;
//	}
//#endif /* CAUTIOUS */

	assert( dim1 >= 0 && dim1 < N_DIMENSIONS );
	assert( dim2 >= 0 && dim2 < N_DIMENSIONS );

	EXCHANGE_DIMS(OBJ_TYPE_DIMS(dp),dim1,dim2)
	EXCHANGE_DIMS(OBJ_MACH_DIMS(dp),dim1,dim2)

	EXCHANGE_INCS(OBJ_TYPE_INCS(dp),dim1,dim2)
	EXCHANGE_INCS(OBJ_MACH_INCS(dp),dim1,dim2)

	/* should this be CAUTIOUS??? */ 
	if( set_shape_flags(OBJ_SHAPE(dp),dp,AUTO_SHAPE) < 0 )
		NWARN("gen_xpose:  RATS!?");

	check_contiguity(dp);
}

double get_dobj_il_flg(QSP_ARG_DECL  Data_Obj *dp)
{
	if( INTERLACED_SHAPE( OBJ_SHAPE(dp) ) ) return(1.0);
	else return(0.0);
}

const char *get_dobj_prec_name(QSP_ARG_DECL  Data_Obj *dp)
{
//#ifdef CAUTIOUS
//	if( dp == NO_OBJ ){
//		ERROR1("CAUTIOUS:  null dp in get_dobj_prec_name()");
//		IOS_RETURN_VAL(NULL)
//	}
//#endif /* CAUTIOUS */
	assert( dp != NO_OBJ );

	return OBJ_PREC_NAME(dp);
}

double get_dobj_size(QSP_ARG_DECL  Data_Obj *dp,int index)
{
//#ifdef CAUTIOUS
//	if( dp == NO_OBJ ){
//		ERROR1("CAUTIOUS:  null dp in get_dobj_size()");
//		IOS_RETURN_VAL(-1)
//	}
//	if( index < 0 || index > N_DIMENSIONS ){
//		ERROR1("CAUTIOUS:  dimension index out of range");
//		IOS_RETURN_VAL(-1)
//	}
//#endif /* CAUTIOUS */
	assert( dp != NO_OBJ );
	assert( index >= 0 && index < N_DIMENSIONS );

	return( (double) OBJ_TYPE_DIM(dp,index) );
}

// We give the position relative to the parent object...

static double get_dobj_posn(QSP_ARG_DECL  Item *ip, int index )
{
	double d=(-1);
	Data_Obj *dp;
	index_t pix_offset;
	int i;
	index_t offsets[N_DIMENSIONS];
	Data_Obj *parent;

	dp = (Data_Obj *)ip;

	if( dp == NO_OBJ ) return 0.0;
	parent = OBJ_PARENT(dp);
	if( parent == NO_OBJ ) return 0.0;

	// The position is relative to the parent
	//
	// The position offsets are not stored when we create
	// a subimage, they are used to create a data ptr offset.
	// We use this offset together with the parent's shape
	// to convert back to coordinates...

	pix_offset = OBJ_OFFSET( dp );

	for(i=N_DIMENSIONS-1;i>=0;i--){
		// MACH_INC is 0 when the corresponding dimension is 1
		if( OBJ_MACH_INC(parent,i) == 0 ){
			offsets[i] = 0;
		} else {
			offsets[i] = pix_offset / OBJ_MACH_INC(parent,i);
			pix_offset -= offsets[i] * OBJ_MACH_INC(parent,i);
		}
	}
//fprintf(stderr,"get_dobj_posn:  offsets = %d %d %d %d %d\n",
//offsets[0],offsets[1],offsets[2],offsets[3],offsets[4]);

	switch(index){
		case 0: d = offsets[1]; break;
		case 1: d = offsets[2]; break;
//#ifdef CAUTIOUS
		default:
//			ERROR1("CAUTIOUS:  get_dobj_posn:  bad index!?");
			assert( AERROR("get_dobj_posn:  bad index!?") );
			break;
//#endif // CAUTIOUS
	}
	return(d);
}

static Size_Functions dobj_sf={
	(double (*)(QSP_ARG_DECL  Item *,int))		get_dobj_size,
	(const char * (*)(QSP_ARG_DECL  Item *))	get_dobj_prec_name
};

static Interlace_Functions dobj_if={
	(double (*)(QSP_ARG_DECL  Item *))		get_dobj_il_flg
};

static Position_Functions dobj_pf={
	get_dobj_posn
};

static Subscript_Functions dobj_ssf={
	(Item * (*)(QSP_ARG_DECL  Item *,index_t))	d_subscript,
	(Item * (*)(QSP_ARG_DECL  Item *,index_t))	c_subscript
};

void dataobj_init(SINGLE_QSP_ARG_DECL)
{
	static int dobj_inited=0;

	if( dobj_inited ){
		/* We need to call this from other menus
		 * like fileio and rawvol, so no assertion here.
		 */
		return;
	}

	debug_data = add_debug_module(QSP_ARG  "data");

	INSURE_QS_DOBJ_ASCII_INFO(THIS_QSP)
	init_dobj_ascii_info(QSP_ARG  QS_DOBJ_ASCII_INFO(THIS_QSP) );
    
	init_dobjs(SINGLE_QSP_ARG);		/* initialize items */

	// update to use platforms...
	//ram_area_p=area_init(QSP_ARG  "ram",NULL,0L,MAX_RAM_CHUNKS,DA_RAM);
	vl2_init_platform(SINGLE_QSP_ARG);	// this initializes ram_area_p

	init_tmp_dps(SINGLE_QSP_ARG);

	set_del_method(QSP_ARG  dobj_itp,(void (*)(QSP_ARG_DECL  Item *))delvec);

	init_dfuncs(SINGLE_QSP_ARG);

	set_obj_funcs(
                  get_obj,
                  dobj_of,
                  d_subscript,
                  c_subscript);
    
	init_dobj_expr_funcs(SINGLE_QSP_ARG);

//	/* BUG need to make context items sizables too!? */
	add_sizable(QSP_ARG  dobj_itp,&dobj_sf,
		(Item * (*)(QSP_ARG_DECL  const char *))hunt_obj);
	add_positionable(QSP_ARG  dobj_itp,&dobj_pf,
		(Item * (*)(QSP_ARG_DECL  const char *))hunt_obj);
	add_interlaceable(QSP_ARG  dobj_itp,&dobj_if,
		(Item * (*)(QSP_ARG_DECL  const char *))hunt_obj);
	add_subscriptable(QSP_ARG  dobj_itp,&dobj_ssf,
		(Item * (*)(QSP_ARG_DECL  const char *))hunt_obj);

	// This was commented out - why?
	// Maybe because initial iOS implementation didn't include ports menu?

	/* set up additional port data type */

	define_port_data_type(QSP_ARG  P_DATA,"data","name of data object",
		recv_obj,
		/* null_proc, */
		(const char *(*)(QSP_ARG_DECL  const char *))pick_obj,
		(void (*)(QSP_ARG_DECL Port *,const void *,int)) xmit_obj
		);

	/* Version control */
//	verdata(SINGLE_QSP_ARG);

	dobj_inited =1;
}

void xfer_dobj_flag(Data_Obj *dpto, Data_Obj *dpfr, uint32_t flagbit)
{
	if( OBJ_FLAGS(dpfr) & flagbit ){
		SET_OBJ_FLAG_BITS(dpto, flagbit);
	} else {
		CLEAR_OBJ_FLAG_BITS(dpto, flagbit);
	}
}


static void propagate_up(Data_Obj *dp,uint32_t flagbit)
{
	if( dp->dt_parent != NO_OBJ ){
		xfer_dobj_flag(dp->dt_parent,dp,flagbit);
		propagate_up(dp->dt_parent,flagbit);
		// Do we need to propagate down from parent,
		// to get the siblings?
		// That could lead to an infinite regress if
		// we don't somehow mark the objects...
		// But then we would have to clear the marks!?
	}
}

static void propagate_down(Data_Obj *dp,uint32_t flagbit)
{
	Node *np;

	if( dp->dt_children != NO_LIST ){
		np=dp->dt_children->l_head;
		while(np!=NO_NODE){
			Data_Obj *child_dp;
			child_dp = (Data_Obj *)np->n_data;
			xfer_dobj_flag(child_dp,dp,flagbit);
			propagate_down(child_dp,flagbit);
			np = np->n_next;
		}
	}
}

void propagate_flag(Data_Obj *dp,uint32_t flagbit)
{
	propagate_up(dp,flagbit);
	propagate_down(dp,flagbit);
}

