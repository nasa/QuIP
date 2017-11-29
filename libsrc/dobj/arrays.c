#include "quip_config.h"

#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "quip_prot.h"
#include "data_obj.h"
#include "debug.h"
#include "platform.h"
#include "dobj_private.h"

/* Sbscripted objects are implemented from a fixed-size pool of temporary
 * objects.  The thinking is that these will be used in an expression
 * of finited depth, and once the operation has completed they are no
 * longer required to persist.
 *
 * BUG We hate to have fixed-size arrays...
 *
 * Will this mechanism break for multi-threaded code?
 * Only the list heads are global, they should not change (except
 * when initialized - should we lock then?), and the list manipulation
 * routines should be thread-safe...
 */

#define N_TMP_DP	24

/* 0 for C style indexing, 1 for fortran, matlab */
static index_t base_index=0;

static List *free_tmpobj_lp=NULL;
static List *used_tmpobj_lp=NULL;

void _init_tmp_dps(SINGLE_QSP_ARG_DECL)
{
	/* this doesn't work, because it gets called too often??? */

	add_cmd_callback( QSP_ARG  _unlock_all_tmp_objs );
}

void _list_temp_dps(QSP_ARG_DECL  FILE *fp)
{
	Node *np;
	int n, nl, nc, nr;

	if( used_tmpobj_lp == NULL ){
		advise("no temp objects");
		return;
	}
	np=QLIST_HEAD(used_tmpobj_lp);
	n=nl=nc=nr=0;
	while(np!=NULL){
		Data_Obj *dp;
		dp=(Data_Obj *)NODE_DATA(np);
		if( DOBJ_IS_LOCKED(dp) ) nl++;
		if( OBJ_CHILDREN(dp) != NULL ) nc++;
		if( OBJ_REFCOUNT(dp) > 0 ) nr++;
		n++;
		np=NODE_NEXT(np);
	}
	sprintf(MSG_STR,
	"%d temporary objects, %d locked, %d w/ children, %d referenced",
		n,nl,nc,nr);
	fprintf(fp,"%s\n",MSG_STR);
}

/*
 * temp_replica - find an unused temporery obj struct, and copy the source into it if possible
 *
 * Return an object structure to use as an array element.
 * These are allocated from a static pool of N_TMP_DP objects.
 * The global var freedp is the index of the one last used,
 * we cycle until we find a free one.
 * A temporary object will not be recycled if:
 *    it has children.
 *    it is the one being replicated! (more than 1 subscript).
 *    non-zero refcount (displayed in viewer)
 *
 * This scheme relies on the fact
 * that these objects are not needed for long (BUG?)
 *
 * A very subtle bug was discovered, difficult to expose:
 * bad things happened when two temp objects occur in the same
 * command line, and the first is freed to make room for the second!?
 * A quick fix might be to make up a list of all available
 * for deletion, and pick the one which was least-recently referenced...
 * this might fix the immediate problem, but doesn't really fix the
 * bug...  the real problem is, we don't know when we are really
 * done!?  This led to the introduction of a LOCKED flag,
 * which is cleared when a new command is fetched.
 *
 * With the introduction of the expression parser, we discover that we need to
 * do this more frequently.  Once per statement is conservative, but probably
 * safe.
 *
 * The original implementation went through the whole table,
 * but this ended up consuming a significant amt of processor time to
 * clear the flag bits of over 100 entries...  Hopefully a linked list
 * will perform better.
 *
 * A separate but related issue is that we may wish to keep around a pointer
 * to an image component (e.g.  rgb{0}).  In this case, we want to keep the thing
 * locked for many commands.  To handle this, we introduce *another* flag, VOLATILE,
 * and we only unlock temp objects when this is set.
 *
 */

#define temp_replica(dp) _temp_replica(QSP_ARG  dp)

static Data_Obj * _temp_replica(QSP_ARG_DECL  Data_Obj *dp)
{
	Data_Obj *newdp;

	newdp = find_free_temp_dp(dp);

//	memcpy( newdp, dp, sizeof(*dp) );

	DOBJ_COPY(newdp,dp);
	SET_OBJ_DECLFILE(newdp,NULL);	// parent is ok

	// What about the shape??
	SET_OBJ_SHAPE(newdp,ALLOC_SHAPE);

	COPY_SHAPE(OBJ_SHAPE(newdp),OBJ_SHAPE(dp));	

	return( newdp );
}

Data_Obj * _temp_scalar(QSP_ARG_DECL  const char *s, Precision *prec_p)
{
	Data_Obj *dp, *newdp;

	dp = mk_scalar(s,prec_p);
	newdp=temp_replica(dp);
	delvec(dp);
	SET_OBJ_NAME(newdp,savestr(s));
	return newdp;
}

/* release_tmpobj_resources
 *
 * returns -1 on failure, 0 for success
 */

static int release_tmpobj_resources(Data_Obj *dp, Data_Obj *parent_dp)
{
	/* a used data_obj should have its name set */
	assert( OBJ_NAME(dp) != NULL );

	if( OBJ_CHILDREN(dp) != NULL ){
		return(-1);	/* failure */
	}
	if( dp == parent_dp ){
		return(-1);	/* failure */
	}
	if( DOBJ_IS_LOCKED(dp) ){
		return(-1);	/* failure */
	}
	if( OBJ_REFCOUNT(dp) > 0 ){
		return(-1);	/* failure */
	}

	/* If we are called from delvec, then disown
	 * child has already been called!?
	 *
	 * So we don't do it here any more...
	 */

	/* release the allocated shape stuff to prevent a memory leak */

	rls_shape(OBJ_SHAPE(dp));

	rls_str((char *)OBJ_NAME(dp));	// release_tmpobj_resources
	SET_OBJ_NAME(dp, NULL);	/* may not be needed,
				 * but helps us catch other bugs...
				 */

	return(0);
} // release_tmpobj_resources

static void add_tmpobjs(List *lp, int n)
{
	Data_Obj *dp;
	int i;

	for(i=0;i<n;i++){
		Node *np;
		NEW_DATA_OBJ(dp);
		// allocates but does not zero???
		np = mk_node(dp);
		addTail(lp,np);
	}
}

static void insure_tmpobj_free_list(void)
{
	if( free_tmpobj_lp == NULL ){
		free_tmpobj_lp = new_list();
		add_tmpobjs(free_tmpobj_lp,N_TMP_DP);
	}
}

static Data_Obj *check_tmpobj_free_list(void)
{
	Node *np;

	insure_tmpobj_free_list();

	if( eltcount(free_tmpobj_lp) == 0 ) return NULL;

	np = remHead(free_tmpobj_lp);

	if( used_tmpobj_lp == NULL )
		used_tmpobj_lp = new_list();
	addHead(used_tmpobj_lp,np);

	return((Data_Obj *)NODE_DATA(np));
}

#define recycle_used_tmpobj(dp) _recycle_used_tmpobj(QSP_ARG  dp)

static Data_Obj *_recycle_used_tmpobj(QSP_ARG_DECL  Data_Obj *dp)
{
	Node *np,*first_np;
	Data_Obj *new_dp;

	np = remTail(used_tmpobj_lp);
	first_np = np;
	new_dp = (Data_Obj *)NODE_DATA(np);
	while( release_tmpobj_resources(new_dp,dp) < 0 ){
		addHead(used_tmpobj_lp,np);
		np=remTail(used_tmpobj_lp);
		if( np == first_np )
			NERROR1("out of temporary objects!?");
		new_dp = (Data_Obj *)NODE_DATA(np);
	}
	/* The deallocate routine has freed the old name etc */
	disown_child(new_dp);

	/* put this one at the head of the used list... */
	addHead(used_tmpobj_lp,np);
	return(new_dp);
}

Data_Obj * _find_free_temp_dp(QSP_ARG_DECL  Data_Obj *dp)
{
	Data_Obj *new_dp;

	new_dp = check_tmpobj_free_list();
	if( new_dp != NULL ) return new_dp;

	/* They are all in use, we have to release a used one... */
	return recycle_used_tmpobj(dp);
}


/* Create a child object with the given name.
 * These names are not hashed, and the objects are initially locked.
 */

Data_Obj *_temp_child( QSP_ARG_DECL  const char *name, Data_Obj *dp )
{
	Data_Obj *newdp;
	newdp=temp_replica(dp);

#ifdef QUIP_DEBUG
/*
if( debug & debug_data ){
sprintf(ERROR_STRING,"saving child name \"%s\"",name);
advise(ERROR_STRING);
}
*/
#endif /* QUIP_DEBUG */

	SET_OBJ_NAME(newdp, savestr(name));

	/* not added to item table!!! */

	/*
	 * Originally not added to parent's children list.
	 * They were acknowledged as children so subimage relocation
	 * would work; but now when a temp object is recycled, we
	 * have to delete references from the parent!
	 */
	parent_relationship(dp,newdp);

	/*
	 * The parent's refcount may be set if it is displayed
	 * in a viewer, but we don't care about its children!
	 */

	SET_OBJ_REFCOUNT(newdp,0);

#ifdef QUIP_DEBUG
if( debug & debug_data ){
printf("locking temp object %s\n",OBJ_NAME(newdp));
}
#endif /* QUIP_DEBUG */

	// temp object begin life as locked and volatile...
	// "volatile" means that we can unlock at end of command cycle.

	SET_OBJ_FLAG_BITS(newdp, DT_TEMP | DT_LOCKED | DT_VOLATILE ) ;
	/* may be contiguous etc, but we have to check... */
	CLEAR_OBJ_FLAG_BITS(newdp, DT_CONTIG | DT_EVENLY | DT_CHECKED) ;

	return(newdp);
}

/* 
 * unlock_all_tmp_objs - the basic idea is that temp objects
 * only need to persist for the duration of one command line..
 * After the command is finished we can call this, which unlocks
 * but doesn't delete the objects.
 * 
 * when should we call this?  After all references to new objects
 * have been made and finished...
 *
 * This comment seems to be out-of-date:
 * There is a problem, though, if we release many of these, then
 * we can encounter a situation in which we assign a[0], then it
 * gets released, then we use a[0] on the RHS, and get a warning
 * about used before initialized...  One way to do this would be to
 * sort the list to be reused based on time of last access...
 * This could get rather slow!?  On the other hand, we can't really keep
 * around data_obj structs for every pixel...  Maybe we can
 * introduce a "lock" function?
 *
 * Should temporary objects be associated with a particular qsp?  I think so!
 * This function also does not appear to be thread-safe, because the list
 * should be locked from other threads while the flag bits are being diddled.
 * Actually, no harm is done if another process skips over an object that
 * is being unlocked, but when an object is locked (where?), then THAT has
 * to be thread-safe!!  BUG
 */

void _unlock_all_tmp_objs(SINGLE_QSP_ARG_DECL)
{
	Node *np;
	Data_Obj *dp;
	if( used_tmpobj_lp == NULL ){
		return;
	}

	np = QLIST_HEAD(used_tmpobj_lp);
	while(np!=NULL){
		dp=(Data_Obj *)NODE_DATA(np);
		if( OBJ_FLAGS(dp) & DT_VOLATILE ){
			CLEAR_OBJ_FLAG_BITS(dp, DT_LOCKED);
		}
		np = NODE_NEXT(np);
	}
}

// unlock_children was introduced to fix a memory leak caused by x display
// creating an image of different depth for display, copying data, displaying
// then deleting.  The subscripted components were still locked at delete time!?
//

void unlock_children(Data_Obj *dp)
{
	if( OBJ_CHILDREN(dp) != NULL ){
		Node *np;

		np = QLIST_HEAD( OBJ_CHILDREN(dp) );
		while( np != NULL ){
			Data_Obj *child_dp;

			child_dp = (Data_Obj *) NODE_DATA(np);
			unlock_children(child_dp);

			np = NODE_NEXT(np);
		}
	}
	SET_OBJ_FLAGS( dp, OBJ_FLAGS(dp) & ~DT_LOCKED );
}

/*
 * gen_subscript() subscripts the requested dimension.  The resulting data object
 * can be named two ways, delim tells us which to use.
 *
 *
 * which_dim
 *	0			component	color_pixel{i}
 *
 *						pixel[i]
 *						row[*][i]
 *						image[*][*][i]
 *						movie[*][*][*][i]
 *
 *	1			pixel (column)	color_row{*}{i}
 *						bw_row{i}
 *
 *						row[i]
 *						image[*][i]
 *						movie[*][*][i]
 *
 *	2			row		color_image{*}{*}{i}
 *						bw_image{*}{i}
 *
 *						image[i]
 *						movie[*][i]
 *
 *	3			frame		color_movie{*}{*}{*}{i}
 *						bw_movie{*}{*}{i}
 *
 *						movie[i]
 *
 *	We use the field dt_maxdim to remember the largest index of a dimension>1 
 *
 *	pixel			maxdim = 0
 *	row			max_dim = 1
 *	image			max_dim = 2
 *	sequence		max_dim = 3
 *
 *	For square brace subscripting, the number of brackets with *'s is
 *	equal to (max_dim-which_dim).
 *
 *	We use the field dt_mindim to remember the smallest index of a dimension>1 
 *
 *	color_pixel		min_dim = 0
 *	bw_pixel		min_dim = 1
 *	bw_column_vector	min_dim = 2
 *
 *	For curly bracket subscripting, the number of brackets with *'s is
 *	equal to (which_dim-min_dim)
 */

#define MAX_INDEX_STRING_LEN	32	// how many digits can actually print???

void _make_array_name( QSP_ARG_DECL  char *target_str, int buflen, Data_Obj *dp, index_t index, int which_dim, int subscr_type )
{
	int left_delim, right_delim, nstars;
	int i;
	char index_string[MAX_INDEX_STRING_LEN];

	if( subscr_type == SQUARE ){
		left_delim  = '[';
		right_delim = ']';
		nstars = OBJ_MAXDIM(dp) - which_dim;
	} else if( subscr_type == CURLY ){
		left_delim  = '{';
		right_delim = '}';
		nstars = which_dim - OBJ_MINDIM(dp);
	}
	else {
		assert( AERROR("Bad subscript type!?") );
	}

	/* now make the name */
	strcpy(target_str,OBJ_NAME(dp));
	for(i=0;i<nstars;i++){
		sprintf(index_string,"%c*%c",left_delim,right_delim);
		strcat(target_str,index_string);
	}
	sprintf(index_string,"%c%d%c",left_delim,index,right_delim);
	strcat(target_str,index_string);
}

static Node *existing_tmpobj_node(const char *name)
{
	Node *np;
	Data_Obj *dp;

	if( used_tmpobj_lp == NULL ) return(NULL);
	np=QLIST_HEAD(used_tmpobj_lp);
	while(np!=NULL){
		dp = (Data_Obj *)NODE_DATA(np);
		if( OBJ_NAME(dp)==NULL ) {
			NERROR1("existing_tmpobj_node:  null object!?");
			return NULL; // NOTREACHED - silence static analyzer
		}
		if( !strcmp(OBJ_NAME(dp),name) )
			return(np);
		np=NODE_NEXT(np);
	}
	return(NULL);
}

#define MAX_OBJ_NAME_LEN	128

Data_Obj *_gen_subscript( QSP_ARG_DECL  Data_Obj *dp, int which_dim, index_t index, int subscr_type )
{
	Data_Obj *newdp=NULL;
	Node *np;
	char name_with_subscript[MAX_OBJ_NAME_LEN];	/* BUG how long can this be really? */
	int i;

	if( dp==NULL ) return(dp);

fprintf(stderr,"gen_subscript %s %d %d BEGIN\n",OBJ_NAME(dp),which_dim,index);
	if( which_dim < 0 || which_dim >= N_DIMENSIONS ){
		warn("gen_subscript:  dimension index out of range");
		return(NULL);
	}
	/* We used to disallow subscripting dimensions with only one element,
	 * but we use mindim and maxdim to keep track of that...
	 * As long as the subscript value itself is legal...
	 */

	if( index < base_index ){
		sprintf(ERROR_STRING,
		"%s subscript too small (%d, min %u) for object %s",
			dimension_name[which_dim],index,
			base_index,OBJ_NAME(dp));
		warn(ERROR_STRING);
		return(NULL);
	}

	/* We test against mach_dim so that we can subscript complex and color
	 * objects...  but for bitmaps we need to use type_dim!
	 */

	if( ! IS_BITMAP(dp) ){
		/* indices are unsigned ! */
		if( index-base_index >= OBJ_MACH_DIM(dp,which_dim) ){
			sprintf(ERROR_STRING,
			"%s subscript too large (%u, max %u) for object %s",
				dimension_name[which_dim],index,
				OBJ_MACH_DIM(dp,which_dim)+base_index-1,OBJ_NAME(dp));
			warn(ERROR_STRING);
			return(NULL);
		}
	} else {	/* bitmap */
		if( index-base_index >= OBJ_TYPE_DIM(dp,which_dim) ){
			sprintf(ERROR_STRING,
			"%s subscript too large (%u, max %u) for object %s",
				dimension_name[which_dim],index,
				OBJ_MACH_DIM(dp,which_dim)+base_index-1,OBJ_NAME(dp));
			warn(ERROR_STRING);
			return(NULL);
		}
	}

	make_array_name(name_with_subscript,MAX_OBJ_NAME_LEN,dp,index,which_dim,subscr_type);

	/* first see if this subobject may exist!? */
	/* these names aren't hashed, so just scan the list */
	/* maybe we would be better off with a linked list?? */

	np = existing_tmpobj_node(name_with_subscript);
	if( np != NULL ){
		/* before returning, move this node up the the head of the list.
		 *
		 * The point is to keep the list priority sorted, where
		 * priority is based on the recency of the access...
		 */
		if( np != QLIST_HEAD(used_tmpobj_lp) ){
			np=remNode(used_tmpobj_lp,np);
			/* do we need to check the return value?? */
			addHead(used_tmpobj_lp,np);
		}
		return((Data_Obj *)NODE_DATA(np));
	}

	/* the object doesn't exist, so create it */

	newdp=temp_child(name_with_subscript,dp);
	SET_OBJ_MACH_DIM(newdp,which_dim, 1 );
	SET_OBJ_MACH_INC(newdp,which_dim, 0 );
	SET_OBJ_TYPE_DIM(newdp,which_dim, 1 );
	SET_OBJ_TYPE_INC(newdp,which_dim, 0 );

	index -= base_index;

	if( ! IS_BITMAP(dp) || which_dim > 1 ){
	} else {
		// We're indexing bits, do nothing

	}
	if( IS_BITMAP(newdp) && which_dim <= 1 ){
		SET_OBJ_BIT0(newdp, OBJ_BIT0(newdp) + index % BITS_PER_BITMAP_WORD );
		if( OBJ_BIT0(newdp) >= BITS_PER_BITMAP_WORD ){
			SET_OBJ_BIT0(newdp, OBJ_BIT0(newdp) - BITS_PER_BITMAP_WORD );
			index += BITS_PER_BITMAP_WORD;
		}
		SET_OBJ_OFFSET(newdp, sizeof(BITMAP_DATA_TYPE)*(index/BITS_PER_BITMAP_WORD) );
	} else {
		index *= OBJ_MACH_INC(dp,which_dim);
		SET_OBJ_OFFSET(newdp, index);		/* offset is in bytes! */
	}
	// We used to update the data ptr here, but for OpenCL we need to do
	// something different if the object is complex, so we moved it to after
	// the code below hwere we update the flags...
	/*
	if( OBJ_DATA_PTR(newdp) != NULL )
		( * PF_OFFSET_DATA_FN(OBJ_PLATFORM(dp)) ) (QSP_ARG  newdp,
								OBJ_OFFSET(newdp) );
								*/

	/* dt_n_mach_elts is the total number of component elements */
	SET_OBJ_N_MACH_ELTS(newdp, OBJ_N_MACH_ELTS(newdp)/OBJ_MACH_DIM(dp,which_dim) );
	/* dt_n_type_elts is the total number of elements, where a complex number is counted as 1 */
	SET_OBJ_N_TYPE_ELTS(newdp, OBJ_N_TYPE_ELTS(newdp)/OBJ_TYPE_DIM(dp,which_dim) );

	if( SET_OBJ_SHAPE_FLAGS(newdp) < 0 )
		warn("holy mole batman!?");

	check_contiguity(newdp);

	if( IS_COMPLEX( OBJ_PARENT(newdp) ) && OBJ_MACH_DIM(newdp,0)!=2 ){
		//CLEAR_OBJ_PREC_BITS(newdp, COMPLEX_PREC_BITS);
		SET_OBJ_PREC_PTR( newdp, OBJ_MACH_PREC_PTR(OBJ_PARENT(newdp)) );
		CLEAR_OBJ_FLAG_BITS(newdp, DT_COMPLEX);
		CLEAR_OBJ_FLAG_BITS(newdp, DT_CONTIG);
		for(i=0;i<N_DIMENSIONS;i++)
			SET_OBJ_TYPE_INC(newdp,i, OBJ_MACH_INC(newdp,i) );
	}
	if( IS_QUAT( OBJ_PARENT(newdp) ) && OBJ_MACH_DIM(newdp,0) !=4 ){
		//CLEAR_OBJ_PREC_BITS(newdp,QUAT_PREC_BITS);
		SET_OBJ_PREC_PTR( newdp, OBJ_MACH_PREC_PTR(OBJ_PARENT(newdp)) );
		CLEAR_OBJ_FLAG_BITS(newdp, DT_QUAT|DT_CONTIG );
		for(i=0;i<N_DIMENSIONS;i++)
			SET_OBJ_TYPE_INC(newdp,i, OBJ_MACH_INC(newdp,i) );
	}

	/* now change mindim/maxdim! */
	/*
	if( subscr_type == SQUARE )
		newdp->dt_maxdim -- ;
	else if( subscr_type == CURLY )
		newdp->dt_mindim ++ ;
		*/

	if( OBJ_DATA_PTR(newdp) != NULL )
		( * PF_OFFSET_DATA_FN(OBJ_PLATFORM(dp)) ) (QSP_ARG  newdp,
								OBJ_OFFSET(newdp) );

	return(newdp);
}

void release_tmp_obj(Data_Obj *dp)
{
	Node *np;

	np = remData(used_tmpobj_lp,dp);
	assert( np != NULL );

	release_tmpobj_resources(dp,NULL);
	addHead(free_tmpobj_lp,np);
}

/*
 * reset the data pointer.
 *
 * The dimensions and increments are unchanged, and for speed we won't
 * change the name...  Hopefully this is not a BUG?
 */

void _reindex( QSP_ARG_DECL  Data_Obj *dp, int which_dim, index_t index )
{
#ifdef QUIP_DEBUG
if( debug & debug_data ){
	char str[MAX_OBJ_NAME_LEN];

	/* BUG don't necessarily know it's square!? */
	/* But this is usually called from dp_vectorize(), then it is... */
	/* the name isn't always right here, but it is useful for debugging... */

	make_array_name(str,MAX_OBJ_NAME_LEN, OBJ_PARENT(dp) ,index,which_dim,SQUARE);
	rls_str((char *)OBJ_NAME(dp));	// reindex
	SET_OBJ_NAME(dp,savestr(str));
}
#endif /* QUIP_DEBUG */
	index -= base_index;
	index *= OBJ_MACH_INC( OBJ_PARENT(dp) ,which_dim) ;
	SET_OBJ_OFFSET(dp, index);
	/* why a check for non-nul ptr above? */
	( * PF_OFFSET_DATA_FN(OBJ_PLATFORM(OBJ_PARENT(dp))) ) (QSP_ARG  dp, index );
}

/* reduce the dimensionality of dp */

Data_Obj *_reduce_from_end( QSP_ARG_DECL  Data_Obj *dp, index_t index, int subscr_type )
{
	Data_Obj *newdp=NULL;
	int dim;

	if( dp==NULL ) return(dp);

	assert( subscr_type == SQUARE || subscr_type == CURLY );

	if( subscr_type == SQUARE )     dim=OBJ_MAXDIM(dp);
	else		/* CURLY */	dim=OBJ_MINDIM(dp);

	newdp =  gen_subscript(dp,dim,index,subscr_type);

	return(newdp);
}

/*
 * sequence -> frame
 * frame -> row
 * row -> point
 * point -> component
 */

Data_Obj *_d_subscript( QSP_ARG_DECL  Data_Obj *dp, index_t index )
{
	/* find the highest dimension != 1 */
	return( reduce_from_end(dp,index,SQUARE) );
}

/* this kind of subscripting works from the other end;
 * whereas image[0] is the first row, image{0} is the first column
 * if image is complex, then image{0} is the real image
 * and image{1} is the imaginary image
 */

Data_Obj *_c_subscript( QSP_ARG_DECL  Data_Obj *dp, index_t index )
{
	return( reduce_from_end(dp,index,CURLY) );
}

int is_in_string(int c,const char *s)
{
	while( *s )
		if( c == *s++ ) return(1);
	return(0);
}

// My guess is that the purpose of this is to support 1-based indexing for matlab...

void _set_array_base_index(QSP_ARG_DECL  int index)
{
	if( index < 0 || index > 1 ){
		sprintf(ERROR_STRING,
	"set_base_index:  requested base index (%d) is out of range (0-1)",
			index);
		warn(ERROR_STRING);
		return;
	}

	base_index = index;
}

