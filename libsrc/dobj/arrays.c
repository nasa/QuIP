#include "quip_config.h"

char VersionId_dataf_arrays[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "data_obj.h"
#include "debug.h"
#include "savestr.h"

/* BUG We hate to have fixed-size arrays...  Also, this mechanism will break
 * for multi-threaded code...
 */

#define N_TMP_DP	128

//static int freedp=0;
static index_t base_index=0;	/* 0 for C style indexing, 1 for fortran, matlab */

static List *free_tmpobj_lp=NO_LIST;
static List *used_tmpobj_lp=NO_LIST;

/* local prototypes */
static Data_Obj * temp_replica(Data_Obj *);

const char *dimension_name[N_DIMENSIONS]={
	"component",
	"column",
	"row",
	"frame",
	"sequence"
};

/* 
 * we take this out on systems without yacc, since we often use expressions
 * for the index
 */

#define PARSE_INDEX

void init_tmp_dps()
{
	/* this doesn't work, because it gets called too often??? */

	add_cmd_callback( unlock_all_tmp_objs );
}

void list_temp_dps()
{
	Node *np;
	int n, nl, nc, nr;

	if( used_tmpobj_lp == NO_LIST ){
		advise("no temp objects");
		return;
	}
	np=used_tmpobj_lp->l_head;
	n=nl=nc=nr=0;
	while(np!=NO_NODE){
		Data_Obj *dp;
		dp=(Data_Obj *)np->n_data;
		if( DOBJ_IS_LOCKED(dp) ) nl++;
		if( dp->dt_children != NO_LIST ) nc++;
		if( dp->dt_refcount > 0 ) nr++;
		n++;
		np=np->n_next;
	}
	sprintf(msg_str,
	"%d temporary objects, %d locked, %d w/ children, %d referenced",
		n,nl,nc,nr);
	prt_msg(msg_str);
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
 */

static Data_Obj * temp_replica(Data_Obj *dp)
{
	Data_Obj *newdp;

	newdp = find_free_temp_dp(dp);

	memcpy( newdp, dp, sizeof(*dp) );

	return( newdp );
}

static int release_tmpobj_resources(Data_Obj *dp, Data_Obj *parent_dp)
{
	/* a used data_obj should have its name set */
#ifdef CAUTIOUS
	if( dp->dt_name == NULL )
		NERROR1("CAUTIOUS:  release_tmpobj_resources:  obj has null name!?");
#endif /* CAUTIOUS */

	if(	dp->dt_children != NO_LIST	||
		dp == parent_dp			||
		DOBJ_IS_LOCKED(dp)		||
		dp->dt_refcount > 0 ){

		return(-1);	/* failure */
	}

	/* If we are called from delvec, then disown child has already been called!?
	 *
	 * So we don't do it here any more...
	 */
	//disown_child(dp);

	rls_str((char *)dp->dt_name);
	dp->dt_name = NULL;	/* should not be needed, but helps us catch other bugs... */

	return(0);
}

static void add_tmpobjs(List *lp, int n)
{
	Data_Obj *data_obj_tbl;
	int i;

	data_obj_tbl=(Data_Obj *)getbuf( n * sizeof(*data_obj_tbl) );

	for(i=0;i<n;i++){
		Node *np;
		data_obj_tbl[i].dt_name=NULL;
		np = mk_node(&data_obj_tbl[i]);
		addTail(free_tmpobj_lp,np);
	}
}

Data_Obj * find_free_temp_dp(Data_Obj *dp)
{
	Data_Obj *new_dp;
	Node *np,*first_np;

	if( free_tmpobj_lp == NO_LIST ){

		free_tmpobj_lp = new_list();
		add_tmpobjs(free_tmpobj_lp,N_TMP_DP);
	}

	if( eltcount(free_tmpobj_lp) > 0 ){
		np = remHead(free_tmpobj_lp);

		if( used_tmpobj_lp == NO_LIST )
			used_tmpobj_lp = new_list();
		addHead(used_tmpobj_lp,np);

		return((Data_Obj *)np->n_data);
	}

	/* They are all in use, we have to release a free one... */
	np = remTail(used_tmpobj_lp);
	first_np = np;
	new_dp = (Data_Obj *)np->n_data;
	while( release_tmpobj_resources(new_dp,dp) < 0 ){
		addHead(used_tmpobj_lp,np);
		np=remTail(used_tmpobj_lp);
		if( np == first_np )
			NERROR1("out of temporary objects!?");
		new_dp = (Data_Obj *)np->n_data;
	}
	/* The deallocate routine has freed the old name etc */
	disown_child(new_dp);

	/* put this one at the head of the used list... */
	addHead(used_tmpobj_lp,np);
	return(new_dp);
}


/* Create a child object with the given name.
 * These names are not hashed, and the objects are initially locked.
 */

Data_Obj *temp_child( const char *name, Data_Obj *dp )
{
	Data_Obj *newdp;
	
	newdp=temp_replica(dp);

#ifdef DEBUG
/*
if( debug & debug_data ){
sprintf(error_string,"saving child name \"%s\"",name);
advise(error_string);
}
*/
#endif /* DEBUG */

	newdp->dt_name = savestr(name);

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

	newdp->dt_refcount = 0;

#ifdef DEBUG
if( debug & debug_data ){
printf("locking temp object %s\n",newdp->dt_name);
}
#endif /* DEBUG */

	newdp->dt_flags |= DT_TEMP | DT_LOCKED | DT_VOLATILE ;
	/* may be contiguous etc, but we have to check... */
	newdp->dt_flags &= ~(DT_CONTIG | DT_EVENLY | DT_CHECKED) ;

	return(newdp);
}

/* when should we call this?  After all references to new objects
 * have been made and finished...
 *
 * There is a problem, though, if we release many of these, then
 * we can encounter a situation in which we assign a[0], then it
 * gets released, then we use a[0] on the RHS, and get a warning
 * about used before initialized...  One way to do this would be to
 * sort the list to be reused based on time of last access...
 * This could get rather slow!?  On the other hand, we can't really keep
 * around data_obj structs for every pixel...  Maybe we can
 * introduce a "lock" function?
 */

void unlock_all_tmp_objs(void)
{
	Node *np;
	Data_Obj *dp;

	if( used_tmpobj_lp == NO_LIST ) return;

	np = used_tmpobj_lp->l_head;
	while(np!=NO_NODE){
		dp=(Data_Obj *)np->n_data;
		if( dp->dt_flags & DT_VOLATILE ){
			dp->dt_flags &= ~DT_LOCKED;
		}
		np = np->n_next;
	}
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

void make_array_name( QSP_ARG_DECL  char *target_str, Data_Obj *dp, index_t index, int which_dim, int subscr_type )
{
	int left_delim, right_delim, nstars;
	int i;
	char str2[LLEN];

	if( subscr_type == SQUARE ){
		left_delim  = '[';
		right_delim = ']';
		nstars = dp->dt_maxdim - which_dim;
	} else if( subscr_type == CURLY ){
		left_delim  = '{';
		right_delim = '}';
		nstars = which_dim - dp->dt_mindim;
	}
#ifdef CAUTIOUS
	else {
		WARN("CAUTIOUS:  unrecognized subscript type");
		left_delim  = '{';
		right_delim = '}';
		nstars = which_dim - dp->dt_mindim;
	}
#endif /* CAUTIOUS */

	/* now make the name */
	strcpy(target_str,dp->dt_name);
	for(i=0;i<nstars;i++){
		sprintf(str2,"%c*%c",left_delim,right_delim);
		strcat(target_str,str2);
	}
	sprintf(str2,"%c%d%c",left_delim,index,right_delim);
	strcat(target_str,str2);
}

static Node *existing_tmpobj_node(const char *name)
{
	Node *np;
	Data_Obj *dp;

	if( used_tmpobj_lp == NO_LIST ) return(NO_NODE);
	np=used_tmpobj_lp->l_head;
	while(np!=NO_NODE){
		dp = (Data_Obj *)np->n_data;
if( dp->dt_name==NULL ) NERROR1("existing_tmpobj_node:  null object!?");
		if( !strcmp(dp->dt_name,name) )
			return(np);
		np=np->n_next;
	}
	return(NO_NODE);
}

Data_Obj *gen_subscript( QSP_ARG_DECL  Data_Obj *dp, int which_dim, index_t index, int subscr_type )
{
	Data_Obj *newdp=NO_OBJ;
	Node *np;
	char str[LLEN];	/* BUG how long can this be really? */
	int i;

	if( dp==NO_OBJ ) return(dp);

	if( which_dim < 0 || which_dim >= N_DIMENSIONS ){
		WARN("gen_subscript:  dimension index out of range");
		return(NO_OBJ);
	}
	/* We used to disallow subscripting dimensions with only one element,
	 * but we use mindim and maxdim to keep track of that...
	 * As long as the subscript value itself is legal...
	 */
	/*
	if( dp->dt_mach_dim[which_dim] == 1 ){
		sprintf(error_string,
			"Can't subscript object %s, has %s dimension = 1",
			dp->dt_name,dimension_name[which_dim]);
		WARN(error_string);
		return(NO_OBJ);
	}
	*/

	/* indices are unsigned ! */
	if( index < base_index ){
		sprintf(error_string,
		"%s subscript too small (%d, min %u) for object %s",
			dimension_name[which_dim],index,
			base_index,dp->dt_name);
		WARN(error_string);
		return(NO_OBJ);
	}

	if( index-base_index >= dp->dt_mach_dim[which_dim] ){
		sprintf(error_string,
		"%s subscript too large (%u, max %u) for object %s",
			dimension_name[which_dim],index,
			dp->dt_mach_dim[which_dim]+base_index-1,dp->dt_name);
		WARN(error_string);
		return(NO_OBJ);
	}

	make_array_name(QSP_ARG  str,dp,index,which_dim,subscr_type);

	/* first see if this subobject may exist!? */
	/* these names aren't hashed, so just scan the list */
	/* maybe we would be better off with a linked list?? */

	np = existing_tmpobj_node(str);
	if( np != NO_NODE ){
		/* before returning, move this node up the the head of the list.
		 *
		 * The point is to keep the list priority sorted, where
		 * priority is based on the recency of the access...
		 */
		if( np != used_tmpobj_lp->l_head ){
			np=remNode(used_tmpobj_lp,np);
			/* do we need to check the return value?? */
			addHead(used_tmpobj_lp,np);
		}
		return((Data_Obj *)np->n_data);
	}

	/* the object doesn't exist, so create it */

	newdp=temp_child(str,dp);
	newdp->dt_mach_dim[which_dim] = 1;
	newdp->dt_mach_inc[which_dim] = 0;
	newdp->dt_type_dim[which_dim] = 1;
	newdp->dt_type_inc[which_dim] = 0;

	index -= base_index;
	index *= dp->dt_mach_inc[which_dim] * ELEMENT_INC_SIZE(dp);
	newdp->dt_offset = index;		/* offset is in bytes! */

	if( IS_BITMAP(newdp) ){
		if( newdp->dt_data != NULL ){
			newdp->dt_data = ((char *)dp->dt_data) + 4*(index/32);
			newdp->dt_bit0 += index % 32;
			if( newdp->dt_bit0 >= 32 ){
				newdp->dt_bit0 -= 32;
				newdp->dt_data = ((char *)newdp->dt_data) + 4;
			}
		}
	} else {
		if( newdp->dt_data != NULL )
			newdp->dt_data = ((char *)dp->dt_data) + newdp->dt_offset;
	}

	/* dt_n_mach_elts is the total number of component elements */
	newdp->dt_n_mach_elts /= dp->dt_mach_dim[which_dim];
	/* dt_n_type_elts is the total number of elements, where a complex number is counted as 1 */
	newdp->dt_n_type_elts /= dp->dt_type_dim[which_dim];

	if( set_shape_flags(&newdp->dt_shape,newdp,AUTO_SHAPE) < 0 )
		WARN("holy mole batman!?");

	check_contiguity(newdp);

	if( IS_COMPLEX(newdp->dt_parent) && newdp->dt_mach_dim[0]!=2 ){
		newdp->dt_prec &= ~COMPLEX_PREC_BITS;
		newdp->dt_flags &= ~DT_COMPLEX;
		newdp->dt_flags &= ~DT_CONTIG;
		for(i=0;i<N_DIMENSIONS;i++)
			newdp->dt_type_inc[i] = newdp->dt_mach_inc[i];
	}
	if( IS_QUAT(newdp->dt_parent) && newdp->dt_mach_dim[0] !=4 ){
		newdp->dt_prec &= ~QUAT_PREC_BITS;
		newdp->dt_flags &= ~DT_QUAT;
		newdp->dt_flags &= ~DT_CONTIG;
		for(i=0;i<N_DIMENSIONS;i++)
			newdp->dt_type_inc[i] = newdp->dt_mach_inc[i];
	}

	/* now change mindim/maxdim! */
	/*
	if( subscr_type == SQUARE )
		newdp->dt_maxdim -- ;
	else if( subscr_type == CURLY )
		newdp->dt_mindim ++ ;
		*/

	return(newdp);
} /* end gen_subscript */

void release_tmp_obj(Data_Obj *dp)
{
	Node *np;

	np = remData(used_tmpobj_lp,dp);
#ifdef CAUTIOUS
	if( np == NO_NODE ){
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  delete_tmp_obj:  %s not found",dp->dt_name);
		NERROR1(DEFAULT_ERROR_STRING);
	}
#endif /* CAUTIOUS */

	release_tmpobj_resources((Data_Obj *)np->n_data,NO_OBJ);
	addHead(free_tmpobj_lp,np);
}

/*
 * reset the data pointer.
 *
 * The dimensions and increments are unchanged, and for speed we won't
 * change the name...  Hopefully this is not a BUG?
 */

void reindex( QSP_ARG_DECL  Data_Obj *dp, int which_dim, index_t index )
{
#ifdef DEBUG
if( debug & debug_data ){
	char str[LLEN];

	/* BUG don't necessarily know it's square!? */
	/* But this is usually called from dp_vectorize(), then it is... */
	/* the name isn't always right here, but it is useful for debugging... */

	make_array_name(QSP_ARG  str,dp->dt_parent,index,which_dim,SQUARE);
	rls_str((char *)dp->dt_name);
	dp->dt_name=savestr(str);
}
#endif /* DEBUG */
	index -= base_index;
	index *= dp->dt_parent->dt_mach_inc[which_dim] * ELEMENT_SIZE(dp);
	dp->dt_offset = index;
	/* why a check for non-nul ptr above? */
	dp->dt_data = ((char *)dp->dt_parent->dt_data) + index;
}

/* reduce the dimensionality of dp */

Data_Obj *reduce_from_end( QSP_ARG_DECL  Data_Obj *dp, index_t index, int subscr_type )
{
	Data_Obj *newdp=NO_OBJ;
	int dim;

	if( dp==NO_OBJ ) return(dp);

	if( subscr_type == SQUARE )     dim=dp->dt_maxdim;
#ifdef CAUTIOUS
	else if( subscr_type == CURLY ) dim=dp->dt_mindim;
	else {
		dim=0;		/* eliminate warning */
		NERROR1("bad subscript type, reduce_from_end()");
	}
#else
	else dim=dp->dt_mindim;
#endif /* CAUTIOUS */

	newdp =  gen_subscript(QSP_ARG  dp,dim,index,subscr_type);

	return(newdp);
}

#ifdef PARSE_INDEX
/*
 * sequence -> frame
 * frame -> row
 * row -> point
 * point -> component
 */

Data_Obj *d_subscript( QSP_ARG_DECL  Data_Obj *dp, index_t index )
{
	/* find the highest dimension != 1 */
	return( reduce_from_end(QSP_ARG  dp,index,SQUARE) );
}

/* this kind of subscripting works from the other end;
 * whereas image[0] is the first row, image{0} is the first column
 * if image is complex, then image{0} is the real image
 * and image{1} is the imaginary image
 */

Data_Obj *c_subscript( QSP_ARG_DECL  Data_Obj *dp, index_t index )
{
	return( reduce_from_end(QSP_ARG  dp,index,CURLY) );
}

int is_in_string(int c,const char *s)
{
	while( *s )
		if( c == *s++ ) return(1);
	return(0);
}


void set_array_base_index(QSP_ARG_DECL  int index)
{
	if( index < 0 || index > 1 ){
		sprintf(error_string,
	"set_base_index:  requested base index (%d) is out of range (0-1)",
			index);
		WARN(error_string);
		return;
	}

	base_index = index;
}

#endif /* PARSE_INDEX */
