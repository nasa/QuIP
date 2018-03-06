
#include "quip_config.h"
#include <string.h>

#include "quip_prot.h"
#include "data_obj.h"
#include "dobj_private.h"

//#include "fio_api.h"


#define PARENT_PROMPT	"parent data object"

//#define N_OBJ_KINDS	5

static const char *parlist[]={"even","odd"};

#ifdef NOT_YET
static Data_Obj *get_obj_or_file(QSP_ARG_DECL const char *name);
#endif /* NOT_YET */
static COMMAND_FUNC( do_tellprec );

Precision * get_precision(SINGLE_QSP_ARG_DECL);



#define INSIST_POSITIVE_DIM( var, dim_name, subrt_name )					\
												\
	if( var <= 0 ){										\
		sprintf(ERROR_STRING,"%s %s:  number of %ss (%ld) must be positive",		\
			subrt_name,obj_name,dim_name,var);					\
		warn(ERROR_STRING);								\
		return;										\
	}

#define INSIST_POSITIVE_NUM( var, desc_str, subrt_name )					\
												\
	if( var <= 0 ){										\
		sprintf(ERROR_STRING,"%s:  %s (%ld) must be positive",				\
			subrt_name,desc_str,var);						\
		warn(ERROR_STRING);								\
		return;										\
	}


#define INSIST_NONNEGATIVE( var, var_string, subrt_name )					\
												\
	if( var < 0 ){										\
		sprintf(ERROR_STRING,"%s %s:  %s (%ld) must be positive",			\
			subrt_name,obj_name,var_string,var);					\
		warn(ERROR_STRING);								\
		return;										\
	}

static COMMAND_FUNC( do_create_area )
{
	const char *area_name;
	long n;
	long siz;

	area_name = NAMEOF("name for this area");
	siz = (long) how_many("size of data area in bytes");
	n = (long) how_many("maximum number of data objects for this area");

	INSIST_POSITIVE_NUM(siz,"number of bytes","create_area");
	INSIST_POSITIVE_NUM(n,"maximum number of objects","create_area");

	curr_ap=new_area(area_name, (dimension_t) siz,(unsigned int)n);	// do_create_area
}

static COMMAND_FUNC( do_select_area )
{
	Data_Area *ap;

	ap = pick_data_area("");
	if( ap != NULL )
		curr_ap=ap;
	else if( curr_ap != NULL ){
		sprintf(ERROR_STRING,"Unable to change data area, current area remains %s.",
			AREA_NAME(curr_ap));
		advise(ERROR_STRING);
	}
}

static COMMAND_FUNC( show_area )
{
	Data_Area *ap;

	ap = pick_data_area("");
	if( ap == NULL ) return;

	show_area_space(ap);
}

static COMMAND_FUNC( do_area_info )
{
	Data_Area *ap;

	ap = pick_data_area("");
	if( ap == NULL ) return;

	data_area_info(ap);
}

static COMMAND_FUNC( do_match_area )
{
	Data_Obj *dp;

	dp = pick_obj("object");
	if( dp == NULL ) return;

	curr_ap=OBJ_AREA(dp);
}

static COMMAND_FUNC( do_list_data_areas )
{ list_data_areas(tell_msgfile()); }

static COMMAND_FUNC( do_get_area )
{
	const char *s;

	s=NAMEOF("script variable for area name");

	assert( curr_ap != NULL );

	assign_var(s, AREA_NAME(curr_ap));
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(areas_menu,s,f,h)

MENU_BEGIN(areas)

ADD_CMD( new_area,	do_create_area,	create new data area	)
ADD_CMD( get_area,	do_get_area,	fetch name of current data area )
ADD_CMD( match,		do_match_area,	select data area to match an object	)
ADD_CMD( select,	do_select_area,	select default data area	)
ADD_CMD( space,		show_area,	show space allocation within a data area	)
ADD_CMD( list,		do_list_data_areas,	list all data areas	)
ADD_CMD( info,		do_area_info,	print infomation about an area	)

MENU_END(areas)


static COMMAND_FUNC( do_area )
{
	CHECK_AND_PUSH_MENU(areas);
}

/* Create and push a new context */

// Can we push an existing context from the menu?

static COMMAND_FUNC( do_push_context )
{
	const char *s;
	Item_Context *icp;

	s=NAMEOF("name for new context");
	NEW_ITEM_CONTEXT(icp);	// allocates memory and pts icp, but does not initalize?
	SET_CTX_NAME(icp,savestr(s));
	push_dobj_context(icp);
}

static COMMAND_FUNC( do_pop_context )
{
	pop_dobj_context();
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(contexts_menu,s,f,h)

MENU_BEGIN(contexts)
ADD_CMD( push,	do_push_context,	push a new context	)
ADD_CMD( pop,	do_pop_context,		pop current context	)
MENU_END(contexts)

static COMMAND_FUNC( do_context )
{
	CHECK_AND_PUSH_MENU(contexts);
}

Precision * get_precision(SINGLE_QSP_ARG_DECL)
{
	if( prec_itp == NULL )
		init_precisions();

	return (Precision *) pick_item(prec_itp, "data precision" );
}

#define finish_obj(obj_name,dsp,prec_p,type_flag)	_finish_obj(QSP_ARG  obj_name,dsp,prec_p,type_flag)

static void _finish_obj(QSP_ARG_DECL  const char *obj_name, Dimension_Set *dsp, Precision *prec_p, uint32_t type_flag)
{
	assert(prec_p!=NULL);

	if( COLOR_PRECISION(PREC_CODE(prec_p)) ){
		if( DIMENSION(dsp,0) != 1 ){
			sprintf(ERROR_STRING,"object %s, number of rgb triples per pixel should be 1",obj_name);
			warn(ERROR_STRING);
		}
		SET_DIMENSION(dsp,0,3);
		prec_p = get_prec("float");
	}

	if( make_dobj_with_shape(obj_name,dsp,prec_p,type_flag) == NULL ) {
		sprintf(ERROR_STRING,"couldn't create data object \"%s\"", obj_name);
		warn(ERROR_STRING);
	}
}

static COMMAND_FUNC( new_hyperseq )
{
	Dimension_Set ds1, *dsp=(&ds1);
	const char *obj_name;
	long ns, nf, nr, nc, ncomps;
	Precision *prec_p;

	obj_name=NAMEOF("object name");

	ns = (long) how_many("number of sequences");
	nf = (long) how_many("number of frames");
	nr = (long) how_many("number of rows");
	nc = (long) how_many("number of columns");
	ncomps = (long) how_many("number of components");
	prec_p = get_precision(SINGLE_QSP_ARG);

	if( prec_p == NULL ) return;

	INSIST_POSITIVE_DIM(ns,"sequence","new_hyperseq");

	SET_DIMENSION(dsp,4,ns);
	SET_DIMENSION(dsp,3,nf);
	SET_DIMENSION(dsp,2,nr);
	SET_DIMENSION(dsp,1,nc);
	SET_DIMENSION(dsp,0,ncomps);

	finish_obj(obj_name,dsp,prec_p,DT_HYPER_SEQ);
}

static COMMAND_FUNC( new_seq )
{
	Dimension_Set ds1, *dsp=(&ds1);
	long nf, nr, nc, ncomps;
	const char *obj_name;
	Precision *prec_p;

	obj_name=NAMEOF("object name");

	nf = (long) how_many("number of frames");
	nr = (long) how_many("number of rows");
	nc = (long) how_many("number of columns");
	ncomps = (long) how_many("number of components");

	prec_p = get_precision(SINGLE_QSP_ARG);

	if( prec_p == NULL ) return;
	INSIST_POSITIVE_DIM(nf,"frame","new_seq");
	INSIST_POSITIVE_DIM(nr,"row","new_seq");
	INSIST_POSITIVE_DIM(nc,"column","new_seq");
	INSIST_POSITIVE_DIM(ncomps,"component","new_seq");

	SET_DIMENSION(dsp,4,1);
	SET_DIMENSION(dsp,3,nf);
	SET_DIMENSION(dsp,2,nr);
	SET_DIMENSION(dsp,1,nc);
	SET_DIMENSION(dsp,0,ncomps);

	finish_obj(obj_name,dsp,prec_p,DT_SEQUENCE);
}

static COMMAND_FUNC( new_frame )
{
	Dimension_Set ds1;
	Dimension_Set *dsp=(&ds1);
	const char *obj_name;
	long nr, nc, ncomps;
	Precision *prec_p;

	obj_name=NAMEOF("object name");

	nr = (long) how_many("number of rows");
	nc = (long) how_many("number of columns");
	ncomps = (long) how_many("number of components");

	prec_p = get_precision(SINGLE_QSP_ARG);

	if( prec_p == NULL ) return;
	INSIST_POSITIVE_DIM(nr,"row","new_frame");
	INSIST_POSITIVE_DIM(nc,"column","new_frame");
	INSIST_POSITIVE_DIM(ncomps,"component","new_frame");

	SET_DIMENSION(dsp,4,1);
	SET_DIMENSION(dsp,3,1);
	SET_DIMENSION(dsp,2,nr);
	SET_DIMENSION(dsp,1,nc);
	SET_DIMENSION(dsp,0,ncomps);

	finish_obj(obj_name,dsp,prec_p,DT_IMAGE);
}

static COMMAND_FUNC( new_gen_obj )
{
	Dimension_Set ds1, *dsp=(&ds1);
	const char *obj_name;
	long ns, nf, nr, nc, ncomps;
	Precision *prec_p;

	obj_name=NAMEOF("object name");

	ns = (long) how_many("number of sequences");
	nf = (long) how_many("number of frames");
	nr = (long) how_many("number of rows");
	nc = (long) how_many("number of columns");
	ncomps = (long) how_many("number of components");

	prec_p = get_precision(SINGLE_QSP_ARG);

	if( prec_p == NULL ) return;
	INSIST_POSITIVE_DIM(ns,"sequence","new_gen_obj");
	INSIST_POSITIVE_DIM(nf,"frame","new_gen_obj");
	INSIST_POSITIVE_DIM(nr,"row","new_gen_obj");
	INSIST_POSITIVE_DIM(nc,"column","new_gen_obj");
	INSIST_POSITIVE_DIM(ncomps,"component","new_gen_obj");

	SET_DIMENSION(dsp,4,ns);
	SET_DIMENSION(dsp,3,nf);
	SET_DIMENSION(dsp,2,nr);
	SET_DIMENSION(dsp,1,nc);
	SET_DIMENSION(dsp,0,ncomps);

	if( *obj_name == 0 ){
		warn("new_gen_obj:  Null object name!?");
		return;	// shouldn't happen, but can - HOW???
		// If it can, then should move this check to finish_obj???
	}

	finish_obj(obj_name,dsp,prec_p,AUTO_SHAPE);
}

#ifdef NOT_YET
static COMMAND_FUNC( new_obj_list )
{
	int n;
	const char *s;
	List *lp;
	Node *np;

	s=NAMEOF("object name");

	n=how_many("number of objects in this list");
	lp=new_list();
	while(n--){
		Data_Obj *dp;

		dp = pick_obj("");
		if( dp != NULL ){
			np=mk_node(dp);
			addTail(lp,np);
		}
	}

	if( make_obj_list(s,lp) == NULL ){
		sprintf(ERROR_STRING,"error making object list %s");
		warn(ERROR_STRING);
	}
}
#endif /* NOT_YET */

static COMMAND_FUNC( new_row )
{
	Dimension_Set ds1, *dsp=(&ds1);
	const char *obj_name;
	long nc, ncomps;
	Precision *prec_p;

	obj_name=NAMEOF("object name");

	nc = (long) how_many("number of elements");
	ncomps = (long) how_many("number of components");

	prec_p = get_precision(SINGLE_QSP_ARG);

	if( prec_p == NULL ) return;
	INSIST_POSITIVE_DIM(nc,"element","new_col")
	INSIST_POSITIVE_DIM(ncomps,"component","new_col")

	SET_DIMENSION(dsp,4,1);
	SET_DIMENSION(dsp,3,1);
	SET_DIMENSION(dsp,2,1);
	SET_DIMENSION(dsp,1,nc);
	SET_DIMENSION(dsp,0,ncomps);

	finish_obj(obj_name,dsp,prec_p,DT_ROWVEC);
}

static COMMAND_FUNC( new_col )
{
	Dimension_Set ds1, *dsp=(&ds1);
	const char *obj_name;
	long nr, ncomps;
	Precision *prec_p;

	obj_name=NAMEOF("object name");

	nr = (long) how_many("number of elements");
	ncomps = (long) how_many("number of components");

	prec_p = get_precision(SINGLE_QSP_ARG);

	if( prec_p == NULL ) return;
	INSIST_POSITIVE_DIM(nr,"element","new_col")
	INSIST_POSITIVE_DIM(ncomps,"component","new_col")

	SET_DIMENSION(dsp,4,1);
	SET_DIMENSION(dsp,3,1);
	SET_DIMENSION(dsp,2,nr);
	SET_DIMENSION(dsp,1,1);
	SET_DIMENSION(dsp,0,ncomps);

	finish_obj(obj_name,dsp,prec_p,DT_COLVEC);
}

static COMMAND_FUNC( new_scalar )
{
	Dimension_Set ds1;
	Dimension_Set *dsp=(&ds1);
	const char *obj_name;
	long ncomps;
	Precision *prec_p;

	obj_name=NAMEOF("object name");

	ncomps = (long) how_many("number of components");
	prec_p = get_precision(SINGLE_QSP_ARG);

	if( prec_p == NULL ) return;
	INSIST_POSITIVE_DIM(ncomps,"component","new_scalar");

	SET_DIMENSION(dsp,4,1);
	SET_DIMENSION(dsp,3,1);
	SET_DIMENSION(dsp,2,1);
	SET_DIMENSION(dsp,1,1);
	SET_DIMENSION(dsp,0,ncomps);

	finish_obj(obj_name,dsp,prec_p,DT_SCALAR);
}

static COMMAND_FUNC( do_delvec )
{
	Data_Obj *dp;

	dp=pick_obj("");
	if( dp==NULL ) return;
	delvec(dp);
}

static COMMAND_FUNC( do_dobj_info )
{
	Data_Obj *dp;

	dp=pick_obj("");
	if( dp==NULL ) return;

	longlist(dp);
}

static COMMAND_FUNC( mksubimg )
{
	const char *obj_name;
	Data_Obj *dp, *newdp;
	long rows, cols;
	long xos, yos;

	obj_name=NAMEOF("name for subimage");

	dp=pick_obj(PARENT_PROMPT);

	cols=(long) how_many("number of columns");
	rows=(long) how_many("number of rows");

	xos=(long) how_many("x offset");
	yos=(long) how_many("y offset");

	if( dp==NULL ) return;
	INSIST_POSITIVE_DIM(rows,"row","mksubimg")
	INSIST_POSITIVE_DIM(cols,"column","mksubimg")

	INSIST_NONNEGATIVE(xos,"x offset","mksubimg");
	INSIST_NONNEGATIVE(yos,"y offset","mksubimg");

	newdp=mk_subimg(dp,(index_t)xos,(index_t)yos,obj_name,(dimension_t)rows,(dimension_t)cols);
	if( newdp == NULL )
		warn("couldn't create subimage");
}

static COMMAND_FUNC( mksubsequence )
{
	const char *obj_name;
	Data_Obj *dp, *newdp;
	index_t offsets[N_DIMENSIONS];
	Dimension_Set ds1, *dsp=(&ds1);
	long x_offset, y_offset, t_offset;
	long nr,nc,nf;

	obj_name=NAMEOF("name for subsequence");

	dp=pick_obj(PARENT_PROMPT);

	nc = (long) how_many("number of columns");
	nr = (long) how_many("number of rows");
	nf = (long) how_many("number of frames");

	x_offset=(long) how_many("x offset");
	y_offset=(long) how_many("y offset");
	t_offset=(long) how_many("t offset");

	if( dp==NULL ) return;

	INSIST_POSITIVE_DIM(nc,"column","mksubsequence");
	INSIST_POSITIVE_DIM(nr,"row","mksubsequence");
	INSIST_POSITIVE_DIM(nf,"frame","mksubsequence");

	INSIST_NONNEGATIVE(x_offset,"x offset","mksubsequence");
	INSIST_NONNEGATIVE(y_offset,"y offset","mksubsequence");
	INSIST_NONNEGATIVE(t_offset,"t offset","mksubsequence");

	offsets[0]=0;
	offsets[1]=(index_t)x_offset;
	offsets[2]=(index_t)y_offset;
	offsets[3]=(index_t)t_offset;
	offsets[4]=0;

	SET_DIMENSION(dsp,0,OBJ_COMPS(dp));
	SET_DIMENSION(dsp,1,nc);
	SET_DIMENSION(dsp,2,nr);
	SET_DIMENSION(dsp,3,nf);
	SET_DIMENSION(dsp,4,1);

	newdp=mk_subseq(obj_name,dp,offsets,dsp);
	if( newdp == NULL )
		warn("couldn't create subimage");
}

static COMMAND_FUNC( mksubstring )
{
	const char *obj_name;
	Data_Obj *dp, *newdp;
	long len;
	long sos;

	obj_name=NAMEOF("name for substring");

	dp=pick_obj(PARENT_PROMPT);

	len=(long) how_many("number of characters");

	sos=(index_t)how_many("offset");

	if( dp==NULL ) return;

	INSIST_POSITIVE_DIM(len,"character","mksubstring")
	INSIST_NONNEGATIVE(sos,"x offset","mksubstring")

	newdp=mk_substring(dp,(index_t)sos,obj_name,(dimension_t)len);
	if( newdp == NULL )
		warn("couldn't create substring");
}

static COMMAND_FUNC( mksubvector )
{
	const char *obj_name;
	Data_Obj *dp, *newdp;
	dimension_t rows;
	index_t yos;
	long cols;
	long xos;

	obj_name=NAMEOF("name for subvector");

	dp=pick_obj(PARENT_PROMPT);

	cols=(long) how_many("number of elements");
	rows=1;

	xos=(index_t)how_many("offset");
	yos=0;

	if( dp==NULL ) return;

	INSIST_POSITIVE_DIM(cols,"element","mksubvector")
	INSIST_NONNEGATIVE(xos,"x offset","mksubvector")

	newdp=mk_subimg(dp,(index_t)xos,yos,obj_name,rows,(dimension_t)cols);
	if( newdp == NULL )
		warn("couldn't create subvector");
}

static COMMAND_FUNC( mksubscalar )
{
	const char *obj_name;
	Data_Obj *dp, *newdp;
	index_t offsets[N_DIMENSIONS];
	Dimension_Set ds1, *dsp=(&ds1);
	long ncomps, comp_offset;

	obj_name=NAMEOF("name for subscalar");

	dp=pick_obj(PARENT_PROMPT);

	ncomps = (long) how_many("number of components");
	comp_offset = (long) how_many("component offset");

	if( dp==NULL ) return;

	INSIST_POSITIVE_DIM(ncomps,"component","mksubscalar");
	INSIST_NONNEGATIVE(comp_offset,"component offset","mksubscalar");

	SET_DIMENSION(dsp,0,ncomps);
	SET_DIMENSION(dsp,1,1);
	SET_DIMENSION(dsp,2,1);
	SET_DIMENSION(dsp,3,1);
	SET_DIMENSION(dsp,4,1);

	offsets[0]=(index_t)comp_offset;
	offsets[1]=0;
	offsets[2]=0;
	offsets[3]=0;
	offsets[4]=0;

	newdp=mk_subseq(obj_name,dp,offsets,dsp);
	if( newdp == NULL )
		warn("couldn't create subscalar");
}

static COMMAND_FUNC( do_ilace )
{
	const char *obj_name;
	Data_Obj *dp, *newdp;
	int parity;

	obj_name=NAMEOF("name for subimage");

	dp=get_img( QSP_ARG  NAMEOF("name of parent image") );
	if( dp==NULL ) return;

	parity=WHICH_ONE("parity of selected lines",2,parlist);
	if( parity < 0 ) return;

	newdp=mk_ilace(dp,obj_name,parity);
	if( newdp == NULL )
		warn("couldn't create interlaced subimage");
}

static COMMAND_FUNC( mkcast )
{
	const char *obj_name;
	Data_Obj *dp, *newdp;
	long rows, cols, tdim;
	long xos, yos;

	obj_name=NAMEOF("name for cast");

	dp=pick_obj(PARENT_PROMPT);
	if( dp==NULL ) return;

	cols=(long) how_many("number of columns");
	rows=(long) how_many("number of rows");
	xos=(long) how_many("x offset");
	yos=(long) how_many("y offset");
	tdim=(long) how_many("type dimension");

	INSIST_POSITIVE_DIM(cols,"column","mkcast")
	INSIST_POSITIVE_DIM(rows,"row","mkcast")
	INSIST_POSITIVE_DIM(tdim,"component","mkcast")

	INSIST_NONNEGATIVE(xos,"x offset","mkcast")
	INSIST_NONNEGATIVE(yos,"y offset","mkcast")

	newdp=nmk_subimg(dp,(index_t)xos,(index_t)yos,obj_name,(dimension_t)rows,(dimension_t)cols,(dimension_t)tdim);
	if( newdp == NULL )
		warn("couldn't create subimage");
}

static COMMAND_FUNC( equivalence )
{
	const char *obj_name;
	Data_Obj *dp;
	Precision * prec_p;
	Dimension_Set ds1, *dsp=(&ds1);
	long ns,nf,nr,nc,nd;

	obj_name=NAMEOF("name for equivalent image");

	dp=pick_obj(PARENT_PROMPT);

	ns=(long) how_many("number of sequences");
	nf=(long) how_many("number of frames");
	nr=(long) how_many("number of rows");
	nc=(long) how_many("number of columns");
	nd=(long) how_many("number of components");

	prec_p = get_precision(SINGLE_QSP_ARG);

	if( dp==NULL ) return;
	if( prec_p == NULL ) return;

	INSIST_POSITIVE_DIM(ns,"sequence","equivalence")
	INSIST_POSITIVE_DIM(nf,"frame","equivalence")
	INSIST_POSITIVE_DIM(nr,"row","equivalence")
	INSIST_POSITIVE_DIM(nc,"column","equivalence")
	INSIST_POSITIVE_DIM(nd,"component","equivalence")

	SET_DIMENSION(dsp,4,ns);
	SET_DIMENSION(dsp,3,nf);
	SET_DIMENSION(dsp,2,nr);
	SET_DIMENSION(dsp,1,nc);
	SET_DIMENSION(dsp,0,nd);

	if( COMPLEX_PRECISION(PREC_CODE(prec_p)) ){
		if( DIMENSION(dsp,0) != 1 ){
			warn("Sorry, can only have 1 complex component");
			return;
		}
		//SET_DIMENSION(dsp,0,2);
	} else if( QUAT_PRECISION(PREC_CODE(prec_p)) ){
		if( DIMENSION(dsp,0) != 1 ){
			warn("Sorry, can only have 1 quaternion component");
			return;
		}
		//SET_DIMENSION(dsp,0,2);
	} else if( COLOR_PRECISION(PREC_CODE(prec_p)) ){
		if( DIMENSION(dsp,0) != 1 ){
			warn("Sorry, can only have 1 color triple per pixel");
			return;
		}
advise("component dim 3 for color");
		//SET_DIMENSION(dsp,0,3);
	}

	if( make_equivalence(obj_name,dp,dsp,prec_p) == NULL )
		warn("error making equivalence");
}

#define MAX_PMPT_LEN	128	// BUG check for overrun

static COMMAND_FUNC( mk_subsample )
{
	const char *obj_name;
	Data_Obj *dp;

	Dimension_Set ds1, *dsp=(&ds1);
	index_t offsets[N_DIMENSIONS];
	long l_offset[N_DIMENSIONS];
	incr_t incrs[N_DIMENSIONS];
	long size[N_DIMENSIONS];
	char pmpt[MAX_PMPT_LEN];
	int i;

	obj_name=NAMEOF("name for subsample object");

	dp=pick_obj(PARENT_PROMPT);

	/* We violate the rule of returning before getting
	 * all arguments, because the fields of dp are needed
	 * to determine what to prompt for!
	 */

	if( dp==NULL ) return;

	for(i=0;i<N_DIMENSIONS;i++){
		/* BUG? should we prompt for all dimensions, instead of just those > 1 ?
		 * If we did, then we could defer the return above...
		 */
		if( OBJ_TYPE_DIM(dp,i) > 1 ){
			if( i < (N_DIMENSIONS-1) )
				// BUG check length
				sprintf(pmpt,"number of %ss per %s",dimension_name[i], dimension_name[i+1]);
			else
				sprintf(pmpt,"number of %ss",dimension_name[i]);

			size[i]=(long) how_many(pmpt);

			sprintf(pmpt,"%s offset",dimension_name[i]);
			l_offset[i] = (long) how_many(pmpt);
			sprintf(pmpt,"%s increment",dimension_name[i]);
			incrs[i] =(incr_t)how_many(pmpt);	// this can be negative...
		} else {
			size[i] = 1;
			l_offset[i]=0;
			incrs[i]=1;
		}
	}
	for(i=0;i<N_DIMENSIONS;i++){
		char offset_descr[LLEN];
		INSIST_POSITIVE_DIM(size[i],dimension_name[i],"mk_subsample");
		sprintf(offset_descr,"%s offset",dimension_name[i]);
		INSIST_NONNEGATIVE(l_offset[i],offset_descr,"mk_subsample");
	}
	for(i=0;i<N_DIMENSIONS;i++){
		SET_DIMENSION(dsp,i,size[i]);
		offsets[i] = (index_t) l_offset[i];
	}

	// make_subsamp checks the increments...

	if( make_subsamp(obj_name,dp,dsp,offsets,incrs) == NULL )
		warn("error making subsamp object");
}

static COMMAND_FUNC( do_relocate )
{
	Data_Obj *dp;
	long x,y,t;
	const char *obj_name;

	dp=pick_obj("subimage");
	x=(long) how_many("x offset");
	y=(long) how_many("y offset");
	t=(long) how_many("t offset");

	if( dp==NULL ) return;
	obj_name = OBJ_NAME(dp);
	INSIST_NONNEGATIVE(x,"x offset","relocate");
	INSIST_NONNEGATIVE(y,"y offset","relocate");
	INSIST_NONNEGATIVE(t,"t offset","relocate");

	if( OBJ_PARENT(dp) == NULL ){
		sprintf(ERROR_STRING,
	"relocate:  object \"%s\" is not a subimage",
			OBJ_NAME(dp));
		warn(ERROR_STRING);
		return;
	}
	relocate(dp,(index_t)x,(index_t)y,(index_t)t);
}

static COMMAND_FUNC( do_gen_xpose )
{
	Data_Obj *dp;
	int d1,d2;

	dp = pick_obj("");
	d1=(int)how_many("dimension index #1");
	d2=(int)how_many("dimension index #2");

	if( dp == NULL ) return;

	gen_xpose(dp,d1,d2);
}

#ifdef NOT_YET
static Data_Obj *get_obj_or_file(QSP_ARG_DECL const char *name)
{
	Data_Obj *dp;
#ifndef PC
	Image_File *ifp;
#endif /* ! PC */

	/* dp = dobj_of(name); */

	/* use hunt_obj() in order to pick up indexed strings */
	dp = hunt_obj(name);
	if( dp != NULL ) return(dp);

#ifndef PC
	ifp = img_file_of(name);
	if( ifp!=NULL ) return(ifp->if_dp);

	sprintf(ERROR_STRING,"No object or open file \"%s\"",name);
	warn(ERROR_STRING);
#else /* PC */
	sprintf(ERROR_STRING,
		"No object \"%s\" (not checking for files!?)",name);
	warn(ERROR_STRING);
#endif /* ! PC */

	return(dp);
}
#endif /* NOT_YET */

static COMMAND_FUNC( do_tellprec )
{
	Data_Obj *dp;
	const char *s;

	dp = get_obj( QSP_ARG NAMEOF("data object") );
	s = NAMEOF("variable name");
	if( dp == NULL ) return;
	assign_var(s,OBJ_PREC_NAME(dp));
}

static COMMAND_FUNC( do_get_align )
{
	int a;

	a=(int)how_many("alignment (in bytes, negative to disable)");
	set_dp_alignment(a);
}

static COMMAND_FUNC( do_list_dobjs ) { list_dobjs(tell_msgfile()); }
static COMMAND_FUNC( do_list_temp_dps ) { list_temp_dps(tell_msgfile()); }
static COMMAND_FUNC( do_unlock_all_tmp_objs ) { unlock_all_tmp_objs(); }

static COMMAND_FUNC( do_protect )
{
	Data_Obj *dp;

	dp=pick_obj("");
	if( dp == NULL ) return;
	if( IS_STATIC(dp) ){
		sprintf(ERROR_STRING,"do_protect:  Object %s is already static!?",OBJ_NAME(dp));
		warn(ERROR_STRING);
		return;
	}
	SET_OBJ_FLAG_BITS(dp,DT_STATIC);
}

/* dm_init may be called from other modules as well...
 */

void dm_init(SINGLE_QSP_ARG_DECL)
{
	static int dm_inited=0;

	if( dm_inited ) return;
	dataobj_init(SINGLE_QSP_ARG);

	/* Version control */
	//verdatam(SINGLE_QSP_ARG);

	dm_inited=1;
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(data_menu,s,f,h)

MENU_BEGIN(data)

ADD_CMD( object,	new_gen_obj,	create auto-shaped object	)
ADD_CMD( image,		new_frame,	create new image	)
ADD_CMD( vector,	new_row,	create new row vector	)
ADD_CMD( column,	new_col,	create new column vector	)
ADD_CMD( scalar,	new_scalar,	create new scalar	)
ADD_CMD( sequence,	new_seq,	create new sequence	)
ADD_CMD( hyperseq,	new_hyperseq,	create new hyper-sequence	)
#ifdef NOT_YET
ADD_CMD( obj_list,	new_obj_list,	create new list object	)
#endif /* NOT_YET */

ADD_CMD( delete,	do_delvec,	delete object	)
ADD_CMD( subsample,	mk_subsample,	sub- or resample a data object	)
ADD_CMD( subimage,	mksubimg,	create a subimage	)
ADD_CMD( substring,	mksubstring,	create a substring	)
ADD_CMD( subvector,	mksubvector,	create a subvector	)
ADD_CMD( subscalar,	mksubscalar,	create a subscalar	)
ADD_CMD( subsequence,	mksubsequence,	create a subsequence	)
ADD_CMD( relocate,	do_relocate,	relocate a subimage	)
ADD_CMD( equivalence,	equivalence,	equivalence an image to another type	)
ADD_CMD( transpose,	do_gen_xpose,	generalized transpose (in-place)	)
ADD_CMD( interlace,	do_ilace,	create a interlaced subimage	)
ADD_CMD( protect,	do_protect,	make object static	)
ADD_CMD( cast,		mkcast,		recast object dimensions	)
ADD_CMD( list,		do_list_dobjs,	list all data objects	)
ADD_CMD( temp_list,	do_list_temp_dps,	list temporary data objects	)
ADD_CMD( info,		do_dobj_info,	give info concerning particular data object	)
/* ADD_CMD( unsigned,	do_mk_unsigned,	make an integer object unsigned	) */
ADD_CMD( alignment,	do_get_align,	specify data buffer alignment	)
ADD_CMD( precision,	do_tellprec,	fetch object precision	)
ADD_CMD( areas,		do_area,	data area submenu	)
ADD_CMD( contexts,	do_context,	data context submenu	)
ADD_CMD( ascii,		asciimenu,	read and write ascii data	)
ADD_CMD( operate,	buf_ops,	simple operations on buffers	)
ADD_CMD( unlock_temp_objs,	do_unlock_all_tmp_objs,	unlock temp objs (when callbacks inhibited)	)

MENU_END(data)

COMMAND_FUNC( do_dobj_menu )
{
	static int inited=0;
	if( !inited ){
		dm_init(SINGLE_QSP_ARG);
		inited=1;
	}

	CHECK_AND_PUSH_MENU(data);
}

