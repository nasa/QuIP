
#include "quip_config.h"

#include "quip_prot.h"
#include "data_obj.h"

//#include "fio_api.h"


#define PARENT_PROMPT	"parent data object"

//#define N_OBJ_KINDS	5

static const char *parlist[]={"even","odd"};

#ifdef NOT_YET
static Data_Obj *get_obj_or_file(QSP_ARG_DECL const char *name);
#endif /* NOT_YET */
static COMMAND_FUNC( do_tellprec );

Precision * get_precision(SINGLE_QSP_ARG_DECL);


/* BUG - dimension_t is unsigned, because it naturally is, but then we
 * have a problem because how_many can return a negative number,
 * which gets converted to a big unsigned number.  Should we be able to
 * check for these bad values?  Should dimension_t just be signed?
 *
 * For now we leave dimension_t unsigned, but prompt for values using a signed long.
 * This means we can't enter the largest values, but that is better than having
 * the program blow up if we accidentally input a negative number...
 */

#define CHECK_POSITIVE( var, var_string, subrt_name, obj_name )					\
												\
	if( var < 0 ){										\
		sprintf(ERROR_STRING,"%s %s:  number of %s (%d) must be positive",		\
			subrt_name,obj_name,var_string,var);					\
		WARN(ERROR_STRING);								\
		return;										\
	}

static COMMAND_FUNC( do_create_area )
{
	const char *s;
	/* BUG check for negative input? */
	u_int n;
	dimension_t siz;

	s = NAMEOF("name for this area");
	siz = (dimension_t)HOW_MANY("size of data area in bytes");
	n = (u_int) HOW_MANY("maximum number of data objects for this area");

	curr_ap=new_area(QSP_ARG  s,siz,n);	// do_create_area
}

static COMMAND_FUNC( do_select_area )
{
	Data_Area *ap;

	ap = PICK_DATA_AREA("");
	if( ap != NO_AREA )
		curr_ap=ap;
	else if( curr_ap != NO_AREA ){
		sprintf(ERROR_STRING,"Unable to change data area, current area remains %s.",
			AREA_NAME(curr_ap));
		advise(ERROR_STRING);
	}
}

static COMMAND_FUNC( show_area )
{
	Data_Area *ap;

	ap = PICK_DATA_AREA("");
	if( ap == NO_AREA ) return;

	show_area_space(QSP_ARG  ap);
}

static COMMAND_FUNC( do_area_info )
{
	Data_Area *ap;

	ap = PICK_DATA_AREA("");
	if( ap == NO_AREA ) return;

	data_area_info(QSP_ARG  ap);
}

static COMMAND_FUNC( do_match_area )
{
	Data_Obj *dp;

	dp = PICK_OBJ("object");
	if( dp == NO_OBJ ) return;

	curr_ap=OBJ_AREA(dp);
}

static COMMAND_FUNC( do_list_data_areas )
{ list_data_areas(SINGLE_QSP_ARG); }

static COMMAND_FUNC( do_get_area )
{
	const char *s;

	s=NAMEOF("script variable for area name");

//#ifdef CAUTIOUS
//	if( curr_ap == NO_AREA ){
//		WARN("CAUTIOUS:  do_get_area:  No current data area!?");
//		return;
//	}
//#endif // CAUTIOUS
	assert( curr_ap != NO_AREA );

	assign_var(QSP_ARG  s, AREA_NAME(curr_ap));
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
	PUSH_MENU(areas);
}

/* Create and push a new context */

static COMMAND_FUNC( do_push_context )
{
	const char *s;
	Item_Context *icp;

	/* BUG should check dobj_itp? */

	s=NAMEOF("name for new context");
	//icp=create_item_context(QSP_ARG  dobj_itp,s);
	//icp = [[Item_Context alloc] initWithName : [[NSString alloc] initWithUTF8String : s] ];
	NEW_ITEM_CONTEXT(icp);
	//icp = [[Item_Context alloc] initWithName : [[NSString alloc] initWithUTF8String : s] ];
	SET_CTX_NAME(icp,savestr(s));
	//if( icp == NO_ITEM_CONTEXT ) return;

	//PUSH_ITEM_CONTEXT(dobj_itp,icp);
	//[DataObj pushContext : icp];
	push_dobj_context(QSP_ARG  icp);
}

static COMMAND_FUNC( do_pop_context )
{
	pop_dobj_context(SINGLE_QSP_ARG);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(contexts_menu,s,f,h)

MENU_BEGIN(contexts)
ADD_CMD( push,	do_push_context,	push a new context	)
ADD_CMD( pop,	do_pop_context,		pop current context	)
MENU_END(contexts)

static COMMAND_FUNC( do_context )
{
	PUSH_MENU(contexts);
}

Precision * get_precision(SINGLE_QSP_ARG_DECL)
{
	if( prec_itp == NO_ITEM_TYPE )
		init_precisions(SINGLE_QSP_ARG);

	return (Precision *) pick_item(QSP_ARG  prec_itp, "data precision" );
}

static void finish_obj(QSP_ARG_DECL   const char *s, Dimension_Set *dsp, uint32_t type_flag)
{
	Precision * prec_p;

	prec_p = get_precision(SINGLE_QSP_ARG);

	if( prec_p == NO_PRECISION ) return;

	if( COLOR_PRECISION(PREC_CODE(prec_p)) ){
		if( DIMENSION(dsp,0) != 1 ){
			sprintf(ERROR_STRING,"object %s, number of rgb triples per pixel should be 1",s);
			WARN(ERROR_STRING);
		}
		SET_DIMENSION(dsp,0,3);
		prec_p = get_prec(QSP_ARG  "float");
	}

	if( make_dobj_with_shape(QSP_ARG  s,dsp,prec_p,type_flag) == NO_OBJ ) {
		sprintf(ERROR_STRING,"couldn't create data object \"%s\"", s);
		WARN(ERROR_STRING);
	}
}

static COMMAND_FUNC( new_hyperseq )
{
	Dimension_Set ds1, *dsp=(&ds1);
	const char *s;

	s=NAMEOF("object name");

	SET_DIMENSION(dsp,4,HOW_MANY("number of sequences"));
	SET_DIMENSION(dsp,3,HOW_MANY("number of frames"));
	SET_DIMENSION(dsp,2,HOW_MANY("number of rows"));
	SET_DIMENSION(dsp,1,HOW_MANY("number of columns"));
	SET_DIMENSION(dsp,0,HOW_MANY("number of components"));

	finish_obj(QSP_ARG  s,dsp,DT_HYPER_SEQ);
}

static COMMAND_FUNC( new_seq )
{
	Dimension_Set ds1, *dsp=(&ds1);
	const char *s;

	s=NAMEOF("object name");

	SET_DIMENSION(dsp,4,1);
	SET_DIMENSION(dsp,3,HOW_MANY("number of frames"));
	SET_DIMENSION(dsp,2,HOW_MANY("number of rows"));
	SET_DIMENSION(dsp,1,HOW_MANY("number of columns"));
	SET_DIMENSION(dsp,0,HOW_MANY("number of components"));

	finish_obj(QSP_ARG  s,dsp,DT_SEQUENCE);
}

static COMMAND_FUNC( new_frame )
{
	Dimension_Set ds1;
	Dimension_Set *dsp=(&ds1);
	const char *s;

	s=NAMEOF("object name");

	SET_DIMENSION(dsp,4,1);
	SET_DIMENSION(dsp,3,1);
	SET_DIMENSION(dsp,2,HOW_MANY("number of rows"));
	SET_DIMENSION(dsp,1,HOW_MANY("number of columns"));
	SET_DIMENSION(dsp,0,HOW_MANY("number of components"));
	finish_obj(QSP_ARG  s,dsp,DT_IMAGE);
}

static COMMAND_FUNC( new_gen_obj )
{
	Dimension_Set ds1, *dsp=(&ds1);
	const char *s;

	s=NAMEOF("object name");

	SET_DIMENSION(dsp,4,HOW_MANY("number of sequences"));
	SET_DIMENSION(dsp,3,HOW_MANY("number of frames"));
	SET_DIMENSION(dsp,2,HOW_MANY("number of rows"));
	SET_DIMENSION(dsp,1,HOW_MANY("number of columns"));
	SET_DIMENSION(dsp,0,HOW_MANY("number of components"));
	finish_obj(QSP_ARG  s,dsp,AUTO_SHAPE);
}

#ifdef NOT_YET
static COMMAND_FUNC( new_obj_list )
{
	int n;
	const char *s;
	List *lp;
	Node *np;

	s=NAMEOF("object name");

	n=HOW_MANY("number of objects in this list");
	lp=new_list();
	while(n--){
		Data_Obj *dp;

		dp = PICK_OBJ("");
		if( dp != NO_OBJ ){
			np=mk_node(dp);
			addTail(lp,np);
		}
	}

	if( make_obj_list(QSP_ARG  s,lp) == NO_OBJ ){
		sprintf(ERROR_STRING,"error making object list %s",s);
		WARN(ERROR_STRING);
	}
}
#endif /* NOT_YET */

static COMMAND_FUNC( new_row )
{
	Dimension_Set ds1, *dsp=(&ds1);
	const char *s;

	s=NAMEOF("object name");

	SET_DIMENSION(dsp,4,1);
	SET_DIMENSION(dsp,3,1);
	SET_DIMENSION(dsp,2,1);
	SET_DIMENSION(dsp,1,HOW_MANY("number of elements"));
	SET_DIMENSION(dsp,0,HOW_MANY("number of components"));

	finish_obj(QSP_ARG  s,dsp,DT_ROWVEC);
}

static COMMAND_FUNC( new_col )
{
	Dimension_Set ds1, *dsp=(&ds1);
	const char *s;

	s=NAMEOF("object name");

	SET_DIMENSION(dsp,4,1);
	SET_DIMENSION(dsp,3,1);
	SET_DIMENSION(dsp,2,HOW_MANY("number of elements"));
	SET_DIMENSION(dsp,1,1);
	SET_DIMENSION(dsp,0,HOW_MANY("number of components"));

	finish_obj(QSP_ARG  s,dsp,DT_COLVEC);
}

static COMMAND_FUNC( new_scalar )
{
	Dimension_Set ds1;
	Dimension_Set *dsp=(&ds1);
	const char *s;

	s=NAMEOF("object name");

	SET_DIMENSION(dsp,4,1);
	SET_DIMENSION(dsp,3,1);
	SET_DIMENSION(dsp,2,1);
	SET_DIMENSION(dsp,1,1);
	SET_DIMENSION(dsp,0,HOW_MANY("number of components"));

	finish_obj(QSP_ARG  s,dsp,DT_SCALAR);
}

static COMMAND_FUNC( do_delvec )
{
	Data_Obj *dp;

	dp=PICK_OBJ("");
	if( dp==NO_OBJ ) return;
	delvec(QSP_ARG  dp);
}

static COMMAND_FUNC( do_dobj_info )
{
	Data_Obj *dp;

	dp=PICK_OBJ("");
	if( dp==NO_OBJ ) return;

	LONGLIST(dp);
}

/* BUG	deleteing an object should automatically cause the
	deletion of all subobjects!! */

static COMMAND_FUNC( mksubimg )
{
	const char *s;
	Data_Obj *dp, *newdp;
	incr_t rows, cols;
	incr_t xos, yos;

	s=NAMEOF("name for subimage");

	dp=PICK_OBJ(PARENT_PROMPT);

	cols=(incr_t)HOW_MANY("number of columns");
	rows=(incr_t)HOW_MANY("number of rows");

	xos=(incr_t)HOW_MANY("x offset");
	yos=(incr_t)HOW_MANY("y offset");

	if( dp==NO_OBJ ) return;

	CHECK_POSITIVE(rows,"rows","mksubimg",s)
	CHECK_POSITIVE(cols,"columns","mksubimg",s)

	newdp=mk_subimg(QSP_ARG  dp,xos,yos,s,(dimension_t)rows,(dimension_t)cols);
	if( newdp == NO_OBJ )
		WARN("couldn't create subimage");
}

static COMMAND_FUNC( mksubsequence )
{
	const char *s;
	Data_Obj *dp, *newdp;
	index_t offsets[N_DIMENSIONS];
	Dimension_Set ds1, *dsp=(&ds1);

	s=NAMEOF("name for subsequence");

	dp=PICK_OBJ(PARENT_PROMPT);

	SET_DIMENSION(dsp,1,HOW_MANY("number of columns"));
	SET_DIMENSION(dsp,2,HOW_MANY("number of rows"));
	SET_DIMENSION(dsp,3,HOW_MANY("number of frames"));
	SET_DIMENSION(dsp,4,1);

	offsets[0]=0;
	offsets[1]=(index_t)HOW_MANY("x offset");
	offsets[2]=(index_t)HOW_MANY("y offset");
	offsets[3]=(index_t)HOW_MANY("t offset");
	offsets[4]=0;

	if( dp==NO_OBJ ) return;
	SET_DIMENSION(dsp,0,OBJ_COMPS(dp));

	newdp=mk_subseq(QSP_ARG  s,dp,offsets,dsp);
	if( newdp == NO_OBJ )
		WARN("couldn't create subimage");
}

static COMMAND_FUNC( mksubvector )
{
	const char *s;
	Data_Obj *dp, *newdp;
	dimension_t rows;
	dimension_t cols;
	index_t xos, yos;

	s=NAMEOF("name for subvector");

	dp=PICK_OBJ(PARENT_PROMPT);

	cols=(dimension_t)HOW_MANY("number of elements");
	rows=1;

	xos=(index_t)HOW_MANY("offset");
	yos=0;

	if( dp==NO_OBJ ) return;

	CHECK_POSITIVE(cols,"elements","mksubvector",s)

	newdp=mk_subimg(QSP_ARG  dp,xos,yos,s,rows,cols);
	if( newdp == NO_OBJ )
		WARN("couldn't create subimage");
}

static COMMAND_FUNC( mksubscalar )
{
	const char *s;
	Data_Obj *dp, *newdp;
	index_t offsets[N_DIMENSIONS];
	Dimension_Set ds1, *dsp=(&ds1);

	s=NAMEOF("name for subscalar");

	dp=PICK_OBJ(PARENT_PROMPT);

	SET_DIMENSION(dsp,0,HOW_MANY("number of components"));
	SET_DIMENSION(dsp,1,1);
	SET_DIMENSION(dsp,2,1);
	SET_DIMENSION(dsp,3,1);
	SET_DIMENSION(dsp,4,1);

	offsets[0]=(index_t)HOW_MANY("component offset");
	offsets[1]=0;
	offsets[2]=0;
	offsets[3]=0;
	offsets[4]=0;

	if( dp==NO_OBJ ) return;

	newdp=mk_subseq(QSP_ARG  s,dp,offsets,dsp);
	if( newdp == NO_OBJ )
		WARN("couldn't create subscalar");
}

static COMMAND_FUNC( do_ilace )
{
	const char *s;
	Data_Obj *dp, *newdp;
	int parity;

	s=NAMEOF("name for subimage");

	dp=get_img( QSP_ARG  NAMEOF("name of parent image") );
	if( dp==NO_OBJ ) return;

	parity=WHICH_ONE("parity of selected lines",2,parlist);
	if( parity < 0 ) return;

	newdp=mk_ilace(QSP_ARG  dp,s,parity);
	if( newdp == NO_OBJ )
		WARN("couldn't create interlaced subimage");
}

static COMMAND_FUNC( mkcast )
{
	const char *s;
	Data_Obj *dp, *newdp;
	dimension_t rows, cols, tdim;
	index_t xos, yos;

	s=NAMEOF("name for cast");

	dp=PICK_OBJ(PARENT_PROMPT);
	if( dp==NO_OBJ ) return;

	cols=(dimension_t)HOW_MANY("number of columns");
	rows=(dimension_t)HOW_MANY("number of rows");
	xos=(index_t)HOW_MANY("x offset");
	yos=(index_t)HOW_MANY("y offset");
	tdim=(dimension_t)HOW_MANY("type dimension");

	CHECK_POSITIVE(cols,"columns","mkcast",s)
	CHECK_POSITIVE(rows,"rows","mkcast",s)
	CHECK_POSITIVE(tdim,"type dimension","mkcast",s)

	newdp=nmk_subimg(QSP_ARG  dp,xos,yos,s,rows,cols,tdim);
	if( newdp == NO_OBJ )
		WARN("couldn't create subimage");
}

static COMMAND_FUNC( equivalence )
{
	const char *s;
	Data_Obj *dp;
	Precision * prec_p;
	Dimension_Set ds1, *dsp=(&ds1);
	dimension_t ns,nf,nr,nc,nd;

	s=NAMEOF("name for equivalent image");

	dp=PICK_OBJ(PARENT_PROMPT);

	ns=(dimension_t)HOW_MANY("number of sequences");
	nf=(dimension_t)HOW_MANY("number of frames");
	nr=(dimension_t)HOW_MANY("number of rows");
	nc=(dimension_t)HOW_MANY("number of columns");
	nd=(dimension_t)HOW_MANY("number of components");

	prec_p = get_precision(SINGLE_QSP_ARG);

	if( dp==NO_OBJ ) return;
	if( prec_p == NO_PRECISION ) return;

	CHECK_POSITIVE(ns,"sequences","equivalence",s)
	CHECK_POSITIVE(nf,"frames","equivalence",s)
	CHECK_POSITIVE(nr,"rows","equivalence",s)
	CHECK_POSITIVE(nc,"columns","equivalence",s)
	CHECK_POSITIVE(nd,"components","equivalence",s)

	SET_DIMENSION(dsp,4,ns);
	SET_DIMENSION(dsp,3,nf);
	SET_DIMENSION(dsp,2,nr);
	SET_DIMENSION(dsp,1,nc);
	SET_DIMENSION(dsp,0,nd);

	if( COMPLEX_PRECISION(PREC_CODE(prec_p)) ){
		if( DIMENSION(dsp,0) != 1 ){
			WARN("Sorry, can only have 1 complex component");
			return;
		}
		//SET_DIMENSION(dsp,0,2);
	} else if( QUAT_PRECISION(PREC_CODE(prec_p)) ){
		if( DIMENSION(dsp,0) != 1 ){
			WARN("Sorry, can only have 1 quaternion component");
			return;
		}
		//SET_DIMENSION(dsp,0,2);
	} else if( COLOR_PRECISION(PREC_CODE(prec_p)) ){
		if( DIMENSION(dsp,0) != 1 ){
			WARN("Sorry, can only have 1 color triple per pixel");
			return;
		}
advise("component dim 3 for color");
		//SET_DIMENSION(dsp,0,3);
	}

	if( make_equivalence(QSP_ARG  s,dp,dsp,prec_p) == NO_OBJ )
		WARN("error making equivalence");
}

static COMMAND_FUNC( mk_subsample )
{
	const char *s;
	Data_Obj *dp;

	Dimension_Set ds1, *dsp=(&ds1);
	index_t offsets[N_DIMENSIONS];
	incr_t incrs[N_DIMENSIONS];
	char pmpt[LLEN];
	int i;

	s=NAMEOF("name for subsample object");

	dp=PICK_OBJ(PARENT_PROMPT);

	/* BUG?  We violate the rule of returning before getting
	 * all arguments, because the fields of dp are needed
	 * to determine what to prompt for!?
	 */

	if( dp==NO_OBJ ) return;

	for(i=0;i<N_DIMENSIONS;i++){
		/* BUG? should we prompt for all dimensions? */
		if( OBJ_TYPE_DIM(dp,i) > 1 ){
			long l;

			if( i < (N_DIMENSIONS-1) )
		sprintf(pmpt,"number of %ss per %s",dimension_name[i],
		dimension_name[i+1]);
			else
		sprintf(pmpt,"number of %ss",dimension_name[i]);

			l=HOW_MANY(pmpt);
			SET_DIMENSION(dsp,i,l);

			sprintf(pmpt,"%s offset",dimension_name[i]);
			l=HOW_MANY(pmpt);
			offsets[i] = (index_t) l;
			sprintf(pmpt,"%s increment",dimension_name[i]);
			incrs[i] =(incr_t)HOW_MANY(pmpt);
		} else {
			SET_DIMENSION(dsp,i,1);
			offsets[i]=0;
			incrs[i]=1;
		}
	}

	if( make_subsamp(QSP_ARG  s,dp,dsp,offsets,incrs) == NO_OBJ )
		WARN("error making subsamp object");
}

static COMMAND_FUNC( relocate )
{
	Data_Obj *dp;
	index_t x,y,t;

	dp=PICK_OBJ("subimage");
	x=(index_t)HOW_MANY("x offset");
	y=(index_t)HOW_MANY("y offset");
	t=(index_t)HOW_MANY("t offset");
	if( dp==NO_OBJ ) return;
	if( OBJ_PARENT(dp) == NO_OBJ ){
		sprintf(ERROR_STRING,
	"relocate:  object \"%s\" is not a subimage",
			OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}
	_relocate(QSP_ARG  dp,x,y,t);
}

static COMMAND_FUNC( do_gen_xpose )
{
	Data_Obj *dp;
	int d1,d2;

	dp = PICK_OBJ("");
	d1=(int)HOW_MANY("dimension index #1");
	d2=(int)HOW_MANY("dimension index #2");

	if( dp == NO_OBJ ) return;

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
	dp = hunt_obj(QSP_ARG  name);
	if( dp != NO_OBJ ) return(dp);

#ifndef PC
	ifp = img_file_of(QSP_ARG  name);
	if( ifp!=NO_IMAGE_FILE ) return(ifp->if_dp);

	sprintf(ERROR_STRING,"No object or open file \"%s\"",name);
	WARN(ERROR_STRING);
#else /* PC */
	sprintf(ERROR_STRING,
		"No object \"%s\" (not checking for files!?)",name);
	WARN(ERROR_STRING);
#endif /* ! PC */

	return(dp);
}
#endif /* NOT_YET */

static COMMAND_FUNC( do_tellprec )
{
	Data_Obj *dp;
	const char *s;

	//dp = get_obj_or_file( QSP_ARG NAMEOF("data object or open image file") );
	dp = get_obj( QSP_ARG NAMEOF("data object") );
	s = NAMEOF("variable name");
	if( dp == NO_OBJ ) return;

	/* BUG should write this in a way that doesn't depend
	 * on the hard-coded constants...
	 */

	ASSIGN_VAR(s,OBJ_PREC_NAME(dp));
}

static COMMAND_FUNC( do_get_align )
{
	int a;

	a=(int)HOW_MANY("alignment (in bytes, negative to disable)");
	set_dp_alignment(a);
}

static COMMAND_FUNC( do_stringify )
{
	Data_Obj *dp;
	const char *s;

	s = NAMEOF("name of variable to hold string value");
	dp = PICK_OBJ("");

	if( dp == NO_OBJ ) return;

	if(  !STRING_PRECISION(OBJ_PREC(dp)) ){
		sprintf(ERROR_STRING,"do_stringify:  Sorry, %s does not have string precision",OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}
	ASSIGN_VAR(s,(char *)OBJ_DATA_PTR(dp));
}

static COMMAND_FUNC( do_import_string )
{
	Data_Obj *dp;
	const char *s;

	dp=PICK_OBJ("");
	s=NAMEOF("string");

	if( dp==NO_OBJ ) return;

	if(  !STRING_PRECISION(OBJ_PREC(dp)) ){
		sprintf(ERROR_STRING,
"do_import_string:  Sorry, object %s (%s) does not have string precision",
			OBJ_NAME(dp),OBJ_PREC_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}

	if( strlen(s)+1 > OBJ_N_TYPE_ELTS(dp) ){
		sprintf(ERROR_STRING,"do_import_string:  object %s is too small for string \"%s\"",
			OBJ_NAME(dp),s);
		WARN(ERROR_STRING);
		return;
	}

	/* BUG check for contiguity */
	strcpy((char *)OBJ_DATA_PTR(dp),s);
}

static COMMAND_FUNC( do_list_dobjs ) { list_dobjs(SINGLE_QSP_ARG); }
static COMMAND_FUNC( do_list_temp_dps ) { list_temp_dps(SINGLE_QSP_ARG); }
static COMMAND_FUNC( do_unlock_all_tmp_objs ) { unlock_all_tmp_objs(SINGLE_QSP_ARG); }

static COMMAND_FUNC( do_protect )
{
	Data_Obj *dp;

	dp=PICK_OBJ("");
	if( dp == NO_OBJ ) return;
	if( IS_STATIC(dp) ){
		sprintf(ERROR_STRING,"do_protect:  Object %s is already static!?",OBJ_NAME(dp));
		WARN(ERROR_STRING);
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

// BUG DataObj needs to have a fancier get to parse subscripts...

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
ADD_CMD( subvector,	mksubvector,	create a subvector	)
ADD_CMD( subscalar,	mksubscalar,	create a subscalar	)
ADD_CMD( subsequence,	mksubsequence,	create a subsequence	)
ADD_CMD( relocate,	relocate,	relocate a subimage	)
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
ADD_CMD( stringify,	do_stringify,	set variable from a string object	)
ADD_CMD( import_string,	do_import_string,	set data object from string	)
ADD_CMD( unlock_temp_objs,	do_unlock_all_tmp_objs,	unlock temp objs (when callbacks inhibited)	)

MENU_END(data)

COMMAND_FUNC( do_dobj_menu )
{
	static int inited=0;
	if( !inited ){
		dm_init(SINGLE_QSP_ARG);
		inited=1;
	}

	PUSH_MENU(data);
}

