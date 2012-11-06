#include "quip_config.h"

char VersionId_datamenu_datamenu[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "data_obj.h"
#include "dataprot.h"

#include "debug.h"
#include "getbuf.h"
#include "query.h"
#include "menuname.h"
#include "fio_api.h"

#define PARENT_PROMPT	"parent data object"

#define N_OBJ_KINDS	5

static const char *parlist[]={"even","odd"};


static COMMAND_FUNC( cr_area );
static COMMAND_FUNC( sel_area );
static COMMAND_FUNC( show_area );

static COMMAND_FUNC( new_hyperseq );
static COMMAND_FUNC( new_seq );
static COMMAND_FUNC( new_frame );
static COMMAND_FUNC( new_row );
static COMMAND_FUNC( new_col );
static COMMAND_FUNC( new_scalar );
static COMMAND_FUNC( do_delvec );
static COMMAND_FUNC( infovec );
/* static COMMAND_FUNC( do_mk_unsigned ); */
static COMMAND_FUNC( mksubimg );
static COMMAND_FUNC( mksubsequence );
static COMMAND_FUNC( mksubvector );
static COMMAND_FUNC( mksubscalar );
static COMMAND_FUNC( do_ilace );
static COMMAND_FUNC( mkcast );
static COMMAND_FUNC( equivalence );
static COMMAND_FUNC( relocate );
static COMMAND_FUNC( do_gen_xpose );
static Data_Obj *get_obj_or_file(QSP_ARG_DECL const char *name);
static COMMAND_FUNC( do_tellprec );

static void finish_obj(QSP_ARG_DECL  const char *s,Dimension_Set *dsp);



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

static COMMAND_FUNC( cr_area )
{
	const char *s;
	/* BUG check for negative input? */
	u_int n;
	dimension_t siz;

	s = NAMEOF("name for this area");
	siz = HOW_MANY("size of data area in bytes");
	n = (u_int) HOW_MANY("maximum number of data objects for this area");

	curr_ap=new_area(QSP_ARG  s,siz,n);
}

static COMMAND_FUNC( sel_area )
{
	Data_Area *ap;

	ap = PICK_DATA_AREA("");
	if( ap != NO_AREA )
		curr_ap=ap;
	else if( curr_ap != NO_AREA ){
		sprintf(ERROR_STRING,"Unable to change data area, current area remains %s.",
			curr_ap->da_name);
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

	data_area_info(ap);
}

static COMMAND_FUNC( do_match_area )
{
	Data_Obj *dp;

	dp = PICK_OBJ("object");
	if( dp == NO_OBJ ) return;

	curr_ap=dp->dt_ap;
}

static COMMAND_FUNC( do_list_data_areas )
{ list_data_areas(SINGLE_QSP_ARG); }

Command ar_ctbl[]={
{ "new_area",	cr_area,	"create new data area"			},
{ "match",	do_match_area,	"select data area to match an object"	},
{ "select",	sel_area,	"select default data area"		},
{ "space",	show_area,	"show space allocation within a data area"},
{ "list",	do_list_data_areas,	"list all data areas"		},
{ "info",	do_area_info,	"print infomation about an area"	},
#ifndef MAC
{ "quit",	popcmd,		"quit"					},
#endif
{ NULL_COMMAND								}
};

COMMAND_FUNC( do_area )
{
	PUSHCMD(ar_ctbl,AREA_MENU_NAME);
}

/* Create and push a new context */

static COMMAND_FUNC( do_push_context )
{
	const char *s;
	Item_Context *icp;

	/* BUG should check dobj_itp? */

#ifdef CAUTIOUS
	if( dobj_itp==NO_ITEM_TYPE )
		ERROR1("CAUTIOUS:  data_obj item type not initialized");
#endif /* CAUTIOUS */

	s=NAMEOF("name for new context");
	icp=create_item_context(QSP_ARG  dobj_itp,s);
	if( icp == NO_ITEM_CONTEXT ) return;

	PUSH_ITEM_CONTEXT(dobj_itp,icp);
}

static COMMAND_FUNC( do_pop_context )
{
	Item_Context *icp;

	icp=pop_item_context(QSP_ARG  dobj_itp);
	/* pop context does not delete the context... */
	delete_item_context(QSP_ARG  icp);
}

Command context_ctbl[]={
{ "push",	do_push_context,	"push a new context"		},
{ "pop",	do_pop_context,		"pop current context"		},
{ "quit",	popcmd,			"exit submenu"			},
{ NULL_COMMAND								}
};

COMMAND_FUNC( do_context )
{
	PUSHCMD(context_ctbl,CONTEXT_MENU_NAME);
}

/* BUG doesn't support double precision complex */

int get_precision(SINGLE_QSP_ARG_DECL)
{
	int p;

	p=WHICH_ONE("data precision",N_NAMED_PRECS,prec_name);
	if( p >= N_MACHINE_PRECS ){
		pseudo_prec pp;

		pp=(pseudo_prec)(p-N_MACHINE_PRECS);
		switch(pp){
			case PP_QUAT:	p = PREC_QUAT;		break;
			case PP_CPX:	p = PREC_CPX;		break;
			case PP_DBLCPX:	p = PREC_DBLCPX;	break;
			case PP_COLOR:	p = PREC_COLOR;		break;
			case PP_VOID:	p = PREC_VOID;		break;
			case PP_STRING: p = PREC_STR;		break;
			case PP_NORM:	p = PREC_SP;		break;
			case PP_BIT:	p = PREC_BIT;		break;
			case PP_LIST:	p = PREC_LIST;		break;
			case PP_CHAR:	p = PREC_CHAR;		break;
#ifdef CAUTIOUS
			case N_PSEUDO_PRECS:
				WARN("CAUTIOUS:  get_precision:  illegal pseudo precision!?");
				break;
			default:
				WARN("CAUTIOUS:  unhandled case in get_precision");
				break;
#endif /* CAUTIOUS */
		}
	}
	return(p);
}

static void finish_obj(QSP_ARG_DECL   const char *s, Dimension_Set *dsp)
{
	prec_t prec;

	prec = get_precision(SINGLE_QSP_ARG);

	if( prec == BAD_PREC ) return;

	if( COLOR_PRECISION(prec) ){
		if( dsp->ds_dimension[0] != 1 ){
			sprintf(ERROR_STRING,"object %s, number of rgb triples per pixel should be 1",s);
			WARN(ERROR_STRING);
		}
		dsp->ds_dimension[0]=3;
		prec = PREC_SP;
	}

	if( make_dobj(QSP_ARG  s,dsp,prec) == NO_OBJ ) {
		sprintf(ERROR_STRING,"couldn't create data object \"%s\"", s);
		WARN(ERROR_STRING);
		return;
	}
}

static COMMAND_FUNC( new_hyperseq )
{
	Dimension_Set dimset;
	const char *s;

	s=NAMEOF("object name");

	dimset.ds_dimension[4]=HOW_MANY("number of sequences");
	dimset.ds_dimension[3]=HOW_MANY("number of frames");
	dimset.ds_dimension[2]=HOW_MANY("number of rows");
	dimset.ds_dimension[1]=HOW_MANY("number of columns");
	dimset.ds_dimension[0]=HOW_MANY("number of components");

	finish_obj(QSP_ARG  s,&dimset);
}

static COMMAND_FUNC( new_seq )
{
	Dimension_Set dimset;
	const char *s;

	s=NAMEOF("object name");

	dimset.ds_dimension[4]=1;
	dimset.ds_dimension[3]=HOW_MANY("number of frames");
	dimset.ds_dimension[2]=HOW_MANY("number of rows");
	dimset.ds_dimension[1]=HOW_MANY("number of columns");
	dimset.ds_dimension[0]=HOW_MANY("number of components");

	finish_obj(QSP_ARG  s,&dimset);
}

static COMMAND_FUNC( new_frame )
{
	Dimension_Set dimset;
	const char *s;

	s=NAMEOF("object name");

	dimset.ds_dimension[4]=1;
	dimset.ds_dimension[3]=1;
	dimset.ds_dimension[2]=HOW_MANY("number of rows");
	dimset.ds_dimension[1]=HOW_MANY("number of columns");
	dimset.ds_dimension[0]=HOW_MANY("number of components");

	finish_obj(QSP_ARG  s,&dimset);
}

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

static COMMAND_FUNC( new_row )
{
	Dimension_Set dimset;
	const char *s;

	s=NAMEOF("object name");

	dimset.ds_dimension[4]=1;
	dimset.ds_dimension[3]=1;
	dimset.ds_dimension[2]=1;
	dimset.ds_dimension[1]=HOW_MANY("number of elements");
	dimset.ds_dimension[0]=HOW_MANY("number of components");

	finish_obj(QSP_ARG  s,&dimset);
}

static COMMAND_FUNC( new_col )
{
	Dimension_Set dimset;
	const char *s;

	s=NAMEOF("object name");

	dimset.ds_dimension[4]=1;
	dimset.ds_dimension[3]=1;
	dimset.ds_dimension[2]=HOW_MANY("number of elements");
	dimset.ds_dimension[1]=1;
	dimset.ds_dimension[0]=HOW_MANY("number of components");

	finish_obj(QSP_ARG  s,&dimset);
}

static COMMAND_FUNC( new_scalar )
{
	Dimension_Set dimset;
	const char *s;

	s=NAMEOF("object name");

	dimset.ds_dimension[4]=1;
	dimset.ds_dimension[3]=1;
	dimset.ds_dimension[2]=1;
	dimset.ds_dimension[1]=1;
	dimset.ds_dimension[0]=HOW_MANY("number of components");

	finish_obj(QSP_ARG  s,&dimset);
}

static COMMAND_FUNC( do_delvec )
{
	Data_Obj *dp;

	dp=PICK_OBJ("");
	if( dp==NO_OBJ ) return;
	delvec(QSP_ARG  dp);
}

static COMMAND_FUNC( infovec )
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

	cols=HOW_MANY("number of columns");
	rows=HOW_MANY("number of rows");

	xos=HOW_MANY("x offset");
	yos=HOW_MANY("y offset");

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
	Dimension_Set dimset;

	s=NAMEOF("name for subsequence");

	dp=PICK_OBJ(PARENT_PROMPT);

	dimset.ds_dimension[1]=HOW_MANY("number of columns");
	dimset.ds_dimension[2]=HOW_MANY("number of rows");
	dimset.ds_dimension[3]=HOW_MANY("number of frames");
	dimset.ds_dimension[4]=1;

	offsets[0]=0;
	offsets[1]=HOW_MANY("x offset");
	offsets[2]=HOW_MANY("y offset");
	offsets[3]=HOW_MANY("t offset");
	offsets[4]=0;

	if( dp==NO_OBJ ) return;
	dimset.ds_dimension[0]=dp->dt_comps;

	newdp=mk_subseq(QSP_ARG  s,dp,offsets,&dimset);
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

	cols=HOW_MANY("number of elements");
	rows=1;

	xos=HOW_MANY("offset");
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
	Dimension_Set dimset;

	s=NAMEOF("name for subscalar");

	dp=PICK_OBJ(PARENT_PROMPT);

	dimset.ds_dimension[0]=HOW_MANY("number of components");
	dimset.ds_dimension[1]=1;
	dimset.ds_dimension[2]=1;
	dimset.ds_dimension[3]=1;
	dimset.ds_dimension[4]=1;

	offsets[0]=HOW_MANY("component offset");
	offsets[1]=0;
	offsets[2]=0;
	offsets[3]=0;
	offsets[4]=0;

	if( dp==NO_OBJ ) return;

	newdp=mk_subseq(QSP_ARG  s,dp,offsets,&dimset);
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

	cols=HOW_MANY("number of columns");
	rows=HOW_MANY("number of rows");
	xos=HOW_MANY("x offset");
	yos=HOW_MANY("y offset");
	tdim=HOW_MANY("type dimension");

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
	int prec;
	Dimension_Set dimset;
	dimension_t ns,nf,nr,nc,nd;

	s=NAMEOF("name for equivalent image");

	dp=PICK_OBJ(PARENT_PROMPT);

	ns=HOW_MANY("number of sequences");
	nf=HOW_MANY("number of frames");
	nr=HOW_MANY("number of rows");
	nc=HOW_MANY("number of columns");
	nd=HOW_MANY("number of components");

	prec = get_precision(SINGLE_QSP_ARG);

	if( dp==NO_OBJ ) return;
	if( prec == BAD_PREC ) return;

	CHECK_POSITIVE(ns,"sequences","equivalence",s)
	CHECK_POSITIVE(nf,"frames","equivalence",s)
	CHECK_POSITIVE(nr,"rows","equivalence",s)
	CHECK_POSITIVE(nc,"columns","equivalence",s)
	CHECK_POSITIVE(nd,"components","equivalence",s)

	dimset.ds_dimension[4]=ns;
	dimset.ds_dimension[3]=nf;
	dimset.ds_dimension[2]=nr;
	dimset.ds_dimension[1]=nc;
	dimset.ds_dimension[0]=nd;

	if( COMPLEX_PRECISION(prec) ){
		if( dimset.ds_dimension[0] != 1 ){
			WARN("Sorry, can only have 1 complex component");
			return;
		}
		//dimset.ds_dimension[0]=2;
	} else if( QUAT_PRECISION(prec) ){
		if( dimset.ds_dimension[0] != 1 ){
			WARN("Sorry, can only have 1 quaternion component");
			return;
		}
		//dimset.ds_dimension[0]=2;
	} else if( COLOR_PRECISION(prec) ){
		if( dimset.ds_dimension[0] != 1 ){
			WARN("Sorry, can only have 1 color triple per pixel");
			return;
		}
advise("component dim 3 for color");
		//dimset.ds_dimension[0]=3;
	}

	if( make_equivalence(QSP_ARG  s,dp,&dimset,prec) == NO_OBJ )
		WARN("error making equivalence");
}

static COMMAND_FUNC( mk_subsample )
{
	const char *s;
	Data_Obj *dp;

	Dimension_Set dimset;
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
		if( dp->dt_type_dim[i] > 1 ){

			if( i < (N_DIMENSIONS-1) )
		sprintf(pmpt,"number of %ss per %s",dimension_name[i],
		dimension_name[i+1]);
			else
		sprintf(pmpt,"number of %ss",dimension_name[i]);

			dimset.ds_dimension[i] = HOW_MANY(pmpt);
			sprintf(pmpt,"%s offset",dimension_name[i]);
			offsets[i] = HOW_MANY(pmpt);
			sprintf(pmpt,"%s increment",dimension_name[i]);
			incrs[i] = HOW_MANY(pmpt);
		} else {
			dimset.ds_dimension[i]=1;
			offsets[i]=0;
			incrs[i]=1;
		}
	}

	if( make_subsamp(QSP_ARG  s,dp,&dimset,offsets,incrs) == NO_OBJ )
		WARN("error making subsamp object");
}

static COMMAND_FUNC( relocate )
{
	Data_Obj *dp;
	index_t x,y,t;

	dp=PICK_OBJ("subimage");
	x=HOW_MANY("x offset");
	y=HOW_MANY("y offset");
	t=HOW_MANY("t offset");
	if( dp==NO_OBJ ) return;
	if( dp->dt_parent == NO_OBJ ){
		sprintf(ERROR_STRING,
	"relocate:  object \"%s\" is not a subimage",
			dp->dt_name);
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

static COMMAND_FUNC( do_tellprec )
{
	Data_Obj *dp;
	const char *s;

	dp = get_obj_or_file( QSP_ARG NAMEOF("data object or open image file") );
	s = NAMEOF("variable name");
	if( dp == NO_OBJ ) return;

	/* BUG should write this in a way that doesn't depend
	 * on the hard-coded constants...
	 */

	ASSIGN_VAR(s,name_for_prec(dp->dt_prec));
}

static COMMAND_FUNC( do_get_align )
{
	int a;

	a=HOW_MANY("alignment (in bytes, negative to disable)");
	set_dp_alignment(a);
}

static COMMAND_FUNC( do_stringify )
{
	Data_Obj *dp;
	const char *s;

	s = NAMEOF("name of variable to hold string value");
	dp = PICK_OBJ("");

	if( dp == NO_OBJ ) return;

	if(  !STRING_PRECISION(dp->dt_prec) ){
		sprintf(ERROR_STRING,"do_stringify:  Sorry, %s does not have string precision",dp->dt_name);
		WARN(ERROR_STRING);
		return;
	}
	ASSIGN_VAR(s,(char *)dp->dt_data);
}

static COMMAND_FUNC( do_import_string )
{
	Data_Obj *dp;
	const char *s;

	dp=PICK_OBJ("");
	s=NAMEOF("string");

	if( dp==NO_OBJ ) return;

	if(  !STRING_PRECISION(dp->dt_prec) ){
		sprintf(ERROR_STRING,
"do_import_string:  Sorry, object %s (%s) does not have string precision",
			dp->dt_name,name_for_prec(dp->dt_prec));
		WARN(ERROR_STRING);
		return;
	}

	if( strlen(s)+1 > dp->dt_n_type_elts ){
		sprintf(ERROR_STRING,"do_import_string:  object %s is too small for string \"%s\"",
			dp->dt_name,s);
		WARN(ERROR_STRING);
		return;
	}

	/* BUG check for contiguity */
	strcpy((char *)dp->dt_data,s);
}

static COMMAND_FUNC( do_list_dobjs ) { list_dobjs(SINGLE_QSP_ARG); }
static COMMAND_FUNC( do_list_temp_dps ) { list_temp_dps(); }
static COMMAND_FUNC( do_unlock_all_tmp_objs ) { unlock_all_tmp_objs(); }

static COMMAND_FUNC( do_protect )
{
	Data_Obj *dp;

	dp=PICK_OBJ("");
	if( dp == NO_OBJ ) return;
	if( IS_STATIC(dp) ){
		sprintf(ERROR_STRING,"do_protect:  Object %s is already static!?",dp->dt_name);
		WARN(ERROR_STRING);
		return;
	}
	dp->dt_flags |= DT_STATIC;
}


Command cr_ctbl[]={
{ "hyperseq",	new_hyperseq,	"create new hyper-sequence"		},
{ "sequence",	new_seq,	"create new sequence"			},
{ "image",	new_frame,	"create new image"			},
{ "vector",	new_row,	"create new row vector"			},
{ "column",	new_col,	"create new column vector"		},
{ "scalar",	new_scalar,	"create new scalar"			},
{ "obj_list",	new_obj_list,	"create new list object"		},

{ "delete",	do_delvec,	"delete object"				},
{ "subsample",	mk_subsample,	"sub- or resample a data object"	},
{ "subimage",	mksubimg,	"create a subimage"			},
{ "subvector",	mksubvector,	"create a subvector"			},
{ "subscalar",	mksubscalar,	"create a subscalar"			},
{ "subsequence",mksubsequence,	"create a subsequence"			},
{ "relocate",	relocate,	"relocate a subimage"			},
{ "equivalence",equivalence,	"equivalence an image to another type"	},
{ "transpose",	do_gen_xpose,	"generalized transpose (in-place)"	},
{ "interlace",	do_ilace,	"create a interlaced subimage"		},
{ "protect",	do_protect,	"make object static"			},
{ "cast",	mkcast,		"recast object dimensions"		},
{ "list",	do_list_dobjs,	"list all data objects"			},
{ "temp_list",	do_list_temp_dps,	"list temporary data objects"		},
{ "info",	infovec,	"give info concerning particular data object"},
/* { "unsigned",	do_mk_unsigned,	"make an integer object unsigned"	}, */
{ "alignment",	do_get_align,	"specify data buffer alignment"		},
{ "precision",	do_tellprec,	"fetch object precision"		},
#ifndef MAC
{ "areas",	do_area,	"data area submenu"			},
{ "contexts",	do_context,	"data context submenu"			},
{ "ascii",	asciimenu,	"read and write ascii data"		},
{ "operate",	buf_ops,	"simple operations on buffers"		},
{ "stringify",	do_stringify,	"set variable from a string object"	},
{ "import_string",	do_import_string,	"set data object from string"	},
{ "unlock_temp_objs",	do_unlock_all_tmp_objs,	"unlock temp objs (when callbacks inhibited)"	},
{ "quit",	popcmd,		"exit submenu"				},
#endif /* ! MAC */

{ NULL_COMMAND								}
};

void dm_init(SINGLE_QSP_ARG_DECL)
{
	static int dm_inited=0;

	if( dm_inited ) return;
	dataobj_init(SINGLE_QSP_ARG);

	/* Version control */
	verdatam(SINGLE_QSP_ARG);

	dm_inited=1;
}

COMMAND_FUNC( datamenu )			/** data object submenu */
{
	dm_init(SINGLE_QSP_ARG);

#ifdef MAC
	/* load the submenus */
	do_area();
	asciimenu();
	buf_ops();
	fiomenu();	/* unix version take this out of data menu!? */
			/* maybe should do this somewhere else on mac? */
#endif	/* MAC */

	PUSHCMD(cr_ctbl,DATA_MENU_NAME);
}


