#include "quip_config.h"

char VersionId_qutil_function[] = QUIP_VERSION_STRING;

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>		/* stat(2) */
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	/* drand48 */
#endif

#include "function.h"
#include "query.h"
#include "rn.h"
#include "fileck.h"
#include "vecgen.h"
#include "query.h"		/* BUG dependency problem? */
#include "macros.h"		/* BUG dependency problem? */

/* local prototypes */
static void init_sizable_class(SINGLE_QSP_ARG_DECL);
static void init_tsable_class(SINGLE_QSP_ARG_DECL);

#ifdef SINE_TBL

/*
 * Table lookup of sin's and cosine's is probably faster than the math
 * library on machines without floating point assist - but on the Sparc2,
 * table lookup (as implemented here) is twice as slow!?
 */

double t_sin(double);
double t_cos(double);

static double *sine_tbl=NULL;
static double pi,one_over_pi;

#define N_SINE_TBL	0x400		/* must be a power of 2 */
#define SINE_TBL_MASK	((N_SINE_TBL*2)-1)

static void init_sine_tbl()
{
	int i;
	double arg, arginc;

	sine_tbl = getbuf( sizeof(double) * N_SINE_TBL );
	pi = 4*atan(1);

	advise("Initializing sine table...");
	arg=0.0;
	arginc = pi / N_SINE_TBL;
	for(i=0;i<N_SINE_TBL;i++){
		sine_tbl[i] = sin(arg);
		arg += arginc;
	}
	advise("Done initializing sine table.");
}

double t_sin(arg)
double arg;
{
	int index;

	if( sine_tbl == NULL )
		init_sine_tbl();

	/* map to a table index */

	index = N_SINE_TBL * arg * one_over_pi;
	index &= SINE_TBL_MASK;
	if( index >= N_SINE_TBL ){
		index -= N_SINE_TBL;
		return( - sine_tbl[index] );
	} else
		return(sine_tbl[index]);
}

double t_cos(arg)
double arg;
{
	int index;

	if( sine_tbl == NULL )
		init_sine_tbl();

	/* map to a table index */

	index = N_SINE_TBL * arg * one_over_pi;
	index += N_SINE_TBL>>1;
	index &= SINE_TBL_MASK;
	if( index >= N_SINE_TBL ){
		index -= N_SINE_TBL;
		return( - sine_tbl[index] );
	} else
		return(sine_tbl[index]);
}
#endif /* SINE_TBL */


/* Dummy placeholder functions for the functions from the data library -
 * They are used to initialize the function vectors, which are reset when
 * the data library initialized...
 */

double nullvfunc(void) { return(0.0); }
double nulld0func(void) { return(0.0); }
double nulld1func(double d) { return(0.0); }
double nulld2func(double d1,double d2) { return(0.0); }
double nulldofunc(Data_Obj *dp) { return(0.0); }
double nullszfunc(QSP_ARG_DECL  Item *ip) { return(0.0); }
double nulltsfunc(QSP_ARG_DECL  Item *ip, dimension_t frame ) { return(0.0); }
double nulls1func(QSP_ARG_DECL  const char *s) { return(0.0); }
double nulls2func(const char *s1,const char *s2) { return(0.0); }
double nulls3func(const char *s1,const char *s2,int n) { return(0.0); }

double modtimefunc(QSP_ARG_DECL  const char *s)
{
	struct stat statb;

	if( stat(s,&statb) < 0 ){
		tell_sys_error(s);
		return(0.0);
	}
	return( (double) statb.st_mtime );
}

static double rn_uni(double arg)		/* arg is not used... */
{
	rninit(SGL_DEFAULT_QSP_ARG);
	return( drand48() );
}

static double rn_number(double dlimit)
{
	double dret;
	int ilimit, iret;

	ilimit=(int)dlimit;
	iret=(int)rn(ilimit);
	dret=iret;
	return(dret);
}

static double dstrstr(const char *s1,const char *s2)
{
	char *s;
	s=strstr(s1,s2);
	if( s == NULL ) return(-1);
	else return( (double)(s - s1) );
}

static double dstrcmp(const char *s1,const char *s2)
{
	double d;
	d=strcmp(s1,s2);
	return(d);
}

static double dstrncmp(const char *s1,const char *s2,int n)
{
	double d;
	d=strncmp(s1,s2,n);
	return(d);
}

static double dstrlen(QSP_ARG_DECL  const char *s1)
{
	double d;
	d=strlen(s1);
	return(d);
}

static double dexists(QSP_ARG_DECL  const char *fname)
{
	if( path_exists(fname) )
		return(1.0);
	else	return(0.0);
}

static double bitsum(double num)
{
	long l;
	long bit;
	int i,sum;

	sum=0;
	l=(long)num;
	bit=1;
	for(i=0;i<32;i++){
		if( l & bit ) sum++;
		bit <<= 1;
	}
	num = sum;
	return( num );
}

static double d_isdigit(double num)
{
	long l;

	l=num;
	if( l < 0 || l > 255 ) return(-1.0);

	if( isdigit(l) )
		return(1.0);
	else
		return(0.0);
}

static double maxfunc(double a1,double a2)
{
	if( a1 >= a2 ) return(a1);
	else return(a2);
}

static double minfunc(double a1,double a2)
{
	if( a1 <= a2 ) return(a1);
	else return(a2);
}

static Item_Class *siz_icp=NO_ITEM_CLASS;
static Item_Class *ts_icp=NO_ITEM_CLASS;

static void init_sizable_class(SINGLE_QSP_ARG_DECL)
{
#ifdef CAUTIOUS
	if( siz_icp != NO_ITEM_CLASS ){
		WARN("CAUTIOUS:  redundant call to init_sizable_class");
		return;
	}
#endif /* CAUTIOUS */
	siz_icp = new_item_class(QSP_ARG  "sizable");
}

void add_sizable(QSP_ARG_DECL  Item_Type *itp,Size_Functions *sfp,Item *(*lookup)(QSP_ARG_DECL  const char *))
{
	if( siz_icp == NO_ITEM_CLASS )
		init_sizable_class(SINGLE_QSP_ARG);

	add_items_to_class(siz_icp,itp,sfp,lookup);
}

Item *find_sizable(QSP_ARG_DECL  const char *name)
{
	if( siz_icp == NO_ITEM_CLASS )
		init_sizable_class(SINGLE_QSP_ARG);

	return( get_member(QSP_ARG  siz_icp,name) );
}

static void init_tsable_class(SINGLE_QSP_ARG_DECL)
{
#ifdef CAUTIOUS
	if( ts_icp != NO_ITEM_CLASS ){
		WARN("CAUTIOUS:  redundant call to init_tsable_class");
		return;
	}
#endif /* CAUTIOUS */
	ts_icp = new_item_class(QSP_ARG  "tsable");
}

void add_tsable(QSP_ARG_DECL  Item_Type *itp,Timestamp_Functions *sfp,Item *(*lookup)(QSP_ARG_DECL  const char *))
{
	if( ts_icp == NO_ITEM_CLASS )
		init_tsable_class(SINGLE_QSP_ARG);

	add_items_to_class(ts_icp,itp,sfp,lookup);
}

Item *find_tsable(QSP_ARG_DECL  const char *name)
{
	if( ts_icp == NO_ITEM_CLASS )
		init_tsable_class(SINGLE_QSP_ARG);

	return( get_member(QSP_ARG  ts_icp,name) );
}

static Size_Functions *get_size_functions(QSP_ARG_DECL  Item *ip)
{
	Member_Info *mip;

	if( siz_icp == NO_ITEM_CLASS )
		init_sizable_class(SINGLE_QSP_ARG);

	mip = get_member_info(QSP_ARG  siz_icp,ip->item_name);
#ifdef CAUTIOUS
	if( mip == NO_MEMBER_INFO ){
		sprintf(ERROR_STRING,
			"CAUTIOUS:  get_size_functions %s, missing member info #2",
			ip->item_name);
		NERROR1(ERROR_STRING);
	}
#endif /* CAUTIOUS */

	return( (Size_Functions *) mip->mi_data );
}

double get_object_size(QSP_ARG_DECL  Item *ip,int d_index)
{
	Size_Functions *sfp;

	if( ip == NO_ITEM ) return(0.0);

	sfp = get_size_functions(QSP_ARG  ip);

#ifdef CAUTIOUS
	if( sfp == NULL ) ERROR1("CAUTIOUS:  get_object_size:  shouldn't happen");
#endif /* CAUTIOUS */

	return( (*sfp->sz_func)(ip,d_index) );
}

double get_interlace_flag(QSP_ARG_DECL  Item *ip)
{
	Size_Functions *sfp;

	if( ip == NO_ITEM ) return(0.0);

	sfp = get_size_functions(QSP_ARG  ip);

#ifdef CAUTIOUS
	if( sfp == NULL ) ERROR1("CAUTIOUS:  get_interlace_flag:  shouldn't happen");
#endif /* CAUTIOUS */

	if( sfp->il_func == NULL ){
		sprintf(ERROR_STRING,"Sorry, is_interlaced() is not defined for object %s",ip->item_name);
		WARN(ERROR_STRING);
		return(0.0);
	}

	return( (*sfp->il_func)(ip) );
}

double get_timestamp(QSP_ARG_DECL  Item *ip,int func_index,dimension_t frame)
{
	Timestamp_Functions *tsfp;
	Member_Info *mip;
	double d;

	if( ip == NO_ITEM ) return(0.0);

	if( ts_icp == NO_ITEM_CLASS )
		init_tsable_class(SINGLE_QSP_ARG);

	mip = get_member_info(QSP_ARG  ts_icp,ip->item_name);

#ifdef CAUTIOUS
	if( mip == NO_MEMBER_INFO ){
		sprintf(ERROR_STRING,
			"CAUTIOUS:  get_timestamp %s %d, missing member info",
			ip->item_name,func_index);
		ERROR1(ERROR_STRING);
	}
#endif /* CAUTIOUS */


	tsfp = (Timestamp_Functions *) mip->mi_data;

	d = (*(tsfp->ts_func[func_index]))(ip,frame);
	return( d );
}

Item *sub_sizable(QSP_ARG_DECL  Item *ip,index_t index)
{
	Size_Functions *sfp;
	Member_Info *mip;

	/* currently data objects are the only sizables
		which can be subscripted */

	if( ip == NO_ITEM ) return(ip);

	if( siz_icp == NO_ITEM_CLASS )
		init_sizable_class(SINGLE_QSP_ARG);

	mip = get_member_info(QSP_ARG  siz_icp,ip->item_name);

#ifdef CAUTIOUS
	if( mip == NO_MEMBER_INFO )
		ERROR1("CAUTIOUS:  missing member info #3");
#endif /* CAUTIOUS */

	sfp = (Size_Functions *) mip->mi_data;

	if( sfp->subscript == NULL ){
		sprintf(ERROR_STRING,"Can't subscript sizable object %s",
			ip->item_name);
		WARN(ERROR_STRING);
		return(NO_ITEM);
	}

	return( (*sfp->subscript)(ip,index) );
}

Item *csub_sizable(QSP_ARG_DECL  Item *ip,index_t index)
{
	Size_Functions *sfp;
	Member_Info *mip;
	Item *ret_ip;

	/* currently data objects are the only sizables
		which can be subscripted */

	if( ip == NO_ITEM ){
ERROR1("csub_sizable passed null item!?");
		return(ip);
	}

	if( siz_icp == NO_ITEM_CLASS )
		init_sizable_class(SINGLE_QSP_ARG);

	mip = get_member_info(QSP_ARG  siz_icp,ip->item_name);

#ifdef CAUTIOUS
	if( mip == NO_MEMBER_INFO )
		ERROR1("CAUTIOUS:  missing member info #1");
#endif /* CAUTIOUS */

	sfp = (Size_Functions *) mip->mi_data;

	if( sfp->csubscript == NULL ){
		sprintf(ERROR_STRING,"Can't subscript sizable object %s",
			ip->item_name);
		WARN(ERROR_STRING);
		return(NO_ITEM);
	}

	ret_ip = (*sfp->csubscript)(ip,index);
#ifdef CAUTIOUS
	if( ret_ip == NO_ITEM ){
		sprintf(ERROR_STRING,"CAUTIOUS:  csub_sizable:  csubscript func returned NO_ITEM!?");
		ERROR1(ERROR_STRING);
	}
#endif /* CAUTIOUS */
	return( ret_ip );
}

static double _ilfunc(QSP_ARG_DECL  Item *ip)
{ return( get_interlace_flag(QSP_ARG  ip) ); }

static double _dpfunc(QSP_ARG_DECL  Item *ip)
{ return( get_object_size(QSP_ARG  ip,0) ); }

static double _colfunc(QSP_ARG_DECL  Item *ip)
{ return( get_object_size(QSP_ARG  ip,1) ); }

static double _rowfunc(QSP_ARG_DECL  Item *ip)
{ return( get_object_size(QSP_ARG  ip,2) ); }

static double _frmfunc(QSP_ARG_DECL  Item *ip)
{ return( get_object_size(QSP_ARG  ip,3) ); }

static double _seqfunc(QSP_ARG_DECL  Item *ip)
{ return( get_object_size(QSP_ARG  ip,4) ); }

static double _nefunc(QSP_ARG_DECL  Item *ip)
{
	int i;
	double d;

	d=1;
	for(i=0;i<N_DIMENSIONS;i++)
		d *= get_object_size(QSP_ARG  ip,i);

	return(d);
}

static double _secfunc(QSP_ARG_DECL  Item *ip, dimension_t frame )
{ return( get_timestamp(QSP_ARG  ip,0,frame)); }

static double _msecfunc(QSP_ARG_DECL  Item *ip, dimension_t frame )
{ return( get_timestamp(QSP_ARG  ip,1,frame) ); }

static double _usecfunc(QSP_ARG_DECL  Item *ip, dimension_t frame )
{ return( get_timestamp(QSP_ARG  ip,2,frame) ); }


static double signfunc(double x)
{ if( x > 0 ) return(1.0); else if( x < 0 ) return(-1.0); else return(0.0); }

#ifndef HAVE_ROUND
double round(double arg) { return floor(arg+0.5); }
#endif /* ! HAVE_ROUND */


/*
 * ANAL BUG?  doesn't compile with strict prototyping
 * of the function pointer...  jbm 3/17/95
 */

/* Even though uni() doesn't need an arg, we'd like to be able to
 * give it a vector arg to indicate that we want a vector value,
 * not a scalar value:
 *
 * x = uni();		# same as:  tmp_scalar = uni(); x = tmp_scalar;
 * vs.  x = uni(x)	# every pixel of x set to a different value...
 */

#define M0_FN( name )		VOID_FUNCTYP , { v_func:name }
Function math0_functbl[]={
/*
{ "uni",	M0_FN(rn_uni),	FVUNI,		INVALID_VFC,	INVALID_VFC },
*/
{ "",		M0_FN(nulld0func),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC }
};

#define M1_FN( name )		D1_FUNCTYP , {d1_func:name}

Function math1_functbl[]={
{ "uni",	M1_FN(rn_uni),		FVUNI,		INVALID_VFC,	INVALID_VFC },
{ "sqrt",	M1_FN(sqrt),		FVSQRT,		INVALID_VFC,	INVALID_VFC },
{ "sin",	M1_FN(sin),		FVSIN,		INVALID_VFC,	INVALID_VFC },
{ "cos",	M1_FN(cos),		FVCOS,		INVALID_VFC,	INVALID_VFC },
{ "erf",	M1_FN(erf),		FVERF,		INVALID_VFC,	INVALID_VFC },
#ifdef SINE_TBL
{ "t_sin",	M1_FN(t_sin),	FVSIN,		INVALID_VFC,	INVALID_VFC },
{ "t_cos",	M1_FN(t_cos),	FVCOS,		INVALID_VFC,	INVALID_VFC },
#endif /* SINE_TBL */
{ "exp",	M1_FN(exp),		FVEXP,		INVALID_VFC,	INVALID_VFC },
{ "log",	M1_FN(log),		FVLOG,		INVALID_VFC,	INVALID_VFC },
{ "log10",	M1_FN(log10),	FVLOG10,	INVALID_VFC,	INVALID_VFC },
{ "random",	M1_FN(rn_number),	FVRAND,		INVALID_VFC,	INVALID_VFC },
{ "atan",	M1_FN(atan),		FVATAN,		INVALID_VFC,	INVALID_VFC },
{ "acos",	M1_FN(acos),		FVACOS,		INVALID_VFC,	INVALID_VFC },
{ "asin",	M1_FN(asin),		FVASIN,		INVALID_VFC,	INVALID_VFC },
{ "floor",	M1_FN(floor),	FVFLOOR,	INVALID_VFC,	INVALID_VFC },
{ "round",	M1_FN(round),	FVROUND,	INVALID_VFC,	INVALID_VFC },
{ "ceil",	M1_FN(ceil),		FVCEIL,		INVALID_VFC,	INVALID_VFC },
{ "rint",	M1_FN(rint),		FVRINT,		INVALID_VFC,	INVALID_VFC },
{ "abs",	M1_FN(fabs),		FVABS,		INVALID_VFC,	INVALID_VFC },
{ "sign",	M1_FN(signfunc),	FVSIGN,		INVALID_VFC,	INVALID_VFC },
#ifndef MAC
#ifndef _WINDOWS
{ "j0",		M1_FN(j0),		INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
#endif /* !_WINDOWS */
#endif /* !MAC */
{ "tan",	M1_FN(tan),		FVTAN,		INVALID_VFC,	INVALID_VFC },
{ "bitsum",	M1_FN(bitsum),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "isdigit",	M1_FN(d_isdigit),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "",		M1_FN(nulld1func),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC }
};

#define D2_FN( name )	D2_FUNCTYP , {d2_func:name}

Function math2_functbl[]={
/* BUG need to implement vpow, vspow (2 flavors), and vsatan2... */
{ "atan2",	D2_FN(atan2),	FVATAN2,	FVSATAN2,	FVSATAN22 },
{ "pow",	D2_FN(pow),		FVPOW,		FVSPOW,		FVSPOW2	 },
{ "max",	D2_FN(maxfunc),	FVMAX,		FVSMAX,		FVSMAX	 },
{ "min",	D2_FN(minfunc),	FVMIN,		FVSMIN,		FVSMIN	 },
{ "",		D2_FN(nulld2func),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC }
};

#define DO_FN( name )	DOBJ_FUNCTYP , {dobj_func:name}

Function data_functbl[]={
{ "value",	DO_FN(nulldofunc),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "Re",		DO_FN(nulldofunc),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "Im",		DO_FN(nulldofunc),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "is_contiguous",DO_FN(nulldofunc),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "",		DO_FN(nulldofunc),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC } /* just an end marker */
};

/* The order of these matters, because the index is used as a dimension index in vectree/evaltree.c */

#define SZ_FN( name )	SIZE_FUNCTYP , {sz_func:name}

Function size_functbl[]={
{ "depth",	SZ_FN(_dpfunc),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "ncols",	SZ_FN(_colfunc),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "nrows",	SZ_FN(_rowfunc),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "nframes",	SZ_FN(_frmfunc),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "nseqs",	SZ_FN(_seqfunc),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "ncomps",	SZ_FN(_dpfunc),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "nelts",	SZ_FN(_nefunc),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "is_interlaced",SZ_FN(_ilfunc),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "",		SZ_FN(nullszfunc),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC } /* just an end marker */
};

#define TS_FN(name)	TS_FUNCTYP , {ts_func:name}

Function timestamp_functbl[]={
{ "seconds",	TS_FN(_secfunc),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "milliseconds",	TS_FN(_msecfunc),	INVALID_VFC,		INVALID_VFC,	INVALID_VFC },
{ "microseconds",	TS_FN(_usecfunc),	INVALID_VFC,		INVALID_VFC,	INVALID_VFC },
{ "",		TS_FN(nulltsfunc),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC }
};

#define V_FN(name)	VOID_FUNCTYP , {v_func:name}
Function misc_functbl[]={
	/* these two are for omdr */
{ "currframe",	V_FN(nullvfunc),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "recordable",	V_FN(nullvfunc),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "",		V_FN(nullvfunc),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC } /* just an end marker */
};

#define S1_FN(name)	STR1_FUNCTYP , {str1_func:name}

Function str1_functbl[]={
{ "strlen",	S1_FN(dstrlen),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "var_exists",	S1_FN(dvarexists),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "macro_exists",	S1_FN(dmacroexists),	INVALID_VFC,		INVALID_VFC,	INVALID_VFC },
{ "iof_exists",	S1_FN(nulls1func),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },

{ "exists",	S1_FN(dexists),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "file_exists",	S1_FN(dexists),	INVALID_VFC,		INVALID_VFC,	INVALID_VFC },
{ "obj_exists",	S1_FN(nulls1func),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "viewer_exists",S1_FN(nulls1func),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "panel_exists",	S1_FN(nulls1func),	INVALID_VFC,		INVALID_VFC,	INVALID_VFC },
{ "mod_time",	S1_FN(modtimefunc),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
/* return 1 if a dobj */
/*
{ "is_directory",	S1_FN(disdir),		INVALID_VFC,		INVALID_VFC,	INVALID_VFC },
{ "is_regfile",	S1_FN(disreg),		INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
*/
{ "",		S1_FN(nulls1func),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC }
};

#define S2_FN( name )	STR2_FUNCTYP , {str2_func:name}

Function str2_functbl[]={
{ "strcmp",	S2_FN(dstrcmp),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "strstr",	S2_FN(dstrstr),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "",		S2_FN(nulls2func),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC }
};

#define S3_FN( name )	STR3_FUNCTYP , {str3_func:name}

Function str3_functbl[]={
{ "strncmp",	S3_FN(dstrncmp),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC },
{ "",		S3_FN(nulls3func),	INVALID_VFC,	INVALID_VFC,	INVALID_VFC }
};

int assgn_func(Function *tbl,const char *name,double (*func)(void))
{
	int i;

	i=whfunc(tbl,name);
	if( i==(-1) ){
		return(-1);
	}
	tbl[i].fn_func.v_func = func;
	return(0);
}

void setdatafunc(const char *name,double (*func)(Data_Obj *))
{
	if( assgn_func((Function *)(void *)data_functbl,name,(double (*)(void))func) == 0 ) return;
	/* if( assgn_func((Function *)str1_functbl,name,func) == 0 ) return; */
#ifdef CAUTIOUS
	sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  no data function \"%s\"",
		name);
	NWARN(DEFAULT_ERROR_STRING);
#endif /* CAUTIOUS */
}

void setstrfunc(const char *name,double (*func)(QSP_ARG_DECL  const char *))
{
	if( assgn_func((Function *)(void *)str1_functbl,name,(double (*)(void))func) == 0 ) return;
#ifdef CAUTIOUS
	sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  no string function \"%s\"",
		name);
	NWARN(DEFAULT_ERROR_STRING);
#endif /* CAUTIOUS */
}

void setmiscfunc(const char *name,double (*func)(void))
{
	if( assgn_func((Function *)(void *)misc_functbl,name,func) != 0 ){
		sprintf(DEFAULT_ERROR_STRING,"no misc function \"%s\"",name);
		NWARN(DEFAULT_ERROR_STRING);
	}
}

