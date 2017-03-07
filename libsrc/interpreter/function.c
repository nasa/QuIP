#include <string.h>
#include <ctype.h>
#include "quip_config.h"
#include "quip_prot.h"
#include "function.h"
#include "func_helper.h"
#include "variable.h"
#include "item_type.h"
#include "warn.h"
#include "rn.h"
#include "data_obj.h"
#include "item_prot.h"
#include "veclib/vecgen.h"
#include "sizable.h"
#include "fileck.h"

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif /* HAVE_SYS_STAT_H */

#ifdef HAVE_MATH_H
#include <math.h>
#endif /* HAVE_MATH_H */

//#ifdef HAVE_GSL
//#include "gsl/gsl_sf_gamma.h"
//#endif /* HAVE_GSL */

//#import "nexpr_func.h"


//void addFunctionWithCode(Quip_Function *f,int code);
//const char *function_name(Quip_Function *f);

static Item_Type *function_itp=NULL;

ITEM_INIT_FUNC(Quip_Function,function,0)
ITEM_NEW_FUNC(Quip_Function,function)
ITEM_CHECK_FUNC(Quip_Function,function)

#define DECLARE_CHAR_FUNC( fname )					\
									\
static int c_##fname( char c )						\
{									\
	if( fname(c) ) return 1;					\
	else return 0;							\
}

DECLARE_CHAR_FUNC( islower )
DECLARE_CHAR_FUNC( isupper )
DECLARE_CHAR_FUNC( isalpha )
DECLARE_CHAR_FUNC( isalnum )
DECLARE_CHAR_FUNC( isdigit )
DECLARE_CHAR_FUNC( isspace )
DECLARE_CHAR_FUNC( iscntrl )
DECLARE_CHAR_FUNC( isblank )

/*
static int c_tolower( char c ) { return( tolower(c) ); }
static int c_toupper( char c ) { return( toupper(c) ); }
*/

#define DECLARE_STRINGMAP_FUNCTION( funcname, test_macro, map_macro )	\
									\
static const char * funcname(QSP_ARG_DECL  const char *s )				\
{									\
	char *t, *p;							\
									\
	t = getbuf(strlen(s)+1);					\
	p=t;								\
	while( *s ){							\
		if( test_macro(*s) )					\
			*p = (char) map_macro(*s);				\
		else							\
			*p = *s;					\
		p++;							\
		s++;							\
	}								\
	*p = 0;								\
	return( t );							\
}

DECLARE_STRINGMAP_FUNCTION(s_tolower,isupper,tolower)
DECLARE_STRINGMAP_FUNCTION(s_toupper,islower,toupper)

static double modtimefunc(QSP_ARG_DECL  const char *s)
{
#ifdef HAVE_SYS_STAT_H
	struct stat statb;

	if( stat(s,&statb) < 0 ){
		tell_sys_error(s);
		return(0.0);
	}
	return( (double) statb.st_mtime );
#else /* ! HAVE_SYS_STAT_H */

	return 0.0;

#endif /* ! HAVE_SYS_STAT_H */
}

#ifndef BUILD_FOR_IOS
int is_portrait(void)
{
	NADVISE("is_portrait() called in non-tablet environment!?");

	return 0;
}
#endif // ! BUILD_FOR_IOS

static double is_landscape_dbl(void)
{
	if( is_portrait() )
		return 0.0;
	else
		return 1.0;
}

static double is_portrait_dbl(void)
{
	return (double) is_portrait();
}

static double rn_uni(double arg)		/* arg is not used... */
{
	double d;
	rninit(SGL_DEFAULT_QSP_ARG);
#ifdef HAVE_DRAND48
	d=drand48();
#else
	NWARN("rn_uni:  no drand48!?");
	d=1.0;
#endif // ! HAVE_DRAND48
	return( d );
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

static double ascii_val(QSP_ARG_DECL  const char *s)
{
	unsigned long l;
	if( (l=strlen(s)) == 0 ){
		WARN("ascii() passed an empty string!?");
		return 0;
	}
	if( l > 1 ){
		sprintf(ERROR_STRING,"ascii() passed a string of length %lu, returning value of 1st char.",l);
		WARN(ERROR_STRING);
	}
	return (double) s[0];
}

static double dvarexists(QSP_ARG_DECL  const char *s)
{
	Variable *vp;

	vp = var_of(QSP_ARG  s);
	if( vp == NO_VARIABLE ) return 0.0;
	return 1.0;
}

static double dmacroexists(QSP_ARG_DECL  const char *s)
{
	Macro *mp;

	mp = macro_of(QSP_ARG  s);
	if( mp == NULL ) return 0.0;
	return 1.0;
}

static double dexists(QSP_ARG_DECL  const char *fname)
{
	if( path_exists(QSP_ARG  fname) )
		return(1.0);
	else	return(0.0);
}

static double disdir(QSP_ARG_DECL  const char *path)
{
	if( directory_exists(QSP_ARG path) )
		return 1.0;
	else	return 0.0;
}

static double disreg(QSP_ARG_DECL  const char *path)
{
	if( regfile_exists(QSP_ARG path) )
		return 1.0;
	else	return 0.0;
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

static Item_Class *sizable_icp=NO_ITEM_CLASS;
static Item_Class *interlaceable_icp=NO_ITEM_CLASS;
static Item_Class *positionable_icp=NO_ITEM_CLASS;
static Item_Class *tsable_icp=NO_ITEM_CLASS;
static Item_Class *subscriptable_icp=NO_ITEM_CLASS;

#define DECLARE_CLASS_INITIALIZER(type_stem)				\
									\
static void init_##type_stem##_class(SINGLE_QSP_ARG_DECL)		\
{									\
	if( type_stem##_icp != NO_ITEM_CLASS ){				\
		sprintf(ERROR_STRING,					\
	"Redundant call to %s class initializer",#type_stem);		\
		WARN(ERROR_STRING);					\
		return;							\
	}								\
	type_stem##_icp = new_item_class(QSP_ARG  #type_stem );		\
}

DECLARE_CLASS_INITIALIZER(sizable)
DECLARE_CLASS_INITIALIZER(tsable)
DECLARE_CLASS_INITIALIZER(subscriptable)
DECLARE_CLASS_INITIALIZER(interlaceable)
DECLARE_CLASS_INITIALIZER(positionable)


// We would like to be able to use named IOS items here too...
// how can we handle that, when one our our Items can't point to an IOS
// object?  This is really ugly, but we could make up another item
// type that provides regular item name lookup, and then handles
// the size functions...

#define DECLARE_ADD_FUNCTION(type_stem,func_type)			\
									\
void add_##type_stem(QSP_ARG_DECL  Item_Type *itp,			\
		func_type *func_str_ptr,				\
		Item *(*lookup)(QSP_ARG_DECL  const char *))		\
{									\
	if( type_stem##_icp == NO_ITEM_CLASS )				\
		init_##type_stem##_class(SINGLE_QSP_ARG);		\
	add_items_to_class(type_stem##_icp,itp,func_str_ptr,lookup);	\
}

DECLARE_ADD_FUNCTION(sizable,Size_Functions)
DECLARE_ADD_FUNCTION(tsable,Timestamp_Functions)
DECLARE_ADD_FUNCTION(interlaceable,Interlace_Functions)
DECLARE_ADD_FUNCTION(positionable,Position_Functions)
// For the most part, the subscript functions aren't really functions with names...
// But originally, they were packed into the size function struct, so here they
// are...
DECLARE_ADD_FUNCTION(subscriptable,Subscript_Functions)

#define DECLARE_FIND_FUNCTION(type_stem)				\
									\
Item *find_##type_stem(QSP_ARG_DECL  const char *name )			\
{									\
	if( type_stem##_icp == NO_ITEM_CLASS )				\
		init_##type_stem##_class(SINGLE_QSP_ARG);		\
									\
	return( get_member(QSP_ARG  type_stem##_icp,name) );		\
}

DECLARE_FIND_FUNCTION(sizable)
DECLARE_FIND_FUNCTION(tsable)
DECLARE_FIND_FUNCTION(interlaceable)
DECLARE_FIND_FUNCTION(positionable)
DECLARE_FIND_FUNCTION(subscriptable)


#ifdef BUILD_FOR_OBJC
#define OBJC_CHECK(type_stem)						\
	if( ip == NULL )						\
		ip = (__bridge Item *) check_ios_##type_stem(QSP_ARG  name);
#else // ! BUILD_FOR_OBJC
#define OBJC_CHECK(type_stem)
#endif // ! BUILD_FOR_OBJC

#define DECLARE_CHECK_FUNC(type_stem)					\
Item *check_##type_stem(QSP_ARG_DECL  const char *name )		\
{									\
	Item *ip;							\
	if( type_stem##_icp == NO_ITEM_CLASS )				\
		init_##type_stem##_class(SINGLE_QSP_ARG);		\
									\
	ip = check_member(QSP_ARG  type_stem##_icp, name );		\
	OBJC_CHECK(type_stem)						\
	return ip;							\
}

DECLARE_CHECK_FUNC(sizable)

//DECLARE_CHECK_FUNC(tsable)

#define DECLARE_GETFUNCS_FUNC(type_stem,func_type)			\
									\
func_type *get_##type_stem##_functions(QSP_ARG_DECL  Item *ip)		\
{									\
	Member_Info *mip;						\
	if( type_stem##_icp == NO_ITEM_CLASS )				\
		init_##type_stem##_class(SINGLE_QSP_ARG);		\
	mip = get_member_info(QSP_ARG  type_stem##_icp,ip->item_name);	\
	/*MEMBER_CAUTIOUS_CHECK(type_stem)*/				\
	assert( mip != NO_MEMBER_INFO );				\
	return (func_type *) mip->mi_data;				\
}

DECLARE_GETFUNCS_FUNC(sizable,Size_Functions)		// get_sizable_functions
DECLARE_GETFUNCS_FUNC(tsable,Timestamp_Functions)	// get_tsable_functions
DECLARE_GETFUNCS_FUNC(interlaceable,Interlace_Functions)	// get_interlaceable_functions
DECLARE_GETFUNCS_FUNC(positionable,Position_Functions)	// get_positionable_functions
DECLARE_GETFUNCS_FUNC(subscriptable,Subscript_Functions)	// get_subscriptable_functions


#ifndef BUILD_FOR_OBJC

// If we are building for IOS, use the version in ios_sizable, which handles
// both types of object...
//
// The initial approach didn't work, because we can't cast void * to IOS_Item...
// A cleaner approach would be for items to carry around their item type ptr,
// and for the item type struct to have a pointer to the size functions
// (when they exist).
// That approach burns a bit more memory, but probably insignificant?

const char *get_object_prec_string(QSP_ARG_DECL  Item *ip )	// non-iOS
{
	Size_Functions *sfp;

	if( ip == NO_ITEM ) return("u_byte");

	sfp = get_sizable_functions(QSP_ARG  ip);

//#ifdef CAUTIOUS
//	if( sfp == NULL ) ERROR1("CAUTIOUS:  precision_string:  shouldn't happen");
//#endif /* CAUTIOUS */
	assert( sfp != NULL );

	return( (*sfp->prec_func)(QSP_ARG  ip) );
}

double get_object_size(QSP_ARG_DECL  Item *ip,int d_index)
{
	Size_Functions *sfp;

	if( ip == NO_ITEM ) return(0.0);

	sfp = get_sizable_functions(QSP_ARG  ip);

//#ifdef CAUTIOUS
//	if( sfp == NULL ) ERROR1("CAUTIOUS:  get_object_size:  shouldn't happen");
//#endif /* CAUTIOUS */
	assert( sfp != NULL );

	return( (*sfp->sz_func)(QSP_ARG  ip,d_index) );
}

#endif /* ! BUILD_FOR_OBJC */

static double get_posn(QSP_ARG_DECL  Item *ip, int index)
{
	Position_Functions *pfp;
	if( ip == NO_ITEM ) return(0.0);
	pfp = get_positionable_functions(QSP_ARG  ip);
//#ifdef CAUTIOUS
//	if( pfp == NULL )
//		ERROR1("CAUTIOUS:  get_posn:  null func struct ptr!?");
	assert( pfp != NULL && index >= 0 && index <= 1 );

//	if( index < 0 || index > 1 )
//		ERROR1("CAUTIOUS:  get_posn:  bad index!?");
//#endif // CAUTIOUS

	return( (*pfp->posn_func)(QSP_ARG  ip,index) );
}

static double get_interlace_flag(QSP_ARG_DECL  Item *ip)
{
	Interlace_Functions *ifp;

	if( ip == NO_ITEM ) return(0.0);

	ifp = get_interlaceable_functions(QSP_ARG  ip);

//#ifdef CAUTIOUS
//	if( ifp == NULL ) ERROR1("CAUTIOUS:  get_interlace_flag:  shouldn't happen");
//#endif /* CAUTIOUS */
	assert( ifp != NULL );

//#ifdef CAUTIOUS
//	if( ifp->ilace_func == NULL ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  get_interlace_flag:  Sorry, is_interlaced() is not defined for object %s",ip->item_name);
//		WARN(ERROR_STRING);
//		return(0.0);
//	}
//#endif // CAUTIOUS
	assert( ifp->ilace_func != NULL );

	return( (*ifp->ilace_func)(QSP_ARG  ip) );
}

static double get_timestamp(QSP_ARG_DECL  Item *ip,int func_index,dimension_t frame)
{
	Timestamp_Functions *tsfp;
	Member_Info *mip;
	double d;

	if( ip == NO_ITEM ) return(0.0);

	if( tsable_icp == NO_ITEM_CLASS )
		init_tsable_class(SINGLE_QSP_ARG);

	mip = get_member_info(QSP_ARG  tsable_icp,ip->item_name);

//#ifdef CAUTIOUS
//	if( mip == NO_MEMBER_INFO ){
//		sprintf(ERROR_STRING,
//			"CAUTIOUS:  get_timestamp %s %d, missing member info",
//			ip->item_name,func_index);
//		ERROR1(ERROR_STRING);
//	}
//#endif /* CAUTIOUS */
	assert( mip != NO_MEMBER_INFO );


	tsfp = (Timestamp_Functions *) mip->mi_data;

	d = (*(tsfp->timestamp_func[func_index]))(QSP_ARG  ip,frame);
	return( d );
}

Item *sub_sizable(QSP_ARG_DECL  Item *ip,index_t index)
{
	Subscript_Functions *sfp;
	Member_Info *mip;

	/* currently data objects are the only sizables
		which can be subscripted */

	if( ip == NO_ITEM ) return(ip);

	if( subscriptable_icp == NO_ITEM_CLASS )
		init_subscriptable_class(SINGLE_QSP_ARG);

	mip = get_member_info(QSP_ARG  subscriptable_icp,ip->item_name);

//#ifdef CAUTIOUS
//	if( mip == NO_MEMBER_INFO )
//		ERROR1("CAUTIOUS:  missing member info #3");
//#endif /* CAUTIOUS */
	assert( mip != NO_MEMBER_INFO );

	sfp = (Subscript_Functions *) mip->mi_data;

	if( sfp->subscript == NULL ){
		sprintf(ERROR_STRING,"Can't subscript object %s!?",
			ip->item_name);
		WARN(ERROR_STRING);
		return(NO_ITEM);
	}

	return( (*sfp->subscript)(QSP_ARG  ip,index) );
}

Item *csub_sizable(QSP_ARG_DECL  Item *ip,index_t index)
{
	Subscript_Functions *sfp;
	Member_Info *mip;

	/* currently data objects are the only sizables
		which can be subscripted */

	if( ip == NO_ITEM ) return(ip);

	if( subscriptable_icp == NO_ITEM_CLASS )
		init_subscriptable_class(SINGLE_QSP_ARG);

	mip = get_member_info(QSP_ARG  subscriptable_icp,ip->item_name);

//#ifdef CAUTIOUS
//	if( mip == NO_MEMBER_INFO )
//		ERROR1("CAUTIOUS:  missing member info #1");
//#endif /* CAUTIOUS */
	assert( mip != NO_MEMBER_INFO );

	sfp = (Subscript_Functions *) mip->mi_data;

	if( sfp->csubscript == NULL ){
		sprintf(ERROR_STRING,"Can't subscript object %s",
			ip->item_name);
		WARN(ERROR_STRING);
		return(NO_ITEM);
	}

	return( (*sfp->csubscript)(QSP_ARG  ip,index) );
}

// We want to be able to pass these functions any type
// of object, but we can't cast from void * to IOS_Object...

static double _ilfunc(QSP_ARG_DECL  Item *ip)
{ return( get_interlace_flag(QSP_ARG  ip) ); }

static double _x_func(QSP_ARG_DECL  Item *ip)
{ return get_posn(QSP_ARG  ip,0); }

static double _y_func(QSP_ARG_DECL  Item *ip)
{ return get_posn(QSP_ARG  ip,1); }

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

static const char *_precfunc(QSP_ARG_DECL  const char *s)
{
	Item *ip;
	ip = find_sizable( DEFAULT_QSP_ARG  s );
	return get_object_prec_string(QSP_ARG  ip);
}

static const char *strcat_func(QSP_ARG_DECL  const char *s1, const char *s2 )
{
	char *s;
	s = getbuf( strlen(s1) + strlen(s2) + 1 );
	strcpy(s,s1);
	strcat(s,s2);
	return s;
}

static double _nefunc(QSP_ARG_DECL  Item *ip)
{
	int i;
	double d;

	d=1;
	for(i=0;i<N_DIMENSIONS;i++)
		d *= get_object_size(QSP_ARG  ip,i);

	return(d);
}

#define SIGN(x)		(x<0?-1.0:1.0)
#define SIGNF(x)	(x<0?-1.0f:1.0f)

// This approximation to erfinv comes from wikipedia, which cites:
// Winitzki, Sergei (6 February 2008). "A handy approximation for
// the error function and its inverse" (PDF). Retrieved 2011-10-03.

double erfinv(double x)
{
	double y;
	static double pi=0.0;
	static double a=0.0;

	if( pi == 0.0 ){
		pi = 4*atan(1);
		a = 8 * ( pi - 3 ) / ( 3 * pi * ( 4 - pi ) );
	}

	y = SIGN(x) *
	    sqrt(
	    sqrt( pow( 2/(pi*a) + log(1-x*x)/2, 2 ) - log(1-x*x)/a )
	     - ( 2/(pi*a) + log(1-x*x)/2 )
	     );

	return y;
}

float erfinvf(float x)
{
	float y;
	static float pi=0.0;
	static float a=0.0;

	if( pi == 0.0 ){
		pi = 4*atanf(1);
		a = 8 * ( pi - 3 ) / ( 3 * pi * ( 4 - pi ) );
	}

	y = SIGNF(x) *
	    sqrtf(
	    sqrtf( powf( 2.0f/(pi*a) + logf(1.0f-x*x)/2.0f, 2.0f ) - logf(1.0f-x*x)/a )
	     - ( 2.0f/(pi*a) + logf(1.0f-x*x)/2.0f )
	     );

	return y;
}


static double _secfunc(QSP_ARG_DECL  Item *ip, dimension_t frame )
{ return( get_timestamp(QSP_ARG  ip,0,frame)); }

static double _msecfunc(QSP_ARG_DECL  Item *ip, dimension_t frame )
{ return( get_timestamp(QSP_ARG  ip,1,frame) ); }

static double _usecfunc(QSP_ARG_DECL  Item *ip, dimension_t frame )
{ return( get_timestamp(QSP_ARG  ip,2,frame) ); }


static double signfunc(double x)
{ if( x > 0 ) return(1.0); else if( x < 0 ) return(-1.0); else return(0.0); }

static int isnanfunc(double x)
{ return isnan(x) ; }

static int isinffunc(double x)
{ return isinf(x) ; }

static int isnormalfunc(double x)
{ return isnormal(x) ; }

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

int func_serial=0;

void declare_functions( SINGLE_QSP_ARG_DECL )
{

DECLARE_D0_FUNCTION(	is_portrait,	is_portrait_dbl,		INVALID_VFC,	INVALID_VFC,	INVALID_VFC	)
DECLARE_D0_FUNCTION(	is_landscape,	is_landscape_dbl,		INVALID_VFC,	INVALID_VFC,	INVALID_VFC	)

DECLARE_D1_FUNCTION(	uni,	rn_uni,		FVUNI,		INVALID_VFC,	INVALID_VFC	)
DECLARE_D1_FUNCTION(	sqrt,	sqrt,		FVSQRT,		INVALID_VFC,	INVALID_VFC	)
#ifdef HAVE_ERF
DECLARE_D1_FUNCTION(	erf,	erf,		FVERF,		INVALID_VFC,	INVALID_VFC	)
#endif // HAVE_ERF
DECLARE_D1_FUNCTION(	erfinv,	erfinv,		FVERFINV,	INVALID_VFC,	INVALID_VFC	)
#ifdef SINE_TBL
DECLARE_D1_FUNCTION(	sin,	t_sin,		FVSIN,		INVALID_VFC,	INVALID_VFC	)
DECLARE_D1_FUNCTION(	cos,	t_cos,		FVCOS,		INVALID_VFC,	INVALID_VFC	)
#else /* ! SINE_TBL */
DECLARE_D1_FUNCTION(	sin,	sin,		FVSIN,		INVALID_VFC,	INVALID_VFC	)
DECLARE_D1_FUNCTION(	cos,	cos,		FVCOS,		INVALID_VFC,	INVALID_VFC	)
#endif /* ! SINE_TBL */
//#ifdef HAVE_GSL
//#ifdef FOOBAR
// disable this while linking with texinfo...
DECLARE_D1_FUNCTION(	gamma,	tgamma,		FVGAMMA,	INVALID_VFC,	INVALID_VFC	)
DECLARE_D1_FUNCTION(	lngamma, lgamma,	FVLNGAMMA,	INVALID_VFC,	INVALID_VFC	)
//#endif // FOOBAR
//#endif /* HAVE_GSL */
DECLARE_D1_FUNCTION(	exp,	exp,		FVEXP,		INVALID_VFC,	INVALID_VFC	)
DECLARE_D1_FUNCTION(	log,	log,		FVLOG,		INVALID_VFC,	INVALID_VFC	)
DECLARE_D1_FUNCTION(	log10,	log10,		FVLOG10,	INVALID_VFC,	INVALID_VFC	)
DECLARE_D1_FUNCTION(	random,	rn_number,	FVRAND,		INVALID_VFC,	INVALID_VFC	)
DECLARE_D1_FUNCTION(	abs,	fabs,		FVABS,		INVALID_VFC,	INVALID_VFC	)
DECLARE_D1_FUNCTION(	sign,	signfunc,	FVSIGN,		INVALID_VFC,	INVALID_VFC	)
DECLARE_I1_FUNCTION(	isnan,	isnanfunc,	FVISNAN,	INVALID_VFC,	INVALID_VFC	)
DECLARE_I1_FUNCTION(	isinf,	isinffunc,	FVISINF,	INVALID_VFC,	INVALID_VFC	)
DECLARE_I1_FUNCTION(	isnormal, isnormalfunc,	FVISNORM,	INVALID_VFC,	INVALID_VFC	)
DECLARE_D1_FUNCTION(	atan,	atan,		FVATAN,		INVALID_VFC,	INVALID_VFC	)
DECLARE_D1_FUNCTION(	acos,	acos,		FVACOS,		INVALID_VFC,	INVALID_VFC	)
DECLARE_D1_FUNCTION(	asin,	asin,		FVASIN,		INVALID_VFC,	INVALID_VFC	)
DECLARE_D1_FUNCTION(	floor,	floor,		FVFLOOR,	INVALID_VFC,	INVALID_VFC	)
DECLARE_D1_FUNCTION(	trunc,	trunc,		FVTRUNC,	INVALID_VFC,	INVALID_VFC	)
#ifdef HAVE_ROUND
DECLARE_D1_FUNCTION(	round,	round,		FVROUND,	INVALID_VFC,	INVALID_VFC	)
#endif // HAVE_ROUND
DECLARE_D1_FUNCTION(	ceil,	ceil,		FVCEIL,		INVALID_VFC,	INVALID_VFC	)
#ifdef HAVE_RINT
DECLARE_D1_FUNCTION(	rint,	rint,		FVRINT,		INVALID_VFC,	INVALID_VFC	)
#endif // HAVE_RINT
#ifndef MAC
#ifndef _WINDOWS
DECLARE_D1_FUNCTION(	j0,	j0,		INVALID_VFC,	INVALID_VFC,	INVALID_VFC	)
#endif /* !_WINDOWS */
#endif /* !MAC */
DECLARE_D1_FUNCTION(	ptoz,	ptoz,		INVALID_VFC,	INVALID_VFC,	INVALID_VFC	)
DECLARE_D1_FUNCTION(	ztop,	ztop,		INVALID_VFC,	INVALID_VFC,	INVALID_VFC	)
DECLARE_D1_FUNCTION(	tan,	tan,		FVTAN,		INVALID_VFC,	INVALID_VFC	)
DECLARE_D1_FUNCTION(	bitsum,	bitsum,		INVALID_VFC,	INVALID_VFC,	INVALID_VFC	)

DECLARE_D2_FUNCTION(	atan2,	atan2,		FVATAN2,	FVSATAN2,	FVSATAN22	)
DECLARE_D2_FUNCTION(	pow,	pow,		FVPOW,		FVSPOW,		FVSPOW2		)
DECLARE_D2_FUNCTION(	max,	maxfunc,	FVMAX,		FVSMAX,		FVSMAX		)
DECLARE_D2_FUNCTION(	min,	minfunc,	FVMIN,		FVSMIN,		FVSMIN		)

DECLARE_STR1_FUNCTION(	strlen,		dstrlen		)
DECLARE_STR1_FUNCTION(	var_exists,	dvarexists	)
DECLARE_STR1_FUNCTION(	macro_exists,	dmacroexists	)
DECLARE_STR1_FUNCTION(	exists,		dexists		)
DECLARE_STR1_FUNCTION(	file_exists,	dexists	 	)
DECLARE_STR1_FUNCTION(	mod_time,	modtimefunc	)
DECLARE_STR1_FUNCTION(  ascii,		ascii_val	)
DECLARE_STR1_FUNCTION(	is_directory,	disdir	 )
DECLARE_STR1_FUNCTION(	is_regfile,	disreg	 )

DECLARE_STR2_FUNCTION(	strcmp,	dstrcmp		)
DECLARE_STR2_FUNCTION(	strstr,	dstrstr		)
DECLARE_STR3_FUNCTION(	strncmp,dstrncmp	)

// shouldn't these be string valued functions that translat a whole string?
// But what about the mapping functions???
DECLARE_STRV_FUNCTION(	tolower, s_tolower,	FVTOLOWER, INVALID_VFC,	INVALID_VFC	)
DECLARE_STRV_FUNCTION(	toupper, s_toupper,	FVTOUPPER, INVALID_VFC,	INVALID_VFC	)

DECLARE_CHAR_FUNCTION(	islower, c_islower,	FVISLOWER, INVALID_VFC, INVALID_VFC	)
DECLARE_CHAR_FUNCTION(	isupper, c_isupper,	FVISUPPER, INVALID_VFC, INVALID_VFC	)
DECLARE_CHAR_FUNCTION(	isalpha, c_isalpha,	FVISALPHA, INVALID_VFC, INVALID_VFC	)
DECLARE_CHAR_FUNCTION(	isdigit, c_isdigit,	FVISDIGIT, INVALID_VFC, INVALID_VFC	)
DECLARE_CHAR_FUNCTION(	isalnum, c_isalnum,	FVISALNUM, INVALID_VFC, INVALID_VFC	)
DECLARE_CHAR_FUNCTION(	isspace, c_isspace,	FVISSPACE, INVALID_VFC, INVALID_VFC	)
DECLARE_CHAR_FUNCTION(	iscntrl, c_iscntrl,	FVISCNTRL, INVALID_VFC, INVALID_VFC	)
DECLARE_CHAR_FUNCTION(	isblank, c_isblank,	FVISBLANK, INVALID_VFC, INVALID_VFC	)

// This was called prec_name, but that was defined by a C macro to be prec_item.item_name!?
DECLARE_STRV_FUNCTION(	precision,	_precfunc, INVALID_VFC, INVALID_VFC, INVALID_VFC	)
DECLARE_STRV2_FUNCTION(	strcat,		strcat_func	)

DECLARE_SIZE_FUNCTION(	depth,	_dpfunc,	0	)
DECLARE_SIZE_FUNCTION(	ncols,	_colfunc,	1	)
DECLARE_SIZE_FUNCTION(	nrows,	_rowfunc,	2	)
DECLARE_SIZE_FUNCTION(	nframes,_frmfunc,	3	)
DECLARE_SIZE_FUNCTION(	nseqs,	_seqfunc,	4	)
DECLARE_SIZE_FUNCTION(	ncomps,	_dpfunc,	0	)
DECLARE_SIZE_FUNCTION(	nelts,	_nefunc,	-1	)

//DECLARE_SIZE_FUNCTION(	is_interlaced,_ilfunc,	-1	)
DECLARE_ILACE_FUNCTION(	is_interlaced,	_ilfunc		)

DECLARE_POSITION_FUNCTION(	x_position,	_x_func,	0	)
DECLARE_POSITION_FUNCTION(	y_position,	_y_func,	1	)

DECLARE_TS_FUNCTION(	seconds,	_secfunc	)
DECLARE_TS_FUNCTION(	milliseconds,	_msecfunc	)
DECLARE_TS_FUNCTION(	microseconds,	_usecfunc	)

}


#ifdef NOT_NEEDED
void assign_func_ptr(const char *name,double (*func)(void))
{
	Quip_Function *func_p;

	func_p = function_of(DEFAULT_QSP_ARG  name);
//#ifdef CAUTIOUS
//	if( func_p == NO_FUNCTION ){
//		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  assgn_func:  no function for \"%s\"!?",name);
//		NERROR1(DEFAULT_ERROR_STRING);
//	}
//#endif /* CAUTIOUS */
	assert( func_p != NO_FUNCTION );

	func_p->fn_u.d0_func = func;
}
#endif /* NOT_NEEDED */

double evalD0Function( Quip_Function *func_p )
{
	return (*(func_p->fn_u.d0_func))();
}

double evalD1Function( Quip_Function *func_p, double arg )
{
	return (*(func_p->fn_u.d1_func))(arg);
}

double evalD2Function( Quip_Function *func_p, double arg1, double arg2 )
{
	return (*(func_p->fn_u.d2_func))(arg1,arg2);
}

int evalI1Function( Quip_Function *func_p, double arg )
{
	return (*(func_p->fn_u.i1_func))(arg);
}

double evalStr1Function( QSP_ARG_DECL  Quip_Function *func_p, const char *s )
{
	return (*(func_p->fn_u.str1_func))(QSP_ARG  s);
}

#ifdef FOOBAR
// original
void evalStrVFunction( Quip_Function *func_p, char *dst, const char *s )
{
	(*(func_p->fn_u.strv_func))(dst,s);
}

// new
const char * evalStrVFunction( QSP_ARG_DECL  Quip_Function *func_p, Item *ip )
{
	return (*(func_p->fn_u.strv_func))(QSP_ARG  ip);
}
#endif // FOOBAR

int evalCharFunction( Quip_Function *func_p, char c )
{
	return (*(func_p->fn_u.char_func))(c);
}

double evalStr2Function( Quip_Function *func_p, const char *s1, const char *s2 )
{
	return (*(func_p->fn_u.str2_func))(s1,s2);
}

double evalStr3Function( Quip_Function *func_p, const char *s1, const char *s2, int n )
{
	return (*(func_p->fn_u.str3_func))(s1,s2,n);
}

#ifndef BUILD_FOR_OBJC
const char *default_prec_name(QSP_ARG_DECL  Item *ip)
{
	return "u_byte";	// could point to precision struct?  to save 7 bytes!?
}
#endif // ! BUILD_FOR_OBJC

