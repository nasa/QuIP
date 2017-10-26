#include "quip_config.h"

#include <stdio.h>
#include "debug.h"
#include "data_obj.h"
#include "rn.h"
#include "warn.h"
#include "variable.h"			/* assign_var() */
#include "quip_prot.h"			/* assign_var() */

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>		/* drand48 */
#endif

// memory.h had the prototype for strcpy on old SUN?
#ifdef HAVE_MEMORY_H
#include <memory.h>
#endif

#define DEFAULT_WHENCE(s)		if( whence == NULL ) whence=s;

#define MAX_VECTORIZABLE	max_vectorizable(SINGLE_QSP_ARG)
#define SET_MAX_VECTORIZABLE(v)	set_max_vectorizable(QSP_ARG  v)

// We probably can eliminate this!  BUG?

void getmean( QSP_ARG_DECL  Data_Obj *dp )
{
	u_long i;
	u_long n;
	double sum, sos, f;
	float max,min;

	if( dp== NULL ) return;
	if( OBJ_PREC(dp) != PREC_SP && OBJ_PREC(dp) != PREC_IN ){
		NWARN("sorry, only float or short objects");
		return;
	}
	if( OBJ_MACH_DIM(dp,0) != 1 ){
		sprintf(ERROR_STRING,"ALERT:  getmean:  object %s has %d components!?",
			OBJ_NAME(dp),OBJ_MACH_DIM(dp,0));
		advise(ERROR_STRING);
	}
	if( ! IS_CONTIGUOUS(dp) ){
		NWARN("sorry, can only compute mean of contiguous objects");
		return;
	}
	sum=sos=0.0;
	n=OBJ_N_MACH_ELTS(dp);
	if( OBJ_PREC(dp) == PREC_SP ){
		float *fnp;

		fnp=(float *)OBJ_DATA_PTR(dp);
		max=min=(*fnp);
		for(i=0;i<n;i++){
			f=(*fnp);
			if( f > max ) max=(float)f;
			else if( f < min ) min=(float)f;
			sos += (double)(f * f);
			sum += (double)f;
			fnp++;
		}
	} else if( OBJ_PREC(dp) == PREC_IN ){
		short *inp;

		inp=(short *)OBJ_DATA_PTR(dp);
		if( IS_UNSIGNED(dp) )
			max=min=(*(u_short *)inp);
		else
			max=min=(*inp);
		for(i=0;i<n;i++){
			if( IS_UNSIGNED(dp) )
				f=(*(u_short *)inp);
			else
				f=(*inp);
			if( f > max ) max=(float)f;
			else if( f < min ) min=(float)f;
			sos += (double)(f * f);
			sum += (double)f;
			inp++;
		}
	}

#ifdef CAUTIOUS
	/* this line is just to shut up a compiler warning... */
	else { max = min = 0 ; }	/* NOTREACHED */
#endif /* CAUTIOUS */

	sum /= (double) n;
	sos -= n * sum * sum;

	if( verbose ){
		sprintf(ERROR_STRING,
			"mean:  %f\nvariance:  %f\nmax:  %f\nmin:  %f",
			sum,sos,max,min);
		advise(ERROR_STRING);
	}

	sprintf(ERROR_STRING,"%f",sum);
	assign_var("mean",ERROR_STRING);
	sprintf(ERROR_STRING,"%f",sos);
	assign_var("variance",ERROR_STRING);
	sprintf(ERROR_STRING,"%f",max);
	assign_var("max",ERROR_STRING);
	sprintf(ERROR_STRING,"%f",min);
	assign_var("min",ERROR_STRING);
}

static double equate_value;

/* equate a contiguous block of data */

#define EQUATE_IT( type )						\
		type * ptr;						\
		type value;						\
									\
		ptr = ( type * ) OBJ_DATA_PTR(dp);				\
		value = ( type ) equate_value;				\
		for(i=0;i<n;i++){					\
			*ptr = value;					\
			ptr += inc;					\
		}

static void fast_equate( Data_Obj *dp )
{
	u_long i;
	u_long n;
	long inc;

	n=OBJ_N_MACH_ELTS(dp);
	inc = OBJ_MACH_INC(dp, OBJ_MINDIM(dp) );

#ifdef QUIP_DEBUG
if( debug & debug_data ){
sprintf(DEFAULT_ERROR_STRING,"fast_equate %s",OBJ_NAME(dp));
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	if( OBJ_PREC(dp) == PREC_BY ){
		EQUATE_IT( char )
	} else if( OBJ_PREC(dp) == PREC_UBY ){
		EQUATE_IT( u_char )
	} else if( OBJ_PREC(dp) == PREC_SP ){
		EQUATE_IT( float )
	} else if( OBJ_PREC(dp) == PREC_DP ){
		EQUATE_IT( double )
	} else if( OBJ_PREC(dp) == PREC_IN ){
		EQUATE_IT( short )
	} else if( OBJ_PREC(dp) == PREC_UIN ){
		EQUATE_IT( u_short )
	} else if( OBJ_PREC(dp) == PREC_DI ){
		EQUATE_IT( long )
	} else if( OBJ_PREC(dp) == PREC_UDI ){
		EQUATE_IT( u_long )
	} else NWARN("fast_equate:  unsupported pixel type");
}

void dp_equate( QSP_ARG_DECL  Data_Obj *dp, double v )
{
	if( dp==NULL ) return;

	equate_value=v;

	SET_MAX_VECTORIZABLE(N_DIMENSIONS-1);     /* default: vectorize over all */
	check_vectorization(QSP_ARG  dp);
	dp1_vectorize(QSP_ARG  (int)(N_DIMENSIONS-1),dp,fast_equate);
}

/* this version works for contiguous objects only */

static void contig_copy( Data_Obj *dp_to, Data_Obj *dp_fr )
{
	u_long nb;

#ifdef QUIP_DEBUG
if( debug & debug_data ){
sprintf(DEFAULT_ERROR_STRING,"contig_copy:  %s  %s",OBJ_NAME(dp_to),OBJ_NAME(dp_fr));
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	nb = OBJ_N_MACH_ELTS(dp_to) * ELEMENT_SIZE(dp_to);

	if( IS_BITMAP(dp_to) )
		nb = (nb+(BITS_PER_BITMAP_WORD-1)) / BITS_PER_BITMAP_WORD;

	/* BUG memcpy() is not guaranteed to do the correct
	 * thing on overlapping moves, but we use if for
	 * now, because memmove() does not seem to be
	 * available on SUN
	 */

#ifndef PC
	memcpy(OBJ_DATA_PTR(dp_to),OBJ_DATA_PTR(dp_fr),nb);
#else /* PC */
	if( nb <= 0x7fff )
		memcpy(OBJ_DATA_PTR(dp_to),OBJ_DATA_PTR(dp_fr),(int)nb);
	else
		/* BUG should go ahead and copy blocks... */
		NWARN("Sorry, can't copy large blocks");
#endif /* PC */
	return;
}

/* copy a pair of evenly spaced (not necessarily contiguous) objects */

#define COPY_IT( type )							\
		type *pto,*pfr;						\
									\
		pto=(type *)OBJ_DATA_PTR(dp_to);				\
		pfr=(type *)OBJ_DATA_PTR(dp_fr);				\
		for(i=0;i<OBJ_N_MACH_ELTS(dp_to);i++){				\
			*pto = *pfr;					\
			pto += to_inc;					\
			pfr += fr_inc;					\
		}

static void fast_copy( Data_Obj *dp_to, Data_Obj *dp_fr )
{
	u_long i;
	long to_inc,fr_inc;

	if( N_IS_CONTIGUOUS(dp_to) && N_IS_CONTIGUOUS(dp_fr) ){
		contig_copy(dp_to,dp_fr);
		return;
	}


#ifdef QUIP_DEBUG
if( debug & debug_data ){
sprintf(DEFAULT_ERROR_STRING,"fast_copy:  %s  %s",OBJ_NAME(dp_to),OBJ_NAME(dp_fr));
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	/* Because these objects are known to be evenly spaced,
	 * dt_mach_inc[dt_mindim] must be the increment!
	 */

	to_inc = OBJ_MACH_INC(dp_to, OBJ_MINDIM(dp_to) );
	fr_inc = OBJ_MACH_INC(dp_fr, OBJ_MINDIM(dp_fr) );

#ifdef QUIP_DEBUG
if( debug & debug_data ){
sprintf(DEFAULT_ERROR_STRING,"fast_copy'ing %s (inc %ld) to %s (inc %ld)",
OBJ_NAME(dp_fr),fr_inc,OBJ_NAME(dp_to),to_inc);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif

	if( OBJ_MACH_PREC(dp_to) == PREC_BY || OBJ_MACH_PREC(dp_to) == PREC_UBY ){
		COPY_IT( u_char )
	} else if( OBJ_MACH_PREC(dp_to) == PREC_IN || OBJ_MACH_PREC(dp_to) == PREC_UIN ){
		COPY_IT( short )
	} else if( OBJ_MACH_PREC(dp_to) == PREC_DI || OBJ_MACH_PREC(dp_to) == PREC_UDI ){
		COPY_IT( long )
	} else if( OBJ_MACH_PREC(dp_to) == PREC_SP){
		COPY_IT( float )
	} else if( OBJ_MACH_PREC(dp_to) == PREC_DP){
		COPY_IT( double )
	}

	else {
		assert( AERROR("Unsupported precision in fast_copy!?") );
	}
}


/* general purpose copy */

void dp_copy( QSP_ARG_DECL  Data_Obj *dp_to, Data_Obj *dp_fr )
{
	if( ! dp_same(QSP_ARG  dp_to,dp_fr,"dp_copy") ) return;

	if( IS_CONTIGUOUS(dp_to) && IS_CONTIGUOUS(dp_fr) )
		contig_copy(dp_to,dp_fr);
	else {
		SET_MAX_VECTORIZABLE(N_DIMENSIONS-1);     /* default: vectorize over all */
		check_vectorization(QSP_ARG  dp_to);
		check_vectorization(QSP_ARG  dp_fr);
		dp2_vectorize(QSP_ARG  N_DIMENSIONS-1,dp_to,dp_fr,fast_copy);
	}
}

/* simple integer randomization */

static int _imin,_imax;

static void fast_rand( Data_Obj *dp )
{
	u_long i;
	u_char *cp;
	long inc;

	cp=(u_char *)OBJ_DATA_PTR(dp);
	i=OBJ_N_MACH_ELTS(dp);
	inc = OBJ_MACH_INC(dp,OBJ_MINDIM(dp));

	while(i--){
		*cp = (u_char)(_imin + rn((long)_imax));
		cp += inc;
	}
}

void i_rnd( QSP_ARG_DECL  Data_Obj *dp, int imin, int imax )
{
	if( dp==NULL ) return;
	if( OBJ_MACH_PREC(dp) != PREC_BY && OBJ_MACH_PREC(dp) != PREC_UBY ){
		sprintf(ERROR_STRING,"i_rnd:  object %s (%s) should have %s or %s precision",
				OBJ_NAME(dp),OBJ_PREC_NAME(dp),NAME_FOR_PREC_CODE(PREC_BY),NAME_FOR_PREC_CODE(PREC_UBY));
		WARN(ERROR_STRING);
		return;
	}

	imax-=imin;

	_imax=imax;
	_imin=imin;

	SET_MAX_VECTORIZABLE(N_DIMENSIONS-1);     /* default: vectorize over all */
	check_vectorization(QSP_ARG  dp);
	dp1_vectorize(QSP_ARG  N_DIMENSIONS-1,dp,fast_rand);
}

static void fast_uni( Data_Obj *dp )
{
#ifdef HAVE_DRAND48
	u_long i;
#endif /* HAVE_DRAND48 */
    u_long n;
	float *fptr;
	long inc;

	fptr=(float *)OBJ_DATA_PTR(dp);
	n=OBJ_N_MACH_ELTS(dp);
	inc = OBJ_MACH_INC(dp,OBJ_MINDIM(dp));

#ifdef HAVE_DRAND48
	for(i=0;i<n;i++){
		*fptr = (float)drand48();
		fptr += inc;
	}
#else
	NERROR1("Sorry, no implementation of drand48() on this configuration...");
#endif

}

void dp_uni( QSP_ARG_DECL  Data_Obj *dp )
{
	/* need to seed this generator... */

	if( dp==NULL ) return;

	if( OBJ_PREC(dp) != PREC_SP ){
		NWARN("Uniform random numbers require FLOAT precision");
		return;
	}

	rninit(SINGLE_QSP_ARG);	/* initialize random number generator */
				/* BUG this assumes lib support compiled for drand48() */

	SET_MAX_VECTORIZABLE(N_DIMENSIONS-1);     /* default: vectorize over all */
	check_vectorization(QSP_ARG  dp);
	dp1_vectorize(QSP_ARG  N_DIMENSIONS-1,dp,fast_uni);
}


int dp_same_dims(QSP_ARG_DECL  Data_Obj *dp1, Data_Obj *dp2, int index1, int index2, const char *whence )
{
	int i, result=1;
	DEFAULT_WHENCE("dp_same_dims")
	for(i=index1;i<=index2;i++)
		result &= dp_same_dim(QSP_ARG  dp1,dp2,i,whence);
	return(result);
}

int dp_equal_dims(QSP_ARG_DECL  Data_Obj *dp1, Data_Obj *dp2, int index1, int index2 )
{
	int i, result=1;
	for(i=index1;i<=index2;i++)
		result &= dp_equal_dim(dp1,dp2,i);
	return(result);
}

/* dp_same_dim squawks if there is a mismatch */

int dp_same_dim(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,int index,const char *whence)
{
	if( OBJ_TYPE_DIM(dp1,index) == OBJ_TYPE_DIM(dp2,index) ) return(1);

	DEFAULT_WHENCE("dp_same_dim")
	sprintf(ERROR_STRING,
	"%s:  Objects %s and %s differ in %s size (%u, %u)",
		whence,
		OBJ_NAME(dp1),OBJ_NAME(dp2),dimension_name[index],
		OBJ_TYPE_DIM(dp1,index),
		OBJ_TYPE_DIM(dp2,index));
	WARN(ERROR_STRING);
	return(0);
}

/* dp_equal_dim is like dp_same_dim, but returns the result silently */

int dp_equal_dim(Data_Obj *dp1,Data_Obj *dp2,int index)
{
	if( OBJ_TYPE_DIM(dp1,index) == OBJ_TYPE_DIM(dp2,index) ) return(1);
	return(0);
}

/*
 * Return 1 if all sizes match, 0 otherwise
 *
 * This function expects them to be the same, and prints a msg if not.
 */

int dp_same_size( QSP_ARG_DECL  Data_Obj *dp1, Data_Obj *dp2, const char *whence )
{
	int i;

	DEFAULT_WHENCE("dp_same_size");
	for(i=0;i<N_DIMENSIONS;i++)
		if( ! dp_same_dim(QSP_ARG  dp1,dp2,i,whence) ) return(0);
	return(1);
}

/*
 * Return 1 if all sizes match, 0 otherwise
 *
 * This function does not necessarily expect them to be the same, and is silent if not.
 */

int dp_same_size_query( Data_Obj *dp1, Data_Obj *dp2 )
{
	int i;

	for(i=0;i<N_DIMENSIONS;i++)
		if( OBJ_TYPE_DIM(dp1,i) != OBJ_TYPE_DIM(dp2,i) ){
			return(0);
		}
	return(1);
}

// _rc is for real/complex - but this function ignores type dimension

int dp_same_size_query_rc( Data_Obj *dp1, Data_Obj *dp2 )
{
	int i;

	for(i=1;i<N_DIMENSIONS;i++)
		if( OBJ_TYPE_DIM(dp1,i) != OBJ_TYPE_DIM(dp2,i) ){
			return(0);
		}
	return(1);
}

int dp_same_prec(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2, const char *whence )
{
	if( whence == NULL ) whence = "sp_same_prec";
	if( OBJ_PREC(dp1) != OBJ_PREC(dp2) ){
		sprintf(ERROR_STRING,
			"%s:  Objects %s (%s) and %s (%s) differ in precision",
			whence,
			OBJ_NAME(dp1),OBJ_PREC_NAME(dp1),
			OBJ_NAME(dp2),OBJ_PREC_NAME(dp2) );
		WARN(ERROR_STRING);
		return(0);
	}
	return(1);
}

int dp_same_mach_prec(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2, const char *whence )
{
	DEFAULT_WHENCE("dp_same_mach_prec")
	if( OBJ_MACH_PREC(dp1) != OBJ_MACH_PREC(dp2) ){
		sprintf(ERROR_STRING,
	"%s:  Objects %s (%s) and %s (%s) differ in machine precision",
			whence,
			OBJ_NAME(dp1),OBJ_MACH_PREC_NAME(dp1),
			OBJ_NAME(dp2),OBJ_MACH_PREC_NAME(dp2));
		WARN(ERROR_STRING);
		return(0);
	}
	return(1);
}

int dp_same_pixel_type( QSP_ARG_DECL  Data_Obj *dp1, Data_Obj *dp2, const char *whence )
{
	if( whence == NULL ) whence = "dp_same_pixel_type";

	if( !dp_same_prec(QSP_ARG  dp1,dp2,whence) ) return(0);

	if( OBJ_MACH_DIM(dp1,0) != OBJ_MACH_DIM(dp2,0) ){
		sprintf(ERROR_STRING,"%s:  type dimension mismatch between %s (%d) and %s (%d)",
			whence,
			OBJ_NAME(dp1),OBJ_MACH_DIM(dp1,0),
			OBJ_NAME(dp2),OBJ_MACH_DIM(dp2,0));
		WARN(ERROR_STRING);
		return(0);
	}
	return(1);
}

/*
 * return 1 if these are the same in every way
 */

int dp_same( QSP_ARG_DECL  Data_Obj *dp1, Data_Obj *dp2, const char *whence )
{
	if( whence == NULL ) whence = "dp_same";
	if( !dp_same_size(QSP_ARG  dp1,dp2,whence) ) return(0);
	if( !dp_same_prec(QSP_ARG  dp1,dp2,whence) ) return(0);
	return(1);
}

/**********************/


/* sets thread var qs_max_vectorizable
 * What is this used for?  SIMD splitting?
 * It appears to be used in dp1_vectorize (recursive function application).
 */

void check_vectorization(QSP_ARG_DECL  Data_Obj *dp)
{
	int max_v;
	int i,j;

	assert( dp != NULL );

	max_v = N_DIMENSIONS-1;	/* default:  vectorize over everything */

	for(i=0;i<(N_DIMENSIONS-1);i++){
		if( OBJ_TYPE_DIM(dp,i) > 1 ){
			/* find the next biggest dimension > 1 */
			for(j=i+1;j<N_DIMENSIONS;j++){
				if( OBJ_TYPE_DIM(dp,j) > 1 ){
					if( OBJ_TYPE_INC(dp,j) != 
						((long)OBJ_TYPE_DIM(dp,i))
						* OBJ_TYPE_INC(dp,i) ){
						max_v = i;

						/* break out of i loop */
						i=N_DIMENSIONS;
					}
					/* break out of j loop */
					j=N_DIMENSIONS;
				}
			}
		}
	}

	// Where is MAX_VECTORIZABLE initialized???  BUG?
	if( max_v < MAX_VECTORIZABLE )
		SET_MAX_VECTORIZABLE(max_v);

	/* special case :  bitmaps for selection, if the row size is not a multiple
	 * of 32, then we can't vectorize across rows...
	 */

	/* We need to do a special test for bitmap
	 */

	if( IS_BITMAP(dp) ){
		sprintf(ERROR_STRING,"check_vectorization:  may be incorrect for bitmap object %s!?",
			OBJ_NAME(dp));
		advise(ERROR_STRING);
	}

#ifdef QUIP_DEBUG
/*
if( debug & debug_data ){
sprintf(ERROR_STRING,"check_vectorization %s:  max_vectorizable = %d",OBJ_NAME(dp),MAX_VECTORIZABLE);
advise(ERROR_STRING);
}
*/
#endif /* QUIP_DEBUG */

}


/*
 * if we are vectorizing over rows (as in subimage), then vectorize(wtp,2)
 * does one frame; vectorize(wtp,3) will do the sequence
 */

void dp2_vectorize(QSP_ARG_DECL  int level,Data_Obj *dpto,Data_Obj *dpfr,void (*dp_func)(Data_Obj *,Data_Obj *) )
{

#ifdef QUIP_DEBUG
if( debug & debug_data ){
sprintf(ERROR_STRING,"level = %d, max_v = %d",level,MAX_VECTORIZABLE);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	if( level == MAX_VECTORIZABLE ){
		(*dp_func)(dpto,dpfr);
	} else if(OBJ_TYPE_DIM(dpto,level)==1){
		dp2_vectorize(QSP_ARG  level-1,dpto,dpfr,dp_func);
	} else {
		dimension_t i;
		Data_Obj *_dpto,*_dpfr;


#ifdef QUIP_DEBUG
if( debug & debug_data ){
sprintf(ERROR_STRING,"dp2_vec:  subscripting at level %d, n=%u (max_vec = %d)",
level,OBJ_TYPE_DIM(dpto,level),MAX_VECTORIZABLE);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

		_dpto = gen_subscript(QSP_ARG  dpto,level,0L,SQUARE);
		_dpfr = gen_subscript(QSP_ARG  dpfr,level,0L,SQUARE);
		for(i=0;i<OBJ_TYPE_DIM(dpto,level);i++){
			reindex(QSP_ARG  _dpto,level,i);
			reindex(QSP_ARG  _dpfr,level,i);
			dp2_vectorize(QSP_ARG  level-1,_dpto,_dpfr,dp_func);
		}

		/* We reset the indices so that we get the correct
		 * offset when referred to by the original name
		 * (because reindex doesn't fix the name).
		 */

		reindex(QSP_ARG  _dpto,level,0L);
		reindex(QSP_ARG  _dpfr,level,0L);
	}
}

// dp1_vectorize applies a function to an object (if level == max_vectorizable),
// or applies the function to an array of indexed subobjects (possibly recursively)...

void dp1_vectorize(QSP_ARG_DECL  int level,Data_Obj *dp,void (*dp_func)(Data_Obj *))
{

#ifdef QUIP_DEBUG
if( debug & debug_data ){
sprintf(ERROR_STRING,"level = %d, max_v = %d",level,MAX_VECTORIZABLE);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	if( level == MAX_VECTORIZABLE ){
		(*dp_func)(dp);
	} else if(OBJ_TYPE_DIM(dp,level)==1){
		dp1_vectorize(QSP_ARG  level-1,dp,dp_func);
	} else {
		dimension_t i;
		Data_Obj *_dp;


#ifdef QUIP_DEBUG
if( debug & debug_data ){
sprintf(ERROR_STRING,"dp1_vec:  subscripting at level %d, n=%u (max_vec = %d)",
level,OBJ_TYPE_DIM(dp,level),MAX_VECTORIZABLE);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

		_dp = gen_subscript(QSP_ARG  dp,level,0L,SQUARE);
		for(i=0;i<OBJ_TYPE_DIM(dp,level);i++){
			reindex(QSP_ARG  _dp,level,i);
			dp1_vectorize(QSP_ARG  level-1,_dp,dp_func);
		}

		/* We reset the indices so that we get the correct
		 * offset when referred to by the original name
		 * (because reindex doesn't fix the name).
		 */

		reindex(QSP_ARG  _dp,level,0L);
	}
}

int not_prec(QSP_ARG_DECL  Data_Obj *dp,prec_t prec)
{
	if( OBJ_MACH_PREC(dp) != prec ){
		sprintf(ERROR_STRING,"Object %s has precision %s, expecting %s",
			OBJ_NAME(dp),OBJ_MACH_PREC_NAME(dp),NAME_FOR_PREC_CODE(prec));
		WARN(ERROR_STRING);
		return(1);
	}
	return(0);
}

