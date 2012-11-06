#include "quip_config.h"

char VersionId_dataf_memops[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include "debug.h"
#include "data_obj.h"
#include "rn.h"
#include "query.h"			/* assign_var() */

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

/* local prototypes */
static void contig_copy(Data_Obj *dp_to,Data_Obj *dp_fr);
static void fast_copy(Data_Obj *dp_to,Data_Obj *dp_fr);
static void fast_uni(Data_Obj *dp);
static void fast_rand(Data_Obj *dp);
static void fast_equate(Data_Obj *dp);

#define DEFAULT_WHENCE(s)		if( whence == NULL ) whence=s;

int max_vectorizable;

void getmean( QSP_ARG_DECL  Data_Obj *dp )
{
	u_long i;
	u_long n;
	double sum, sos, f;
	float max,min;

	if( dp== NO_OBJ ) return;
	if( dp->dt_prec != PREC_SP && dp->dt_prec != PREC_IN ){
		NWARN("sorry, only float or short objects");
		return;
	}
	if( dp->dt_mach_dim[0] != 1 ){
		sprintf(error_string,"ALERT:  getmean:  object %s has %d components!?",
			dp->dt_name,dp->dt_mach_dim[0]);
		advise(error_string);
	}
	if( ! IS_CONTIGUOUS(dp) ){
		NWARN("sorry, can only compute mean of contiguous objects");
		return;
	}
	sum=sos=0.0;
	n=dp->dt_n_mach_elts;
	if( dp->dt_prec == PREC_SP ){
		float *fnp;

		fnp=(float *)dp->dt_data;
		max=min=(*fnp);
		for(i=0;i<n;i++){
			f=(*fnp);
			if( f > max ) max=(float)f;
			else if( f < min ) min=(float)f;
			sos += (double)(f * f);
			sum += (double)f;
			fnp++;
		}
	} else if( dp->dt_prec == PREC_IN ){
		short *inp;

		inp=(short *)dp->dt_data;
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
		sprintf(error_string,
			"mean:  %f\nvariance:  %f\nmax:  %f\nmin:  %f",
			sum,sos,max,min);
		advise(error_string);
	}

	sprintf(error_string,"%f",sum);
	ASSIGN_VAR("mean",error_string);
	sprintf(error_string,"%f",sos);
	ASSIGN_VAR("variance",error_string);
	sprintf(error_string,"%f",max);
	ASSIGN_VAR("max",error_string);
	sprintf(error_string,"%f",min);
	ASSIGN_VAR("min",error_string);
}

static double equate_value;

/* equate a contiguous block of data */

#define EQUATE_IT( type )						\
		type * ptr;						\
		type value;						\
									\
		ptr = ( type * ) dp->dt_data;				\
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

	n=dp->dt_n_mach_elts;
	inc = dp->dt_mach_inc[ dp->dt_mindim ];

#ifdef DEBUG
if( debug & debug_data ){
sprintf(DEFAULT_ERROR_STRING,"fast_equate %s",dp->dt_name);
advise(DEFAULT_ERROR_STRING);
}
#endif /* DEBUG */

	if( dp->dt_prec == PREC_BY ){
		EQUATE_IT( char )
	} else if( dp->dt_prec == PREC_UBY ){
		EQUATE_IT( u_char )
	} else if( dp->dt_prec == PREC_SP ){
		EQUATE_IT( float )
	} else if( dp->dt_prec == PREC_DP ){
		EQUATE_IT( double )
	} else if( dp->dt_prec == PREC_IN ){
		EQUATE_IT( short )
	} else if( dp->dt_prec == PREC_UIN ){
		EQUATE_IT( u_short )
	} else if( dp->dt_prec == PREC_DI ){
		EQUATE_IT( long )
	} else if( dp->dt_prec == PREC_UDI ){
		EQUATE_IT( u_long )
	} else NWARN("fast_equate:  unsupported pixel type");
}

void dp_equate( QSP_ARG_DECL  Data_Obj *dp, double v )
{
	if( dp==NO_OBJ ) return;

	equate_value=v;

	max_vectorizable=N_DIMENSIONS-1;     /* default: vectorize over all */
	check_vectorization(dp);
	dp1_vectorize(QSP_ARG  (int)(N_DIMENSIONS-1),dp,fast_equate);
}

/* this version works for contiguous objects only */

static void contig_copy( Data_Obj *dp_to, Data_Obj *dp_fr )
{
	u_long nb;

#ifdef DEBUG
if( debug & debug_data ){
sprintf(DEFAULT_ERROR_STRING,"contig_copy:  %s  %s",dp_to->dt_name,dp_fr->dt_name);
advise(DEFAULT_ERROR_STRING);
}
#endif /* DEBUG */

	nb = dp_to->dt_n_mach_elts * ELEMENT_SIZE(dp_to);

	if( IS_BITMAP(dp_to) )
		nb = (nb+(BITS_PER_BITMAP_WORD-1)) / BITS_PER_BITMAP_WORD;

	/* BUG memcpy() is not guaranteed to do the correct
	 * thing on overlapping moves, but we use if for
	 * now, because memmove() does not seem to be
	 * available on SUN
	 */

#ifndef PC
	memcpy(dp_to->dt_data,dp_fr->dt_data,nb);
#else /* PC */
	if( nb <= 0x7fff )
		memcpy(dp_to->dt_data,dp_fr->dt_data,(int)nb);
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
		pto=(type *)dp_to->dt_data;				\
		pfr=(type *)dp_fr->dt_data;				\
		for(i=0;i<dp_to->dt_n_mach_elts;i++){				\
			*pto = *pfr;					\
			pto += to_inc;					\
			pfr += fr_inc;					\
		}

static void fast_copy( Data_Obj *dp_to, Data_Obj *dp_fr )
{
	u_long i;
	long to_inc,fr_inc;

	if( IS_CONTIGUOUS(dp_to) && IS_CONTIGUOUS(dp_fr) ){
		contig_copy(dp_to,dp_fr);
		return;
	}


#ifdef DEBUG
if( debug & debug_data ){
sprintf(DEFAULT_ERROR_STRING,"fast_copy:  %s  %s",dp_to->dt_name,dp_fr->dt_name);
advise(DEFAULT_ERROR_STRING);
}
#endif /* DEBUG */

	/* Because these objects are known to be evenly spaced,
	 * dt_mach_inc[dt_mindim] must be the increment!
	 */

	to_inc = dp_to->dt_mach_inc[ dp_to->dt_mindim ];
	fr_inc = dp_fr->dt_mach_inc[ dp_fr->dt_mindim ];

#ifdef DEBUG
if( debug & debug_data ){
sprintf(DEFAULT_ERROR_STRING,"fast_copy'ing %s (inc %ld) to %s (inc %ld)",
dp_fr->dt_name,fr_inc,dp_to->dt_name,to_inc);
advise(DEFAULT_ERROR_STRING);
}
#endif

	if( MACHINE_PREC(dp_to) == PREC_BY || MACHINE_PREC(dp_to) == PREC_UBY ){
		COPY_IT( u_char )
	} else if( MACHINE_PREC(dp_to) == PREC_IN || MACHINE_PREC(dp_to) == PREC_UIN ){
		COPY_IT( short )
	} else if( MACHINE_PREC(dp_to) == PREC_DI || MACHINE_PREC(dp_to) == PREC_UDI ){
		COPY_IT( long )
	} else if( MACHINE_PREC(dp_to) == PREC_SP){
		COPY_IT( float )
	} else if( MACHINE_PREC(dp_to) == PREC_DP){
		COPY_IT( double )
	}

#ifdef CAUTIOUS
	else {
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  fast_copy:  unsupported precision %s",
			prec_name[MACHINE_PREC(dp_to)]);
		NWARN(DEFAULT_ERROR_STRING);
	}
#endif /* CAUTIOUS */

}


/* general purpose copy */

void dp_copy( QSP_ARG_DECL  Data_Obj *dp_to, Data_Obj *dp_fr )
{
	if( ! dp_same(QSP_ARG  dp_to,dp_fr,"dp_copy") ) return;

	if( IS_CONTIGUOUS(dp_to) && IS_CONTIGUOUS(dp_fr) )
		contig_copy(dp_to,dp_fr);
	else {
		max_vectorizable=N_DIMENSIONS-1;     /* default: vectorize over all */
		check_vectorization(dp_to);
		check_vectorization(dp_fr);
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

	cp=(u_char *)dp->dt_data;
	i=dp->dt_n_mach_elts;
	inc = dp->dt_mach_inc[dp->dt_mindim];

	while(i--){
		*cp = (u_char)(_imin + rn((long)_imax));
		cp += inc;
	}
}

void i_rnd( QSP_ARG_DECL  Data_Obj *dp, int imin, int imax )
{
	if( dp==NO_OBJ ) return;
	if( MACHINE_PREC(dp) != PREC_BY && MACHINE_PREC(dp) != PREC_UBY ){
		sprintf(ERROR_STRING,"i_rnd:  object %s (%s) should have %s or %s precision",
				dp->dt_name,PNAME(dp),prec_name[PREC_BY],prec_name[PREC_UBY]);
		WARN(ERROR_STRING);
		return;
	}

	imax-=imin;

	_imax=imax;
	_imin=imin;

	max_vectorizable=N_DIMENSIONS-1;     /* default: vectorize over all */
	check_vectorization(dp);
	dp1_vectorize(QSP_ARG  N_DIMENSIONS-1,dp,fast_rand);
}

static void fast_uni( Data_Obj *dp )
{
	u_long i,n;
	float *fptr;
	long inc;

	fptr=(float *)dp->dt_data;
	n=dp->dt_n_mach_elts;
	inc = dp->dt_mach_inc[dp->dt_mindim];

#ifdef HAVE_DRAND48
	for(i=0;i<n;i++){
		*fptr = drand48();
		fptr += inc;
	}
#else
	error1("Sorry, no implementation of drand48() on this configuration...");
#endif

}

void dp_uni( QSP_ARG_DECL  Data_Obj *dp )
{
	/* need to seed this generator... */

	if( dp==NO_OBJ ) return;

	if( dp->dt_prec != PREC_SP ){
		NWARN("Uniform random numbers require FLOAT precision");
		return;
	}

	rninit(SINGLE_QSP_ARG);	/* initialize random number generator */
				/* BUG this assumes lib support compiled for drand48() */

	max_vectorizable=N_DIMENSIONS-1;     /* default: vectorize over all */
	check_vectorization(dp);
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
	if( dp1->dt_type_dim[index] == dp2->dt_type_dim[index] ) return(1);

	DEFAULT_WHENCE("dp_same_dim")
	sprintf(ERROR_STRING,
	"%s:  Objects %s and %s differ in %s size (%u, %u)",
		whence,
		dp1->dt_name,dp2->dt_name,dimension_name[index],
		dp1->dt_type_dim[index],
		dp2->dt_type_dim[index]);
	WARN(ERROR_STRING);
	return(0);
}

/* dp_equal_dim is like dp_same_dim, but returns the result silently */

int dp_equal_dim(Data_Obj *dp1,Data_Obj *dp2,int index)
{
	if( dp1->dt_type_dim[index] == dp2->dt_type_dim[index] ) return(1);
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
		if( dp1->dt_type_dim[i] != dp2->dt_type_dim[i] ){
			return(0);
		}
	return(1);
}

#ifdef FOOBAR

/* We don't need this now that we have type_dim. */

/* like dp_same_size_query(), but don't check component dimension.
 * This is useful for comparing real and complex objects.
 */

int dp_same_size_query_rc( Data_Obj *dp1, Data_Obj *dp2 )
{
	int i;

	// First, make sure that the first obj is real and the second is cpx
	if( dp1->dt_tdim != 1 ) return(0);
	if( ! IS_COMPLEX( dp2 ) ) return(0);

	for(i=1;i<N_DIMENSIONS;i++)
		if( dp1->dt_dimension[i] != dp2->dt_dimension[i] ){
			return(0);
		}
	return(1);
}
#endif /* FOOBAR */


#ifdef FOOBAR
int dp_same_len( Data_Obj *dp1, Data_Obj *dp2 )
{
	int i;

	for(i=1;i<N_DIMENSIONS;i++)
		if( dp1->dt_dimension[i] != dp2->dt_dimension[i] ){
			return(0);
		}
	return(1);
}
#endif /* FOOBAR */

int dp_same_prec(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2, const char *whence )
{
	if( whence == NULL ) whence = "sp_same_prec";
	if( dp1->dt_prec != dp2->dt_prec ){
		sprintf(ERROR_STRING,
			"%s:  Objects %s (%s) and %s (%s) differ in precision",
			whence,
			dp1->dt_name,name_for_prec(dp1->dt_prec),
			dp2->dt_name,name_for_prec(dp2->dt_prec) );
		WARN(ERROR_STRING);
		return(0);
	}
	return(1);
}

int dp_same_mach_prec(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2, const char *whence )
{
	DEFAULT_WHENCE("dp_same_mach_prec")
	if( MACHINE_PREC(dp1) != MACHINE_PREC(dp2) ){
		sprintf(ERROR_STRING,
	"%s:  Objects %s (%s) and %s (%s) differ in machine precision",
			whence,
			dp1->dt_name,prec_name[MACHINE_PREC(dp1)],
			dp2->dt_name,prec_name[MACHINE_PREC(dp2)]);
		WARN(ERROR_STRING);
		return(0);
	}
	return(1);
}

int dp_same_pixel_type( QSP_ARG_DECL  Data_Obj *dp1, Data_Obj *dp2, const char *whence )
{
	if( whence == NULL ) whence = "dp_same_pixel_type";

	if( !dp_same_prec(QSP_ARG  dp1,dp2,whence) ) return(0);

	if( dp1->dt_mach_dim[0] != dp2->dt_mach_dim[0] ){
		sprintf(ERROR_STRING,"%s:  type dimension mismatch between %s (%d) and %s (%d)",
			whence,
			dp1->dt_name,dp1->dt_mach_dim[0],
			dp2->dt_name,dp2->dt_mach_dim[0]);
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


void check_vectorization(Data_Obj *dp)		/** sets global max_vectorizable */
	/* NOT thread-safe - FIXME! */
{
	int max_v;
	int i,j;
#ifdef FOOBAR
	int start_dim;
#endif /* FOOBAR */

#ifdef CAUTIOUS
	if( dp == NO_OBJ ){
		NWARN("CAUTIOUS:  check_vectorization called with NULL arg!?");
		return;
	}
#endif /* CAUTIOUS */

	max_v = N_DIMENSIONS-1;	/* default:  vectorize over everything */

#ifdef FOOBAR
	if( IS_COMPLEX(dp) )
		start_dim=1;
	else	start_dim=0;
#endif /* FOOBAR */

	for(i=/*start_dim*/0;i<(N_DIMENSIONS-1);i++){
		if( dp->dt_type_dim[i] > 1 ){
			/* find the next biggest dimension > 1 */
			for(j=i+1;j<N_DIMENSIONS;j++){
				if( dp->dt_type_dim[j] > 1 ){
					if( dp->dt_type_inc[j] != 
						((long)dp->dt_type_dim[i])
						* dp->dt_type_inc[i] ){
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

	if( max_v < max_vectorizable )
		max_vectorizable = max_v;

	/* special case :  bitmaps for selection, if the row size is not a multiple
	 * of 32, then we can't vectorize across rows...
	 */

	/* We need to do a special test for bitmaps, but the logic is different now that
	 * we are not requiring an integral number of words per row.
	 * FIXME
	 */

#ifdef DEBUG
/*
if( debug & debug_data ){
sprintf(error_string,"check_vectorization %s:  max_vectorizable = %d",dp->dt_name,max_vectorizable);
advise(error_string);
}
*/
#endif /* DEBUG */

}


/*
 * if we are vectorizing over rows (as in subimage), then vectorize(wtp,2)
 * does one frame; vectorize(wtp,3) will do the sequence
 */

void dp2_vectorize(QSP_ARG_DECL  int level,Data_Obj *dpto,Data_Obj *dpfr,void (*dp_func)(Data_Obj *,Data_Obj *) )
{

#ifdef DEBUG
if( debug & debug_data ){
sprintf(error_string,"level = %d, max_v = %d",level,max_vectorizable);
advise(error_string);
}
#endif /* DEBUG */

	if( level == max_vectorizable ){
		(*dp_func)(dpto,dpfr);
	} else if(dpto->dt_type_dim[level]==1){
		dp2_vectorize(QSP_ARG  level-1,dpto,dpfr,dp_func);
	} else {
		dimension_t i;
		Data_Obj *_dpto,*_dpfr;


#ifdef DEBUG
if( debug & debug_data ){
sprintf(error_string,"dp2_vec:  subscripting at level %d, n=%u (max_vec = %d)",
level,dpto->dt_type_dim[level],max_vectorizable);
advise(error_string);
}
#endif /* DEBUG */

		_dpto = gen_subscript(QSP_ARG  dpto,level,0L,SQUARE);
		_dpfr = gen_subscript(QSP_ARG  dpfr,level,0L,SQUARE);
		for(i=0;i<dpto->dt_type_dim[level];i++){
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


void dp1_vectorize(QSP_ARG_DECL  int level,Data_Obj *dp,void (*dp_func)(Data_Obj *))
{

#ifdef DEBUG
if( debug & debug_data ){
sprintf(error_string,"level = %d, max_v = %d",level,max_vectorizable);
advise(error_string);
}
#endif /* DEBUG */

	if( level == max_vectorizable ){
		(*dp_func)(dp);
	} else if(dp->dt_type_dim[level]==1){
		dp1_vectorize(QSP_ARG  level-1,dp,dp_func);
	} else {
		dimension_t i;
		Data_Obj *_dp;


#ifdef DEBUG
if( debug & debug_data ){
sprintf(error_string,"dp1_vec:  subscripting at level %d, n=%u (max_vec = %d)",
level,dp->dt_type_dim[level],max_vectorizable);
advise(error_string);
}
#endif /* DEBUG */

		_dp = gen_subscript(QSP_ARG  dp,level,0L,SQUARE);
		for(i=0;i<dp->dt_type_dim[level];i++){
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
	if( MACHINE_PREC(dp) != prec ){
		sprintf(ERROR_STRING,"Object %s has precision %s, expecting %s",
			dp->dt_name,prec_name[MACHINE_PREC(dp)],prec_name[prec]);
		WARN(ERROR_STRING);
		return(1);
	}
	return(0);
}

