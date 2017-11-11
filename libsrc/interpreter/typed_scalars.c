#include <string.h>
#include "quip_config.h"
#include "quip_prot.h"
#include "data_obj.h"
#include "typed_scalar.h"
#include "debug.h"

#define MAX_TYPED_SCALARS	32

static Typed_Scalar ts_array[MAX_TYPED_SCALARS];
static int ts_idx=(-1);	// BUG - not thread-safe...

static Typed_Scalar *available_typed_scalar(void)
{
	int i=0,j=0;

	// first time initialization
	if( ts_idx < 0 ){
		for(i=0;i<MAX_TYPED_SCALARS;i++){
			ts_array[i].ts_flags=TS_FREE;
		}
		ts_idx=0;
	}

	assert( ts_idx < MAX_TYPED_SCALARS );

	// search for an available scalar
	i=ts_idx;
	for(j=0;j<MAX_TYPED_SCALARS;j++){
		if( FREE_SCALAR(&ts_array[i]) ){
			ts_idx=i+1;
			if( ts_idx >= MAX_TYPED_SCALARS )
				ts_idx=0;
			return( &ts_array[i] );
		}
		i++;
		if( i >= MAX_TYPED_SCALARS ) i=0;
	}
	NERROR1("avaliable_typed_scalar:  no free scalars!?");
	return NULL;
}

#define TS_PREC(tsp)	((tsp)->ts_prec_code)

int scalars_are_equal(Typed_Scalar *tsp1, Typed_Scalar *tsp2)
{
	if( TS_PREC(tsp1) == PREC_DP ){
		if( TS_PREC(tsp2) == PREC_DP ){
			if( tsp1->ts_value.u_d ==
					tsp2->ts_value.u_d )
				return 1;
			else	return 0;
		} else if( TS_PREC(tsp2) == PREC_LI ){
			if( tsp1->ts_value.u_d ==
					(double) tsp2->ts_value.u_ll )
				return 1;
			else	return 0;
		}
		  else {
		  	assert( AERROR("scalars_are_equal:  Unexpected type #1!?") );
		}
	} else if( TS_PREC(tsp1) == PREC_LI ){
		if( TS_PREC(tsp2) == PREC_DP ){
			if( (double) tsp1->ts_value.u_ll ==
					tsp2->ts_value.u_d )
				return 1;
			else	return 0;
		} else if( TS_PREC(tsp2) == PREC_LI ){
			if( tsp1->ts_value.u_ll ==
					tsp2->ts_value.u_ll )
				return 1;
			else	return 0;
		}
		  else {
		  	assert( AERROR("scalars_are_equal:  Unexpected type #1!?") );
		}
	}	
	  else {
	  	assert( AERROR("scalars_are_equal:  Unexpected type #3!?") );
	}
	return 0;
}

int _has_zero_value(QSP_ARG_DECL  Typed_Scalar *tsp )
{
	if( TS_PREC(tsp) == PREC_DP ){
		if( tsp->ts_value.u_d == 0.0 ) return 1;
		else return 0;
	} else if( TS_PREC(tsp) == PREC_LI ){
		if( tsp->ts_value.u_ll == 0 ) return 1;
		else return 0;
	} else if( TS_PREC(tsp) == PREC_STR ){
		char *s;
		s=tsp->ts_value.u_vp;
		if( !strcmp(s,"true") ) return 1;
		else if( !strcmp(s,"false") ) return 0;
		else {
	  		sprintf(ERROR_STRING,
"has_zero_value:  string argument \"%s\", expected \"true\" or \"false\"",s);
			warn(ERROR_STRING);
			return 0;
		}
	}
			
	  else {
		assert( AERROR("has_zero_value:  Unexpected type (%"PRId64")!?") );
	}
	return 0;
}

Typed_Scalar *scalar_for_long(long v)
{
	Typed_Scalar *tsp;

	tsp = available_typed_scalar();
	tsp->ts_prec_code = PREC_LI;
	tsp->ts_value.u_ll = v;
	return tsp;
}

Typed_Scalar *scalar_for_llong(int64_t v)
{
	Typed_Scalar *tsp;

	tsp = available_typed_scalar();
	tsp->ts_prec_code = PREC_LI;
	tsp->ts_value.u_ll = v;
	return tsp;
}

Typed_Scalar *scalar_for_double(double v)
{
	Typed_Scalar *tsp;

	tsp = available_typed_scalar();
	tsp->ts_prec_code = PREC_DP;
	tsp->ts_value.u_d = v;
	return tsp;
}

// returns a typed scalar that points to the given string...

Typed_Scalar *scalar_for_string(const char *s)
{
	Typed_Scalar *tsp;

	if( s == NULL ) return NULL;
	tsp = available_typed_scalar();
	tsp->ts_prec_code = PREC_STR;
	tsp->ts_value.u_vp = (void *)s;
	return tsp;
}

double _double_for_scalar(QSP_ARG_DECL  Typed_Scalar *tsp)
{
	if( TS_PREC(tsp) == PREC_DP ){
		return tsp->ts_value.u_d;
	} else if( TS_PREC(tsp) == PREC_LI ){
		return (double) tsp->ts_value.u_ll;
	} else if( TS_PREC(tsp) == PREC_STR ){
        	sprintf(ERROR_STRING,
	"double_for_scalar:  passed string \"%s\" instead of a numeric scalar!?",
		(const char *)(tsp->ts_value.u_vp));
		warn(ERROR_STRING);
	}
		
	  else {
	  	assert( AERROR("double_for_scalar:  Unexpected type code!?") );
	}
	return 0.0;
}

int64_t llong_for_scalar(Typed_Scalar *tsp)
{
	if( TS_PREC(tsp) == PREC_DP ){
		return (int64_t) tsp->ts_value.u_d;
	} else if( TS_PREC(tsp) == PREC_LI ){
		return tsp->ts_value.u_ll;
	}
	  else {
	  	assert( AERROR("llong_for_scalar:  Unexpected type code!?") );
	}
	return 0;
}

int32_t _long_for_scalar(QSP_ARG_DECL  Typed_Scalar *tsp)
{
	if( TS_PREC(tsp) == PREC_DP ){
		return (int32_t) tsp->ts_value.u_d;
	} else if( TS_PREC(tsp) == PREC_LI ){
		int64_t ll;
		ll = (int64_t)tsp->ts_value.u_ll;
		if( ll > 0x7fffffff || ll < -0x80000000 ){
			warn("long_for_scalar:  losing precision!?");
		}
		return (int32_t) ll;
	}
	  else {
	  	assert( AERROR( "long_for_scalar:  Unexpected type code!?") );
	}
	return 0;
}

// index_t is uint32_t...
//
// cast a scalar to a value that can be used as an index...

index_t _index_for_scalar(QSP_ARG_DECL  Typed_Scalar *tsp)
{
	int64_t v;

	if( TS_PREC(tsp) == PREC_DP ){
		v = (int64_t) tsp->ts_value.u_d;
	} else if( TS_PREC(tsp) == PREC_LI ){
		v = tsp->ts_value.u_ll;
	}
	  else {
          fprintf(stderr,"index_for_scalar:  tsp at 0x%llx has precision code %d - expected %d (PREC_DP) or %d (PREC_LI)\n",
                  (long long) tsp,TS_PREC(tsp),PREC_DP,PREC_LI);
		v=0;
	  	assert( AERROR( "index_for_scalar:  Unexpected type code!?") );
	}
	if( v < 0 ){
		sprintf(ERROR_STRING,
	"index_for_scalar:  value (%"PRId64") is negative!?",v);
		warn(ERROR_STRING);
		v=0;
	}
	if( v > 0xffffffff ){
		sprintf(ERROR_STRING,
	"index_for_scalar:  value (0x%"PRIx64") is out of range!?",v);
		warn(ERROR_STRING);
		v=0;
	}
	return (index_t) v;
}

void _string_for_typed_scalar(QSP_ARG_DECL  char *buf,int buflen, Typed_Scalar *tsp)
{
	// BUG should check buffer length even for the numbers...
	if( TS_PREC(tsp) == PREC_DP ){
		sprintf(buf,"%g",tsp->ts_value.u_d);
	} else if( TS_PREC(tsp) == PREC_LI ){
		sprintf(buf,"0x%"PRIx64, tsp->ts_value.u_ll);
	} else if( TS_PREC(tsp) == PREC_STR ){
		if( strlen( (char *) tsp->ts_value.u_vp ) >= buflen ){
			warn("string_for_typed_scalar:  value would overrun fixed-size buffer!?");
		}
		//sprintf(buf,"%s",(char *)tsp->ts_value.u_vp);
		strncpy(buf,tsp->ts_value.u_vp,buflen);
	} else {
		/*
		sprintf(ERROR_STRING,
	"string_for_typed_scalar:  bad prec code %d (%s)",
			TS_PREC(tsp),PREC_NAME(PREC_FOR_CODE(TS_PREC(tsp))));
		NERROR1(ERROR_STRING);
		*/
		NERROR1("string_for_typed_scalar:  bad prec code");
	}
}

#define BUFLEN	80

void _show_typed_scalar(QSP_ARG_DECL  Typed_Scalar *tsp)
{
	char str[BUFLEN];

	string_for_typed_scalar(str,BUFLEN,tsp);

	if( TS_PREC(tsp) == PREC_DP ){
		sprintf(MSG_STR,"double precision float:  %s\n",str);
	} else if( TS_PREC(tsp) == PREC_LI ){
		sprintf(MSG_STR,"long integer:  %"PRId64" (%s)\n",
			tsp->ts_value.u_ll, str);
	} else if( TS_PREC(tsp) == PREC_STR ){
		sprintf(MSG_STR,"string:  \"%s\"",(char *)tsp->ts_value.u_vp);
	} else {
		error1("show_typed_scalar:  bad prec code");
	}
	prt_msg(MSG_STR);
}

