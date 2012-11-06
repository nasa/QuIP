
#include "quip_config.h"

char VersionId_newvec_fftsupp[] = QUIP_VERSION_STRING;

#include "nvf.h"

/* local prototypes */

static void mk_bitrev(dimension_t);
static int log_2(dimension_t n);

dimension_t bitrev_size=0;
dimension_t *bitrev_data;

void bitrev_init(dimension_t len)
{
	if( bitrev_size > 0 ){
		givbuf(bitrev_data);
	}
	bitrev_data = (dimension_t *)getbuf( sizeof(*bitrev_data) * len );
	mk_bitrev(len);
}

static void mk_bitrev(dimension_t n)
{
	dimension_t i,j;
	dimension_t m;
		
	/*	bit reverse	*/

	/*		i	j
			0	0
			1	n/2
			2	n/4
			3	3n/4
	*/

	j = 0;
	for(i=0;i<n;i++){
		bitrev_data[i]=j;
		m = n/2;
		while (j>=m) {
			j -= m;
			m = (m+1)/2;
		}
		j += m;
	}
	bitrev_size = n;
}

static int log_2(dimension_t n)
{
	dimension_t p;
	int w;

	p = 0;
	w = 0;	
	while (n >= 2 && p == 0) {
		++w;
		p = n % 2;
		n = n / 2;
	}
	if (p != 0) return(-1);
	else return(w);
}

int fft_row_size_ok(QSP_ARG_DECL  Data_Obj *dp)
{
	if( log_2(dp->dt_cols) == -1 ){
		sprintf(error_string,
	"Number of columns of image %s (%d) is not a power of two for FFT",
			dp->dt_name,dp->dt_cols);
		WARN(error_string);
		LONGLIST(dp);
		return(-1);
	}

	return(0);
}

int fft_size_ok(QSP_ARG_DECL  Data_Obj *dp)
{
	if( fft_row_size_ok(QSP_ARG  dp) < 0 ) return(-1);
	if( fft_col_size_ok(QSP_ARG  dp) < 0 ) return(-1);
	return(0);
}

int fft_col_size_ok(QSP_ARG_DECL  Data_Obj *dp)
{
	if( log_2(dp->dt_rows) == -1 ){
		sprintf(error_string,
	"Number of rows of image %s (%d) is not a power of two for FFT",
			dp->dt_name,dp->dt_rows);
		WARN(error_string);
		LONGLIST(dp);
		return(-1);
	}
	return(0);
}

int real_row_fft_check(QSP_ARG_DECL  Data_Obj *real_dp,Data_Obj *cpx_dp,const char *funcname)
{
	if( (cpx_dp->dt_cols-1) != (real_dp->dt_cols/2) ){
		sprintf(error_string,
"%s:  complex %s ncols (%d) should be 1 plus half real %s ncols (%d)",
			funcname,cpx_dp->dt_name,cpx_dp->dt_cols,
			real_dp->dt_name,real_dp->dt_cols);
		WARN(error_string);
		return(-1);
	}
	if( cpx_dp->dt_rows != real_dp->dt_rows ){
		sprintf(error_string,
			"%s:  row count mismatch, %s (%d) and %s (%d)",
			funcname,cpx_dp->dt_name,cpx_dp->dt_rows,
			real_dp->dt_name,real_dp->dt_rows);
		WARN(error_string);
		return(-1);
	}
	if( ! IS_COMPLEX(cpx_dp) ){
		sprintf(error_string,
			"%s:  %s must be complex",funcname,cpx_dp->dt_name);
		WARN(error_string);
		return(-1);
	}
	if( ! IS_REAL(real_dp) ){
		sprintf(error_string,
			"%s:  %s must be real",funcname,real_dp->dt_name);
		WARN(error_string);
		return(-1);
	}
	if( ! FLOATING_OBJ( cpx_dp ) ){
		sprintf(error_string,
			"%s:  precision must be float or double",funcname);
		WARN(error_string);
		return(-1);
	}

	if( !dp_same_mach_prec(QSP_ARG  cpx_dp,real_dp,funcname) ){
		sprintf(error_string,
	"fft_real_check (%s):  complex object (%s,%s) and target (%s,%s) must have same precision",
			funcname,cpx_dp->dt_name,prec_name[MACHINE_PREC(cpx_dp)],
			real_dp->dt_name,prec_name[MACHINE_PREC(real_dp)]);
		WARN(error_string);
		return(-1);
	}
	if( fft_row_size_ok(QSP_ARG  real_dp) < 0 ) return(-1);
	return(0);
}

int real_fft_check(QSP_ARG_DECL  Data_Obj *real_dp,Data_Obj *cpx_dp,const char *funcname)
{
	if( real_row_fft_check(QSP_ARG  real_dp,cpx_dp,funcname) < 0 ) return(-1);
	if( fft_col_size_ok(QSP_ARG  real_dp) < 0 ) return(-1);

	return(0);
}

