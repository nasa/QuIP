
/* included file, so no version string */

#ifdef HAVE_MATH_H
#include <math.h>
#endif

/* this file is included in sptest.c and dptest.c */

#define MIN_PARALLEL_PROCESSORS	100	// a nonsense value to inhibit the switches
					// set to 2 to use these features on a dual proc machine.

static dimension_t last_cpx_len=0;
static std_cpx *twiddle;

static dimension_t last_real_len=0;
static std_type *_isinfact=NULL;
static std_type *_sinfact=NULL;

/* local prototypes */
static void init_sinfact(dimension_t n);
static void init_twiddle(dimension_t);
static void _c_fft(FFT_Args *fap);
static int fft_ok(QSP_ARG_DECL  Data_Obj *dp);
static int row_fft_ok(QSP_ARG_DECL  Data_Obj *dp);

#define SET_FFT_INC( which_inc, dp )					\
									\
	if( IS_ROWVEC(dp) ){						\
		fa.which_inc = (dp)->dt_pinc;				\
	} else if( IS_COLVEC(dp) ){					\
		fa.which_inc = (dp)->dt_rinc;				\
	} else {							\
		NWARN(DEFAULT_ERROR_STRING);				\
		sprintf(DEFAULT_ERROR_STRING,				\
	"rvfft:  %s is neither a row nor a column!?",(dp)->dt_name);	\
		return;							\
	}								\
	if( IS_COMPLEX(dp) )						\
		fa.which_inc /= 2;


static void init_twiddle(dimension_t len)
{
	double pi,theta;
	dimension_t i;

	if( last_cpx_len > 0 ){
		givbuf(twiddle);
	}

	twiddle = (std_cpx *)getbuf( sizeof(*twiddle) * (len/2) );


	pi = 8.0*atan(1.0);	/* This is two pi!? */

	/* W -kn N , W N = exp ( -j twopi/N ) */

	for(i=0;i<len/2;i++){
		theta = pi*(double)(i)/(double)len;
		twiddle[i].re = cos(theta);
		twiddle[i].im = sin(theta);
	}
	last_cpx_len=len;
}

/* This is usually called in-place, but doesn't have to be.
 *
 * isi = -1   ->  forward xform
 * isi =  1   ->  inverse xform
 */


#define MAX_FFT_LEN	4096L
static char *revdone=NULL;
static u_int max_fft_len=(-1);

#ifdef FOOBAR
static void show_fa(FFT_Args *fap)
{
	sprintf(error_string,"dst_addr = 0x%lx, inc = %ld",(int_for_addr)fap->dst_addr,fap->dst_inc);
	advise(error_string);
	sprintf(error_string,"src_addr = 0x%lx, inc = %ld",(int_for_addr)fap->src_addr,fap->src_inc);
	advise(error_string);
	sprintf(error_string,"len = %ld, isi = %d",fap->len,fap->isi);
	advise(error_string);
}
#endif /* FOOBAR */

/* _c_fft takes separate dest and source args, but works for in-place(?)
 */

static void _c_fft(FFT_Args *fap)
{
	dimension_t i,j;
	dimension_t len;
	std_cpx temp,*wp;
	std_cpx *source, *dest;
	dimension_t m, mmax, istep;
	incr_t inc1;
	/* BUG we really don't want to allocate and deallocate revdone each time... */
	/* anyway, this is no good because getbuf/givbuf are not thread-safe!
	 * I can't see a way to do this without passing the thread index on the stack...
	 * OR having the entire revdone array on the stack?
	 */
	/* char revdone[MAX_FFT_LEN]; */

	if( ! for_real ) return;

	len = fap->len;

	if( revdone==NULL ){
		revdone=(char *)getbuf(len);
		max_fft_len = len;
	}
	if( len > max_fft_len ){
		givbuf(revdone);
		revdone=(char *)getbuf(len);
		max_fft_len = len;
	}

	if( len != last_cpx_len ) {
#if N_PROCESSORS >= MIN_PARALLEL_PROCESSORS
		if( n_processors > 1 ) WARN("_c_fft:  init_twiddle is not thread-safe!?");
#endif
		init_twiddle(len);
	}

	dest=(std_cpx *)fap->dst_addr;
	source=(std_cpx *)fap->dst_addr;
	inc1 = fap->dst_inc;
	/* inc1 should be in units of complex */

	if( len != bitrev_size ){
#if N_PROCESSORS >= MIN_PARALLEL_PROCESSORS
		if( n_processors > 1 ) WARN("_c_fft:  bitrev_init is not thread-safe!?");
#endif
		bitrev_init(len);
	}

	/* init revdone */
		
	for(i=0;i<len;i++) revdone[i]=0;
	for(i=0;i<len;i++){
		dimension_t di, dj;
		std_cpx tmp;

		if( !revdone[i] ){
			di = i * inc1;
			dj = bitrev_data[i] * inc1;
			if( di != dj ){
				tmp.re = source[di].re;
				tmp.im = source[di].im;
				dest[di].re = source[dj].re;
				dest[di].im = source[dj].im;
				dest[dj].re = tmp.re;
				dest[dj].im = tmp.im;
			}
			revdone[i]=1;
			revdone[ bitrev_data[i] ]=1;
		}
	}

	/*	now compute the butterflies 	*/
	/* this section is trashing some memory!? */

	mmax = 1;
	while( mmax<len ){
		istep = 2*mmax;
		for(m=0;m<mmax;m++){
			dimension_t index;

			index = m*(len/(mmax<<1));

			/* make index modulo len/2 */
			/* hope this works for negative index!! */
			index &= ((len>>1)-1);

			/* if( index < 0 ) index += len; */

			wp = (& twiddle[index]);

			for(i=m;i<len;i+=istep){
				dimension_t dj, di;

				j = i+mmax;
				dj = j * inc1;
				di = i * inc1;
				temp.re = wp->re*dest[dj].re
					- fap->isi * wp->im*dest[dj].im;
				temp.im = wp->re*dest[dj].im
					+ fap->isi * wp->im*dest[dj].re;
				dest[dj].re = dest[di].re-temp.re;
				dest[dj].im = dest[di].im-temp.im;
				dest[di].re += temp.re;
				dest[di].im += temp.im;
			}
		}
		mmax = istep;
	}

	if (fap->isi>=0){
		std_type fn;

		fn = (std_type) len;
		for(i=0;i<len;i++){
			dimension_t di;

			di = i * inc1;
			dest[di].re /= fn;
			dest[di].im /= fn;
		}
	}
}

void FFT_METHOD_NAME( cvfft )( FFT_Args *fap )
{
	/* BUG - we don't really need to copy the args!? */
	FFT_Args fa;

	fa.dst_addr = fap->dst_addr;
	fa.len = fap->len;
	fa.dst_inc = fap->dst_inc;
	fa.src_addr = fap->src_addr;	// added?  needed?
	fa.src_inc = fap->src_inc;	// added?  needed?
	fa.isi = FWD_FFT;

	_c_fft(&fa);
}

void FFT_METHOD_NAME( cvift )( FFT_Args *fap )
{
	FFT_Args fa;

	fa.dst_addr = fap->dst_addr;
	fa.len = fap->len;
	fa.dst_inc = fap->dst_inc;
	fa.isi = INV_FFT;

	_c_fft(&fa);
}

void OBJ_METHOD_NAME( cvfft )( Vec_Obj_Args *oap )
{
	FFT_Args fa;
	
	/* _c_fft computes in-place (the destination).
	 * Therefore, if the src and dest differ,
	 * we need to copy the source into the destination first.
	 */
	if( oap->oa_dest != oap->oa_1 )
		vmov(oap);

	fa.dst_addr = (std_cpx *)oap->oa_dest->dt_data;
	SET_FFT_INC(dst_inc,oap->oa_dest);

	/*fa.src_addr = oap->oa_1->dt_data; */
	/*SET_FFT_INC(src_inc,oap->oa_1); */

#ifdef FOOBAR
	/* Compute in-place */
	fa.src_addr = (std_cpx *)oap->oa_dest->dt_data;
	SET_FFT_INC(src_inc,oap->oa_dest);
#endif /* FOOBAR */

	fa.len = oap->oa_dest->dt_n_type_elts;	/* complex */
	fa.isi = (-1);

	FFT_METHOD_NAME(cvfft)( &fa );
}

void OBJ_METHOD_NAME( cvift )( Vec_Obj_Args *oap )
{
	FFT_Args fa;
	
	fa.dst_addr = (std_cpx *)oap->oa_dest->dt_data;
	fa.src_addr = (std_cpx *)oap->oa_1->dt_data;
	SET_FFT_INC(dst_inc,oap->oa_dest);
	SET_FFT_INC(src_inc,oap->oa_1);
	fa.len = oap->oa_dest->dt_n_type_elts;	/* complex */
	fa.isi = (1);

	FFT_METHOD_NAME(cvift)( &fa );
}


static void init_sinfact(dimension_t n)
{
	dimension_t i;
	std_type arginc, arg;

	last_real_len = n;
	n /= 2;

	if( _sinfact != (std_type *)NULL )
		givbuf(_sinfact);
	_sinfact = (std_type *)getbuf( n * sizeof(std_type) );

	if( _isinfact != (std_type *)NULL )
		givbuf(_isinfact);
	_isinfact = (std_type *)getbuf( n * sizeof(std_type) );

	arginc = 4 * atan(1.0) / n;
	arg = 0.0;

	for(i=1;i<n;i++){
		arg += arginc;
		_isinfact[i] = 2 * sin(arg);
		_sinfact[i] = 1.0 / _isinfact[i];
	}
}


/* this real fft is based on the method given in problem 19
 * (p. 56) of Elliott & Rao, "Fast Transforms"
 *
 * We insist that the source and dest are not the same object
 * (i.e., this transform cannot be done in-place).
 * We transform the real source into the half-length
 * complex desination, and then do a regular in-place
 * complex DFT on that.
 */

#define RECOMBINE							\
									\
	for(i=1;i<len/4;i++){						\
		std_type s1,s2,d1,d2;					\
									\
		ctop--;							\
		cbot++;							\
									\
		s1 = 0.5 * ( cbot->re + cbot->im );			\
		s2 = 0.5 * ( ctop->re + ctop->im );			\
		d1 = 0.5 * ( cbot->re - cbot->im );			\
		d2 = 0.5 * ( ctop->re - ctop->im );			\
									\
		cbot->re = s1 + d2;					\
		cbot->im = s1 - d2;					\
		ctop->re = s2 + d1;					\
		ctop->im = s2 - d1;					\
	}								\



/*
 * One-dimensional real fft.
 *
 * This routine seems to assume that the data are contiguous...
 * Increments are not used, checked...
 */

void FFT_METHOD_NAME( rvfft )( FFT_Args *fap)
{
	std_cpx *cbot, *ctop;
	std_type *top, *bottom;
	dimension_t i;
	double B0;
	dimension_t len;
	std_type *source;
	std_cpx *dest;
	FFT_Args fa;

	if( ! for_real ) return;

	/* len the length of the real data or complex data?... */
	len = fap->len;
	source = (std_type *)fap->src_addr;
	dest = (std_cpx *)fap->dst_addr;

	if( len != last_real_len ){
		init_sinfact(len);
	}

/* we assume that the destination has 1+len/2 complex entries */
/* transform the input while copying into dest */

	cbot = dest;
	bottom = source;
	top = source+len-1;

	/* after we do the first one,
	 * we don't have to worry about boundary conds
	 */

	cbot->re = *bottom;	/* lots of terms cancel */
	cbot->im = *top - *(bottom+1);
	B0 = *(bottom+1);

	for(i=1;i<len/2;i++){
		std_type p,q;

		cbot ++;
		bottom += 2;		/* BUG we might use increments here...? */
		top -= 2;

		p = *bottom + *(bottom+1) - *(bottom-1);
		q = *(top+1) + *top - *(top+2) ;
		B0 += *(bottom+1);

		cbot->re = 0.5 * ( p + q );	/* y1 even */
		cbot->im = 0.5 * ( q - p );	/* y2 odd */
	}

	fa.dst_addr = fap->dst_addr;
	fa.dst_inc = fap->dst_inc;
	fa.src_addr = NULL;
	fa.src_inc = 0;
	fa.len = fap->len/2;
	fa.isi = FWD_FFT;
	_c_fft(&fa);

	cbot = dest;
	cbot->im = B0;
	for(i=1;i<len/2;i++){
		cbot++;
		cbot->im *= _sinfact[i];
	}

	/* now make it look like the correct answer */

	cbot = dest;
	ctop = dest+len/2;

	ctop->re = cbot->re - cbot->im;
	cbot->re += cbot->im;
	ctop->im = cbot->im = 0.0;

	RECOMBINE
}

/* One dimensional real inverse fft.
 *
 * This routine seems to be destructive to its source!? ...
 */

void FFT_METHOD_NAME( rvift )( FFT_Args *fap)
{
std_cpx *src;
std_type *dest;
dimension_t len;
	std_cpx *cbot, *ctop;
	std_type *ep, *op;
	std_type diff;
	double B0,total;
	dimension_t i;
	FFT_Args fa;

	if( ! for_real ) return;

	src=(std_cpx *)fap->src_addr;
	dest=(std_type *)fap->dst_addr;
	len=fap->len;		/* length of the real destination */

	if( len != last_real_len ){
		init_sinfact(len);
	}

	/* first fiddle the transform back */

	cbot = src;
	ctop = src + len/2;

	/* the imaginary dc & nyquist are zero */

	cbot->im = cbot->re - ctop->re;
	cbot->re += ctop->re;
	cbot->re *= 0.5;
	cbot->im *= 0.5;

	/* RECOMBINE reassigns cbot, ctop... */
	RECOMBINE

	/* remember value of B0 */
	cbot = src;
	B0 = cbot->im;

	/* multiply B's by inverse sine factor */
	cbot->im = 0.0;
	for(i=1;i<len/2;i++){
		cbot++;
		cbot->im *= _isinfact[i];
	}

	fa.dst_addr = fap->src_addr;
	fa.dst_inc = fap->src_inc;
	fa.src_addr = NULL;
	fa.src_inc = 0;
	fa.len = fap->len/2;
	fa.isi = INV_FFT;

	_c_fft(&fa);

	/* now reconstruct the samples */

	cbot = src;
	ctop = src + len/2;
	ep = dest;
	op = dest+1;

	*ep = cbot->re;
	*op = - cbot->im;
	for(i=1;i<len/2;i++){
		std_type s1, d1;
		ep += 2;
		op += 2;
		cbot++;
		ctop--;
		s1 = ctop->re + ctop->im;	/* F(y1) + F(y2) */
		d1 = cbot->re - cbot->im;	/* y1 - y2 */
		*ep = 0.5 * ( s1 + d1 );
		*op = 0.5 * ( d1 - s1 );
	}

	/* now integrate the odd samples */

	/*	delta's		output
	 *
	 *	1		1
	 *	2		3
	 *	-1		2
	 */

	op = dest+1;
	total = *op;
	for(i=1;i<len/2;i++){
		*(op+2) += *op;
		op += 2;
		total += *op;
	}
	/* the total should equal B0 */
	diff = 2 * ( B0 - total ) / len;
	op = dest+1;
	for(i=0;i<len/2;i++){
		*op += diff;
		op += 2;
	}
	/* done */
}

void OBJ_METHOD_NAME( rvfft )( Vec_Obj_Args *oap )
{
	FFT_Args fa;

	fa.src_addr = (std_type *)oap->oa_1->dt_data;
	fa.dst_addr = (std_cpx *)oap->oa_dest->dt_data;

	SET_FFT_INC(src_inc,oap->oa_1);
	SET_FFT_INC(dst_inc,oap->oa_dest);

	fa.len = oap->oa_1->dt_n_type_elts;
	fa.isi = (-1);
	FFT_METHOD_NAME(rvfft)( &fa );
}

void OBJ_METHOD_NAME( rvift )( Vec_Obj_Args *oap )
{
	FFT_Args fa;

	fa.src_addr = (std_cpx *)oap->oa_1->dt_data;
	fa.dst_addr = (std_type *)oap->oa_dest->dt_data;

	SET_FFT_INC(src_inc,oap->oa_1);
	SET_FFT_INC(dst_inc,oap->oa_dest);

	fa.len = oap->oa_dest->dt_n_type_elts;
	fa.isi = 1;
	FFT_METHOD_NAME(rvfft)( &fa );
}

/* Read 2-D fourier transform.
 *
 * This version has been optimized for SMP execution.
 */


#define ROW_LOOP(real_dp,func,src_typ,dst_typ)				\
									\
	{								\
		for (i = 0; i < real_dp->dt_rows; ++i) {		\
			func(&fa);					\
			fa.src_addr = ((src_typ *)fa.src_addr)		\
					+ oap->oa_1->dt_rinc;		\
			fa.dst_addr = ((dst_typ *)fa.dst_addr)		\
					+ oap->oa_dest->dt_rinc;	\
		}							\
	}


#define MULTIPROCESSOR_ROW_LOOP(real_dp,func,src_typ,dst_typ)		\
	src_typ *_src_p[N_PROCESSORS];					\
	dst_typ *_dest_p[N_PROCESSORS];					\
	Vec_Obj_Args va[N_PROCESSORS];					\
	int n_passes,n_extra,i;						\
									\
	src_p = (src_typ *)src->dt_data;				\
	dest_p = (dst_typ *)dest->dt_data;				\
									\
	n_passes = src->dt_rows / n_processors ;			\
	n_extra = src->dt_rows - n_passes*n_processors;			\
									\
	for(i=0;i<n_processors;i++){					\
		va[i] = args;						\
		_src_p[i] = src_p + i * n_passes * src->dt_rinc;	\
		_dest_p[i] = dest_p + i * n_passes * dest->dt_rinc;	\
	}								\
									\
	if( real_dp->dt_cols/2 != bitrev_size )				\
		bitrev_init(real_dp->dt_cols/2);			\
	if( real_dp->dt_cols/2 != last_cpx_len )			\
		init_twiddle(real_dp->dt_cols/2);			\
	if( real_dp->dt_cols != last_real_len ){			\
		init_sinfact(real_dp->dt_cols);				\
	}								\
									\
	while( n_passes -- ){						\
		for(i=0;i<n_processors;i++){				\
			va[i].arg_v1 = _src_p[i];			\
			va[i].arg_v2 = _dest_p[i];			\
			_src_p[i] += src->dt_rinc;			\
			_dest_p[i] += dest->dt_rinc;			\
		}							\
									\
		launch_threads(QSP_ARG  func,va);			\
	}								\
	/* BUG for n_processors > 2 we would like to do these in parallel! */\
	while( n_extra -- ){						\
		va[0].arg_v1 = _src_p[n_processors-1];			\
		va[0].arg_v2 = _dest_p[n_processors-1];			\
		_src_p[n_processors-1] += src->dt_rinc;			\
		_dest_p[n_processors-1] += dest->dt_rinc;		\
		func(&va[0]);						\
	}



/* In a 2D fft, the column transforms are always complex,
 * regardless of whether the input is real or comples.
 * That is because we do the rows first, so only the row
 * transforms are real->complex.
 */


#define COLUMN_LOOP(dp,func)						\
									\
	{								\
		dimension_t i;						\
									\
		fa.src_addr = NULL;					\
		fa.dst_addr = dp->dt_data;				\
									\
		for(i=0;i<dp->dt_cols;i++){				\
			func(&fa);					\
			fa.dst_addr = ((std_cpx *)fa.dst_addr) + dp->dt_pinc;\
		}							\
	}


/* SMP versions of fft code.
 *
 * Originally, we had the parallel processors work on adjacent rows & cols,
 * but in this case we failed to see a linear speedup, in fact the
 * running time was about the same as a single cpu job (with double
 * the cpu time, although the number of calls was the same!?
 * My current theory is that system ram is locked in page units
 * to a particular thread (or perhaps physical cpu), so that when 
 * a given cpu is accessing a particular page of ram, the other
 * cpu will be put into a wait state if it also wants to access this
 * page.  Therefore, we have separated the memory accesses as much as possible,
 * and after doing this we see the memory speedup that we expect.
 *
 * Changed pinc/2 to pinc...
 */

#define MULTIPROCESSOR_COLUMN_LOOP(dp,func)				\
									\
	Vec_Obj_Args va[N_PROCESSORS];					\
	int n_passes,n_extra,i;						\
	std_cpx *_cp[N_PROCESSORS];					\
									\
	cp = (std_type *)dp->dt_data;					\
									\
	n_passes = dp->dt_cols / n_processors ;				\
	n_extra = dp->dt_cols - n_passes*n_processors;			\
									\
	for(i=0;i<n_processors;i++){					\
		va[i] = args;						\
		_cp[i] = cp + i * n_passes * dp->dt_pinc;		\
	}								\
									\
	/* cvfft calls _c_fft which calls bitrev_init() and init_twiddle()\
	 * which set up global arrays iff the requested			\
	 * fft length is different from the last length...		\
	 * This cannot be called in parallel, so we call it		\
	 * here to avoid any problems...				\
	 */								\
	if( dp->dt_rows != bitrev_size )				\
		bitrev_init(dp->dt_rows);				\
	if( dp->dt_rows != last_cpx_len )				\
		init_twiddle(dp->dt_rows);				\
									\
	while(n_passes--){						\
		for(i=0;i<n_processors;i++){				\
			va[i].arg_v1 = _cp[i];				\
			va[i].arg_v2 = _cp[i];				\
			_cp[i] += dp->dt_pinc/2;			\
		}							\
									\
		launch_threads(QSP_ARG  func,va);			\
	}								\
	/* BUG for n_processors > 2 we would like to do these in parallel! */\
	while( n_extra -- ){						\
		va[0].arg_v1 = _cp[n_processors-1];			\
		va[0].arg_v2 = _cp[n_processors-1];			\
		_cp[n_processors-1] += dp->dt_pinc/2;			\
		func(&va[0]);						\
	}

void OBJ_METHOD_NAME( r_rowfft )(Vec_Obj_Args *oap)
{
	FFT_Args fa;
	dimension_t i;

	if( real_row_fft_check(DEFAULT_QSP_ARG  oap->oa_1,oap->oa_dest,"r_rowfft") < 0 ) return;

	fa.isi = FWD_FFT;			/* not used, but play it safe */
	fa.src_addr = oap->oa_1->dt_data;
	fa.src_inc = oap->oa_1->dt_pinc;
	fa.dst_addr = oap->oa_dest->dt_data;
	fa.dst_inc = oap->oa_dest->dt_pinc;	// was /2
	fa.len = oap->oa_1->dt_cols ;

#if N_PROCESSORS >= MIN_PARALLEL_PROCESSORS
	if( n_processors > 1 ){
		MULTIPROCESSOR_ROW_LOOP(oap->oa_1,FFT_METHOD_NAME(rvfft),std_type,std_cpx)
	} else
#endif /* N_PROCESSORS > 1 */
		ROW_LOOP( oap->oa_1, FFT_METHOD_NAME(rvfft),std_type,std_cpx )

}

void OBJ_METHOD_NAME( r_2dfft )(Vec_Obj_Args *oap)
{
	FFT_Args fa;

	if( real_fft_check(DEFAULT_QSP_ARG  oap->oa_1,oap->oa_dest,"r_2dfft") < 0 ) return;

	fa.isi = FWD_FFT;			/* not used, but play it safe */
	fa.src_inc = oap->oa_1->dt_pinc;
	//fa.dst_inc = oap->oa_dest->dt_pinc/2;
	fa.dst_inc = oap->oa_dest->dt_pinc;

	if( oap->oa_1->dt_cols > 1 ){		/* more than 1 column ? */
		/* Transform the rows */
		dimension_t i;

		fa.src_addr = oap->oa_1->dt_data;
		fa.src_inc = oap->oa_1->dt_pinc;
		fa.dst_addr = oap->oa_dest->dt_data;
		fa.dst_inc = oap->oa_dest->dt_pinc;
		fa.len = oap->oa_1->dt_cols ;

#if N_PROCESSORS >= MIN_PARALLEL_PROCESSORS
		if( n_processors > 1 ){
			MULTIPROCESSOR_ROW_LOOP(oap->oa_1,FFT_METHOD_NAME(rvfft),std_type,std_cpx)
		} else
#endif /* N_PROCESSORS > 1 */
			ROW_LOOP( oap->oa_1,FFT_METHOD_NAME(rvfft),
				std_type,std_cpx)
	}

	/* Now transform the columns */
	/* BUG wrong if columns == 1 */
	/* Then we should copy into the complex target... */

	fa.len = oap->oa_1->dt_rows;
	//fa.dst_inc = oap->oa_dest->dt_rinc/2;
	fa.dst_inc = oap->oa_dest->dt_rinc;

	if( oap->oa_1->dt_rows > 1 ){			/* more than 1 row? */
		fa.len = oap->oa_1->dt_rows ;

#if N_PROCESSORS >= MIN_PARALLEL_PROCESSORS
		if( n_processors > 1 ){
			MULTIPROCESSOR_COLUMN_LOOP(oap->oa_dest, FFT_METHOD_NAME( cvfft ) )
		} else
#endif /* N_PROCESSORS > 1 */
		COLUMN_LOOP(oap->oa_dest,FFT_METHOD_NAME(cvfft))
	}
}

void OBJ_METHOD_NAME( r_2dift )( Vec_Obj_Args *oap )
{
	FFT_Args fa;

	if( real_fft_check(DEFAULT_QSP_ARG  oap->oa_dest,oap->oa_1,"r_2dift") < 0 ) return;

	fa.isi = 1;
	//fa.dst_inc = oap->oa_1->dt_rinc / 2 ;
	fa.dst_inc = oap->oa_1->dt_rinc;

	if( oap->oa_1->dt_rows > 1 ){			/* more than 1 row? */
		/* Transform the columns */
		fa.len = oap->oa_1->dt_rows;
#if N_PROCESSORS >= MIN_PARALLEL_PROCESSORS
		if( n_processors > 1 ){
			MULTIPROCESSOR_COLUMN_LOOP(oap->oa_1,FFT_METHOD_NAME(cvift))
		} else
#endif /* N_PROCESSORS > 1 */
		COLUMN_LOOP(oap->oa_1,FFT_METHOD_NAME(cvift))
	}

	if( oap->oa_1->dt_cols > 1 ){		/* more than 1 column ? */
		dimension_t i;

		fa.src_addr = oap->oa_1->dt_data;
		fa.src_inc = oap->oa_1->dt_pinc;
		fa.dst_addr = oap->oa_dest->dt_data;
		fa.dst_inc = oap->oa_dest->dt_pinc;
		fa.len = oap->oa_dest->dt_cols;	/* use the real len */

#if N_PROCESSORS >= MIN_PARALLEL_PROCESSORS
		if( n_processors > 1 ){
			MULTIPROCESSOR_ROW_LOOP(oap->oa_dest,FFT_METHOD_NAME(rvift),std_cpx,std_type)
		} else
#endif /* N_PROCESSORS > 1 */
			ROW_LOOP(oap->oa_dest,FFT_METHOD_NAME(rvift),
				std_cpx,std_type)
	}
}

void OBJ_METHOD_NAME( r_rowift )( Vec_Obj_Args *oap )
{
	FFT_Args fa;
	dimension_t i;

	if( real_row_fft_check(DEFAULT_QSP_ARG  oap->oa_dest,oap->oa_1,"r_rowift") < 0 ) return;

	fa.isi = 1;
	fa.src_addr = oap->oa_1->dt_data;
	fa.src_inc = oap->oa_1->dt_pinc;		// used to be /2
	fa.dst_addr = oap->oa_dest->dt_data;
	fa.dst_inc = oap->oa_dest->dt_pinc;
	fa.len = oap->oa_dest->dt_cols;

#if N_PROCESSORS >= MIN_PARALLEL_PROCESSORS
	if( n_processors > 1 ){
		MULTIPROCESSOR_ROW_LOOP(oap->oa_dest,FFT_METHOD_NAME(rvift),std_cpx,std_type)
		return;
	} else
#endif /* N_PROCESSORS > 1 */
		ROW_LOOP(oap->oa_dest,FFT_METHOD_NAME(rvift),std_cpx,std_type)
}


static int row_fft_ok(QSP_ARG_DECL  Data_Obj *dp)
{
	if( ! IS_COMPLEX(dp) ){
		sprintf(error_string,
			"Image %s is not complex for FFT",dp->dt_name);
		WARN(error_string);
		return(-1);
	}

	if( fft_row_size_ok(QSP_ARG  dp) < 0 )
		return(-1);

	return(0);
}

static int fft_ok(QSP_ARG_DECL  Data_Obj *dp)
{
	if( ! IS_COMPLEX(dp) ){
		sprintf(error_string,
			"Image %s is not complex for FFT",dp->dt_name);
		WARN(error_string);
		return(-1);
	}

	if( fft_size_ok(QSP_ARG  dp) < 0 )
		return(-1);

	return(0);
}

/*
 * Do an in-place complex FFT
 *
 * No SMP version (yet).
 */

void OBJ_METHOD_NAME( c_2dfft )( Vec_Obj_Args *oap, int is_inv )
{
	dimension_t i;
	FFT_Args fa;

	if( fft_ok(DEFAULT_QSP_ARG  oap->oa_1) < 0 ) return;

	/* transform the columns */

	fa.isi = is_inv;

	if( oap->oa_1->dt_rows > 1 ){	/* more than one row */
		fa.dst_addr = (std_type *)oap->oa_1->dt_data;
		fa.len = oap->oa_1->dt_rows;
		//fa.dst_inc = oap->oa_1->dt_rinc/2;
		fa.dst_inc = oap->oa_1->dt_rinc;

		for (i = 0; i < oap->oa_1->dt_cols; ++i) {
			_c_fft(&fa);
			/* ((std_type *)fa.dst_addr) += oap->oa_1->dt_pinc; */
			fa.dst_addr = ((std_cpx *)fa.dst_addr) + oap->oa_1->dt_pinc;
		}
	}

	/* transform the rows */

	if( oap->oa_1->dt_cols > 1 ){
		fa.dst_addr = (std_type *)oap->oa_1->dt_data;
		fa.len = oap->oa_1->dt_cols;
		/* pixel inc used to be in machine units,
		 * now it's in type units!? */
		//fa.dst_inc = oap->oa_1->dt_pinc/2;
		fa.dst_inc = oap->oa_1->dt_pinc;

		for (i = 0; i < oap->oa_1->dt_rows; ++i) {
			_c_fft(&fa);
			/* ((std_type *)fa.dst_addr) += oap->oa_1->dt_rinc; */
			fa.dst_addr = ((std_cpx *)fa.dst_addr) + oap->oa_1->dt_rinc;
		}
	}
}

void OBJ_METHOD_NAME( c_rowfft )( Vec_Obj_Args *oap, int is_inv )
{
	dimension_t i;
	FFT_Args fa;

	if( row_fft_ok(DEFAULT_QSP_ARG  oap->oa_1) < 0 ) return;

	fa.isi = is_inv;

	/* transform the rows */

	if( oap->oa_1->dt_cols > 1 ){
		fa.dst_addr = (std_type *)oap->oa_1->dt_data;
		fa.len = oap->oa_1->dt_cols;
		// What is pinc??? type units not machine units?
		//fa.dst_inc = oap->oa_1->dt_pinc/2;
		fa.dst_inc = oap->oa_1->dt_pinc;

		for (i = 0; i < oap->oa_1->dt_rows; ++i) {
			_c_fft(&fa);
			/* ((std_type *)fa.dst_addr) += oap->oa_1->dt_rinc; */
			fa.dst_addr = ((std_type *)fa.dst_addr) + oap->oa_1->dt_rinc;
		}
	}
}

