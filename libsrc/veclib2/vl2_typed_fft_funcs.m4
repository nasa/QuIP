// vl2_fft_funcs.m4 BEGIN
/* included file, so no version string */

#ifdef HAVE_MATH_H
#include <math.h>
#endif

extern int n_processors;

/* this file is included in sptest.c and dptest.c */

// BUG need to set this from config!?
define(`N_PROCESSORS',`2')
define(`MIN_PARALLEL_PROCESSORS',`100')	// a nonsense value to inhibit the switches
dnl 					// set to 2 to use these features on a dual proc machine.
dnl define(`MIN_PARALLEL_PROCESSORS',`2')	dnl for testing

define(`MULTI_PROC_TEST',`eval(N_PROCESSORS >= MIN_PARALLEL_PROCESSORS)')

// BUG - static globals are not thread-safe!?


define(`revdone',`TYPED_NAME(_revdone)')
dnl define(`init_twiddle',`TYPED_NAME(_init_twiddle)')
define(`init_twiddle',`TYPED_NAME(`init_twiddle_')')
define(`last_cpx_len',`TYPED_NAME(_last_cpx_len)')
define(`twiddle',`TYPED_NAME(_twiddle)')
define(`last_real_len',`TYPED_NAME(_last_real_len)')
define(`_isinfact',`TYPED_NAME(__isinfact)')
define(`_sinfact',`TYPED_NAME(__sinfact)')
define(`max_fft_len',`TYPED_NAME(_max_fft_len)')
define(`init_sinfact',`TYPED_NAME(_init_sinfact)')


define(`MAX_FFT_LEN',`4096L')

// vl2_fft_funcs.m4 buiding_kernels is set

// How can we have these static vars when this file is included twice!?

/* static fft vars NOT already declared */
define(`DECLARE_STATIC_FFT_VARS',`
/* declare_static_fft_vars DOING IT */
static dimension_t last_cpx_len=0;
static std_cpx *twiddle;

static dimension_t last_real_len=0;
static std_type *_isinfact=NULL;
static std_type *_sinfact=NULL;

static char *revdone=NULL;
static u_int max_fft_len=(-1);
')

dnl	Instead of ifdef, better to have separate file for kernels and host funcs?

ifdef(`BUILDING_KERNELS',`
/* vl2_fft_funcs.m4 declaring static vars */
DECLARE_STATIC_FFT_VARS
',`') dnl endif BUILDING_KERNELS

dnl	XFER_FFT_SINC( func, fap, dp )
define(`XFER_FFT_SINC',`

	if( IS_ROWVEC($3) ){
		SET_FFT_SINC($2, OBJ_PXL_INC( ($3) ) );
	} else if( IS_COLVEC($3) ){
		SET_FFT_SINC($2, OBJ_ROW_INC( ($3) ) );
	} else {
		NWARN(DEFAULT_ERROR_STRING);
		sprintf(DEFAULT_ERROR_STRING,
	"%s:  %s is neither a row nor a column!?","$1",OBJ_NAME($3));
		return;
	}
dnl	This code was probably a relic from when the increment was from mach_inc, not type_inc?
dnl	Doesnt seem to be needed now.
dnl	if( IS_COMPLEX($3) ){
dnl		SET_FFT_SINC($2, FFT_SINC($2)/2);
dnl fprintf(stderr,"xfer_fft_sinc:  complex src inc = %d\\n",FFT_SINC($2));
dnl	}
')


dnl	XFER_FFT_DINC( func, fap, dp )
define(`XFER_FFT_DINC',`

	if( IS_ROWVEC($3) ){
		SET_FFT_DINC($2, OBJ_PXL_INC( ($3) ) );
	} else if( IS_COLVEC($3) ){
		SET_FFT_DINC($2, OBJ_ROW_INC( ($3) ) );
	} else {
		NWARN(DEFAULT_ERROR_STRING);
		sprintf(DEFAULT_ERROR_STRING,
	"%s:  %s is neither a row nor a column!?","$1",OBJ_NAME($3));
		return;
	}
	dnl /* if( IS_COMPLEX($3) )	SET_FFT_DINC($2, FFT_DINC($2)/2); */
')


dnl	#ifdef NOT_USED
dnl	#define SET_FFT_INC( func, which_inc, dp )
dnl
dnl		if( IS_ROWVEC(dp) ){
dnl			fa.which_inc = OBJ_PXL_INC( (dp) );
dnl		} else if( IS_COLVEC(dp) ){
dnl			fa.which_inc = OBJ_ROW_INC( (dp) );
dnl		} else {
dnl			NWARN(DEFAULT_ERROR_STRING);
dnl			sprintf(DEFAULT_ERROR_STRING,
dnl		"%s:  %s is neither a row nor a column!?",#func,OBJ_NAME(dp));
dnl			return;
dnl		}
dnl		if( IS_COMPLEX(dp) )
dnl			fa.which_inc /= 2;
dnl	#endif // NOT_USED

ifdef(`BUILDING_KERNELS',`
// vl2_fft_funcs.m4 buiding_kernels is SET

static void init_twiddle (dimension_t len)
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
		twiddle[i].re = (std_type)cos(theta);
		twiddle[i].im = (std_type)sin(theta);
	}
	last_cpx_len=len;
}

/* This is usually called in-place, but doesnt have to be.
 *
 * isi = -1   ->  forward xform
 * isi =  1   ->  inverse xform
 */


#ifdef ONLY_FOR_DEBUG
static void show_fa(FFT_Args *fap)
{
	sprintf(DEFAULT_ERROR_STRING,"dst_addr = 0x%lx, inc = %ld",
		(int_for_addr)FFT_DST(fap),(long)FFT_DINC(fap));
	NADVISE(DEFAULT_ERROR_STRING);
	sprintf(DEFAULT_ERROR_STRING,"src_addr = 0x%lx, inc = %ld",
		(int_for_addr)FFT_SRC(fap),(long)FFT_SINC(fap));
	NADVISE(DEFAULT_ERROR_STRING);
	sprintf(DEFAULT_ERROR_STRING,"len = %ld, isi = %d",
		(long)FFT_LEN(fap),FFT_ISI(fap));
	NADVISE(DEFAULT_ERROR_STRING);
}
#endif // ONLY_FOR_DEBUG

/* cvfft takes separate dest and source args, but works for in-place(?)
 */

static void PF_FFT_CALL_NAME(cvfft)(FFT_Args *fap)
{
	dimension_t i,j;
	dimension_t len;
	std_cpx temp,*wp;
	std_cpx *source, *dest;
	dimension_t m, mmax, istep;
	incr_t inc1;
	/* BUG we really don_t want to allocate and deallocate revdone each time... */
	/* anyway, this is no good because getbuf/givbuf are not thread-safe!
	 * I can_t see a way to do this without passing the thread index on the stack...
	 * OR having the entire revdone array on the stack?
	 */
	/* char revdone[MAX_FFT_LEN]; */

	//if( ! for_real ) return;

dnl	fprintf(stderr,"PF_FFT_CALL_NAME(cvfft) BEGIN\\n");
	len = FFT_LEN(fap);

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
ifelse(MULTI_PROC_TEST,`1',`
		if( n_processors > 1 ) NWARN("cvfft:  init_twiddle is not thread-safe!?");
')
		init_twiddle (len);
	}

	dest=(std_cpx *)FFT_DST(fap);
	source=(std_cpx *)FFT_DST(fap);
	inc1 = FFT_DINC(fap);
	/* inc1 should be in units of complex */

	if( len != bitrev_size ){
ifelse(MULTI_PROC_TEST,`1',`
		if( n_processors > 1 ) NWARN("cvfft:  bitrev_init is not thread-safe!?");
')
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
					- FFT_ISI(fap) * wp->im*dest[dj].im;
				temp.im = wp->re*dest[dj].im
					+ FFT_ISI(fap) * wp->im*dest[dj].re;
				dest[dj].re = dest[di].re-temp.re;
				dest[dj].im = dest[di].im-temp.im;
				dest[di].re += temp.re;
				dest[di].im += temp.im;
			}
		}
		mmax = istep;
	}

dnl	This block does the scaling, but this is not done by the fftw or cuFFT,
dnl	so maybe it is best to be compatible and skip it...

dnl	if (FFT_ISI(fap)>=0){
dnl		std_type fn;
dnl
dnl		fn = (std_type) len;
dnl		for(i=0;i<len;i++){
dnl			dimension_t di;
dnl
dnl			di = i * inc1;
dnl			dest[di].re /= fn;
dnl			dest[di].im /= fn;
dnl		}
dnl	}

}

static void PF_FFT_CALL_NAME(cvift)( FFT_Args *fap )
{
	FFT_Args fa;
	FFT_Args *new_fap=(&fa);

dnl	fprintf(stderr,"PF_FFT_CALL_NAME(cvift) BEGIN\\n");
	SET_FFT_DST(new_fap, FFT_DST(fap));
	SET_FFT_LEN(new_fap, FFT_LEN(fap));
	SET_FFT_DINC(new_fap, FFT_DINC(fap));
	SET_FFT_ISI(new_fap, INV_FFT);

	PF_FFT_CALL_NAME(cvfft)(new_fap);
}

static void init_sinfact (dimension_t n)
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

	arginc = (std_type)(4 * atan(1.0) / n);
	arg = 0.0;

	for(i=1;i<n;i++){
		arg += arginc;
		_isinfact[i] = 2 * (std_type)sin(arg);
		_sinfact[i] = 1.0f / _isinfact[i];
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

dnl	RECOMBINE(inc)
define(`RECOMBINE',`

	for(i=1;i<len/4;i++){
		std_type s1,s2,d1,d2;

dnl		/*ctop--;
dnl		cbot++;*/
		ctop -= $1;
		cbot += $1;

		s1 = 0.5f * ( cbot->re + cbot->im );
		s2 = 0.5f * ( ctop->re + ctop->im );
		d1 = 0.5f * ( cbot->re - cbot->im );
		d2 = 0.5f * ( ctop->re - ctop->im );

		cbot->re = s1 + d2;
		cbot->im = s1 - d2;
		ctop->re = s2 + d1;
		ctop->im = s2 - d1;
	}
')



/*
 * One-dimensional real fft.
 *
 * This routine seems to assume that the data are contiguous...
 * Increments are not used, checked...
 */

static void PF_FFT_CALL_NAME(rvfft)( const FFT_Args *fap)
{
	std_cpx *cbot, *ctop;
	std_type *top, *bottom;
	dimension_t i;
	double B0;
	dimension_t len;
	std_type *source;
	std_cpx *dest;
	FFT_Args fa;
	FFT_Args *_fap=(&fa);
	incr_t src_inc;
	incr_t dst_inc;

	//if( ! for_real ) return;

	/* len the length of the real data or complex data?... */
	len = FFT_LEN(fap);
	source = (std_type *)FFT_SRC(fap);
	dest = (std_cpx *)FFT_DST(fap);
	src_inc = FFT_SINC(fap);
	dst_inc = FFT_DINC(fap);

	if( len != last_real_len ){
		init_sinfact (len);
	}

/* we assume that the destination has 1+len/2 complex entries */
/* transform the input while copying into dest */

	/* OLD COMMENT:  BUG - we are fixing this to work with source increments
	 * different from 1, but we will defer non-1 destination
	 * increments...
	 *
	 * NEW COMMENT:  In order to perform real transforms of columns, we
	 * now have to implement destination increments...
	 */

	cbot = dest;
	bottom = source;
	top = source+src_inc*(len-1);

	/* after we do the first one,
	 * we don_t have to worry about boundary conds
	 */

	cbot->re = *bottom;	/* lots of terms cancel */
	cbot->im = *top - *(bottom+src_inc);
	B0 = *(bottom+src_inc);

	for(i=1;i<len/2;i++){
		std_type p,q;

		//cbot ++; /* BUG we need to use increment here!!! */
		cbot += dst_inc;
		bottom += 2*src_inc;
		top -= 2*src_inc;

		p = *bottom + *(bottom+src_inc) - *(bottom-src_inc);
		q = *(top+src_inc) + *top - *(top+2*src_inc) ;
		B0 += *(bottom+src_inc);

		cbot->re = 0.5f * ( p + q );	/* y1 even */
		cbot->im = 0.5f * ( q - p );	/* y2 odd */
	}

	// Why are we copying the args?
	// Because we can_t modify the input arg struct...
	SET_FFT_DST(_fap, FFT_DST(fap) );
	SET_FFT_DINC(_fap, FFT_DINC(fap) );
	SET_FFT_SRC(_fap, NULL);
	SET_FFT_SINC(_fap, 0);
	SET_FFT_LEN(_fap, FFT_LEN(fap)/2 );
	SET_FFT_ISI(_fap, FWD_FFT);
	PF_FFT_CALL_NAME(cvfft)(_fap);

	cbot = dest;
	cbot->im = (std_type) B0;
	for(i=1;i<len/2;i++){
		cbot+=dst_inc;
		cbot->im *= _sinfact[i];
	}

	/* now make it look like the correct answer */

	cbot = dest;
	ctop = dest+dst_inc*(len/2);

	ctop->re = cbot->re - cbot->im;
	cbot->re += cbot->im;
	ctop->im = cbot->im = 0.0;

	RECOMBINE(dst_inc)
}

/* One dimensional real inverse fft.
 *
 * This routine seems to be destructive to its source!? ...
 * Yes, because the first complex FFT is done in-place
 */

static void PF_FFT_CALL_NAME(rvift)( FFT_Args *fap)
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
	FFT_Args *_fap=(&fa);
	incr_t dst_inc;
	incr_t src_inc;

	//if( ! for_real ) return;

	src=(std_cpx *)FFT_SRC(fap);
	dest=(std_type *)FFT_DST(fap);
	dst_inc = FFT_DINC(fap);
	src_inc = FFT_SINC(fap);
	len=FFT_LEN(fap);		/* length of the real destination */
/*fprintf(stderr,"dest = 0x%lx inc = %d\\nsrc = 0x%lx inc = %d\nlen = %d\n",
(long)dest,dst_inc,(long)src,src_inc,len);*/

	if( len != last_real_len ){
		init_sinfact (len);
	}

	/* first fiddle the transform back */

	cbot = src;
	ctop = src + src_inc*(len/2);

	/* the imaginary dc & nyquist are zero */

	cbot->im = cbot->re - ctop->re;
	cbot->re += ctop->re;
	cbot->re *= 0.5;
	cbot->im *= 0.5;

	/* RECOMBINE reassigns cbot, ctop... */
	RECOMBINE(src_inc)

	/* remember value of B0 */
	cbot = src;
	B0 = cbot->im;

	/* multiply Bs by inverse sine factor */
	cbot->im = 0.0;
	for(i=1;i<len/2;i++){
		cbot+=src_inc;
		cbot->im *= _isinfact[i];
	}

	SET_FFT_DST( _fap, FFT_SRC(fap) );
	SET_FFT_DINC( _fap, FFT_SINC(fap) );
	SET_FFT_SRC( _fap, NULL );
	SET_FFT_SINC( _fap, 0 );
	SET_FFT_LEN( _fap, FFT_LEN(fap)/2 );
	SET_FFT_ISI( _fap, INV_FFT );

	PF_FFT_CALL_NAME(cvfft)(&fa);

	/* now reconstruct the samples */
	/* BUG we fix destination increment, but assume src inc is 1!? */

	cbot = src;
	ctop = src + src_inc*(len/2);
	ep = dest;
	op = dest+dst_inc;

	*ep = cbot->re;
	*op = - cbot->im;
	for(i=1;i<len/2;i++){
		std_type s1, d1;
		ep += 2*dst_inc;
		op += 2*dst_inc;
		cbot+=src_inc;
		ctop-=src_inc;
		s1 = ctop->re + ctop->im;	/* F(y1) + F(y2) */
		d1 = cbot->re - cbot->im;	/* y1 - y2 */
		*ep = 0.5f * ( s1 + d1 );
		*op = 0.5f * ( d1 - s1 );
	}

	/* now integrate the odd samples */

	/*	deltas		output
	 *
	 *	1		1
	 *	2		3
	 *	-1		2
	 */

	op = dest+dst_inc;
	total = *op;
	for(i=1;i<len/2;i++){
		*(op+2*dst_inc) += *op;
		op += 2*dst_inc;
		total += *op;
	}
	/* the total should equal B0 */
	diff = (std_type)(2 * ( B0 - total ) / len);
	op = dest+dst_inc;
	for(i=0;i<len/2;i++){
		*op += diff;
		op += 2*dst_inc;
	}
	/* done */
}

',` dnl else ! BUILDING_KERNELS

// vl2_fft_funcs.m4 buiding_kernels is NOT SET

static void HOST_TYPED_CALL_NAME_CPX(vfft,type_code)( HOST_CALL_ARG_DECLS )
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);

dnl	fprintf(stderr,"HOST_TYPED_CALL_NAME_CPX(vfft,type_code) BEGIN\\n");
	SET_FFT_DST(fap, (std_cpx *)OBJ_DATA_PTR( OA_DEST(oap) ) );
	SET_FFT_SRC(fap, (std_cpx *)OBJ_DATA_PTR( OA_SRC1(oap) ) );
	XFER_FFT_DINC(cvfft,fap,OA_DEST(oap))
	XFER_FFT_SINC(cvfft,fap,OA_SRC1(oap))
	SET_FFT_LEN(fap, OBJ_N_TYPE_ELTS( OA_DEST(oap) ) );	/* complex */
	SET_FFT_ISI(fap,(-1) );

	PF_FFT_CALL_NAME(cvfft)( &fa );
}

static void HOST_TYPED_CALL_NAME_CPX(vift,type_code)( HOST_CALL_ARG_DECLS )
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);

dnl	fprintf(stderr,"HOST_TYPED_CALL_NAME_CPX(vift,type_code) BEGIN\\n");
	SET_FFT_DST(fap, (std_cpx *)OBJ_DATA_PTR( OA_DEST(oap) ) );
	SET_FFT_SRC(fap, (std_cpx *)OBJ_DATA_PTR( OA_SRC1(oap) ) );
	XFER_FFT_DINC(cvift,fap,OA_DEST(oap))
	XFER_FFT_SINC(cvift,fap,OA_SRC1(oap))
	SET_FFT_LEN(fap, OBJ_N_TYPE_ELTS( OA_DEST(oap) ) );	/* complex */
	SET_FFT_ISI(fap, (1) );

	PF_FFT_CALL_NAME(cvift)( &fa );
}


static void HOST_TYPED_CALL_NAME_REAL(vfft,type_code)( HOST_CALL_ARG_DECLS )
{
	FFT_Args fa;
	FFT_Args *fap;

dnl	fprintf(stderr,"HOST_TYPED_CALL_NAME_REAL(vfft,type_code) BEGIN\\n");
	fap = (&fa);

	SET_FFT_SRC( fap, (std_type *)OBJ_DATA_PTR( OA_SRC1(oap) ) );
	SET_FFT_DST( fap, (std_cpx *)OBJ_DATA_PTR( OA_DEST(oap) ) );

	XFER_FFT_SINC(rvfft,fap,OA_SRC1(oap))
	XFER_FFT_DINC(rvfft,fap,OA_DEST(oap))

	SET_FFT_LEN( fap, OBJ_N_TYPE_ELTS( OA_SRC1(oap) ) );
	SET_FFT_ISI( fap, (-1) );
	PF_FFT_CALL_NAME(rvfft)( &fa );
}

static void HOST_TYPED_CALL_NAME_REAL(vift,type_code)( HOST_CALL_ARG_DECLS )
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);

dnl	fprintf(stderr,"HOST_TYPED_CALL_NAME_REAL(vift,type_code) BEGIN\\n");
	SET_FFT_SRC( fap, (std_cpx *)OBJ_DATA_PTR( OA_SRC1(oap) ) );
	SET_FFT_DST( fap, (std_type *)OBJ_DATA_PTR( OA_DEST(oap) ) );

	XFER_FFT_SINC(rvift,fap,OA_SRC1(oap))
	XFER_FFT_DINC(rvift,fap,OA_DEST(oap))

	SET_FFT_LEN( fap, OBJ_N_TYPE_ELTS( OA_DEST(oap) ) );
	SET_FFT_ISI( fap, 1 );
	//PF_FFT_CALL_NAME(rvfft)( &fa );
	PF_FFT_CALL_NAME(rvift)( &fa );
}

/* Read 2-D fourier transform.
 *
 * This version has been optimized for SMP execution.
 */


dnl	ROW_LOOP(real_dp,func,src_typ,dst_typ)
define(`ROW_LOOP',`

	{
		for (i = 0; i < OBJ_ROWS( $1 ); ++i) {
			$2(fap);
			SET_FFT_SRC( fap, (($3 *)FFT_SRC(fap))
					+ OBJ_ROW_INC( OA_SRC1(oap) ) );
			SET_FFT_DST( fap, (($4 *)FFT_DST(fap))
					+ OBJ_ROW_INC( OA_DEST(oap) ) );
		}
	}
')


dnl	MULTIPROCESSOR_ROW_LOOP(real_dp,func,src_typ,dst_typ)
define(`MULTIPROCESSOR_ROW_LOOP',`
	$3 *_src_p[N_PROCESSORS];
	$4 *_dest_p[N_PROCESSORS];
	$3 *src_p;
	$4 *dest_p;
	Vec_Obj_Args va[N_PROCESSORS];
	int n_passes,n_extra,i;

	src_p = ($3 *)OBJ_DATA_PTR( src );
	dest_p = ($4 *)OBJ_DATA_PTR( dest );

	n_passes = OBJ_ROWS( src ) / n_processors ;
	n_extra = OBJ_ROWS( src ) - n_passes*n_processors;

	for(i=0;i<n_processors;i++){
		va[i] = args;
		_src_p[i] = src_p + i * n_passes * OBJ_ROW_INC( src );
		_dest_p[i] = dest_p + i * n_passes * OBJ_ROW_INC( dest );
	}

	if( OBJ_COLS( $1 )/2 != bitrev_size )
		bitrev_init(OBJ_COLS( $1 )/2);
	if( OBJ_COLS( $1 )/2 != last_cpx_len )
		init_twiddle (OBJ_COLS( $1 )/2);
	if( OBJ_COLS( $1 ) != last_real_len ){
		init_sinfact (OBJ_COLS( $1 ));
	}

	while( n_passes -- ){
		for(i=0;i<n_processors;i++){
			va[i].arg_v1 = _src_p[i];
			va[i].arg_v2 = _dest_p[i];
			_src_p[i] += OBJ_ROW_INC( src );
			_dest_p[i] += OBJ_ROW_INC( dest );
		}

		launch_threads(QSP_ARG  $2,va);
	}
	/* BUG for n_processors > 2 we would like to do these in parallel! */
	while( n_extra -- ){
		va[0].arg_v1 = _src_p[n_processors-1];
		va[0].arg_v2 = _dest_p[n_processors-1];
		_src_p[n_processors-1] += OBJ_ROW_INC( src );
		_dest_p[n_processors-1] += OBJ_ROW_INC( dest );
		$2(&va[0]);
	}
')



/* OLD COMMENT:
 * In a 2D fft, the column transforms are always complex,
 * regardless of whether the input is real or comples.
 * That is because we do the rows first, so only the row
 * transforms are real->complex.
 *
 * NEW COMMENT:
 * The preceding was true before we tried for compatibility
 * with clFFT.  That package appears to transform the columns first
 * in a 2D real FFT.  So we try the same.  But we can_t use exactly
 * the same column loop, see COL_LOOP_2 below.
 */


dnl	COLUMN_LOOP(dp,func)
define(`COLUMN_LOOP',`

	{
		dimension_t i;

		SET_FFT_SRC( fap, NULL );
		SET_FFT_DST( fap, OBJ_DATA_PTR( $1 ) );

		for(i=0;i<OBJ_COLS( $1 );i++){
			$2(fap);
			SET_FFT_DST( fap, ((std_cpx *)FFT_DST(fap)) + OBJ_PXL_INC( $1 ) );
		}
	}
')

dnl	ROW_LOOP_2(dp,func)
define(`ROW_LOOP_2',`

	{
		dimension_t i;

		SET_FFT_SRC( fap, NULL );

		for(i=0;i<OBJ_ROWS( $1 );i++){
			SET_FFT_DST( fap, ((std_cpx *)OBJ_DATA_PTR($1))
						+ i*OBJ_ROW_INC( $1 ) );
			$2(fap);
		}
	}
')

dnl	COL_LOOP_2(real_dp,func,src_typ,dst_typ)
define(`COL_LOOP_2',`

	{
		for (i = 0; i < OBJ_COLS( $1 ); ++i) {
			SET_FFT_SRC( fap, (($3 *)OBJ_DATA_PTR(OA_SRC1(oap)))
						+ i * OBJ_PXL_INC(OA_SRC1(oap)) );
			SET_FFT_DST( fap, (($4 *)OBJ_DATA_PTR(OA_DEST(oap)))
						+ i * OBJ_PXL_INC(OA_DEST(oap)) );
			$2(fap);
		}
	}
')



/* SMP versions of fft code.
 *
 * Originally, we had the parallel processors work on adjacent rows & cols,
 * but in this case we failed to see a linear speedup, in fact the
 * running time was about the same as a single cpu job - with double
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

dnl	MULTIPROCESSOR_COLUMN_LOOP(dp,func)
define(`MULTIPROCESSOR_COLUMN_LOOP',`

	Vec_Obj_Args va[N_PROCESSORS];
	int n_passes,n_extra,i;
	std_cpx *_cp[N_PROCESSORS];

	cp = (std_type *)OBJ_DATA_PTR( $1 );

	n_passes = OBJ_COLS( $1 ) / n_processors ;
	n_extra = OBJ_COLS( $1 ) - n_passes*n_processors;

	for(i=0;i<n_processors;i++){
		va[i] = args;
		_cp[i] = cp + i * n_passes * OBJ_PXL_INC( $1 );
	}

	/* cvfft calls cvfft which calls bitrev_init() and init_twiddle ()
	 * which set up global arrays iff the requested
	 * fft length is different from the last length...
	 * This cannot be called in parallel, so we call it
	 * here to avoid any problems...
	 */
	if( OBJ_ROWS( $1 ) != bitrev_size )
		bitrev_init(OBJ_ROWS( $1 ));
	if( OBJ_ROWS( $1 ) != last_cpx_len )
		init_twiddle (OBJ_ROWS( $1 ));

	while(n_passes--){
		for(i=0;i<n_processors;i++){
			va[i].arg_v1 = _cp[i];
			va[i].arg_v2 = _cp[i];
			_cp[i] += OBJ_PXL_INC( $1 )/2;
		}

		launch_threads(QSP_ARG  $2,va);
	}
	/* BUG for n_processors > 2 we would like to do these in parallel! */
	while( n_extra -- ){
		va[0].arg_v1 = _cp[n_processors-1];
		va[0].arg_v2 = _cp[n_processors-1];
		_cp[n_processors-1] += OBJ_PXL_INC( $1 )/2;
		$2(&va[0]);
	}
')

static void HOST_TYPED_CALL_NAME_REAL(fftrows,type_code)(HOST_CALL_ARG_DECLS)
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);
	dimension_t i;

	if( ! real_row_fft_ok(DEFAULT_QSP_ARG  OA_SRC1(oap),OA_DEST(oap),"rfftrows") ) return;

	SET_FFT_ISI( fap, FWD_FFT );			/* not used, but play it safe */
	SET_FFT_SRC( fap, OBJ_DATA_PTR( OA_SRC1(oap) ) );
	SET_FFT_SINC( fap, OBJ_PXL_INC( OA_SRC1(oap) ) );
	SET_FFT_DST( fap, OBJ_DATA_PTR( OA_DEST(oap) ) );
	SET_FFT_DINC( fap, OBJ_PXL_INC( OA_DEST(oap) ) );	// was /2
	SET_FFT_LEN( fap, OBJ_COLS( OA_SRC1(oap) ) );

ifelse(MULTI_PROC_TEST,`1',` dnl #if N_PROCESSORS >= MIN_PARALLEL_PROCESSORS
	if( n_processors > 1 ){
		MULTIPROCESSOR_ROW_LOOP(OA_SRC1(oap),PF_FFT_CALL_NAME(rvfft),std_type,std_cpx)
	} else
') dnl endif /* N_PROCESSORS > 1 */
		ROW_LOOP( OA_SRC1(oap), PF_FFT_CALL_NAME(rvfft),std_type,std_cpx )

}

// JBMs original implementation of the real transform for 2D images performs real transforms
// of the rows first, then complex transforms of the columns, resulting in a transform with 1+N/2 cols
// and N rows (for a square NxN image).  This seems to be compatible with cuFFT.  However, clFFT
// appears to do the opposite!?  Therefore, it would be a kindness to provide a column-first version...

static void HOST_TYPED_CALL_NAME_REAL(fft2d_1,type_code)(HOST_CALL_ARG_DECLS)
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);

	SET_FFT_ISI( fap, FWD_FFT );			/* not used, but play it safe */
	SET_FFT_SINC( fap, OBJ_PXL_INC( OA_SRC1(oap) ) );
dnl	//SET_FFT_DINC( fap, OBJ_PXL_INC( OA_DEST(oap) )/2 );
	SET_FFT_DINC( fap, OBJ_PXL_INC( OA_DEST(oap) ) );

	if( OBJ_COLS( OA_SRC1(oap) ) > 1 ){		/* more than 1 column ? */
		/* Transform the rows */
		dimension_t i;

		SET_FFT_SRC( fap, OBJ_DATA_PTR( OA_SRC1(oap) ) );
		SET_FFT_SINC( fap, OBJ_PXL_INC( OA_SRC1(oap) ) );
		SET_FFT_DST( fap, OBJ_DATA_PTR( OA_DEST(oap) ) );
		SET_FFT_DINC( fap, OBJ_PXL_INC( OA_DEST(oap) ) );
		SET_FFT_LEN( fap, OBJ_COLS( OA_SRC1(oap) ) );

ifelse(MULTI_PROC_TEST,`1',` dnl #if N_PROCESSORS >= MIN_PARALLEL_PROCESSORS
		if( n_processors > 1 ){
			MULTIPROCESSOR_ROW_LOOP(OA_SRC1(oap),PF_FFT_CALL_NAME(rvfft),std_type,std_cpx)
		} else
') dnl endif /* N_PROCESSORS > 1 */
			ROW_LOOP( OA_SRC1(oap),PF_FFT_CALL_NAME(rvfft),
				std_type,std_cpx)
	}

	/* Now transform the columns */
	/* BUG wrong if columns == 1 */
	/* Then we should copy into the complex target... */

	SET_FFT_LEN( fap, OBJ_ROWS( OA_SRC1(oap) ) );
	//SET_FFT_DINC( fap, OBJ_ROW_INC( OA_DEST(oap) )/2 );
	SET_FFT_DINC( fap, OBJ_ROW_INC( OA_DEST(oap) ) );

	if( OBJ_ROWS( OA_SRC1(oap) ) > 1 ){			/* more than 1 row? */
		SET_FFT_LEN( fap, OBJ_ROWS( OA_SRC1(oap) ) );

ifelse(MULTI_PROC_TEST,`1',` dnl #if N_PROCESSORS >= MIN_PARALLEL_PROCESSORS
		if( n_processors > 1 ){
			MULTIPROCESSOR_COLUMN_LOOP(OA_DEST(oap), PF_FFT_CALL_NAME(cvfft) )
		} else
') dnl endif /* N_PROCESSORS > 1 */
		COLUMN_LOOP(OA_DEST(oap),PF_FFT_CALL_NAME(cvfft))
	}
}

static void HOST_TYPED_CALL_NAME_REAL(fft2d_2,type_code)(HOST_CALL_ARG_DECLS)
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);

	SET_FFT_ISI( fap, FWD_FFT );			/* not used, but play it safe */

	// Transform the columns first!
	/* old
	SET_FFT_SINC( fap, OBJ_PXL_INC( OA_SRC1(oap) ) );
	SET_FFT_DINC( fap, OBJ_PXL_INC( OA_DEST(oap) ) );
	*/
	if( OBJ_ROWS( OA_SRC1(oap) ) > 1 ){		/* more than 1 column ? */
		/* Transform the columns */
		dimension_t i;

		SET_FFT_SINC( fap, OBJ_ROW_INC( OA_SRC1(oap) ) );
		SET_FFT_DINC( fap, OBJ_ROW_INC( OA_DEST(oap) ) );
		SET_FFT_LEN( fap, OBJ_ROWS( OA_SRC1(oap) ) );

ifelse(MULTI_PROC_TEST,`1',` dnl #if N_PROCESSORS >= MIN_PARALLEL_PROCESSORS
		if( n_processors > 1 ){
			MULTIPROCESSOR_ROW_LOOP(OA_SRC1(oap),PF_FFT_CALL_NAME(rvfft),std_type,std_cpx)
		} else
') dnl endif /* N_PROCESSORS > 1 */
			COL_LOOP_2( OA_SRC1(oap),PF_FFT_CALL_NAME(rvfft),
				std_type,std_cpx)
	}

	/* Now transform the rows */
	/* BUG wrong if rows == 1 */
	/* Then we should copy into the complex target... */

	SET_FFT_LEN( fap, OBJ_COLS( OA_SRC1(oap) ) );
	SET_FFT_DINC( fap, OBJ_PXL_INC( OA_DEST(oap) ) );
	// We don_t need to set SINC because transform is done in-place...

	if( OBJ_COLS( OA_SRC1(oap) ) > 1 ){			/* more than 1 col? */
ifelse(MULTI_PROC_TEST,`1',` dnl #if N_PROCESSORS >= MIN_PARALLEL_PROCESSORS
		if( n_processors > 1 ){
			MULTIPROCESSOR_COLUMN_LOOP(OA_DEST(oap), PF_FFT_CALL_NAME(cvfft) )
		} else
') dnl endif /* N_PROCESSORS > 1 */
		//COLUMN_LOOP(OA_DEST(oap),PF_FFT_CALL_NAME(cvfft))
		ROW_LOOP_2(OA_DEST(oap),PF_FFT_CALL_NAME(cvfft))
	}
}

static void HOST_TYPED_CALL_NAME_REAL(fft2d,type_code)(HOST_CALL_ARG_DECLS)
{
	int n;

	switch( (n=real_fft_type(DEFAULT_QSP_ARG  OA_SRC1(oap),OA_DEST(oap),"rfft2d")) ){
		case 1:
			HOST_TYPED_CALL_NAME_REAL(fft2d_1,type_code)(HOST_CALL_ARGS);
			break;
		case 2:
			HOST_TYPED_CALL_NAME_REAL(fft2d_2,type_code)(HOST_CALL_ARGS);
			break;
		// default case is error in inputs, but reported elsewhere...
	}
}

// Inverse read DFT with short rows
// First we inverse transform the columns (complex),
// then the rows

static void HOST_TYPED_CALL_NAME_REAL(ift2d_1,type_code)( HOST_CALL_ARG_DECLS )
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);

	SET_FFT_ISI( fap, 1 );
	//SET_FFT_DINC( fap, OBJ_ROW_INC( OA_SRC1(oap) ) / 2 );
	SET_FFT_DINC( fap, OBJ_ROW_INC( OA_SRC1(oap) ) );

	if( OBJ_ROWS( OA_SRC1(oap) ) > 1 ){			/* more than 1 row? */
		/* Transform the columns */
		SET_FFT_LEN( fap, OBJ_ROWS( OA_SRC1(oap) ) );
ifelse(MULTI_PROC_TEST,`1',` dnl #if N_PROCESSORS >= MIN_PARALLEL_PROCESSORS
		if( n_processors > 1 ){
			MULTIPROCESSOR_COLUMN_LOOP(OA_SRC1(oap),PF_FFT_CALL_NAME(cvift))
		} else
') dnl endif /* N_PROCESSORS > 1 */
		COLUMN_LOOP(OA_SRC1(oap),PF_FFT_CALL_NAME(cvift))
	}

	if( OBJ_COLS( OA_SRC1(oap) ) > 1 ){		/* more than 1 column ? */
		dimension_t i;

		SET_FFT_SRC( fap, OBJ_DATA_PTR( OA_SRC1(oap) ) );
		SET_FFT_SINC( fap, OBJ_PXL_INC( OA_SRC1(oap) ) );
		SET_FFT_DST( fap, OBJ_DATA_PTR( OA_DEST(oap) ) );
		SET_FFT_DINC( fap, OBJ_PXL_INC( OA_DEST(oap) ) );
		SET_FFT_LEN( fap, OBJ_COLS( OA_DEST(oap) ) );	/* use the real len */

ifelse(MULTI_PROC_TEST,`1',` dnl #if N_PROCESSORS >= MIN_PARALLEL_PROCESSORS
		if( n_processors > 1 ){
			MULTIPROCESSOR_ROW_LOOP(OA_DEST(oap),PF_FFT_CALL_NAME(rvift),std_cpx,std_type)
		} else
') dnl endif /* N_PROCESSORS > 1 */
			ROW_LOOP(OA_DEST(oap),PF_FFT_CALL_NAME(rvift),
				std_cpx,std_type)
	}
}

// Inverse read DFT with short colums
// First we inverse transform the rows (complex),
// then the columns

static void HOST_TYPED_CALL_NAME_REAL(ift2d_2,type_code)( HOST_CALL_ARG_DECLS )
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);

	//if( real_fft_check(DEFAULT_QSP_ARG  OA_DEST(oap),OA_SRC1(oap),"rift2d") < 0 ) return;

	SET_FFT_ISI( fap, 1 );
	//SET_FFT_DINC( fap, OBJ_ROW_INC( OA_SRC1(oap) ) / 2 );
	SET_FFT_DINC( fap, OBJ_PXL_INC( OA_SRC1(oap) ) );

	if( OBJ_COLS( OA_SRC1(oap) ) > 1 ){			/* more than 1 column? */
		/* Transform the rows */
		SET_FFT_LEN( fap, OBJ_COLS( OA_SRC1(oap) ) );
ifelse(MULTI_PROC_TEST,`1',` dnl #if N_PROCESSORS >= MIN_PARALLEL_PROCESSORS
		if( n_processors > 1 ){
			MULTIPROCESSOR_COLUMN_LOOP(OA_SRC1(oap),PF_FFT_CALL_NAME(cvift))
		} else
') dnl endif /* N_PROCESSORS > 1 */
		//COLUMN_LOOP(OA_SRC1(oap),PF_FFT_CALL_NAME(cvift))
		ROW_LOOP_2(OA_SRC1(oap),PF_FFT_CALL_NAME(cvift))
	}

	// Now perform the real inverse transform on the columns

	if( OBJ_ROWS( OA_SRC1(oap) ) > 1 ){		/* more than 1 row ? */
		dimension_t i;

		SET_FFT_SRC( fap, OBJ_DATA_PTR( OA_SRC1(oap) ) );
		SET_FFT_SINC( fap, OBJ_ROW_INC( OA_SRC1(oap) ) );
		SET_FFT_DST( fap, OBJ_DATA_PTR( OA_DEST(oap) ) );
		SET_FFT_DINC( fap, OBJ_ROW_INC( OA_DEST(oap) ) );
		SET_FFT_LEN( fap, OBJ_ROWS( OA_DEST(oap) ) );	/* use the real len */

ifelse(MULTI_PROC_TEST,`1',` dnl #if N_PROCESSORS >= MIN_PARALLEL_PROCESSORS
		if( n_processors > 1 ){
			MULTIPROCESSOR_ROW_LOOP(OA_DEST(oap),PF_FFT_CALL_NAME(rvift),std_cpx,std_type)
		} else
') dnl endif /* N_PROCESSORS > 1 */
	/*		ROW_LOOP(OA_DEST(oap),PF_FFT_CALL_NAME(rvift),
				std_cpx,std_type) */
			COL_LOOP_2(OA_DEST(oap),PF_FFT_CALL_NAME(rvift),
				std_cpx,std_type)
	}
}


static void HOST_TYPED_CALL_NAME_REAL(ift2d,type_code)(HOST_CALL_ARG_DECLS)
{
	int n;

	switch( (n=real_fft_type(DEFAULT_QSP_ARG  OA_DEST(oap),OA_SRC1(oap),"rift2d")) ){
		case 1:
			HOST_TYPED_CALL_NAME_REAL(ift2d_1,type_code)(HOST_CALL_ARGS);
			break;
		case 2:
			HOST_TYPED_CALL_NAME_REAL(ift2d_2,type_code)(HOST_CALL_ARGS);
			break;
		// default case is error in inputs, but reported elsewhere...
	}
}

static void HOST_TYPED_CALL_NAME_REAL(iftrows,type_code)( HOST_CALL_ARG_DECLS )
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);
	dimension_t i;

	if( ! real_row_fft_ok(DEFAULT_QSP_ARG  OA_DEST(oap),OA_SRC1(oap),"r_rowift") ) return;

	SET_FFT_ISI( fap, 1 );
	SET_FFT_SRC( fap, OBJ_DATA_PTR( OA_SRC1(oap) ) );
	SET_FFT_SINC( fap, OBJ_PXL_INC( OA_SRC1(oap) ) );		// used to be /2
	SET_FFT_DST( fap, OBJ_DATA_PTR( OA_DEST(oap) ) );
	SET_FFT_DINC( fap, OBJ_PXL_INC( OA_DEST(oap) ) );
	SET_FFT_LEN( fap, OBJ_COLS( OA_DEST(oap) ) );

ifelse(MULTI_PROC_TEST,`1',` dnl #if N_PROCESSORS >= MIN_PARALLEL_PROCESSORS
	if( n_processors > 1 ){
		MULTIPROCESSOR_ROW_LOOP(OA_DEST(oap),PF_FFT_CALL_NAME(rvift),std_cpx,std_type)
		return;
	} else
') dnl endif /* N_PROCESSORS > 1 */
		ROW_LOOP(OA_DEST(oap),PF_FFT_CALL_NAME(rvift),std_cpx,std_type)
}


/*
 * Do an in-place complex FFT
 *
 * No SMP version (yet).
 */

static void HOST_TYPED_CALL_NAME_CPX(fft2d,type_code)( HOST_CALL_ARG_DECLS, int is_inv )
{
	dimension_t i;
	FFT_Args fa;
	FFT_Args *fap=(&fa);

	if( ! cpx_fft_ok(DEFAULT_QSP_ARG  OA_SRC1(oap), STRINGIFY(HOST_TYPED_CALL_NAME_CPX(fft2d,type_code)) ) )
		return;

	/* transform the columns */

	SET_FFT_ISI( fap, is_inv );

	if( OBJ_ROWS( OA_SRC1(oap) ) > 1 ){	/* more than one row */
		SET_FFT_DST( fap, (std_type *)OBJ_DATA_PTR( OA_SRC1(oap) ) );
		SET_FFT_LEN( fap, OBJ_ROWS( OA_SRC1(oap) ) );
		//SET_FFT_DINC( fap, OBJ_ROW_INC( OA_SRC1(oap) )/2 );
		SET_FFT_DINC( fap, OBJ_ROW_INC( OA_SRC1(oap) ) );

		for (i = 0; i < OBJ_COLS( OA_SRC1(oap) ); ++i) {
			PF_FFT_CALL_NAME(cvfft)(&fa);
			/* ((std_type *)fa.dst_addr) += OBJ_PXL_INC( OA_SRC1(oap) ); */
			SET_FFT_DST( fap, ((std_cpx *)FFT_DST(fap)) + OBJ_PXL_INC( OA_SRC1(oap) ) );
		}
	}

	/* transform the rows */

	if( OBJ_COLS( OA_SRC1(oap) ) > 1 ){
		SET_FFT_DST( fap, (std_type *)OBJ_DATA_PTR( OA_SRC1(oap) ) );
		SET_FFT_LEN( fap, OBJ_COLS( OA_SRC1(oap) ) );
		/* pixel inc used to be in machine units,
		 * now it_s in type units!? */
		//SET_FFT_DINC( fap, OBJ_PXL_INC( OA_SRC1(oap) )/2 );
		SET_FFT_DINC( fap, OBJ_PXL_INC( OA_SRC1(oap) ) );

		for (i = 0; i < OBJ_ROWS( OA_SRC1(oap) ); ++i) {
			PF_FFT_CALL_NAME(cvfft)(&fa);
			SET_FFT_DST( fap, ((std_cpx *)FFT_DST(fap)) + OBJ_ROW_INC( OA_SRC1(oap) ) );
		}
	}
}

static void HOST_TYPED_CALL_NAME_CPX(fftrows,type_code)( HOST_CALL_ARG_DECLS, int is_inv )
{
	dimension_t i;
	FFT_Args fa;
	FFT_Args *fap=(&fa);

	if( ! row_fft_ok(DEFAULT_QSP_ARG  OA_SRC1(oap), STRINGIFY(HOST_TYPED_CALL_NAME_CPX(fftrows,type_code)) ) )
		return;

	SET_FFT_ISI( fap, is_inv );

	/* transform the rows */

	if( OBJ_COLS( OA_SRC1(oap) ) > 1 ){
		SET_FFT_DST( fap, (std_type *)OBJ_DATA_PTR( OA_SRC1(oap) ) );
		SET_FFT_LEN( fap, OBJ_COLS( OA_SRC1(oap) ) );
		// What is pinc??? type units not machine units?
		//SET_FFT_DINC( fap, OBJ_PXL_INC( OA_SRC1(oap) )/2 );
		SET_FFT_DINC( fap, OBJ_PXL_INC( OA_SRC1(oap) ) );

		for (i = 0; i < OBJ_ROWS( OA_SRC1(oap) ); ++i) {
			PF_FFT_CALL_NAME(cvfft)(&fa);
			/* why not std_cpx??? */
			SET_FFT_DST( fap, ((std_type *)FFT_DST(fap)) + OBJ_ROW_INC( OA_SRC1(oap) ) );
		}
	}
}

static void HOST_TYPED_CALL_NAME_CPX(iftrows,type_code)(HOST_CALL_ARG_DECLS, int is_inv)
{
	HOST_TYPED_CALL_NAME_CPX(fftrows,type_code)(HOST_CALL_ARGS,1);
}

') dnl endif ! BUILDING_KERNELS

// vl2_fft_funcs.m4 DONE
