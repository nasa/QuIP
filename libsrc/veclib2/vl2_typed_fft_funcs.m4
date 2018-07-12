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
define(`last_real_sinfact_len',`TYPED_NAME(_last_real_sinfact_len)')
define(`last_real_AB_len',`TYPED_NAME(_last_real_AB_len)')

dnl BUG pick one or the other of these two methods for real FFT
define(`_isinfact',`TYPED_NAME(__isinfact)')
define(`_sinfact',`TYPED_NAME(__sinfact)')

define(`A_array',`TYPED_NAME(_A_array)')
define(`B_array',`TYPED_NAME(_B_array)')

define(`max_fft_len',`TYPED_NAME(_max_fft_len)')
define(`init_sinfact',`TYPED_NAME(_init_sinfact)')
define(`init_AB',`TYPED_NAME(_init_AB)')

define(`INV_FFT',`1')
define(`FWD_FFT',`-1')

define(`MAX_FFT_LEN',`4096L')

// vl2_fft_funcs.m4 buiding_kernels is set

// How can we have these static vars when this file is included twice!?

/* static fft vars NOT already declared */
define(`DECLARE_STATIC_FFT_VARS',`
/* declare_static_fft_vars DOING IT */
static dimension_t last_cpx_len=0;
static std_cpx *twiddle;

// BUG - not thread-safe!?
static dimension_t last_real_sinfact_len=0;
static std_type *_isinfact=NULL;
static std_type *_sinfact=NULL;

static dimension_t last_real_AB_len=0;
static std_cpx *A_array=NULL;
static std_cpx *B_array=NULL;

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
		sprintf(ERROR_STRING,
	"%s:  %s is neither a row nor a column!?","$1",OBJ_NAME($3));
		warn(ERROR_STRING);
		return;
	}
')


dnl	XFER_FFT_DINC( func, fap, dp )
define(`XFER_FFT_DINC',`

	if( IS_ROWVEC($3) ){
		SET_FFT_DINC($2, OBJ_PXL_INC( ($3) ) );
	} else if( IS_COLVEC($3) ){
		SET_FFT_DINC($2, OBJ_ROW_INC( ($3) ) );
	} else {
		sprintf(ERROR_STRING,
	"%s:  %s is neither a row nor a column!?","$1",OBJ_NAME($3));
		warn(ERROR_STRING);
		return;
	}
	dnl /* if( IS_COMPLEX($3) )	SET_FFT_DINC($2, FFT_DINC($2)/2); */
')


ifdef(`BUILDING_KERNELS',`
// vl2_fft_funcs.m4 buiding_kernels is SET

// twiddle factors are exp( i theta ), theta runs from 0 to pi, and we get pi to 2pi from symmetry

static void init_twiddle (dimension_t len)
{
	double twopi,theta;
	dimension_t i;

	if( last_cpx_len > 0 ){
		givbuf(twiddle);
	}

	twiddle = (std_cpx *)getbuf( sizeof(*twiddle) * (len/2) );


	twopi = 8.0*atan(1.0);

	/* W -kn N , W N = exp ( -j twopi/N ) */

	for(i=0;i<len/2;i++){
		theta = twopi*(double)(i)/(double)len;
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
	sprintf(DEFAULT_ERROR_STRING,"dst_addr = 0x%"PRIxPTR", inc = %ld",
		(uintptr_t)FFT_DST(fap),(long)FFT_DINC(fap));
	NADVISE(DEFAULT_ERROR_STRING);
	sprintf(DEFAULT_ERROR_STRING,"src_addr = 0x%"PRIxPTR", inc = %ld",
		(uintptr_t)FFT_SRC(fap),(long)FFT_SINC(fap));
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
	incr_t src_inc, dst_inc;

	//if( ! for_real ) return;

	len = FFT_LEN(fap);

	if( revdone==NULL ){
		/* this is no good because getbuf/givbuf are not thread-safe!
		 * I can_t see a way to do this without passing the thread index on the stack...
		 * OR having the entire revdone array on the stack?
		 */
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
		if( n_processors > 1 ) _warn(DEFAULT_QSP_ARG  "cvfft:  init_twiddle is not thread-safe!?");
')
		init_twiddle (len);
	}

	dest=(std_cpx *)FFT_DST(fap);
	dst_inc = FFT_DINC(fap);
	source=(std_cpx *)FFT_SRC(fap);
	src_inc = FFT_SINC(fap);

	/* inc1 should be in units of complex */

	if( len != bitrev_size ){
ifelse(MULTI_PROC_TEST,`1',`
		if( n_processors > 1 ) _warn(DEFAULT_QSP_ARG  "cvfft:  bitrev_init is not thread-safe!?");
')
		bitrev_init(len);
	}

	/* Copy from source to destination, in bit-reversed order.
	 * The use of the tmp storage during the exchanges ensures
	 * this this works when source and destination are the same.
	 */

	/* init revdone */
		
	for(i=0;i<len;i++) revdone[i]=0;
	for(i=0;i<len;i++){
		dimension_t di, dj, si, sj;
		std_cpx tmp;

		if( !revdone[i] ){
			di = i * dst_inc;
			si = i * src_inc;
			dj = bitrev_data[i] * dst_inc;
			sj = bitrev_data[i] * src_inc;
			if( di != dj ){
				// We use tmp so this will still work in-place
				tmp.re = source[si].re;
				tmp.im = source[si].im;
				dest[di].re = source[sj].re;
				dest[di].im = source[sj].im;
				dest[dj].re = tmp.re;
				dest[dj].im = tmp.im;
			} else {
				// Unnecessary if in-place...
				dest[di].re = source[sj].re;
				dest[di].im = source[sj].im;
			}
			revdone[i]=1;
			revdone[ bitrev_data[i] ]=1;
		}
	}

	/*	now compute the butterflies 	*/
	/* this section is trashing some memory!? */

	/* Note that this computation uses the inverse flag,
	 * which MUST be 1 or -1 !!!
	 */

	mmax = 1;
	while( mmax<len ){	// for s = 1 to log(n)
				// m = 2^s
				// w_m = exp(-2 pi i / m)
		istep = 2*mmax;
		for(m=0;m<mmax;m++){		// for k = 0 to n-1 by m
			dimension_t index;

						// w = 1
			index = m*(len/(mmax<<1));		// m * len/2 / mmax

			/* make index modulo len/2 */
			/* hope this works for negative index!! */
			index &= ((len>>1)-1);

			/* if( index < 0 ) index += len; */

			wp = (& twiddle[index]);

			for(i=m;i<len;i+=istep){		// for j = 0 to m/2-1
				dimension_t dj, di;

				j = i+mmax;
				dj = j * dst_inc;
				di = i * dst_inc;
				// if ISI=1, then temp = (*wp) * dest[dj]
				// otherwise its the complex conjugate of (*wp)...

				// t = w A[k+j+m/2] (dj)
				temp.re = wp->re*dest[dj].re
					- FFT_ISI(fap) * wp->im*dest[dj].im;
				temp.im = wp->re*dest[dj].im
					+ FFT_ISI(fap) * wp->im*dest[dj].re;
				// u = A[k+j]  (di)

				// A[k+j+m/2] = u - t
				dest[dj].re = dest[di].re-temp.re;
				dest[dj].im = dest[di].im-temp.im;
				// A[k+j] = u + t
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

	SET_FFT_DST(new_fap, FFT_DST(fap));
	SET_FFT_DINC(new_fap, FFT_DINC(fap));
	SET_FFT_SRC(new_fap, FFT_SRC(fap));
	SET_FFT_SINC(new_fap, FFT_SINC(fap));
	SET_FFT_LEN(new_fap, FFT_LEN(fap));
	SET_FFT_ISI(new_fap, INV_FFT);

	PF_FFT_CALL_NAME(cvfft)(new_fap);
}

// Compute tables isinfact and sinfact.
// Curiously, isinfact appears to hold the sine of arg, while sinfact is
// the inverse sine!?  EXPLANATION:  "inverse" sine factor is used in the
// computation of the inverse DFT!

dnl	the space before the opening paren is important!!!

static void init_sinfact (dimension_t n)
{
	dimension_t i;
	std_type arginc, arg;

	last_real_sinfact_len = n;
	n /= 2;

	if( _sinfact != (std_type *)NULL )
		givbuf(_sinfact);
	_sinfact = (std_type *)getbuf( n * sizeof(std_type) );

	if( _isinfact != (std_type *)NULL )
		givbuf(_isinfact);
	_isinfact = (std_type *)getbuf( n * sizeof(std_type) );

	arginc = (std_type)(4 * atan(1.0) / n);
	arg = 0.0;

	// What about the 0th entries???
	// We don_t want to divide by zero!?
	// Is it never used???
	for(i=1;i<n;i++){
		arg += arginc;
		_isinfact[i] = 2 * (std_type)sin(arg);
		_sinfact[i] = 1.0f / _isinfact[i];
	}
}

dnl	the space before the opening paren is important!!!

static void init_AB (dimension_t n)
{
	dimension_t i;
	std_type arginc, arg;

	last_real_AB_len = n;
	n /= 2;

	if( A_array != (std_cpx *)NULL )
		givbuf(A_array);
	A_array = (std_cpx *)getbuf( n * sizeof(std_cpx) );

	if( B_array != (std_cpx *)NULL )
		givbuf(B_array);
	B_array = (std_cpx *)getbuf( n * sizeof(std_cpx) );

	arginc = (std_type)(4 * atan(1.0) / n);
	arg = 0.0;

	for(i=0;i<n;i++){
		std_type w_re, w_im;

		// These are probably just the twiddle factors!?
		// can we use those from a table instead???

		// A_k = 0.5 * ( 1 - j W_2N^k )
		// B_k = 0.5 * ( 1 + j W_2N^k )
		w_re = cos(arg);
		w_im = sin(arg);

		A_array[i].re = 0.5 * ( 1 + w_im );
		A_array[i].im = 0.5 * (   - w_re );
		B_array[i].re = 0.5 * ( 1 - w_im );
		B_array[i].im = 0.5 * (   + w_re );

		arg += arginc;
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
 *
 * For the forward fft, this macro is called AFTER the transform has
 * been computed...
 */

// The TI documentation calls this the "split operation"
// Found this on the web:
// Efficient FFT Computation of Real Input - on TI wiki...

dnl	RECOMBINE does not touch the first sample...
dnl	Lets assume that we have loaded the complex array with the even
dnl	index samples in the real part, and the odd index samples
dnl	in the imaginary part...
dnl	Then RECOMBINE computes s1,s2 - the part of the original signal
dnl	with even SYMMETRY, and d1,d2 - the part of the signal with odd symm -
dnl	and flips the sign of every other odd symmetric value - (why?)
dnl
dnl	This operation is done on the transform AFTER the forward fft,
dnl	and on the input values BEFORE the inverse fft...


dnl	RECOMBINE(inc)

define(`RECOMBINE',`

	for(i=1;i<len/4;i++){
		std_type s1,s2,d1,d2;

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
 *
 * The original implementation (supposedly based on Elliott & Rao?)
 * didnt use the twiddle factors, but did something
 * like a split before and after the complex fft...
 *
 * The TI document uses a different algorithm that uses the twiddle factors,
 * that looks simpler...
 */

static void PF_FFT_CALL_NAME(rvfft_v1)( const FFT_Args *fap)
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

	if( len != last_real_sinfact_len ){
dnl	the space before the opening paren is important!!!
		init_sinfact (len);
	}

/* we assume that the destination has 1+len/2 complex entries */
/* transform the input while copying into dest */

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
	// Because we can_t (or should not) modify the input arg struct...

	// Compute in-place on the destination array...
	SET_FFT_DST(_fap, FFT_DST(fap) );
	SET_FFT_DINC(_fap, FFT_DINC(fap) );
	SET_FFT_SRC(_fap, FFT_DST(fap));
	SET_FFT_SINC(_fap, FFT_DINC(fap));
	SET_FFT_LEN(_fap, FFT_LEN(fap)/2 );
	SET_FFT_ISI(_fap, FWD_FFT);

	// Compute the FFT
	PF_FFT_CALL_NAME(cvfft)(_fap);

	cbot = dest;
	cbot->im = (std_type) B0;	// shouldn_t this be zero???
	for(i=1;i<len/2;i++){
		cbot+=dst_inc;
		cbot->im *= _sinfact[i];	// rvfft
	}

	/* now make it look like the correct answer */

	cbot = dest;
	ctop = dest+dst_inc*(len/2);

	ctop->re = cbot->re - cbot->im;
	cbot->re += cbot->im;
	ctop->im = cbot->im = 0.0;

	RECOMBINE(dst_inc)	// in-place
}

dnl	Alternate implementation based on TI documentation
dnl
dnl	This is somewhat simpler, but uses twiddle factors...

define(`ADVANCE_CPX_PTRS',`
	cbot += dst_inc;
	ctop -= dst_inc;
	abot++;
	bbot++;
	atop--;
	btop--;
')

dnl	GET_CPX_PROD(cptr,aptr,bptr)
dnl	destination must be distinct from sources!

define(`GET_CPX_PROD',`
	($1)->re = ($2)->re * ($3)->re - ($2)->im * ($3)->im;
	($1)->im = ($2)->re * ($3)->im + ($2)->im * ($3)->re;
')

dnl	conjugate just the first factor...

define(`GET_CPX_CONJ1_PROD',`
	($1)->re = ($2)->re * ($3)->re + ($2)->im * ($3)->im;
	($1)->im = ($2)->re * ($3)->im - ($2)->im * ($3)->re;
')

dnl	conjugate just the second factor...

define(`GET_CPX_CONJ2_PROD',`
	($1)->re = ($2)->re * ($3)->re + ($2)->im * ($3)->im;
	($1)->im = ($2)->im * ($3)->re - ($2)->re * ($3)->im;
')

dnl	conjugate BOTH factors...

define(`GET_CPX_CONJ12_PROD',`
	($1)->re = ($2)->re * ($3)->re - ($2)->im * ($3)->im;
	($1)->im = - ($2)->im * ($3)->re - ($2)->re * ($3)->im;
')

define(`GET_CPX_SUM',`
	($1)->re = ($2)->re + ($3)->re;
	($1)->im = ($2)->im + ($3)->im;
')

dnl	SHOW_SPLIT_DATA(msg)

define(`SHOW_ONE_CPX',`
	sprintf(DEFAULT_ERROR_STRING,"$1 at 0x%lx:   %g   %g",
		(long)$1,$1->re,$1->im);
	NADVISE(DEFAULT_ERROR_STRING);
')

define(`SHOW_SPLIT_DATA',`
	NADVISE("$1");
	SHOW_ONE_CPX(cbot)
	SHOW_ONE_CPX(ctop)
	SHOW_ONE_CPX(abot)
	SHOW_ONE_CPX(atop)
	SHOW_ONE_CPX(bbot)
	SHOW_ONE_CPX(btop)
	NADVISE("");
')

static void PF_FFT_CALL_NAME(rvfft)( const FFT_Args *fap)
{
	std_cpx *ctop, *cbot;
	std_cpx *atop, *abot;
	std_cpx *btop, *bbot;
	dimension_t i;
	dimension_t len;
	std_type *source;
	std_cpx *dest;
	FFT_Args fa;
	FFT_Args *_fap=(&fa);
	incr_t src_inc;
	incr_t dst_inc;
	std_cpx p1, p2, t1, t2;

	//if( ! for_real ) return;

	/* len is the length of the real data */
	len = FFT_LEN(fap);
	source = (std_type *)FFT_SRC(fap);
	dest = (std_cpx *)FFT_DST(fap);
	src_inc = FFT_SINC(fap);
	dst_inc = FFT_DINC(fap);

	// copy the input data

	for(i=0;i<len/2;i++){
		dest->re = *source;
		source += src_inc;
		dest->im = *source;
		source += src_inc;
		dest += dst_inc;
	}
sprintf(DEFAULT_ERROR_STRING,"After copy:");
NADVISE(DEFAULT_ERROR_STRING);
dest = (std_cpx *)FFT_DST(fap);
for(i=0;i<len/2;i++){
sprintf(DEFAULT_ERROR_STRING,"   %g   %g",dest[i].re,dest[i].im);
NADVISE(DEFAULT_ERROR_STRING);
}

	if( len != last_real_AB_len ){
dnl	the space before the opening paren is important!!!
		init_AB (len);
	}

	// Compute in-place on the destination array...
	SET_FFT_DST(_fap, FFT_DST(fap) );
	SET_FFT_DINC(_fap, FFT_DINC(fap) );
	SET_FFT_SRC(_fap, FFT_DST(fap));
	SET_FFT_SINC(_fap, FFT_DINC(fap));
	SET_FFT_LEN(_fap, FFT_LEN(fap)/2 );
	SET_FFT_ISI(_fap, FWD_FFT);

	// Compute the FFT
	PF_FFT_CALL_NAME(cvfft)(_fap);

sprintf(DEFAULT_ERROR_STRING,"After fft:");
NADVISE(DEFAULT_ERROR_STRING);
for(i=0;i<len/2;i++){
sprintf(DEFAULT_ERROR_STRING,"   %g   %g",dest[i].re,dest[i].im);
NADVISE(DEFAULT_ERROR_STRING);
}
	// Perform the "split"
	cbot = (std_cpx *)FFT_DST(fap);
	ctop = cbot + dst_inc * len/2;		// valid because xform has len N/2+1
	// Fix the extra entry before we do the split
	ctop->re = cbot->re - cbot->im;
	ctop->im = 0;

	abot = A_array;
	atop = A_array + len/2;
	bbot = B_array;
	btop = B_array + len/2;

	// 0 is a special case...
SHOW_SPLIT_DATA(before 0)
	GET_CPX_PROD(&p1,cbot,abot)
	GET_CPX_CONJ1_PROD(&p2,cbot,bbot)	// really ctop...
	GET_CPX_SUM(&t1,&p1,&p2)
	*cbot = t1;
SHOW_SPLIT_DATA(after 0)

	ADVANCE_CPX_PTRS

	for(i=1;i<len/4;i++){
		// G(k) = X(k)A(k) + X*(N-k)B(k)
SHOW_SPLIT_DATA(before idx)
		GET_CPX_PROD(&p1,cbot,abot)
		GET_CPX_CONJ1_PROD(&p2,ctop,bbot)
		GET_CPX_SUM(&t1,&p1,&p2)
		// G(N-k) = X(N-k)A(N-k) + X*(k)B(N-k)
		GET_CPX_PROD(&p1,ctop,atop)
		GET_CPX_CONJ1_PROD(&p2,cbot,btop)
		GET_CPX_SUM(&t2,&p1,&p2)

		*cbot = t1;
		*ctop = t2;
SHOW_SPLIT_DATA(after idx)

		ADVANCE_CPX_PTRS
	}

	// Now cbot and ctop should point to the same thing - the sample at len/4
	assert(cbot==ctop);

SHOW_SPLIT_DATA(before last)
	GET_CPX_PROD(&p1,cbot,abot)
	GET_CPX_CONJ1_PROD(&p2,ctop,bbot)
	GET_CPX_SUM(&t1,&p1,&p2)
	*cbot = t1;
SHOW_SPLIT_DATA(after last)

} // rvfft

/* One dimensional real inverse fft.
 *
 * This routine seems to be destructive to its source!? ...
 * Yes, because the first complex FFT is done in-place.
 * That is necessary because there is an extra column,
 * so we would need to allocate scratch space to do it non-destructively.
 *
 * original code based on Elliott & Rao
 */

static void PF_FFT_CALL_NAME(rvift_v2)( FFT_Args *fap)
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

	if( len != last_real_sinfact_len ){
dnl	the space before the opening paren is important!!!
		init_sinfact (len);
	}

	/* first fiddle the transform back */

	cbot = src;
	ctop = src + src_inc*(len/2);

	/* the imaginary dc & nyquist are zero */

	cbot->im = cbot->re - ctop->re;	// imaginary dc
	cbot->re += ctop->re;		// real dc
	cbot->re *= 0.5;
	cbot->im *= 0.5;

	RECOMBINE(src_inc)	// in-place - rvift

	/* remember value of B0 */
	cbot = src;
	B0 = cbot->im;

	/* multiply Bs by inverse sine factor */
	cbot->im = 0.0;
	for(i=1;i<len/2;i++){
		cbot+=src_inc;
		cbot->im *= _isinfact[i];	// rvift
	}

	SET_FFT_DST( _fap, FFT_SRC(fap) );
	SET_FFT_DINC( _fap, FFT_SINC(fap) );
	SET_FFT_SRC( _fap, FFT_SRC(fap) );
	SET_FFT_SINC( _fap, FFT_SINC(fap) );
	SET_FFT_LEN( _fap, FFT_LEN(fap)/2 );
	SET_FFT_ISI( _fap, INV_FFT );

	// compute in-place, overwriting the source...
	PF_FFT_CALL_NAME(cvfft)(_fap);
	// Now that we are not scaling the inverse transform,
	// the returned values here are larger than before by a factor of n/2
	// But they are half of what they should be...
	// We fix this by eliminating the multiplication by 0.5f below

	/* now reconstruct the samples */

	cbot = src;
	ctop = src + src_inc*(len/2);
	ep = dest;
	op = dest+dst_inc;

	*ep = cbot->re * 2;		// because un-scaled!
	*op = - cbot->im * 2;		// because un-scaled?
	for(i=1;i<len/2;i++){
		std_type s1, d1;
		ep += 2*dst_inc;
		op += 2*dst_inc;
		cbot+=src_inc;
		ctop-=src_inc;
		s1 = ctop->re + ctop->im;	/* F(y1) + F(y2) */
		d1 = cbot->re - cbot->im;	/* y1 - y2 */
		*ep = /* 0.5f * */ ( s1 + d1 );
		*op = /* 0.5f * */ ( d1 - s1 );
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
sprintf(DEFAULT_ERROR_STRING,"rvift:  total = %g    B0 = %g",total,B0);
NADVISE(DEFAULT_ERROR_STRING);

	// Not sure what the above comment means...
	// Because they are generally not equal - does this
	// following operation make them equal?
	//
	// This code broke after the normalization
	// was removed from the inverse transform
	// (done for compatibility w/ other libs).
	// 7/10/18 - we are off by a factor of 2 compared to cuda AND un-normed cvfft...
	//
	// WITH normalization:
	//diff = (std_type)(2 * ( B0 - total ) / len);
	// WITHOUT normalization:
	total /= (len);
	diff = (std_type)( 2*(B0 - total) );
sprintf(DEFAULT_ERROR_STRING,"rvift:  after normalizing total = %g    diff = %g",total,diff);
NADVISE(DEFAULT_ERROR_STRING);

	// B0 comes from the transform,
	// while total comes from the output of the
	// inverse transform;

	op = dest+dst_inc;
	for(i=0;i<len/2;i++){
		*op += diff;
		op += 2*dst_inc;
	}
	/* done */
}

// Alternate implementation based on TI white paper
// The forward transform seems to have more numerical error
// than the nVidia solution???

static void PF_FFT_CALL_NAME(rvift)( FFT_Args *fap)
{
	std_cpx *src;
	std_type *dest;
	dimension_t len;
	std_cpx *cbot, *ctop;
	std_cpx *atop, *abot;
	std_cpx *btop, *bbot;
	std_cpx p1, p2, t1, t2;
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

	if( len != last_real_AB_len ){
dnl	the space before the opening paren is important!!!
		init_AB (len);
	}

	/* transform G(k) to X(k) using the split operation */

	cbot = src;
	ctop = src + src_inc*(len/2);

NADVISE("before split");
src = (std_cpx *)FFT_SRC(fap);
for(i=0;i<len/2;i++){
sprintf(DEFAULT_ERROR_STRING,"   %g   %g",src[i].re,src[i].im);
NADVISE(DEFAULT_ERROR_STRING);
}
	abot = A_array;
	atop = A_array + len/2;
	bbot = B_array;
	btop = B_array + len/2;

	// 0 is a special case...
SHOW_SPLIT_DATA(before 0)
	GET_CPX_CONJ2_PROD(&p1,cbot,abot)
	GET_CPX_CONJ12_PROD(&p2,cbot,bbot)	// really ctop...
	GET_CPX_SUM(&t1,&p1,&p2)
	*cbot = t1;
SHOW_SPLIT_DATA(after 0)

	ADVANCE_CPX_PTRS

	for(i=1;i<len/4;i++){
		// X(k) = G(k)A*(k) + G*(N-k)B*(k)
SHOW_SPLIT_DATA(before idx)
		GET_CPX_CONJ2_PROD(&p1,cbot,abot)
		GET_CPX_CONJ12_PROD(&p2,ctop,bbot)
		GET_CPX_SUM(&t1,&p1,&p2)
		// G(N-k) = X(N-k)A(N-k) + X*(k)B(N-k)
		GET_CPX_CONJ2_PROD(&p1,ctop,atop)
		GET_CPX_CONJ12_PROD(&p2,cbot,btop)
		GET_CPX_SUM(&t2,&p1,&p2)

		*cbot = t1;
		*ctop = t2;
SHOW_SPLIT_DATA(after idx)

		ADVANCE_CPX_PTRS
	}

	// Now cbot and ctop should point to the same thing - the sample at len/4
	assert(cbot==ctop);

SHOW_SPLIT_DATA(before last)
	GET_CPX_CONJ2_PROD(&p1,cbot,abot)
	GET_CPX_CONJ12_PROD(&p2,ctop,bbot)
	GET_CPX_SUM(&t1,&p1,&p2)
	*cbot = t1;
SHOW_SPLIT_DATA(after last)

NADVISE("after split / before transform");
src = (std_cpx *)FFT_SRC(fap);
for(i=0;i<len/2;i++){
sprintf(DEFAULT_ERROR_STRING,"   %g   %g",src[i].re,src[i].im);
NADVISE(DEFAULT_ERROR_STRING);
}

	SET_FFT_DST( _fap, FFT_SRC(fap) );
	SET_FFT_DINC( _fap, FFT_SINC(fap) );
	SET_FFT_SRC( _fap, FFT_SRC(fap) );
	SET_FFT_SINC( _fap, FFT_SINC(fap) );
	SET_FFT_LEN( _fap, FFT_LEN(fap)/2 );
	SET_FFT_ISI( _fap, INV_FFT );

	// compute in-place, overwriting the source...
	PF_FFT_CALL_NAME(cvfft)(_fap);
NADVISE("after transform");
src = (std_cpx *)FFT_SRC(fap);
for(i=0;i<len/2;i++){
sprintf(DEFAULT_ERROR_STRING,"   %g   %g",src[i].re,src[i].im);
NADVISE(DEFAULT_ERROR_STRING);
}

	// copy the data to the destination

	for(i=0;i<len/2;i++){
		*dest = src->re;
		dest += dst_inc;
		*dest = src->im;
		dest += dst_inc;
		src += src_inc;
	}
}	// rvift

',` dnl else ! BUILDING_KERNELS

// vl2_fft_funcs.m4 buiding_kernels is NOT SET

void HOST_TYPED_CALL_NAME_CPX(vfft,type_code)( HOST_CALL_ARG_DECLS )
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);

	SET_FFT_DST(fap, (std_cpx *)OBJ_DATA_PTR( OA_DEST(oap) ) );
	SET_FFT_SRC(fap, (std_cpx *)OBJ_DATA_PTR( OA_SRC1(oap) ) );
	XFER_FFT_DINC(cvfft,fap,OA_DEST(oap))
	XFER_FFT_SINC(cvfft,fap,OA_SRC1(oap))
	SET_FFT_LEN(fap, OBJ_N_TYPE_ELTS( OA_DEST(oap) ) );	/* complex */
	SET_FFT_ISI(fap,FWD_FFT);

	PF_FFT_CALL_NAME(cvfft)( fap );
}

void HOST_TYPED_CALL_NAME_CPX(vift,type_code)( HOST_CALL_ARG_DECLS )
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);

	SET_FFT_DST(fap, (std_cpx *)OBJ_DATA_PTR( OA_DEST(oap) ) );
	SET_FFT_SRC(fap, (std_cpx *)OBJ_DATA_PTR( OA_SRC1(oap) ) );
	XFER_FFT_DINC(cvift,fap,OA_DEST(oap))
	XFER_FFT_SINC(cvift,fap,OA_SRC1(oap))
	SET_FFT_LEN(fap, OBJ_N_TYPE_ELTS( OA_DEST(oap) ) );	/* complex */
	SET_FFT_ISI(fap, INV_FFT );

	PF_FFT_CALL_NAME(cvift)( fap );
}


void HOST_TYPED_CALL_NAME_REAL(vfft,type_code)( HOST_CALL_ARG_DECLS )
{
	FFT_Args fa;
	FFT_Args *fap;

	fap = (&fa);

	SET_FFT_SRC( fap, (std_type *)OBJ_DATA_PTR( OA_SRC1(oap) ) );
	SET_FFT_DST( fap, (std_cpx *)OBJ_DATA_PTR( OA_DEST(oap) ) );

	XFER_FFT_SINC(rvfft,fap,OA_SRC1(oap))
	XFER_FFT_DINC(rvfft,fap,OA_DEST(oap))

	SET_FFT_LEN( fap, OBJ_N_TYPE_ELTS( OA_SRC1(oap) ) );
	SET_FFT_ISI( fap, FWD_FFT );
	PF_FFT_CALL_NAME(rvfft)( fap );
}

void HOST_TYPED_CALL_NAME_REAL(vift,type_code)( HOST_CALL_ARG_DECLS )
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);

	SET_FFT_SRC( fap, (std_cpx *)OBJ_DATA_PTR( OA_SRC1(oap) ) );
	SET_FFT_DST( fap, (std_type *)OBJ_DATA_PTR( OA_DEST(oap) ) );

	XFER_FFT_SINC(rvift,fap,OA_SRC1(oap))
	XFER_FFT_DINC(rvift,fap,OA_DEST(oap))

	SET_FFT_LEN( fap, OBJ_N_TYPE_ELTS( OA_DEST(oap) ) );
	SET_FFT_ISI( fap, INV_FFT );
	//PF_FFT_CALL_NAME(rvfft)( fap );
	PF_FFT_CALL_NAME(rvift)( fap );
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
	if( OBJ_COLS( $1 ) != last_real_sinfact_len ){
dnl	the space before the opening paren is important!!!
		init_sinfact (OBJ_COLS( $1 ));	// multiprocessor row loop
		// OR init_AB ???
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

dnl fprintf(stderr,"column_loop starting on object %s\\n",OBJ_NAME($1));
		SET_FFT_SRC( fap, OBJ_DATA_PTR( $1 ) );
		SET_FFT_DST( fap, OBJ_DATA_PTR( $1 ) );

		for(i=0;i<OBJ_COLS( $1 );i++){
			$2(fap);
			SET_FFT_SRC( fap, ((std_cpx *)FFT_SRC(fap)) + OBJ_PXL_INC( $1 ) );
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

void HOST_TYPED_CALL_NAME_REAL(fftrows,type_code)(HOST_CALL_ARG_DECLS)
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

dnl fprintf(stderr,"rvfft2d_1:  row loop done\\n");

	/* Now transform the columns */
	/* BUG wrong if columns == 1 */
	/* Then we should copy into the complex target... */

	SET_FFT_LEN( fap, OBJ_ROWS( OA_SRC1(oap) ) );
	SET_FFT_DINC( fap, OBJ_ROW_INC( OA_DEST(oap) ) );
	SET_FFT_SINC( fap, OBJ_ROW_INC( OA_DEST(oap) ) );

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

void HOST_TYPED_CALL_NAME_REAL(fft2d,type_code)(HOST_CALL_ARG_DECLS)
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

	SET_FFT_ISI( fap, INV_FFT );

	if( OBJ_ROWS( OA_SRC1(oap) ) > 1 ){			/* more than 1 row? */
		SET_FFT_DST( fap, OBJ_DATA_PTR( OA_SRC1(oap) ) );
		SET_FFT_DINC( fap, OBJ_ROW_INC( OA_SRC1(oap) ) );
		SET_FFT_SRC( fap, OBJ_DATA_PTR( OA_SRC1(oap) ) );
		SET_FFT_SINC( fap, OBJ_ROW_INC( OA_SRC1(oap) ) );
		/* Transform the columns in-place */
		// BUG if there is only one column, should not be in-place!?  FIXME
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

	SET_FFT_ISI( fap, INV_FFT );
	//SET_FFT_DINC( fap, OBJ_ROW_INC( OA_SRC1(oap) ) / 2 );
	SET_FFT_DINC( fap, OBJ_PXL_INC( OA_SRC1(oap) ) );

	if( OBJ_COLS( OA_SRC1(oap) ) > 1 ){			/* more than 1 column? */
		/* Transform the rows */
		SET_FFT_LEN( fap, OBJ_COLS( OA_SRC1(oap) ) );
ifelse(MULTI_PROC_TEST,`1',` dnl #if N_PROCESSORS >= MIN_PARALLEL_PROCESSORS
// BUG should be ROW_LOOP???
		if( n_processors > 1 ){
			MULTIPROCESSOR_COLUMN_LOOP(OA_SRC1(oap),PF_FFT_CALL_NAME(cvift))
		} else
') dnl endif /* N_PROCESSORS > 1 */
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


void HOST_TYPED_CALL_NAME_REAL(ift2d,type_code)(HOST_CALL_ARG_DECLS)
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

void HOST_TYPED_CALL_NAME_REAL(iftrows,type_code)( HOST_CALL_ARG_DECLS )
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);
	dimension_t i;

	if( ! real_row_fft_ok(DEFAULT_QSP_ARG  OA_DEST(oap),OA_SRC1(oap),"r_rowift") ) return;

	SET_FFT_ISI( fap, INV_FFT );
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

static void HOST_TYPED_CALL_NAME_CPX(xft2d,type_code)( Vec_Obj_Args *oap, FFT_Args *fap )
{
	dimension_t i;
	incr_t src_row_inc;

	/* transform the columns */

	if( OBJ_ROWS( OA_SRC1(oap) ) > 1 ){	/* more than one row */
		SET_FFT_DST( fap, (std_type *)OBJ_DATA_PTR( OA_DEST(oap) ) );
		SET_FFT_DINC( fap, OBJ_ROW_INC( OA_DEST(oap) ) );
		SET_FFT_SRC( fap, (std_type *)OBJ_DATA_PTR( OA_SRC1(oap) ) );
		SET_FFT_SINC( fap, OBJ_ROW_INC( OA_SRC1(oap) ) );
		SET_FFT_LEN( fap, OBJ_ROWS( OA_SRC1(oap) ) );

		for (i = 0; i < OBJ_COLS( OA_SRC1(oap) ); ++i) {
			PF_FFT_CALL_NAME(cvfft)(fap);
			SET_FFT_DST( fap, ((std_cpx *)FFT_DST(fap)) + OBJ_PXL_INC( OA_DEST(oap) ) );
			SET_FFT_SRC( fap, ((std_cpx *)FFT_SRC(fap)) + OBJ_PXL_INC( OA_SRC1(oap) ) );
		}

		// prepare for row transforms
		SET_FFT_SRC( fap, (float *)OBJ_DATA_PTR( OA_DEST(oap) ) );
		SET_FFT_SINC( fap, OBJ_PXL_INC( OA_DEST(oap) ) );
		src_row_inc = OBJ_ROW_INC( OA_DEST(oap) );
	} else {
		// row cannot be done in place
		SET_FFT_SRC( fap, (float *)OBJ_DATA_PTR( OA_SRC1(oap) ) );
		SET_FFT_SINC( fap, OBJ_PXL_INC( OA_SRC1(oap) ) );
		src_row_inc = OBJ_ROW_INC( OA_SRC1(oap) );
	}

	/* transform the rows */

	if( OBJ_COLS( OA_SRC1(oap) ) > 1 ){
		SET_FFT_DST( fap, (std_type *)OBJ_DATA_PTR( OA_DEST(oap) ) );
		SET_FFT_DINC( fap, OBJ_PXL_INC( OA_DEST(oap) ) );
		SET_FFT_LEN( fap, OBJ_COLS( OA_SRC1(oap) ) );

		for (i = 0; i < OBJ_ROWS( OA_SRC1(oap) ); ++i) {
			PF_FFT_CALL_NAME(cvfft)(fap);
			SET_FFT_DST( fap, ((std_cpx *)FFT_DST(fap)) + OBJ_ROW_INC( OA_DEST(oap) ) );
			SET_FFT_SRC( fap, ((SP_Complex *)FFT_SRC(fap)) + src_row_inc );
		}
	}
}

void HOST_TYPED_CALL_NAME_CPX(fft2d,type_code)( HOST_CALL_ARG_DECLS )
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);

	if( ! cpx_fft_ok(DEFAULT_QSP_ARG  OA_SRC1(oap), STRINGIFY(HOST_TYPED_CALL_NAME_CPX(fft2d,type_code)) ) )
		return;


	SET_FFT_ISI( fap, FWD_FFT );
	HOST_TYPED_CALL_NAME_CPX(xft2d,type_code)(oap,fap);
}

// Duplicates fft2d - but needed for consistency???

void HOST_TYPED_CALL_NAME_CPX(ift2d,type_code)( HOST_CALL_ARG_DECLS )
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);

	if( ! cpx_fft_ok(DEFAULT_QSP_ARG  OA_SRC1(oap), STRINGIFY(HOST_TYPED_CALL_NAME_CPX(fft2d,type_code)) ) )
		return;

	SET_FFT_ISI( fap, INV_FFT );
	HOST_TYPED_CALL_NAME_CPX(xft2d,type_code)(oap,fap);
}

static void HOST_TYPED_CALL_NAME_CPX(xftrows,type_code)( Vec_Obj_Args *oap, FFT_Args *fap )
{
	dimension_t i;

	/* transform the rows */

	if( OBJ_COLS( OA_SRC1(oap) ) > 1 ){
		SET_FFT_DST( fap, (std_type *)OBJ_DATA_PTR( OA_SRC1(oap) ) );
		SET_FFT_LEN( fap, OBJ_COLS( OA_SRC1(oap) ) );
		// What is pinc??? type units not machine units?
		//SET_FFT_DINC( fap, OBJ_PXL_INC( OA_SRC1(oap) )/2 );
		SET_FFT_DINC( fap, OBJ_PXL_INC( OA_SRC1(oap) ) );

		for (i = 0; i < OBJ_ROWS( OA_SRC1(oap) ); ++i) {
			PF_FFT_CALL_NAME(cvfft)(fap);
			/* why not std_cpx??? */
			SET_FFT_DST( fap, ((std_type *)FFT_DST(fap)) + OBJ_ROW_INC( OA_SRC1(oap) ) );
		}
	}
}

void HOST_TYPED_CALL_NAME_CPX(fftrows,type_code)( HOST_CALL_ARG_DECLS )
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);

	if( ! row_fft_ok(DEFAULT_QSP_ARG  OA_SRC1(oap), STRINGIFY(HOST_TYPED_CALL_NAME_CPX(fftrows,type_code)) ) )
		return;

	SET_FFT_ISI( fap, FWD_FFT );

	HOST_TYPED_CALL_NAME_CPX(xftrows,type_code)(oap,fap);
}

void HOST_TYPED_CALL_NAME_CPX(iftrows,type_code)( HOST_CALL_ARG_DECLS )
{
	FFT_Args fa;
	FFT_Args *fap=(&fa);

	if( ! row_fft_ok(DEFAULT_QSP_ARG  OA_SRC1(oap), STRINGIFY(HOST_TYPED_CALL_NAME_CPX(fftrows,type_code)) ) )
		return;

	SET_FFT_ISI( fap, INV_FFT );

	HOST_TYPED_CALL_NAME_CPX(xftrows,type_code)(oap,fap);
}

') dnl endif ! BUILDING_KERNELS

// vl2_fft_funcs.m4 DONE
