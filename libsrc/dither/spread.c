#include "quip_config.h"

/*
 * program to requantize
 *
 * modified 8-22 to monotonicly reduce the total squared error
 * modified 11-14-91 to handle color images
 *
 * 12-1-01 cleaned up, and more comments added.
 */

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>		/* drand48 */
#endif

#ifdef HAVE_TIME_H
#include <time.h>
#endif

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

#include "quip_prot.h"
#include "vec_util.h"
#include "data_obj.h"

/* local prototypes */
static void anneal_pixel(dimension_t x,dimension_t y);
static void mk_prob_tbl(void);

#define NO_VALUE	(-1)
static double the_sos=NO_VALUE;
#define NO_PIXELS 0
static dimension_t _npixels=NO_PIXELS;
static float _k;

/* user-supplied images */
static Data_Obj *_gdp=NO_OBJ;		/* input grayscale image */
static Data_Obj *_hdp=NO_OBJ;		/* output halftone */
static Data_Obj *_fdp=NO_OBJ;		/* filter */
static float *_gptr;
static float *_hptr;
/* static float *_fptr; */

/* program-private images */
static Data_Obj *_edp=NO_OBJ;		/* error image */
static Data_Obj *_fedp=NO_OBJ;		/* doubly filtered error */
static Data_Obj *_ffdp=NO_OBJ;		/* double filter */
static float *_eptr;
static float *_feptr;
/* static float *_ffptr; */

static double _temp;


#define MAX_DELE	16.0
#define N_BINS		1024
#define BIN_WIDTH	(2*MAX_DELE/N_BINS)

static double prob_tbl[N_BINS];

void (*scan_func)(dimension_t,dimension_t,dimension_t,dimension_t *,dimension_t *)=get_xy_random_point;

/* The probablilty table represents energies from -MAX_DELE to MAX_DELE...
 * linearly divided into bins.
 * This is exponentiated, and then a probability is computed as e/(1+e).
 * this approaches 0 for large negative delta_E, and 1 for large positive delta_E,
 * and has a value of 0.5 if delta_E is 0.
 * It is not at all clear why this is the right thing to do, it would seem
 * to introduce a decided bias between on- and off-pixels!?
 */

static void mk_prob_tbl(void)
{
	double val, eval;
	int i;
	static int prob_ready=0;

	if( prob_ready ) return;

	if( verbose )
		NADVISE("Initializing probability table");

	val = - MAX_DELE + BIN_WIDTH/2;

	for(i=0;i<N_BINS;i++){
		eval = exp(val);
		prob_tbl[i] = eval / ( 1 + eval );
		val += BIN_WIDTH;
	}
	prob_ready=1;
}

/* get probability associated with a given delta_E...
 *
 */

double get_prob(double dele)
{
	int index;

	if( dele > MAX_DELE ) dele=MAX_DELE;
	else if( dele < -MAX_DELE ) dele = -MAX_DELE;

	dele += MAX_DELE;	/* range is now 0 to 2*MAX_DELE */
	dele *= N_BINS/(2*MAX_DELE);

	index = (int)dele;
	if( index >= N_BINS ) index = N_BINS-1;

	return( prob_tbl[index] );
}

int setup_requantize(SINGLE_QSP_ARG_DECL)
{
	if( _hdp == NO_OBJ ){
		NWARN("output image not specified");
		return(-1);
	}
	if( _gdp == NO_OBJ ){
		NWARN("input image not specified");
		return(-1);
	}
	if( _fdp == NO_OBJ ){
		NWARN("filter not specified");
		return(-1);
	}
	if( OBJ_ROWS(_hdp) != OBJ_ROWS(_gdp) ||
		OBJ_COLS(_hdp) != OBJ_COLS(_gdp) ){
		NWARN("input/output size mismatch");
		return(-1);
	}

	if( OBJ_ROWS(_gdp) != OBJ_COLS(_gdp)
		&& scan_func==get_xy_scattered_point ){

		NWARN("input image must be square for scattered scanning");
		return(-1);
	}

	_npixels = (OBJ_ROWS(_hdp) * OBJ_COLS(_hdp));

	if( _edp != NO_OBJ )
		delvec(QSP_ARG  _edp);
	_edp = mk_img(QSP_ARG  "HT_error",
		OBJ_ROWS(_hdp),OBJ_COLS(_hdp),1,PREC_FOR_CODE(PREC_SP));
	if( _edp == NO_OBJ ){
		NWARN("couldn't create error image");
		return(-1);
	}
	_eptr = (float *) OBJ_DATA_PTR(_edp);

	if( _fedp != NO_OBJ )
		delvec(QSP_ARG  _fedp);
	_fedp = mk_img(QSP_ARG  "HT_ferror",
		OBJ_ROWS(_hdp),OBJ_COLS(_hdp),1,PREC_FOR_CODE(PREC_SP));
	if( _fedp == NO_OBJ ){
		NWARN("couldn't create filtered error image");
		return(-1);
	}
	_feptr = (float *) OBJ_DATA_PTR(_fedp);

	setup_ffilter(QSP_ARG  _fdp);

	/* We used to seed the random number generator here
	 * (using the time), but this was eliminated because
	 * it made it impossible to seed the generator by hand.
	 */

	return(0);
}

void set_filter(Data_Obj *fdp)
{
	_fdp = fdp;
	/* _fptr = (float *) OBJ_DATA_PTR(fdp); */
}

void set_grayscale(Data_Obj *gdp)
{
	_gdp = gdp;
	_gptr = (float *) OBJ_DATA_PTR(gdp);
}

void set_halftone(Data_Obj *hdp)
{
	_hdp = hdp;
	_hptr = (float *) OBJ_DATA_PTR(hdp);
}

void setup_ffilter(QSP_ARG_DECL  Data_Obj *fdp)
{
	dimension_t i,j;
	float *fptr,val;
	incr_t offset, xos, yos;

	if( _ffdp != NO_OBJ )
		delvec(QSP_ARG  _ffdp);

	_ffdp = mk_img(QSP_ARG   "double_filter",
		OBJ_ROWS(fdp)*2-1,OBJ_COLS(fdp)*2-1,1,PREC_FOR_CODE(PREC_SP));

	if( _ffdp == NO_OBJ ){
		NWARN("couldn't create double filter image");
		return;
	}

	normalize_filter(fdp);

	xos = ((incr_t)OBJ_COLS(_ffdp) - (incr_t)OBJ_COLS(fdp))/2;
	yos = ((incr_t)OBJ_ROWS(_ffdp) - (incr_t)OBJ_ROWS(fdp))/2;
	fptr = (float *) OBJ_DATA_PTR(fdp);

	/* Initialize the double filter by convolving the filter w/ itself */

	img_clear(_ffdp);
	for(j=0;j<OBJ_ROWS(fdp);j++){
		for(i=0;i<OBJ_COLS(fdp);i++){
			offset = (incr_t)(j*OBJ_COLS(fdp)+i);
			val = *(fptr+offset);
			add_impulse(val,_ffdp,fdp,xos+i,yos+j);
		}
	}
}

double get_volume(Data_Obj *dp)
{
	dimension_t i,j;
	double rowsum,sum;
	float *ptr;

	sum=0.0;
	ptr = (float *) OBJ_DATA_PTR(dp);
	for(j=0;j<OBJ_ROWS(dp);j++){
		rowsum=0.0;
		for(i=0;i<OBJ_COLS(dp);i++)
			rowsum += *(ptr+ j*OBJ_ROW_INC(dp)+i );
		sum += rowsum;
	}
	return(sum);
}


void normalize_filter(Data_Obj *fdp)
{
	float *fptr;
	dimension_t i,j;
	double sos, length;
	incr_t offset;

	fptr = (float *) OBJ_DATA_PTR(fdp);

	sos =0.0;
	for(j=0;j<OBJ_ROWS(fdp);j++){
		for(i=0;i<OBJ_COLS(fdp);i++){
			offset = (incr_t)j*OBJ_ROW_INC(fdp) + (incr_t)i;
			sos += *(fptr+offset) * *(fptr+offset);
		}
	}
	if( sos <= 0.0 ){
		NWARN("filter has non-positive vector length!?");
		return;
	}
	length = sqrt(sos);
	if( verbose ){
		sprintf(DEFAULT_ERROR_STRING,
			"Normalizing filter by factor %g",length);
		NADVISE(DEFAULT_ERROR_STRING);
	}
	for(j=0;j<OBJ_ROWS(fdp);j++){
		for(i=0;i<OBJ_COLS(fdp);i++){
			offset = (incr_t)j*OBJ_ROW_INC(fdp) + (incr_t)i;
			*(fptr+offset) /= length;
		}
	}
}

void init_requant(void)
{
	dimension_t i,j;
	dimension_t offset;
	float value;

	/* initialize error image */

	for(j=0;j<OBJ_ROWS(_edp); j++){
		for(i=0;i<OBJ_COLS(_edp); i++){
			offset = OBJ_COLS(_edp)*j+i;
			*(_eptr+offset) = *(_hptr+offset) - *(_gptr+offset);
		}
	}

	/* initialize filtered error image */
	img_clear(_fedp);
	for(j=0;j<OBJ_ROWS(_edp); j++){
		for(i=0;i<OBJ_COLS(_edp); i++){
			offset = (OBJ_COLS(_edp)*j+i);
			value = *(_eptr+offset);
			add_impulse(value,_fedp,_ffdp,i,j);
		}
	}

	mk_prob_tbl();
}

int scan_requant(int ntimes)
{
	dimension_t i;
	int j;
	dimension_t x,y;
	dimension_t n_changed;


	if( _npixels == NO_PIXELS ){
		NWARN("have to tell me which images first!");
		return(0);
	}

	/* scan image, updating y's */

	n_changed=0;
	for(j=0;j<ntimes;j++){
		for(i=0;i<(dimension_t)_npixels;i++){
			(*scan_func)(i,OBJ_COLS(_edp),OBJ_ROWS(_edp),&x,&y);
			n_changed += redo_pixel(x,y);
		}
	}
	return( n_changed );
}

void scan2_requant(QSP_ARG_DECL  int ntimes)
{
	int j;
	dimension_t i, x,y;
	int n_changed;


	if( _npixels == NO_PIXELS ){
		NWARN("have to tell me which images first!");
		return;
	}

	/* scan image, updating y's */

	for(j=0;j<ntimes;j++){
		n_changed=0;
		for(i=0;i<(dimension_t)_npixels;i++){
			(*scan_func)(i,OBJ_COLS(_edp),OBJ_ROWS(_edp),&x,&y);
			n_changed += redo_two_pixels(x,y);
		}
sprintf(ERROR_STRING,"Iteration %d:  %d pixels changed",j+1,n_changed);
advise(ERROR_STRING);
		if( n_changed == 0 ) return;
	}
}

void scan_anneal(double temp,int ntimes)
{
	int j;
	dimension_t i,x,y;

	if( _npixels == NO_PIXELS ){
		NWARN("have to tell me which images first!");
		return;
	}

	/* scan image, updating y's */

	/* this should probably have some other normalizing factors in it... */
	_k = 1 ;

	_temp = temp;
	for(j=0;j<ntimes;j++){
		for(i=0;i<_npixels;i++){
			(*scan_func)(i,OBJ_COLS(_edp),OBJ_ROWS(_edp),&x,&y);
			anneal_pixel(x,y);
		}
	}
}

/*
 * This returns the difference between the error with the pixel
 * set and the error with the pixel cleared
 */

float get_delta(dimension_t x,dimension_t y)
{
	float delta_E;
	dimension_t offset;

	offset = y*OBJ_ROW_INC(_edp) + x;
	/* BUG this assumes that all images have the same rowing */

	delta_E = *(_feptr+offset);		/* doubly filtered error */
	delta_E -= *(_eptr+offset);		/* the local error */
	delta_E -= *(_gptr+offset);		/* subtract desired value */

	return(delta_E);
}


/* redo_pixel changes the state of the pixel at x,y to reduce the total error
 * Returns 1 if flipped, 0 otherwise.
 */

int redo_pixel(dimension_t x,dimension_t y)
{
	dimension_t offset;
	float oldbit;
	float delta_E,olderr,errval;
	double oldsos=0.0;	/* init to eliminate warning */

	offset = y*OBJ_ROW_INC(_edp) + x;

	if( verbose ){
		if( the_sos == NO_VALUE ){
			the_sos = get_sos(_edp,_fdp);
			sprintf(DEFAULT_MSG_STR,"Initial SOS:  %g",the_sos);
			_prt_msg(DEFAULT_QSP_ARG  DEFAULT_MSG_STR);
		}
		oldsos = the_sos;
		/* subtract contribution of this pt from sos */
		add_to_sos(x,y,_edp,_fdp,-1);
	}

	olderr = *(_eptr+offset);

	/* save to check if changed later */
	oldbit = *(_hptr+offset);

	delta_E = get_delta(x,y);

	if( delta_E >= 0 ){
		if( verbose ){
			sprintf(DEFAULT_ERROR_STRING,"clearing pixel at %d, %d",x,y);
			NADVISE(DEFAULT_ERROR_STRING);
		}
		*(_hptr+offset) = -1;
	} else {
		if( verbose ){
			sprintf(DEFAULT_ERROR_STRING,"setting pixel at %d, %d",x,y);
			NADVISE(DEFAULT_ERROR_STRING);
		}
		*(_hptr+offset) =  1;
	}

	errval = *(_hptr+offset) - *(_gptr+offset);
	*(_eptr+offset) = errval ;

	if( verbose )
		add_to_sos(x,y,_edp,_fdp,1);

	if( *(_hptr+offset) == oldbit )	/* no change? */
		return(0);

	/* correct filtered error */
	add_impulse(errval-olderr,_fedp,_ffdp,x,y);

	if( verbose ){
		sprintf(DEFAULT_MSG_STR,
			"Pixel at %d,%d\t SOS:  %g\t\tdelta = %g",
			x,y,the_sos,the_sos-oldsos);
		_prt_msg(DEFAULT_QSP_ARG  DEFAULT_MSG_STR);
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"total sse: %g",get_sos(_edp,_fdp));
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	}
	return(1);
} /* end redo_pixel */

/*
 * Here we investigate the effect of flipping a single pixel
 * and also the effect of exchanging it with a neighboring
 * pixel of opposite sign
 *
 * We keep a table of the energy delta's:
 *
 *   1   2   3
 *   8   0   4
 *   7   6   5
 */

#define HERE	0
#define NW	1
#define NORTH	2
#define NE	3
#define EAST	4
#define SE	5
#define SOUTH	6
#define SW	7
#define WEST	8

#define NLOCS	9
#define NO_LOC	(-1)

static incr_t dx_tbl[NLOCS]={ 0, -1, 0, 1, 1, 1, 0, -1, -1 };
static incr_t dy_tbl[NLOCS]={ 0, 1, 1, 1, 0, -1, -1, -1, 0 };

#define NO_GO	(-10000.0)

/* redo_two_pixels
 * called by "migrate" menu function, this trys flipping a pixel and one of it's neighbors.
 * This is useful because we might obtain lower energy by exchanging two pixels, but with
 * an energy "hump" when only one is flipped...   This routine allows "tunneling" under the
 * energy barrier.
 */

int redo_two_pixels(dimension_t x,dimension_t y)
{
	dimension_t offset;
	float oldbit;
	dimension_t newoffset;
	float delta_E,olderr,errval,olderr2,newerr2;
	double dtbl[NLOCS];
	incr_t xx,yy;
	int loc;
	int bestloc;
	double maxdel;
	int n_changed;

#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"optimizing at %d, %d",x,y);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	offset = y*OBJ_ROW_INC(_edp) + x;
	oldbit = *(_hptr+offset);	/* save to check if changed later */
	olderr = *(_eptr+offset);

	/* Delta_E is the difference in energies between having this bit set vs. clear.
	 * If it is >0, then we want to set the pixel to -1, if it is <0 we set the pixel to 1.
	 */

	delta_E = get_delta(x,y);

	/* If the energy is reduced by flipping this pixel, we do it and quit.
	 * If the energy goes up by flipping, then we consider flipping each of
	 * the 8 neighbors, to see if the net causes a reduction.
	 */


	/* We multiply by oldbit because this represents the energy change for a flip.  If the
	 * original value is 1, then we flip if delta_E is >0, if the original value is -1 we flip
	 * if it is <0.  By multiplying by oldbit, it represents the energy decrease for a flip,
	 * regardless of sign.
	 */
	dtbl[ HERE ] = delta_E  * oldbit ;
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"energy gain from flip:  %g",dtbl[HERE]);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	/* now actually flip the state */

	*(_hptr+offset) = oldbit * -1;
	errval = *(_eptr+offset) = *(_hptr+offset) - *(_gptr+offset);
	/* correct filtered error */
	add_impulse(errval-olderr,_fedp,_ffdp,x,y);

	/* now check all of the directions */
	for(loc=1;loc<NLOCS;loc++){
		xx = (incr_t)x + dx_tbl[loc];
		yy = (incr_t)y + dy_tbl[loc];

		/* skip this location if it is off the edge of the image */
		if( yy < 0 || xx < 0 || yy >= (incr_t)OBJ_ROWS(_edp) ||
			xx >= (incr_t)OBJ_COLS(_edp) ){

			dtbl[ loc ] = NO_GO;
			continue;
		}
		newoffset = yy * OBJ_ROW_INC(_edp) + xx;
		/* do nothing if this pixel has the same value as the original pixel */
		if( oldbit == *(_hptr+newoffset) ){
			dtbl[ loc ] = NO_GO;
			continue;
		}
	/* We multiply by oldbit because this represents the energy change for a flip.  If the
	 * original value is 1, then we flip (set to -1) if delta_E is >0;
	 * if the original value is -1 we flip (set to 1) if it is <0.
	 * By multiplying by oldbit, it represents the energy decrease for a flip,
	 * regardless of sign.
	 * But for the neighboring location, we start with the opposite value as the central pixel.
	 * If oldbit is 1, then the neighbor is -1.  We flip (set to 1) if the delta is <0, so we
	 * combine with a minus sign.
	 */
		dtbl[ loc ] = ( delta_E - get_delta(xx,yy) ) * oldbit;
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"energy gain from exchange w %d %d:  %g",xx,yy,dtbl[loc]);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	}
	bestloc=NO_LOC;
	maxdel = 0.0;
	for(loc=0;loc<NLOCS;loc++){
		if( dtbl[loc] > maxdel ){
			maxdel = dtbl[loc];
			bestloc = loc;
		}
	}
	n_changed = 0;
	if( bestloc == NO_LOC ){	/* restore to original state */
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"no improvement from pair flip");
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		/* correct filtered error */
		add_impulse(olderr-errval,_fedp,_ffdp,x,y);

		*(_hptr+offset) = oldbit ;
		*(_eptr+offset) = *(_hptr+offset) - *(_gptr+offset);
	} else if( bestloc != HERE ){
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"improvement from pair flip at %d %d",dx_tbl[bestloc],dy_tbl[bestloc]);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		xx = (incr_t)x + dx_tbl[bestloc];
		yy = (incr_t)y + dy_tbl[bestloc];
		newoffset = yy * OBJ_ROW_INC(_edp) + xx;

		olderr2 = *(_hptr+newoffset) - *(_gptr+newoffset);
		*(_hptr+newoffset) = oldbit ;
		newerr2 = *(_eptr+newoffset) =
			*(_hptr+newoffset) - *(_gptr+newoffset);
		add_impulse(newerr2-olderr2,_fedp,_ffdp,xx,yy);
		n_changed = 2;
	}
	return(n_changed);
}

void set_temp(double temp)
{
	_temp = temp;
}

static void anneal_pixel(dimension_t x,dimension_t y)
{
	dimension_t offset;
	u_char oldbit;
	double delta_E,olderr,errval;
	double oldsos=0.0;	/* init to elim compiler warning */
	double prob,rnum;

	offset = y*OBJ_ROW_INC(_edp) + x;

	if( verbose ){
		if( the_sos == NO_VALUE ){
			the_sos = get_sos(_edp,_fdp);
			sprintf(DEFAULT_MSG_STR,"Initial SOS:  %g",the_sos);
			_prt_msg(DEFAULT_QSP_ARG  DEFAULT_MSG_STR);
		}
		oldsos = the_sos;
		/* subtract contribution of this pt from sos */
		add_to_sos(x,y,_edp,_fdp,-1);
	}


	olderr = *(_eptr+offset);
	oldbit = (u_char) *(_hptr+offset);	/* save to check if changed later */
	delta_E = *(_feptr+offset);		/* blurred error */
	delta_E -= olderr;
	delta_E -= *(_gptr+offset);		/* subtract desired value */

	/*
	 * If the energy is reduced by flipping states,
	 * then always do it.  Otherwise, do it depending
	 * on delta E.
	 */

	if( oldbit == 1 ){
		if( delta_E > 0 ) *(_hptr+offset) = -1;
		else {
			delta_E *= _k / _temp;		/* scale delE */
			prob = get_prob(delta_E);	/* for negative delta_E, prob is near 0 */
			rnum = drand48();
			if( prob > rnum)
				*(_hptr+offset) =  -1;
		}
	} else {
		if( delta_E < 0 ) *(_hptr+offset) = 1;
		else {
			delta_E *= _k / _temp;		/* scale delE */
			prob = get_prob(delta_E);	/* for positive delta_E, prob is near 1 */
			rnum = drand48();
			if( prob < rnum)
				*(_hptr+offset) =  1;
		}
	}

	if( verbose ){
		*(_eptr+offset) = *(_hptr+offset) - *(_gptr+offset);
		add_to_sos(x,y,_edp,_fdp,1);
	}

	if( *(_hptr+offset) == oldbit )	/* no change? */
		return;

	errval = *(_eptr+offset) = *(_hptr+offset) - *(_gptr+offset);

	/* correct filtered error */
	add_impulse(errval-olderr,_fedp,_ffdp,x,y);

	if( verbose ){
		if( verbose ){
			sprintf(DEFAULT_MSG_STR,
			"Pixel at %d,%d\t SOS:  %g\t\tdelta = %g",
				x,y,the_sos,the_sos-oldsos);
			NADVISE(DEFAULT_MSG_STR);
		}
	}
} /* end anneal_pixel */

void insist_pixel(dimension_t x,dimension_t y)
{
	dimension_t offset;
	double sos_on, sos_off;

	offset = y*OBJ_ROW_INC(_edp) + x;

	*(_hptr+offset) = 1;		/* turn pixel on */
					/* calculate error */
	*(_eptr+offset) = *(_hptr+offset) - *(_gptr+offset);
	sos_on = get_sos(_edp,_fdp);

	*(_hptr+offset) = -1;		/* turn pixel off */
					/* calculate error */
	*(_eptr+offset) = *(_hptr+offset) - *(_gptr+offset);
	sos_off = get_sos(_edp,_fdp);

	if( sos_on < sos_off )
		*(_hptr+offset) = 1;		/* turn pixel on */
	else
		*(_hptr+offset) = -1;		/* turn pixel off */
	*(_eptr+offset) = *(_hptr+offset) - *(_gptr+offset);
}

