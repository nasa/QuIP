#include "quip_config.h"

/*
 * program to requantize
 *
 * This is the time domain version (color capable?)
 *
 */

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>		/* drand48() */
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

static void anneal_pixel3d(incr_t *posn);
static double _temp,_k;

#define NO_VALUE	(-1)
static double the_sos=NO_VALUE;
#define NO_PIXELS	0
static dimension_t _npixels=NO_PIXELS;

/* user-supplied images */
static Data_Obj *_gdp=NULL;		/* input grayscale image */
static Data_Obj *_hdp=NULL;		/* output halftone */
static Data_Obj *_fdp=NULL;		/* filter */
static float *_gptr;
static float *_hptr;

/* program-private images */
static Data_Obj *_edp=NULL;		/* error image */
static Data_Obj *_fedp=NULL;		/* doubly filtered error */
static Data_Obj *_ffdp=NULL;		/* double filter */
static float *_eptr;
static float *_feptr;
/* static float *_ffptr; */



#define MAX_DELE	16.0
#define N_BINS		1024
#define BIN_WIDTH	(2*MAX_DELE/N_BINS)


void (*scan_func3d)(dimension_t,Dimension_Set *, incr_t *)=get_3d_random_point;

static void check_posn(incr_t *posn, Data_Obj *dp)
{
	int i;

	for(i=0;i<N_DIMENSIONS;i++){
		if( posn[i] < 0 || posn[i] >= OBJ_TYPE_DIM(dp,i) ){
			sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  posn[%d] = %d",
				i,posn[i]);
			NERROR1(DEFAULT_ERROR_STRING);
		}
	}
}

int _setup_requantize3d(SINGLE_QSP_ARG_DECL)
{
	if( _hdp == NULL ){
		NWARN("output movie not specified");
		return(-1);
	}
	if( _gdp == NULL ){
		NWARN("input movie not specified");
		return(-1);
	}
	if( _fdp == NULL ){
		NWARN("filter not specified");
		return(-1);
	}
	if( OBJ_ROWS(_hdp) != OBJ_ROWS(_gdp)		||
		OBJ_FRAMES(_hdp) != OBJ_FRAMES(_gdp)	||
		OBJ_COMPS(_hdp) != OBJ_COMPS(_gdp)	||
		OBJ_COLS(_hdp) != OBJ_COLS(_gdp) ){

		sprintf(DEFAULT_ERROR_STRING,"setup_requantize3d:  input/output size mismatch, objects %s and %s",
			OBJ_NAME(_hdp),OBJ_NAME(_gdp));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

	/* what is 3d scattered scanning?? */
	/* must it be square?? */

	_npixels = (dimension_t)(OBJ_FRAMES(_hdp) * OBJ_ROWS(_hdp) * OBJ_COLS(_hdp));

	if( _edp != NULL )
		delvec(_edp);
	_edp = make_obj(QSP_ARG  "HT_error",
		OBJ_FRAMES(_hdp),OBJ_ROWS(_hdp),OBJ_COLS(_hdp),1,PREC_FOR_CODE(PREC_SP));
	if( _edp == NULL ){
		NWARN("couldn't create error image");
		return(-1);
	}
	_eptr = (float *) OBJ_DATA_PTR(_edp);

	if( _fedp != NULL )
		delvec(_fedp);
	_fedp = make_obj(QSP_ARG  "HT_ferror",
		OBJ_FRAMES(_hdp),OBJ_ROWS(_hdp),OBJ_COLS(_hdp),1,PREC_FOR_CODE(PREC_SP));
	if( _fedp == NULL ){
		NWARN("couldn't create filtered error image");
		return(-1);
	}
	_feptr = (float *) OBJ_DATA_PTR(_fedp);

	setup_ffilter3d(QSP_ARG  _fdp);

	/* We used to seed the random number generator here
	 * (using the time), but this was eliminated because
	 * it made it impossible to seed the generator by hand.
	 */

	return(0);
}

void set_filter3d(Data_Obj *fdp)
{
	_fdp = fdp;
}

void set_grayscale3d(Data_Obj *gdp)
{
	_gdp = gdp;
	_gptr = (float *) OBJ_DATA_PTR(gdp);
}

void set_halftone3d(Data_Obj *hdp)
{
	_hdp = hdp;
	_hptr = (float *) OBJ_DATA_PTR(hdp);
}


#define EDGE_OFFSET(level)	((incr_t)OBJ_TYPE_DIM(_ffdp,level) - \
                                 (incr_t)OBJ_TYPE_DIM(fdp,level)     )/2

#define BEGIN(level)										\
	for(var[level]=0;var[level]<(incr_t)OBJ_TYPE_DIM(fdp,level);var[level]++){			\
		os[level] = var[level] * OBJ_TYPE_INC(fdp,level);				\
		posn[level] = var[level] + EDGE_OFFSET(level);

void setup_ffilter3d(QSP_ARG_DECL  Data_Obj *fdp)
{
	incr_t var[N_DIMENSIONS];
	incr_t os[N_DIMENSIONS];
	incr_t posn[N_DIMENSIONS];
	incr_t offset;
	float *fptr,val;

	if( _ffdp != NULL )
		delvec(_ffdp);

	_ffdp = make_obj(QSP_ARG   "double_filter",OBJ_FRAMES(fdp)*2-1,
		OBJ_ROWS(fdp)*2-1,OBJ_COLS(fdp)*2-1,1,PREC_FOR_CODE(PREC_SP));

	if( _ffdp == NULL ){
		NWARN("couldn't create double filter image");
		return;
	}

	normalize_filter3d(fdp);
	
	fptr = (float *) OBJ_DATA_PTR(fdp);

	/* Initialize the double filter by convolving the filter w/ itself */

	/* this doesn't work because the sizes are unequal */
	/* convolve3d(_ffdp,fdp,fdp); */

	img_clear3d(_ffdp);

	posn[4] = 0;

	BEGIN(3)
		BEGIN(2)
			BEGIN(1)
				BEGIN(0)
					offset = os[3]+os[2]+os[1]+os[0];
					val = *( fptr + offset );
					add_impulse3d(val,_ffdp,fdp,posn);

				}
			}
		}
	}
}

#define GV_BEGIN(level)									\
											\
	for(var[level]=0;var[level]<(incr_t)OBJ_TYPE_DIM(dp,level);var[level]++){		\
		isum[level]=0.0;							\
		os[level]=var[level]*OBJ_TYPE_INC(dp,level);

#define GV_END(level)									\
											\
		isum[level+1] += isum[level];						\
	}

#ifdef NOT_USED

double get_volume3d(Data_Obj *dp)
{
	double isum[N_DIMENSIONS];
	incr_t var[N_DIMENSIONS], os[N_DIMENSIONS];
	float *ptr;

	ptr = (float *) OBJ_DATA_PTR(dp);
	isum[4]=0.0;				/* have to have an extra slot for these macros to work! */
	GV_BEGIN(3)
		GV_BEGIN(2)
			GV_BEGIN(1)
				GV_BEGIN(0)
					isum[0] += *(ptr+os[0]+os[1]+os[2]+os[3]);
				GV_END(0)
			GV_END(1)
		GV_END(2)
	GV_END(3)
	return(isum[4]);
}
#endif /* NOT_USED */


#define BEG2(level,dp)								\
	for(var[level]=0;var[level]<(incr_t)OBJ_TYPE_DIM(dp,level);var[level]++){	\
		os[level] = var[level] * OBJ_TYPE_INC(dp,level);

#define BEGIN_NF(level,dp)							\
										\
	isos[level]=0;								\
	BEG2(level,dp)

#define END_NF(level)								\
										\
	}									\
	isos[level+1] += isos[level];

void normalize_filter3d(Data_Obj *fdp)
{
	float *fptr;
	double isos[N_DIMENSIONS], length;
	incr_t offset;
	incr_t var[N_DIMENSIONS], os[N_DIMENSIONS];

	fptr = (float *) OBJ_DATA_PTR(fdp);

	isos[4] =0.0;
	BEGIN_NF(3,fdp)
		BEGIN_NF(2,fdp)
			BEGIN_NF(1,fdp)
				BEGIN_NF(0,fdp)
					offset = os[3]+os[2]+os[1]+os[0];
					isos[0] += *(fptr+offset) * *(fptr+offset);
				END_NF(0)
			END_NF(1)
		END_NF(2)
	END_NF(3)

	if( isos[4] <= 0.0 ){
		NWARN("filter has non-positive vector length!?");
		return;
	}
	length = sqrt(isos[4]);
	if( verbose ){
		sprintf(DEFAULT_ERROR_STRING,
			"Normalizing filter by factor %g",length);
		NADVISE(DEFAULT_ERROR_STRING);
	}
	BEG2(3,fdp)
		BEG2(2,fdp)
			BEG2(1,fdp)
				BEG2(0,fdp)
					offset = os[0]+os[1]+os[2]+os[3];
					*(fptr+offset) /= length;
				}
			}
		}
	}
}

void _init_requant3d(SINGLE_QSP_ARG_DECL)
{
	dimension_t offset;
	/* float value; */
	incr_t var[N_DIMENSIONS],os[N_DIMENSIONS];

	/* initialize error image */

	BEG2(3,_edp)
		BEG2(2,_edp)
			BEG2(1,_edp)
				BEG2(0,_edp)
					offset = os[0]+os[1]+os[2]+os[3];
					*(_eptr+offset) = *(_hptr+offset) - *(_gptr+offset);
				}
			}
		}
	}

	convolve3d(QSP_ARG  _fedp,_edp,_ffdp);
}

int scan_requant3d(int ntimes)
{
	dimension_t i;
	int j;
	uint32_t n_changed;
	incr_t posn[N_DIMENSIONS];


	if( _npixels == NO_PIXELS ){
		NWARN("have to tell me which images first!");
		return(0);
	}

	/* scan image, updating y's */

	n_changed=0;
	for(j=0;j<ntimes;j++){
		for(i=0;i<(dimension_t)_npixels;i++){
			(*scan_func3d)(i,OBJ_TYPE_DIMS(_edp),posn);
			n_changed += redo_pixel3d(posn);
		}
	}
	return( n_changed );
}

void scan2_requant3d(int ntimes)
{
	int j;
	dimension_t i;
	incr_t posn[N_DIMENSIONS];


	if( _npixels == NO_PIXELS ){
		NWARN("have to tell me which images first!");
		return;
	}

	/* scan image, updating y's */

	for(j=0;j<ntimes;j++){
		for(i=0;i<(dimension_t)_npixels;i++){
			(*scan_func3d)(i,OBJ_TYPE_DIMS(_edp),posn);
			redo_two_pixels3d(posn);
		}
	}
}

void scan_anneal3d(double temp,int ntimes)
{
	int j;
	incr_t i,posn[N_DIMENSIONS];

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
			(*scan_func3d)(i,OBJ_TYPE_DIMS(_edp),posn);
			anneal_pixel3d(posn);
		}
	}
}

/*
 * This returns the difference between the error with the pixel
 * set and the error with the pixel cleared
 */

float get_delta3d(incr_t *posn)
{
	float delta_E;
	dimension_t feoffset, eoffset, goffset;
	int i;

	feoffset=eoffset=goffset=0;
	for(i=0;i<N_DIMENSIONS;i++){
		feoffset += posn[i] * OBJ_TYPE_INC(_fedp,i);
		eoffset += posn[i] * OBJ_TYPE_INC(_edp,i);
		goffset += posn[i] * OBJ_TYPE_INC(_gdp,i);
	}

	delta_E = *(_feptr+feoffset);		/* doubly filtered error */
	delta_E -= *(_eptr+eoffset);		/* the local error */
	delta_E -= *(_gptr+goffset);		/* subtract desired value */

	return(delta_E);
}


/* redo_pixel3d changes the state of the pixel at x,y,t to reduce the total error
 * Returns 1 if flipped, 0 otherwise.
 */

int redo_pixel3d(incr_t *posn)
{
	dimension_t eoffset,hoffset,goffset;
	float oldbit;
	float delta_E,olderr,errval;
	double oldsos=0.0;	/* init to eliminate warning */
	int i;

#ifdef CAUTIOUS
	check_posn(posn,_edp);
#endif /* CAUTIOUS */

	eoffset=0;
	hoffset=0;
	goffset=0;
	for(i=0;i<N_DIMENSIONS;i++){
		eoffset += posn[i] * OBJ_TYPE_INC(_edp,i);
		hoffset += posn[i] * OBJ_TYPE_INC(_hdp,i);
		goffset += posn[i] * OBJ_TYPE_INC(_gdp,i);
	}

	if( verbose ){
		if( the_sos == NO_VALUE ){
			the_sos = get_sos3d(_edp,_fdp);
			sprintf(DEFAULT_MSG_STR,"Initial SOS:  %g",the_sos);
			_prt_msg(DEFAULT_QSP_ARG  DEFAULT_MSG_STR);
		}
		oldsos = the_sos;
		/* subtract contribution of this pt from sos */
		add_to_sos3d(posn,_edp,_fdp,-1);
	}

	olderr = *(_eptr+eoffset);

	/* save to check if changed later */
	oldbit = *(_hptr+hoffset);

	delta_E = get_delta3d(posn);

	if( delta_E >= 0 ){
		if( verbose ){
			sprintf(DEFAULT_ERROR_STRING,"clearing pixel at %d, %d, %d",posn[1],posn[2],posn[3]);
			NADVISE(DEFAULT_ERROR_STRING);
		}
		*(_hptr+hoffset) = -1;
	} else {
		if( verbose ){
			sprintf(DEFAULT_ERROR_STRING,"setting pixel at %d, %d, %d",posn[1],posn[2],posn[3]);
			NADVISE(DEFAULT_ERROR_STRING);
		}
		*(_hptr+hoffset) =  1;
	}

	errval = *(_hptr+hoffset) - *(_gptr+goffset);
	*(_eptr+eoffset) = errval ;

	if( verbose )
		add_to_sos3d(posn,_edp,_fdp,1);

	if( *(_hptr+hoffset) == oldbit )	/* no change? */
		return(0);

	/* correct filtered error */
	add_impulse3d(errval-olderr,_fedp,_ffdp,posn);

	if( verbose ){
		sprintf(DEFAULT_MSG_STR,
			"Pixel at %d,%d,%d\t SOS:  %g\t\tdelta = %g",
			posn[1],posn[2],posn[3],the_sos,the_sos-oldsos);
		_prt_msg(DEFAULT_QSP_ARG  DEFAULT_MSG_STR);
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"total sse: %g",get_sos3d(_edp,_fdp));
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

#define NLOCS	27
#define NO_LOC	(-1)

static incr_t dx_tbl[NLOCS]={ 0, -1, 0, 1, 1,  1,  0, -1, -1,
							0, -1, 0, 1, 1,  1,  0, -1, -1,
											0, -1, 0, 1, 1,  1,  0, -1, -1 };
static incr_t dy_tbl[NLOCS]={ 0,  1, 1, 1, 0, -1, -1, -1,  0,
							0,  1, 1, 1, 0, -1, -1, -1,  0,
											0,  1, 1, 1, 0, -1, -1, -1,  0 };
static incr_t dt_tbl[NLOCS]={ 0,  0, 0, 0, 0,  0,  0,  0,  0,
							1,  1, 1, 1, 1,  1,  1,  1,  1,
											-1,-1,-1,-1,-1, -1, -1, -1, -1 };

#define NO_GO	(-10000.0)

#define NEW_OFFSETS									\
		new_eoffset = 0;							\
		new_hoffset = 0;							\
		new_goffset = 0;							\
		for(i=0;i<N_DIMENSIONS;i++){						\
			new_eoffset += new_posn[i] * OBJ_TYPE_INC(_edp,i) ;		\
			new_hoffset += new_posn[i] * OBJ_TYPE_INC(_edp,i) ;		\
			new_goffset += new_posn[i] * OBJ_TYPE_INC(_edp,i) ;		\
		}

/* redo_two_pixels
 * called by "migrate" menu function, this trys flipping a pixel and one of it's neighbors.
 * This is useful because we might obtain lower energy by exchanging two pixels, but with
 * an energy "hump" when only one is flipped...   This routine allows "tunneling" under the
 * energy barrier.
 */

void redo_two_pixels3d(incr_t *posn)
{
	dimension_t eoffset;
	dimension_t goffset;
	dimension_t hoffset;
	incr_t new_posn[N_DIMENSIONS];
	float oldbit;
	dimension_t new_eoffset;
	dimension_t new_goffset;
	dimension_t new_hoffset;
	float delta_E,olderr,errval,olderr2,newerr2;
	double dtbl[NLOCS];
	int loc;
	int bestloc;
	double maxdel;
	int i;

#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"optimizing at %d, %d, %d",posn[1],posn[2],posn[3]);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

#ifdef CAUTIOUS
	check_posn(posn,_edp);
#endif /* CAUTIOUS */

	eoffset = 0;
	goffset = 0;
	hoffset = 0;
	for(i=0;i<N_DIMENSIONS;i++){
		eoffset += posn[i] * OBJ_TYPE_INC(_edp,i);
		hoffset += posn[i] * OBJ_TYPE_INC(_hdp,i);
		goffset += posn[i] * OBJ_TYPE_INC(_gdp,i);
	}

	oldbit = *(_hptr+hoffset);	/* save to check if changed later */
	olderr = *(_eptr+eoffset);

	/* Delta_E is the difference in energies between having this bit set vs. clear.
	 * If it is >0, then we want to set the pixel to -1, if it is <0 we set the pixel to 1.
	 */

	delta_E = get_delta3d(posn);

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

	*(_hptr+hoffset) = oldbit * -1;
	errval = *(_eptr+eoffset) = *(_hptr+hoffset) - *(_gptr+goffset);
	/* correct filtered error */
	add_impulse3d(errval-olderr,_fedp,_ffdp,posn);

	/* now check all of the directions */
	for(loc=1;loc<NLOCS;loc++){
		/* new_posn[0] = posn[0] + dc_tbl[loc]; */
		new_posn[0] = 0;
		new_posn[4] = 0;
		new_posn[1] = posn[1] + dx_tbl[loc];
		new_posn[2] = posn[2] + dy_tbl[loc];
		new_posn[3] = posn[3] + dt_tbl[loc];

		/* skip this location if it is off the edge of the image */
		if(	new_posn[3] < 0				||
			new_posn[2] < 0				||
			new_posn[1] < 0				||
			new_posn[3] >= OBJ_TYPE_DIM(_edp,3)	||
			new_posn[2] >= OBJ_TYPE_DIM(_edp,2)	||
			new_posn[1] >= OBJ_TYPE_DIM(_edp,1)	  ){

			dtbl[ loc ] = NO_GO;
			continue;
		}
		NEW_OFFSETS

		/* do nothing if this pixel has the same value as the original pixel */
		if( oldbit == *(_hptr+new_hoffset) ){
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
		dtbl[ loc ] = ( delta_E - get_delta3d(new_posn) ) * oldbit;
	}
	bestloc=NO_LOC;
	maxdel = 0.0;
	for(loc=0;loc<NLOCS;loc++){
		if( dtbl[loc] > maxdel ){
			maxdel = dtbl[loc];
			bestloc = loc;
		}
	}
	if( bestloc == NO_LOC ){	/* restore to original state */
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"no improvement from pair flip");
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		/* correct filtered error */
		add_impulse3d(olderr-errval,_fedp,_ffdp,posn);

		*(_hptr+hoffset) = oldbit ;
		*(_eptr+eoffset) = *(_hptr+hoffset) - *(_gptr+goffset);
	} else if( bestloc != HERE ){
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"improvement from pair flip at %d %d",dx_tbl[bestloc],dy_tbl[bestloc]);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		new_posn[1] = posn[1] + dx_tbl[bestloc];
		new_posn[2] = posn[2] + dy_tbl[bestloc];
		new_posn[3] = posn[3] + dt_tbl[bestloc];

		NEW_OFFSETS

		olderr2 = *(_hptr+new_hoffset) - *(_gptr+new_goffset);
		*(_hptr+new_hoffset) = oldbit ;
		newerr2 = *(_eptr+new_eoffset) =
			*(_hptr+new_hoffset) - *(_gptr+new_goffset);
		add_impulse3d(newerr2-olderr2,_fedp,_ffdp,new_posn);
	}
}

void set_temp3d(double temp)
{
	_temp = temp;
}

static void anneal_pixel3d(incr_t *posn)
{
	incr_t goffset,hoffset,eoffset,feoffset;
	u_char oldbit;
	double delta_E,olderr,errval;
	double oldsos=0.0;	/* init to elim compiler warning */
	double prob,rnum;
	int i;

	if( verbose ){
		if( the_sos == NO_VALUE ){
			the_sos = get_sos3d(_edp,_fdp);
			sprintf(DEFAULT_MSG_STR,"Initial SOS:  %g",the_sos);
			_prt_msg(DEFAULT_QSP_ARG  DEFAULT_MSG_STR);
		}
		oldsos = the_sos;
		/* subtract contribution of this pt from sos */
		add_to_sos3d(posn,_edp,_fdp,-1);
	}

	eoffset=hoffset=goffset=feoffset=0;
	for(i=0;i<N_DIMENSIONS;i++){
		eoffset += posn[i] * OBJ_TYPE_INC(_edp,i);
		hoffset += posn[i] * OBJ_TYPE_INC(_hdp,i);
		goffset += posn[i] * OBJ_TYPE_INC(_gdp,i);
		feoffset += posn[i] * OBJ_TYPE_INC(_fedp,i);
	}

	olderr = *(_eptr+eoffset);
	oldbit = (u_char) *(_hptr+hoffset);	/* save to check if changed later */
	delta_E = *(_feptr+feoffset);		/* blurred error */
	delta_E -= olderr;
	delta_E -= *(_gptr+goffset);		/* subtract desired value */

	/*
	 * If the energy is reduced by flipping states,
	 * then always do it.  Otherwise, do it depending
	 * on delta E.
	 */

	if( oldbit == 1 ){
		if( delta_E > 0 ) *(_hptr+hoffset) = -1;
		else {
			delta_E *= _k / _temp;		/* scale delE */
			prob = get_prob(delta_E);	/* for negative delta_E, prob is near 0 */
			rnum = drand48();
			if( prob > rnum)
				*(_hptr+hoffset) =  -1;
		}
	} else {
		if( delta_E < 0 ) *(_hptr+hoffset) = 1;
		else {
			delta_E *= _k / _temp;		/* scale delE */
			prob = get_prob(delta_E);	/* for positive delta_E, prob is near 1 */
			rnum = drand48();
			if( prob < rnum)
				*(_hptr+hoffset) =  1;
		}
	}

	if( verbose ){
		*(_eptr+eoffset) = *(_hptr+hoffset) - *(_gptr+goffset);
		add_to_sos3d(posn,_edp,_fdp,1);
	}

	if( *(_hptr+hoffset) == oldbit )	/* no change? */
		return;

	errval = *(_eptr+eoffset) = *(_hptr+hoffset) - *(_gptr+goffset);

	/* correct filtered error */
	add_impulse3d(errval-olderr,_fedp,_ffdp,posn);

	if( verbose ){
		if( verbose ){
			sprintf(DEFAULT_MSG_STR,
			"Pixel at %d,%d,%d\t SOS:  %g\t\tdelta = %g",
				posn[1],posn[2],posn[3],the_sos,the_sos-oldsos);
			NADVISE(DEFAULT_MSG_STR);
		}
	}
} /* end anneal_pixel3d */

#ifdef NOT_USED
void insist_pixel3d(dimension_t *posn)
{
	dimension_t hoffset,goffset,eoffset;
	double sos_on, sos_off;
	int i;

	goffset=eoffset=hoffset=0;
	for(i=0;i<N_DIMENSIONS;i++){
		hoffset += posn[i] * OBJ_TYPE_INC(_hdp,i);
		eoffset += posn[i] * OBJ_TYPE_INC(_edp,i);
		goffset += posn[i] * OBJ_TYPE_INC(_gdp,i);
	}

	*(_hptr+hoffset) = 1;		/* turn pixel on */
					/* calculate error */
	*(_eptr+eoffset) = *(_hptr+hoffset) - *(_gptr+goffset);
	sos_on = get_sos3d(_edp,_fdp);

	*(_hptr+hoffset) = -1;		/* turn pixel off */
					/* calculate error */
	*(_eptr+eoffset) = *(_hptr+hoffset) - *(_gptr+goffset);
	sos_off = get_sos3d(_edp,_fdp);

	if( sos_on < sos_off )
		*(_hptr+hoffset) = 1;		/* turn pixel on */
	else
		*(_hptr+hoffset) = -1;		/* turn pixel off */
	*(_eptr+eoffset) = *(_hptr+hoffset) - *(_gptr+goffset);
}
#endif /* NOT_USED */

static int power_of_two(uint32_t n)
{
	do {
		if( n == 1 ) return(1);
		if( n & 1 ) return(0);
		n >>= 1;
	} while( n > 0 );
	/* should never be reached */
	return(0);
}

void get_3d_scattered_point(dimension_t n,Dimension_Set *dsp,posn_t *posn)
{
	uint32_t src_bit,dst_bit;
	int i;
	uint32_t result;
	uint32_t tot_siz,s;
	int32_t nbits;

	/* We want to bit reverse, but we need to know how
	 * many bits we have...
	 */

	tot_siz = DIMENSION(dsp,0) * DIMENSION(dsp,1) * DIMENSION(dsp,2) * DIMENSION(dsp,3) * DIMENSION(dsp,4);
	if( ! power_of_two(tot_siz) ){
		sprintf(DEFAULT_ERROR_STRING,"get_3d_scattered_point:  total size %d is not a power of two",
			tot_siz);
		NERROR1(DEFAULT_ERROR_STRING);
		return;
	}
    if( tot_siz <= 1 ){
        sprintf(DEFAULT_ERROR_STRING,"get_3d_scattered_point:  total size %d is <= 1!?",tot_siz);
        NERROR1(DEFAULT_ERROR_STRING);
	    return;
    }

	nbits = 0;
	s=tot_siz;
	while( s > 1 ){
		nbits ++;
		s >>= 1;
	}

	src_bit=1;
	dst_bit=1<<(nbits-1);
	result=0;
	for(i=0;i<nbits;i++){
		if( n & src_bit )
			result |= dst_bit;
		src_bit <<= 1;
		dst_bit >>= 1;
	}
	get_3d_raster_point(result,dsp,posn);
	posn[0]=0;
	posn[4]=0;
}

void get_3d_raster_point(dimension_t n,Dimension_Set *dsp,posn_t *posn)
{
	posn[3] = n / (DIMENSION(dsp,0)*DIMENSION(dsp,1)*DIMENSION(dsp,2));
	posn[2] = ( n / (DIMENSION(dsp,1)*DIMENSION(dsp,0)) ) % DIMENSION(dsp,2);
	posn[1] = ( n / DIMENSION(dsp,0) ) % DIMENSION(dsp,1);
	/*
	posn[0] = n % DIMENSION(dsp,0);
	*/
	posn[0]=0;
	posn[4]=0;
}

void get_3d_random_point(dimension_t n,Dimension_Set *dsp,posn_t *posn)
{
	int i;

	for(i=0;i<N_DIMENSIONS;i++){
		posn[i] = (dimension_t)(drand48() * (double)DIMENSION(dsp,i));
		if( posn[i] == DIMENSION(dsp,i) ) posn[i]=0;
	}
	posn[0]=0;
	posn[4]=0;
}

#define FE_BEGIN(level)								\
												\
	var[level] = OBJ_TYPE_DIM(fdp,level);							\
	while( var[level] -- ){									\
		new_posn[level] = posn[level] + var[level] - OBJ_TYPE_DIM(fdp,level)/2;	\
		FE_BOUNDS(level)

#ifdef NOWRAP

#define FE_BOUNDS(level)								\
											\
		if( new_posn[level] < 0 ) continue;					\
		else if( new_posn[level] >= OBJ_TYPE_DIM(edp,level) ) continue;
#else /* ! NOWRAP */

#define FE_BOUNDS(level)												\
															\
		while( new_posn[level] < 0 ) new_posn[level] += OBJ_TYPE_DIM(edp,level);				\
		while( new_posn[level] >= OBJ_TYPE_DIM(edp,level) ) new_posn[level] -= OBJ_TYPE_DIM(edp,level);

#endif /* ! NOWRAP */

#define NEW_FE_OFFSETS									\
		eoffset = 0;							\
		foffset = 0;							\
		for(i=0;i<N_DIMENSIONS;i++){						\
			eoffset += new_posn[i] * OBJ_TYPE_INC(_edp,i) ;		\
			foffset += new_posn[i] * OBJ_TYPE_INC(_fdp,i) ;		\
		}


/* get the filtered error at one point.
 * We sum the product of the filter impulse response with a subimage
 * of the input centered at x,y,t.
 */

static double get_ferror3d(Data_Obj *edp,Data_Obj *fdp,dimension_t *posn)
{
	double err;
	float *eptr, *fptr;
	incr_t var[N_DIMENSIONS];
	dimension_t new_posn[N_DIMENSIONS];
	incr_t eoffset, foffset;
	int i;

	/*
	 * the filter should have odd sizes;
	 * the center is at the middle of the image
	 */

	err=0.0;
	eptr = (float *)OBJ_DATA_PTR(edp);
	fptr = (float *)OBJ_DATA_PTR(fdp);
	new_posn[4]=0;

	FE_BEGIN(3)
		FE_BEGIN(2)
			FE_BEGIN(1)
				FE_BEGIN(0)
					NEW_FE_OFFSETS
					err += *(eptr+eoffset) * *(fptr+foffset);
				}
			}
		}
	}

	return(err);
}

#define GS_BEGIN(level)								\
										\
	isos[level]=0;								\
	for(var[level]=0;var[level]<(incr_t)OBJ_TYPE_DIM(edp,level);var[level]++){

#define GS_END(level)								\
										\
	}									\
	isos[level+1] += isos[level];
/* really the mean square, after we divide by the number of pixels! */

double get_sos3d(Data_Obj *edp,Data_Obj *fdp)		/* get the total sq'd error */
{
	incr_t var[N_DIMENSIONS];
	double isos[N_DIMENSIONS], err;

	isos[4] = 0.0;
	var[4]=0;
	GS_BEGIN(3)
		GS_BEGIN(2)
			GS_BEGIN(1)
				GS_BEGIN(0)
					err = get_ferror3d(edp,fdp,(dimension_t *)var);
					isos[0] += err*err;
				GS_END(0)
			GS_END(1)
		GS_END(2)
	GS_END(3)
			
	/* normalize by number of pixels */
	isos[4] /= OBJ_ROWS(edp)*OBJ_COLS(edp)*OBJ_COMPS(edp)*OBJ_FRAMES(edp);
	return(isos[4]);
}

/* add_to_sos
 * This version recalculates the sum of squared filtered error, with an adjustment
 * of strength factor at x,y.  The filtered error is not stored, we call get_ferror
 * over the support of the filter.  (get_ferror does [inefficient] space domain convolution.)
 * factor is always 1 or -1, so we use this routine to exclude or include the effect of
 * a particular pixel.  Typically, we remove the effect of a pixel, then recalculate for a new
 * value.
 */

#ifdef NOWRAP

#define CHECK_BOUNDS(level)									\
												\
		if( p2[level] >= 0 && p2[level] < (incr_t)OBJ_TYPE_DIM(edp,level) ){

#else /* ! NOWRAP */

#define CHECK_BOUNDS(level)									\
												\
		while( p2[level] < 0 ) p2[level] += OBJ_TYPE_DIM(edp,level);			\
		while( p2[level] >= (incr_t)OBJ_TYPE_DIM(edp,level) )					\
			p2[level] -= OBJ_TYPE_DIM(edp,level);					\
		{

#endif /* ! NOWRAP */

#define AS_BEGIN(level)										\
												\
	for(var[level]=0;var[level]<(incr_t)OBJ_TYPE_DIM(fdp,level);var[level]++){			\
		p2[level] = posn[level] + var[level] - OBJ_TYPE_DIM(fdp,level)/2;		\
		CHECK_BOUNDS(level)

#define AS_END			}}

double add_to_sos3d(posn_t *posn,Data_Obj *edp,Data_Obj *fdp,int factor)
{
	double err,adj;
	incr_t var[N_DIMENSIONS];
	incr_t p2[N_DIMENSIONS];

	/*
	if( the_sos == NO_VALUE )
		the_sos = get_sos3d(edp,fdp);
	*/

	adj =0.0;
	p2[4]=0;
	AS_BEGIN(3)
		AS_BEGIN(2)
			AS_BEGIN(1)
				AS_BEGIN(0)
					err = get_ferror3d(edp,fdp,(dimension_t *)p2);
					adj += err*err;
				AS_END
			AS_END
		AS_END
	AS_END

	/* normalize by number of pixels */
	if( factor == 1 )
		adj /= (OBJ_COLS(edp) * OBJ_ROWS(edp) * OBJ_FRAMES(edp) * OBJ_COMPS(edp));
	else if( factor == -1 )
		adj /= - (OBJ_COLS(edp) * OBJ_ROWS(edp) * OBJ_FRAMES(edp) * OBJ_COMPS(edp));
#ifdef CAUTIOUS
	else {
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  add_to_sos:  factor (%d) is not 1 or -1 !?",factor);
		NWARN(DEFAULT_ERROR_STRING);
		return(0.0);
	}
#endif /* CAUTIOUS */
	/* the_sos += adj; */
	return(adj);
}
	


