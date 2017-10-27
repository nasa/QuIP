#include "quip_config.h"

#define INLINE inline	// BUG - use configure to determine if inline can be used?

/*
 * program to requantize
 *
 * modified 8-22 to monotonicly reduce the total squared error
 * modified 11-19-91 to handle color images
 */

#include "quip_prot.h"

#define ERROR_RETURN	(-1)

#define ON_LEVEL	1
#define OFF_LEVEL	(-1)

#define RED_BIT		1
#define GREEN_BIT	2
#define BLUE_BIT	4

#include <math.h>
#include <time.h>	/* time() */
#include <stdlib.h>	/* srand48 */
#include "data_obj.h"
#include "vec_util.h"


static float act_lum[8], act_rg[8], act_by[8];
static int act_ready=0;
static int n_pixels_changed;

#define NO_VALUE	(-1)
static double the_sos=NO_VALUE;
static double lum_sos=NO_VALUE;
static double rg_sos=NO_VALUE;
static double by_sos=NO_VALUE;
#define NO_PIXELS	0
static dimension_t _npixels=NO_PIXELS;
static int cspread_inited=0;

/* user-supplied images */
static Data_Obj *deslum_dp=NULL;		/* desired luminance image */
static Data_Obj *desrg_dp=NULL;		/* desired red-green image */
static Data_Obj *desby_dp=NULL;		/* desired blue-yellow image */
static Data_Obj *halftone_dp=NULL;		/* output composite halftone */
static Data_Obj *lum_filt_dp=NULL;
static Data_Obj *rg_filt_dp=NULL;
static Data_Obj *by_filt_dp=NULL;
static Data_Obj *rgb2opp_mat;			/* rgb to opponent matrix */

/* program private */
static Data_Obj *lum_ferr_dp=NULL;		/* filtered error images */
static Data_Obj *rg_ferr_dp=NULL;
static Data_Obj *by_ferr_dp=NULL;

static Data_Obj *lum_err_dp=NULL;		/* error images */
static Data_Obj *rg_err_dp=NULL;
static Data_Obj *by_err_dp=NULL;

/* local prototypes */
static int _setup_clr_requantize(SINGLE_QSP_ARG_DECL);
#define setup_clr_requantize()	_setup_clr_requantize(SINGLE_QSP_ARG)
static void init_ferror(void);
static double rgb_sos(void);
static void adjust_rgb_sos(dimension_t x,dimension_t y,double factor);
static void adjust_rgb_ferror(dimension_t x,dimension_t y,double factor);


static INLINE int get_ht( dimension_t x, dimension_t y )
{
	char *bptr;

	bptr=(char *)OBJ_DATA_PTR(halftone_dp);
	bptr+=(x+y*OBJ_ROW_INC(halftone_dp))*OBJ_PXL_INC(halftone_dp);
	return(*bptr);
}

void filter_error(Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *filtdp)
{
	dimension_t i,j;
	float *toptr;
	int row_os, os;

	toptr = (float *) OBJ_DATA_PTR(dpto);

	for(j=0;j<OBJ_ROWS(dpto);j++){
		row_os = j*OBJ_ROW_INC(dpto);
		for(i=0;i<OBJ_COLS(dpto);i++){
			os = ( row_os + i ) * OBJ_PXL_INC(dpto);
			*(toptr+os) = (float) get_ferror(dpfr,filtdp,i,j);
		}
	}
}
			
static void init_ferror(void)
{
	filter_error(lum_ferr_dp,lum_err_dp,lum_filt_dp);
	filter_error(rg_ferr_dp,rg_err_dp,rg_filt_dp);
	filter_error(by_ferr_dp,by_err_dp,by_filt_dp);
}

/*
 * update the filtered error image, either removing the contribution from x,y
 * or adding it in, as factor is -1 or 1
 */

void adjust_ferror(
	Data_Obj *fedp,
	Data_Obj *edp,
	Data_Obj *fdp,
	dimension_t x,
	dimension_t y,
	double factor)
{
	/* we only have to update a filter-sized neighborhood about x,y */

	float value, *fptr;

	/* pick out the value of the error */
	fptr = (float *) OBJ_DATA_PTR(edp);
	value = fptr[ ( y * OBJ_ROW_INC(edp) + x ) * OBJ_PXL_INC(edp) ];
/*
sprintf(ERROR_STRING,"adjust_ferror %d %d:  err = %g",x,y,value);
advise(ERROR_STRING);
*/
	add_impulse(factor*value,fedp,fdp,x,y);
}

void adjust_rgb_ferror(dimension_t x,dimension_t y,double factor)
{
/*
sprintf(ERROR_STRING,"adjust_rgb_ferror %d %d:  factor = %g",x,y,factor);
advise(ERROR_STRING);
*/
	adjust_ferror(lum_ferr_dp,lum_err_dp,lum_filt_dp,x,y,factor);
	adjust_ferror(rg_ferr_dp,rg_err_dp,rg_filt_dp,x,y,factor);
	adjust_ferror(by_ferr_dp,by_err_dp,by_filt_dp,x,y,factor);
}

#ifdef FOOBAR
static double get_sos(Data_Obj *fedp)		/* get the total sq'd error */
{
	long i,j;
	double sos, rowsos;
	float *ptr,v;

	sos = 0.0;
	ptr = (float *) OBJ_DATA_PTR(fedp);
	for(j=0;j<OBJ_ROWS(fedp);j++){
		rowsos=0.0;
		for(i=0;i<OBJ_COLS(fedp);i++){
			v=ptr[(j*OBJ_COLS(fedp)+i)*OBJ_PXL_INC(fedp)];
			rowsos += v*v;
		}
		sos += rowsos;
	}
	/* normalize by number of pixels */
	sos /= OBJ_ROWS(fedp)*OBJ_COLS(fedp);
	return(sos);
}
#endif /* FOOBAR */

#define act_init()	_act_init(SINGLE_QSP_ARG)

static void _act_init(SINGLE_QSP_ARG_DECL)
{
	int mask,r,g,b;
	float *fptr;

/*
sprintf(ERROR_STRING,"BEGIN act_init, matrix = %s",OBJ_NAME(rgb2opp_mat));
advise(ERROR_STRING);
*/
	if( rgb2opp_mat == NULL ){
		NWARN("transformation matrix not defined");
		return;
	}
	if( ! IS_CONTIGUOUS(rgb2opp_mat) ){
		sprintf(DEFAULT_ERROR_STRING,"Matrix %s must be contiguous",OBJ_NAME(rgb2opp_mat));
		NERROR1(DEFAULT_ERROR_STRING);
	}
	if( OBJ_PREC(rgb2opp_mat) != PREC_SP ){
		sprintf(DEFAULT_ERROR_STRING,"Matrix %s (%s) must have %s precision",
			OBJ_NAME(rgb2opp_mat),PREC_NAME(OBJ_PREC_PTR(rgb2opp_mat)),
			PREC_NAME(PREC_FOR_CODE(PREC_SP)));
		NERROR1(DEFAULT_ERROR_STRING);
	}
	if( OBJ_COMPS(rgb2opp_mat) != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"Matrix %s (%d) must have component dimension = 1",
			OBJ_NAME(rgb2opp_mat),OBJ_COMPS(rgb2opp_mat));
		NERROR1(DEFAULT_ERROR_STRING);
	}
	if( OBJ_PXL_INC(rgb2opp_mat) != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"Matrix %s (%d) must have pixel increment = 1",
			OBJ_NAME(rgb2opp_mat),OBJ_PXL_INC(rgb2opp_mat));
		NERROR1(DEFAULT_ERROR_STRING);
	}
	if( OBJ_COLS(rgb2opp_mat) != 3 || OBJ_ROWS(rgb2opp_mat) != 3 ){
		sprintf(DEFAULT_ERROR_STRING,"Matrix %s (%d x %d) must be 3x3",
			OBJ_NAME(rgb2opp_mat),OBJ_ROWS(rgb2opp_mat),OBJ_COLS(rgb2opp_mat));
		NERROR1(DEFAULT_ERROR_STRING);
	}
	fptr = (float *) OBJ_DATA_PTR(rgb2opp_mat);
	for(mask=0;mask<8;mask++){
		if( mask & RED_BIT ) r = ON_LEVEL;
		else r = OFF_LEVEL;
		if( mask & GREEN_BIT ) g = ON_LEVEL;
		else g = OFF_LEVEL;
		if( mask & BLUE_BIT ) b = ON_LEVEL;
		else b = OFF_LEVEL;

		act_lum[mask]  = *(fptr  ) * r;
		act_lum[mask] += *(fptr+1) * g;
		act_lum[mask] += *(fptr+2) * b;
		act_rg[mask]   = *(fptr+3) * r;
		act_rg[mask]  += *(fptr+4) * g;
		act_rg[mask]  += *(fptr+5) * b;
		act_by[mask]   = *(fptr+6) * r;
		act_by[mask]  += *(fptr+7) * g;
		act_by[mask]  += *(fptr+8) * b;
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"act_init:\t%d \t\t%g %g %g",mask,act_lum[mask],act_rg[mask],act_by[mask]);
advise(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	}
	act_ready=1;
}

/* recalc_error fetches the desired values, and updates the error images based
 * on the value of mask (which encodes 3 bits of rgb, and is used to index tables
 * of corresponding luma and chroma.
 * Called from:
 *	calc_errors
 *	init_clr_requant
 *	try_it
 */

#define recalc_error(x,y,m)	_recalc_error(QSP_ARG  x,y,m)

static void _recalc_error(QSP_ARG_DECL  dimension_t x,dimension_t y,int mask)
{
	double lum, rg, by;
	double deslum, desrg, desby;
	float *fptr;


	/* assume the matrix is 3x3 */

	if( !act_ready ) act_init();

	lum = act_lum[mask];
	rg = act_rg[mask];
	by = act_by[mask];

	fptr = (float *) OBJ_DATA_PTR(deslum_dp);
	deslum = *(fptr + (x+y*OBJ_ROW_INC(deslum_dp))*OBJ_PXL_INC(deslum_dp) );
	fptr = (float *) OBJ_DATA_PTR(lum_err_dp);
	*(fptr + (x+y*OBJ_ROW_INC(lum_err_dp))*OBJ_PXL_INC(lum_err_dp) ) = (float)(lum - deslum);

	fptr = (float *) OBJ_DATA_PTR(desrg_dp);
	desrg = *(fptr + (x+y*OBJ_ROW_INC(desrg_dp))*OBJ_PXL_INC(desrg_dp) );
	fptr = (float *) OBJ_DATA_PTR(rg_err_dp);
	*(fptr + (x+y*OBJ_ROW_INC(rg_err_dp))*OBJ_PXL_INC(rg_err_dp) ) = (float)(rg - desrg);

	fptr = (float *) OBJ_DATA_PTR(desby_dp);
	desby = *(fptr + (x+y*OBJ_ROW_INC(desby_dp))*OBJ_PXL_INC(desby_dp) );
	fptr = (float *) OBJ_DATA_PTR(by_err_dp);
	*(fptr + (x+y*OBJ_ROW_INC(by_err_dp))*OBJ_PXL_INC(by_err_dp) ) = (float)(by - desby);
#ifdef QUIP_DEBUG
/*
if( debug & spread_debug ){
sprintf(ERROR_STRING,"recalc_error %d %d  %d:  act %f %f %f",
x,y,mask,lum,rg,by);
advise(ERROR_STRING);
sprintf(ERROR_STRING,"recalc_error %d %d  %d:  des %f %f %f",
x,y,mask,deslum,desrg,desby);
advise(ERROR_STRING);
}
*/
#endif /* QUIP_DEBUG */

}

#define calc_errors()	_calc_errors(SINGLE_QSP_ARG)

static void _calc_errors(SINGLE_QSP_ARG_DECL)
{
	dimension_t x,y;
	int mask;
	char *bptr;

	/*
	advise("calculating initial error image");
	*/
	for(y=0;y<OBJ_ROWS(halftone_dp);y++){
		for(x=0;x<OBJ_COLS(halftone_dp);x++){
			bptr = (char *) OBJ_DATA_PTR(halftone_dp);
			bptr += (x + y*OBJ_ROW_INC(halftone_dp))
				*OBJ_PXL_INC(halftone_dp);
			mask = (*bptr);

			recalc_error(x,y,mask);
		}
	}
	/*
	advise("finished calculating initial error image");
	*/
}


static int _setup_clr_requantize(SINGLE_QSP_ARG_DECL)
{
	long tvec;
	Precision *prec_p;

	if( halftone_dp == NULL ){
		NWARN("output image not specified");
		return(ERROR_RETURN);
	}
	if( deslum_dp == NULL || desrg_dp == NULL || desby_dp == NULL ){
		NWARN("input images not specified");
		return(ERROR_RETURN);
	}

	if( lum_filt_dp == NULL ){
		NWARN("filters not specified");
		return(ERROR_RETURN);
	}

	if( OBJ_ROWS(halftone_dp) != OBJ_ROWS(deslum_dp) ||
		OBJ_COLS(halftone_dp) != OBJ_COLS(deslum_dp) ){
		NWARN("input/output size mismatch");
		return(ERROR_RETURN);
	}

	if( (OBJ_ROWS(deslum_dp) != OBJ_COLS(deslum_dp)) &&
		( scan_func == get_xy_scattered_point) ){

		NWARN("input image must be square for scattered scanning");
		return(ERROR_RETURN);
	}

	_npixels = OBJ_ROWS(halftone_dp) * OBJ_COLS(halftone_dp);

	if( lum_err_dp != NULL ){
		delvec(lum_err_dp);
		delvec(rg_err_dp);
		delvec(by_err_dp);
		delvec(lum_ferr_dp);
		delvec(rg_ferr_dp);
		delvec(by_ferr_dp);
	}
	prec_p = PREC_FOR_CODE(PREC_SP);

	lum_ferr_dp = mk_img("lum_ferror",
		OBJ_ROWS(halftone_dp),OBJ_COLS(halftone_dp),1,prec_p);
	rg_ferr_dp = mk_img("rg_ferror",
		OBJ_ROWS(halftone_dp),OBJ_COLS(halftone_dp),1,prec_p);
	by_ferr_dp = mk_img("by_ferror",
		OBJ_ROWS(halftone_dp),OBJ_COLS(halftone_dp),1,prec_p);
	if( lum_ferr_dp == NULL || rg_ferr_dp == NULL || by_ferr_dp == NULL ){
		NWARN("couldn't create filtered error images");
		return(ERROR_RETURN);
	}

	lum_err_dp = mk_img("lum_error",
		OBJ_ROWS(halftone_dp),OBJ_COLS(halftone_dp),1,prec_p);
	rg_err_dp = mk_img("rg_error",
		OBJ_ROWS(halftone_dp),OBJ_COLS(halftone_dp),1,prec_p);
	by_err_dp = mk_img("by_error",
		OBJ_ROWS(halftone_dp),OBJ_COLS(halftone_dp),1,prec_p);
	if( lum_err_dp == NULL || rg_err_dp == NULL || by_err_dp == NULL ){
		NWARN("couldn't create error images");
		return(ERROR_RETURN);
	}

	/* now need to initialize the errors ! */
	calc_errors();

	time(&tvec);
	tvec &= 4095;
	srand48(tvec);

	return(0);
}

/*
 * Some things need only be done once (creating images)
 * while others should be redone before each run (computing error)
 *
 * Unfortunately, it is not clear which of these things this routine does...
 */

COMMAND_FUNC( init_clr_requant )
{
	dimension_t x,y;
	int mask;

	if( setup_clr_requantize() == ERROR_RETURN ) return;

	/* initialize error images */

	if( halftone_dp == NULL ){
		NWARN("init_clr_requant:  no halftone image specified");
		return;
	}
	/*
	advise("init_clr_requant:  calculating error");
	*/
	for(y=0;y<OBJ_ROWS(halftone_dp);y++){
		for(x=0;x<OBJ_COLS(halftone_dp);x++){
			mask = get_ht( x, y );
			recalc_error(x,y,mask);
		}
	}
	/* filter the error */
	/* advise("filtering error..."); */
	init_ferror();
	the_sos = NO_VALUE;
	lum_sos = NO_VALUE;
	rg_sos  = NO_VALUE;
	by_sos  = NO_VALUE;

	cspread_inited=1;
}

static double rgb_sos(void)
{
	/*
	lum_sos = get_sos( lum_ferr_dp );
	rg_sos = get_sos( rg_ferr_dp );
	by_sos = get_sos( by_ferr_dp );
	*/
	lum_sos = get_sos( lum_err_dp, lum_filt_dp  );
	rg_sos = get_sos( rg_err_dp, rg_filt_dp );
	by_sos = get_sos( by_err_dp, by_filt_dp );
	return( lum_sos + rg_sos + by_sos );
}

/* is this the same as the achrom version??? */

double adjust_sos(dimension_t x,dimension_t y,Data_Obj *fedp,Data_Obj *fdp,double factor)
{
	incr_t i,j;
	float err,adj;
	int xx,yy;
	float *fptr;

	adj =0.0;
	fptr = (float *) OBJ_DATA_PTR(fedp);
	for(j=0;j<(incr_t)OBJ_ROWS(fdp);j++){
		yy = y + j - OBJ_ROWS(fdp)/2;
#ifdef NOWRAP
		if( yy >= 0 && yy < (incr_t)OBJ_ROWS(fedp) ){	/* yy in bounds? */
			for(i=0;i<(incr_t)OBJ_COLS(fdp);i++){	/* scan over x */
				xx = x + i - OBJ_COLS(fdp)/2;
				if( xx >= 0 && xx < (incr_t)OBJ_COLS(fedp) ){	/* xx in bounds? */
					int index;

					index = ( (yy * OBJ_ROW_INC(fedp)) + xx )
						* OBJ_PXL_INC(fedp);
					err = fptr[ index ];
					adj += err*err;
				}
			}
		}
#else /* ! NOWRAP */
		while( yy < 0 ) yy += OBJ_ROWS(fedp);
		while( yy >= (incr_t)OBJ_ROWS(fedp) ) yy -= OBJ_ROWS(fedp);
		for(i=0;i<(incr_t)OBJ_COLS(fdp);i++){
			xx = x + i - OBJ_COLS(fdp)/2;
			while( xx < 0 ) xx += OBJ_COLS(fedp);
			while( xx >= (incr_t)OBJ_COLS(fedp) ) xx -= OBJ_COLS(fedp);
			err = fptr[ ( (yy * OBJ_ROW_INC(fedp)) + xx ) * OBJ_PXL_INC(fedp) ];
			adj += err*err;
		}
#endif /* NOWRAP */
	}

	/* normalize by number of pixels */
	return( (double) (factor * adj / (OBJ_COLS(fedp) * OBJ_ROWS(fedp))) );
}

static void adjust_rgb_sos(dimension_t x,dimension_t y,double factor)
{
	if( the_sos == NO_VALUE )
		the_sos = rgb_sos();
	lum_sos += adjust_sos(x,y,lum_ferr_dp,lum_filt_dp,factor);
	rg_sos += adjust_sos(x,y,rg_ferr_dp,rg_filt_dp,factor);
	by_sos += adjust_sos(x,y,by_ferr_dp,by_filt_dp,factor);
	the_sos = lum_sos + rg_sos + by_sos;
#ifdef QUIP_DEBUG
/*
if( debug & spread_debug ){
sprintf(ERROR_STRING,"adjust_rgb_sos %d %d %g:  lum_sos = %g, rg_sos = %g, by_sos = %g, total = %g",
x,y,factor,lum_sos,rg_sos,by_sos,the_sos);
advise(ERROR_STRING);
}
*/
#endif /* QUIP_DEBUG */

}

#define try_it(m,x,y)	_try_it(QSP_ARG  m,x,y)

static void _try_it(QSP_ARG_DECL  int mask,dimension_t x,dimension_t y)
{
#ifdef QUIP_DEBUG
/*
if( debug & spread_debug ){
sprintf(ERROR_STRING,"try_it %d  %d %d",mask,x,y);
advise(ERROR_STRING);
}
*/
#endif /* QUIP_DEBUG */

	/* subtract contribution from this point */
	adjust_rgb_sos(x,y,-1.0);
	adjust_rgb_ferror(x,y,-1.0);

	/* update error */
	recalc_error(x,y,mask);

	/* add new contribution from this point */
	adjust_rgb_ferror(x,y,1.0);
	adjust_rgb_sos(x,y,1.0);
}

/* Set the halftone image at x,y to the value of mask */

static void set_ht(int mask,dimension_t x,dimension_t y)
{
	char *bptr;

	bptr = (char *) OBJ_DATA_PTR(halftone_dp);
	bptr += (x + y*OBJ_ROW_INC(halftone_dp))*OBJ_PXL_INC(halftone_dp);
	*bptr = (char) mask;
}

/* clr_migrate_pixel
 * instead of trying all possibilities for this pixel, we only examine
 * transformations that preserve the number of rg and b bits...
 * (like "tunnelling" in the achromatic case).
 * For each component (rgb), we try flipping the bit here, and in
 * all the neighbors where it has the opposite initial state.
 *
 *  0  1  2
 *  3     4
 *  5  6  7
 *
 * clr_migrate_pixel2 is similar in that it allows tunneling, but we
 * don't restrict it to just tunnel 1 bit - we check all possible states
 * of this pixel, in conjunction with all possible states of each neighbor.
 */

/* BUG - these defns are inconsistent with similar ones in spread.c ...  */
#define NW	0
#define NORTH	1
#define NE	2
#define WEST	3
#define EAST	4
#define SW	5
#define SOUTH	6
#define SE	7
#define HERE	8

#define N_DIRS	8
#define N_MASKS	8

static int _dx[8]={ -1, 0, 1, -1, 1, -1, 0, 1 };
static int _dy[8]={ 1, 1, 1, 0, 0, -1, -1, -1 };

#define NO_GO	100000.0

static void _clr_migrate_pixel2(QSP_ARG_DECL  incr_t x, incr_t y)
{
	double delta_arr[N_MASKS][N_DIRS][N_MASKS];
	double orig_sos, min_delta;
	int init_state, neighbor;
	int min_dir;
	int mask1,mask2;
	int min_mask1,min_mask2;
	int dir;

	if( the_sos == NO_VALUE )
		the_sos = rgb_sos();
	orig_sos = the_sos;

	init_state = get_ht(x,y);

	//neighbor=0;			// quiet compiler
					// This is totally unnecessary, as
					// a value is always assigned below...
                    // Something to do with optimization?
	// BUT the analyzer calls it a dead store???
    
    min_mask1=0;
	min_mask2=0;

#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"clr_migrate %d %d:  orig_state = %d, orig_sos = %g",x,y,init_state,orig_sos);
advise(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	for(mask1=0;mask1<N_MASKS;mask1++){
		try_it(mask1,x,y);
		/* now for this component, try to migrate in all directions */
		for(dir=0;dir<8;dir++){
			if( (x+_dx[dir]) < 0 || (x+_dx[dir]) >= OBJ_COLS(halftone_dp) ||
			    (y+_dy[dir]) < 0 || (y+_dy[dir]) >= OBJ_ROWS(halftone_dp) ){
				for(mask2=0;mask2<N_MASKS;mask2++)
					delta_arr[mask1][dir][mask2] = NO_GO;
				continue;
			}
			neighbor = get_ht(x+_dx[dir],y+_dy[dir]);
			for(mask2=0;mask2<N_MASKS;mask2++){
				try_it(mask2,x+_dx[dir],y+_dy[dir]);
				delta_arr[mask1][dir][mask2] = the_sos - orig_sos;
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"clr_migrate2 %d %d %d:  the_sos = %g, delta = %g",
mask1,dir,mask2,the_sos,delta_arr[mask1][dir][mask2]);
advise(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			}
			/* restore former state! */
			try_it(neighbor,x+_dx[dir],y+_dy[dir]);
		}
	}
	try_it(init_state,x,y);
	min_delta=NO_GO;
	min_dir=(-1);
	for(mask1=0;mask1<N_MASKS;mask1++){
		for(dir=0;dir<N_DIRS;dir++){
			for(mask2=0;mask2<N_MASKS;mask2++){
				if( delta_arr[mask1][dir][mask2] < min_delta ){
					min_delta = delta_arr[mask1][dir][mask2];
					min_mask1=mask1;
					min_mask2=mask2;
					min_dir=dir;
				}
			}
		}
	}
	if( min_delta > 0 ){
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"clr_migrate2 %d %d, no improvement possible",x,y);
advise(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		return;	/* no improvement anywhere */
	}

	n_pixels_changed += 2;

#ifdef QUIP_DEBUG
// Is this the source of the uninitialized use warning?
/*
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"FINAL clr_migrate2 %d %d %d (was %d), %d %d %d (was %d)",x,y,min_mask1,
init_state,
x+_dx[min_dir],y+_dy[min_dir],min_mask2,neighbor);
advise(DEFAULT_ERROR_STRING);
}
*/
#endif /* QUIP_DEBUG */
	try_it(min_mask1,x,y);
	set_ht(min_mask1,x,y);
	try_it(min_mask2,x+_dx[min_dir],y+_dy[min_dir]);
	set_ht(min_mask2,x+_dx[min_dir],y+_dy[min_dir]);
}

void _clr_migrate_pixel(QSP_ARG_DECL  incr_t x, incr_t y)
{
	double delta_arr[3][8];
	double orig_sos, min_delta;
	int i,j;
	int bit;
	int init_state, neighbor, this_bit;
	int min_bit, min_dir;

	if( the_sos == NO_VALUE )
		the_sos = rgb_sos();
	orig_sos = the_sos;

	init_state = get_ht(x,y);

#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"clr_migrate %d %d:  orig_sos = %g",x,y,orig_sos);
advise(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	bit=1;
	for(i=0;i<3;i++){
		try_it(init_state ^ bit,x,y);
		this_bit = init_state & bit;
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"clr_migrate %d:  bit = %d, init_state = %d, this_bit = %d",
i,bit,init_state,this_bit);
advise(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		/* now for this component, try to migrate in all directions */
		for(j=0;j<8;j++){
			if( (x+_dx[j]) < 0 || (x+_dx[j]) >= OBJ_COLS(halftone_dp) ||
			    (y+_dy[j]) < 0 || (y+_dy[j]) >= OBJ_ROWS(halftone_dp) ){
				delta_arr[i][j] = NO_GO;
				continue;
			}
			neighbor = get_ht(x+_dx[j],y+_dy[j]);
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"clr_migrate %d %d:  neighbor = %d",i,j,neighbor);
advise(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			if( this_bit == (neighbor&bit) ){
				delta_arr[i][j] = NO_GO;
				continue;
			}
			/* Now we know that these two pixels have different states;
			 * Try flipping.
			 */
			try_it(neighbor ^ bit,x+_dx[j],y+_dy[j]);
			delta_arr[i][j] = the_sos - orig_sos;
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"clr_migrate %d %d:  the_sos = %g, delta = %g",i,j,the_sos,delta_arr[i][j]);
advise(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			/* restore former state! */
			try_it(neighbor,x+_dx[j],y+_dy[j]);
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"clr_migrate %d %d:  %d %d  %d, %d %d  %d, delta = %g",i,j,
x,y,init_state^bit,x+_dx[j],y+_dy[j],neighbor^bit,delta_arr[i][j]);
advise(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		}
		try_it(init_state,x,y);

		bit <<= 1;
	}
	bit=1;
	min_delta=NO_GO;
	min_bit=0;
	min_dir=(-1);
	for(i=0;i<3;i++){
		for(j=0;j<8;j++){
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"clr_migrate %d %d, delta = %g",i,j,delta_arr[i][j]);
advise(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			if( delta_arr[i][j] < min_delta ){
				min_delta = delta_arr[i][j];
				min_bit=bit;
				min_dir=j;
			}
		}
		bit <<= 1;
	}
	if( min_delta > 0 ){
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"clr_migrate %d %d, no improvement possible",x,y);
advise(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		return;	/* no improvement anywhere */
	}

	n_pixels_changed += 2;

	try_it(init_state^min_bit,x,y);
	set_ht(init_state^min_bit,x,y);
	neighbor = get_ht(x+_dx[min_dir],y+_dy[min_dir]);
	try_it(neighbor^min_bit,x+_dx[min_dir],y+_dy[min_dir]);
	set_ht(neighbor^min_bit,x+_dx[min_dir],y+_dy[min_dir]);
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"clr_migrate %d %d:  %d %d, final setting %d",
i,j,x,y,init_state^min_bit);
advise(DEFAULT_ERROR_STRING);
sprintf(DEFAULT_ERROR_STRING,"clr_migrate %d %d:  %d %d, final setting %d",
i,j,x+_dx[min_dir],y+_dy[min_dir],neighbor^min_bit);
advise(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

}

/* clr_redo_pixel
 * We try all 8 possibilities for this pixel, and compute the net error for
 * each setting, settling on the lowest.
 *
 * This doesn't work very well; if we are trying to do a 50% gray, and it
 * is initialized with equal dithers in r,g, and b, then any single pixel
 * change will make the luminance error worse...  bit migration should work,
 * but we haven't been able to get it right yet.
 */

void _clr_redo_pixel(QSP_ARG_DECL  incr_t x,incr_t y)
{

	/*
	 * The strategy here is to try all 8 rgb combo's
	 * and pick the one with the lowest total SOS
	 */

	double delta, min_delta=0.0;
	int min_mask,mask;
	double orig_sos;
	double del_sos[8];

	if( !cspread_inited ) {
		WARN("cspread module not initialized");
		return;
	}

	min_mask = 0;		// quiet compiler

	if( x < 0 || x >= OBJ_COLS(halftone_dp) ){
		sprintf(DEFAULT_ERROR_STRING,
"clr_redo_pixel:  x coordinate %d is out of range for image %s (0-%d)",
			x,OBJ_NAME(halftone_dp),OBJ_COLS(halftone_dp));
		WARN(DEFAULT_ERROR_STRING);
		return;
	}

	if( y < 0 || y >= OBJ_ROWS(halftone_dp) ){
		sprintf(DEFAULT_ERROR_STRING,
"clr_redo_pixel:  y coordinate %d is out of range for image %s (0-%d)",
			y,OBJ_NAME(halftone_dp),OBJ_ROWS(halftone_dp));
		WARN(DEFAULT_ERROR_STRING);
		return;
	}

	if( the_sos == NO_VALUE )
		the_sos = rgb_sos();
	orig_sos = the_sos;

#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"clr_redo_pixel %d %d:  orig_sos = %g",x,y,orig_sos);
advise(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	for(mask=0;mask<8;mask++){
		try_it(mask,x,y);
		delta = the_sos - orig_sos;
		del_sos[mask]=delta;
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"clr_redo_pixel %d  %d %d:  the_sos = %g,  delta = %g",mask,x,y,the_sos,delta);
advise(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		if( mask==0 || delta <= min_delta ){
			min_delta = delta;
			min_mask=mask;
		}
	}
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
for(mask=0;mask<8;mask++)
sprintf(DEFAULT_ERROR_STRING,"%d, %d\tdelta SOS for mask %d is %g\n",x,y,mask,del_sos[mask]);
advise(DEFAULT_ERROR_STRING);
sprintf(DEFAULT_ERROR_STRING,"%d, %d\t\tfinal mask is %d\n",x,y,min_mask);
advise(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	/* set error image & sos */
	try_it(min_mask,x,y);
	set_ht(min_mask,x,y);
}

void _clr_scan_requant(QSP_ARG_DECL  uint32_t ntimes)
{
	dimension_t i;
	u_long j;
	dimension_t x,y;


	if( _npixels == NO_PIXELS ){
		NWARN("have to tell me which images first!");
		return;
	}

	if( !cspread_inited ) init_clr_requant(SINGLE_QSP_ARG);

	/* scan image, updating y's */

	for(j=0;j<ntimes;j++){
		for(i=0;i<_npixels;i++){
			(*scan_func)(i,OBJ_COLS(halftone_dp),OBJ_ROWS(halftone_dp),&x,&y);
			clr_redo_pixel(x,y);
		}
	}
}

void _clr_scan_migrate(QSP_ARG_DECL  uint32_t n_times)
{
	dimension_t i;
	u_long j;
	dimension_t x,y;

	if( _npixels == NO_PIXELS ){
		NWARN("have to tell me which images first!");
		return;
	}

	if( !cspread_inited ) init_clr_requant(SINGLE_QSP_ARG);

	/* scan image, updating y's */

	n_pixels_changed=0;

	for(j=0;j<n_times;j++){
		for(i=0;i<_npixels;i++){
			(*scan_func)(i,OBJ_COLS(halftone_dp),OBJ_ROWS(halftone_dp),&x,&y);
			clr_migrate_pixel2(x,y);
		}
	}

	if( n_pixels_changed == 0 ){
		advise("No pixels changed.");
	} else {
		sprintf(DEFAULT_ERROR_STRING,"%d pixels changed.",n_pixels_changed);
		advise(DEFAULT_ERROR_STRING);
	}
}

COMMAND_FUNC( tell_sos )
{
	char str[256];

	if( the_sos == NO_VALUE )
		the_sos = rgb_sos();

	/*
	sprintf(msg_str,"\tlum_sos:  %f",lum_sos);
	prt_msg(msg_str);
	sprintf(msg_str,"\trg_sos:  %f",rg_sos);
	prt_msg(msg_str);
	sprintf(msg_str,"\tby_sos:  %f",by_sos);
	prt_msg(msg_str);
	sprintf(msg_str,"Total:  %f",lum_sos+rg_sos+by_sos);
	prt_msg(msg_str);
	*/

	sprintf(str,"%f",lum_sos);
	assign_var("lum_sos",str);
	sprintf(str,"%f",rg_sos);
	assign_var("rg_sos",str);
	sprintf(str,"%f",by_sos);
	assign_var("by_sos",str);

	/* now recompute... */
	the_sos = rgb_sos();
	sprintf(str,"%f",the_sos);
	assign_var("total_sos",str);

	/*
	advise("RECOMPUTED:");
	*/

	/*
	sprintf(msg_str,"\tlum_sos:  %f",lum_sos);
	prt_msg(msg_str);
	sprintf(msg_str,"\trg_sos:  %f",rg_sos);
	prt_msg(msg_str);
	sprintf(msg_str,"\tby_sos:  %f",by_sos);
	prt_msg(msg_str);
	sprintf(msg_str,"Total:  %f",lum_sos+rg_sos+by_sos);
	prt_msg(msg_str);
	*/

}

#ifdef FOOBAR
double add_to_sos(x,y,edp,fdp,factor)
Data_Obj *edp,*fdp;
int x,y;
int factor;
{
	long i,j;
	double err,adj;
	int xx,yy;

	adj =0.0;
	for(j=0;j<OBJ_ROWS(fdp);j++){
		yy = y + j - OBJ_ROWS(fdp)/2;
#ifdef NOWRAP
		if( yy >= 0 && yy < OBJ_ROWS(edp) ){
			for(i=0;i<OBJ_COLS(fdp);i++){
				xx = x + i - OBJ_COLS(fdp)/2;
				if( xx >= 0 && xx < OBJ_COLS(edp) ){
					err = get_ferror(edp,fdp,xx,yy);
					adj += factor*err*err;
				}
			}
		}
#else
		while( yy < 0 ) yy += OBJ_ROWS(edp);
		while( yy >= OBJ_ROWS(edp) ) yy -= OBJ_ROWS(edp);
		for(i=0;i<OBJ_COLS(fdp);i++){
			xx = x + i - OBJ_COLS(fdp)/2;
			while( xx < 0 ) xx += OBJ_COLS(edp);
			while( xx >= OBJ_COLS(edp) ) xx -= OBJ_COLS(edp);
			err = get_ferror(edp,fdp,xx,yy);
			adj += factor*err*err;
		}
#endif /* NOWRAP */
	}
	/* normalize by number of pixels */
	return( adj / (OBJ_COLS(edp) * OBJ_ROWS(edp)) );
}

static void add_to_rgb_sos(int x,int y,int factor)
{
	if( the_sos == NO_VALUE )
		the_sos = rgb_sos();
	lum_sos += add_to_sos(x,y,lum_err_dp,lum_filt_dp,factor);
	rg_sos += add_to_sos(x,y,rg_err_dp,rg_filt_dp,factor);
	by_sos += add_to_sos(x,y,by_err_dp,by_filt_dp,factor);
	the_sos = lum_sos + rg_sos + by_sos;
}
#endif /* FOOBAR */

/* The problem with having temp dp's here is that we want to remember the pointers,
 * but these may be dangling...  we need a flag bit to set...
 */

Data_Obj *check_not_temp(Data_Obj *dp)
{
	if( IS_TEMP(dp) )
		CLEAR_OBJ_FLAG_BITS(dp,DT_VOLATILE);	/* make it stay locked */
	return(dp);
}

void _set_rgb_input(QSP_ARG_DECL  Data_Obj *lumdp,Data_Obj *rgdp,Data_Obj *bydp)
{
	deslum_dp = check_not_temp(lumdp);
	desrg_dp  = check_not_temp(rgdp);
	desby_dp  = check_not_temp(bydp);

	/* BUG should check sizes match */
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(ERROR_STRING,"input images:  lum %s, rg %s, by %s",
OBJ_NAME(deslum_dp),OBJ_NAME(desrg_dp),OBJ_NAME(desby_dp));
advise(ERROR_STRING);
longlist(deslum_dp);
}
#endif /* QUIP_DEBUG */

}

void set_rgb_output(Data_Obj *hdp)
{
	halftone_dp = check_not_temp(hdp);
}

void set_rgb_filter(Data_Obj *lumdp,Data_Obj *rgdp,Data_Obj *bydp)
{
	lum_filt_dp = check_not_temp(lumdp);
	rg_filt_dp = check_not_temp(rgdp);
	by_filt_dp = check_not_temp(bydp);
}

void set_clr_xform(Data_Obj *matrix)
{
#ifdef QUIP_DEBUG
	float *fptr;
if( debug & spread_debug ){
	fptr=(float *) OBJ_DATA_PTR(matrix);
	fprintf(stderr,"matrix:\n");
	fprintf(stderr,"\t%g\t%g\t%g\n",*fptr,*(fptr+1),*(fptr+2));
	fprintf(stderr,"\t%g\t%g\t%g\n",*(fptr+3),*(fptr+4),*(fptr+5));
	fprintf(stderr,"\t%g\t%g\t%g\n",*(fptr+6),*(fptr+7),*(fptr+8));
	fflush(stderr);
}
#endif /* QUIP_DEBUG */
	rgb2opp_mat = check_not_temp(matrix);
}

COMMAND_FUNC( cspread_tell )
{
	longlist(deslum_dp);
	longlist(desrg_dp);
	longlist(desby_dp);
	longlist(halftone_dp);
	longlist(lum_filt_dp);
	longlist(rg_filt_dp);
	longlist(by_filt_dp);
	longlist(rgb2opp_mat);
	longlist(lum_err_dp);
	longlist(rg_err_dp);
	longlist(by_err_dp);
}


