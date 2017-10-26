#include "quip_config.h"

#define INLINE inline	// BUG - use configure to determine if inline can be used?



/*
 * program to requantize
 *
 * This is a dichromatic version, that we are using to understand the problems
 * with color.
 */

#include "quip_prot.h"

#define ERROR_RETURN	(-1)

#define ON_LEVEL	1
#define OFF_LEVEL	(-1)

#define RED_BIT		1
#define GREEN_BIT	2

#include <math.h>
#include <time.h>	/* time() */
#include <stdlib.h>	/* srand48 */
#include "data_obj.h"
#include "vec_util.h"

#define BOILING_POINT	100.0
#define THERMAL_NOISE( temp )		( drand48() * ( temp ) / BOILING_POINT )

static float act_lum[8], act_rg[8];
static int act_ready=0;
static int n_pixels_changed;

static float _w_lum=1.0, _w_rg=1.0;

#define NO_VALUE	(-1)
static double the_sos=NO_VALUE;
static double lum_sos=NO_VALUE;
static double rg_sos=NO_VALUE;
#define NO_PIXELS 0
static dimension_t _npixels=NO_PIXELS;
static int initialized=0;

/* user-supplied images */
static Data_Obj *deslum_dp=NULL;		/* desired luminance image */
static Data_Obj *desrg_dp=NULL;		/* desired red-green image */
static Data_Obj *halftone_dp=NULL;		/* output composite halftone */
static Data_Obj *lum_filt_dp=NULL;
static Data_Obj *rg_filt_dp=NULL;
static Data_Obj *dich2opp_mat;			/* dichromatic color (RG) to opponent matrix */

/* program private */
static Data_Obj *lum_ferr_dp=NULL;		/* filtered error images */
static Data_Obj *rg_ferr_dp=NULL;

static Data_Obj *lum_err_dp=NULL;		/* error images */
static Data_Obj *rg_err_dp=NULL;

/* local prototypes */

static void init_ferror(void);
static int setup_dich_requantize(SINGLE_QSP_ARG_DECL);
static double dich_sos(void);
static void adjust_dich_sos(dimension_t x,dimension_t y,double factor);
static void adjust_dich_ferror(dimension_t x,dimension_t y,double factor);

static int good_location(incr_t x, incr_t y)
{
	if( !initialized ) {
		NWARN("not initialized");
		return(0);
	}

	if( x < 0 || x >= OBJ_COLS(halftone_dp) ){
		sprintf(DEFAULT_ERROR_STRING,
"good_location:  x coordinate %d is out of range for image %s (0-%d)",
			x,OBJ_NAME(halftone_dp),OBJ_COLS(halftone_dp));
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}

	if( y < 0 || y >= OBJ_ROWS(halftone_dp) ){
		sprintf(DEFAULT_ERROR_STRING,
"good_location:  y coordinate %d is out of range for image %s (0-%d)",
			y,OBJ_NAME(halftone_dp),OBJ_ROWS(halftone_dp));
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}
	return(1);
}

static INLINE int get_ht( dimension_t x, dimension_t y )
{
	char *bptr;

	bptr=(char *)OBJ_DATA_PTR(halftone_dp);
	bptr+=(x+y*OBJ_ROW_INC(halftone_dp))*OBJ_PXL_INC(halftone_dp);
	return(*bptr);
}

static void init_ferror(void)
{
	filter_error(lum_ferr_dp,lum_err_dp,lum_filt_dp);
	filter_error(rg_ferr_dp,rg_err_dp,rg_filt_dp);
}

static void adjust_dich_ferror(dimension_t x,dimension_t y,double factor)
{
/*
sprintf(DEFAULT_ERROR_STRING,"adjust_dich_ferror %d %d:  factor = %g",x,y,factor);
advise(DEFAULT_ERROR_STRING);
*/
	adjust_ferror(lum_ferr_dp,lum_err_dp,lum_filt_dp,x,y,factor);
	adjust_ferror(rg_ferr_dp,rg_err_dp,rg_filt_dp,x,y,factor);
}

static void act_init(SINGLE_QSP_ARG_DECL)
{
	int mask,r,g;
	float *fptr;

	if( dich2opp_mat == NULL ){
		WARN("transformation matrix not defined");
		return;
	}
sprintf(ERROR_STRING,"BEGIN act_init, matrix = %s",OBJ_NAME(dich2opp_mat));
advise(ERROR_STRING);
	if( ! IS_CONTIGUOUS(dich2opp_mat) ){
		sprintf(ERROR_STRING,"Matrix %s must be contiguous",OBJ_NAME(dich2opp_mat));
		error1(ERROR_STRING);
	}
	if( OBJ_PREC(dich2opp_mat) != PREC_SP ){
		sprintf(ERROR_STRING,"Matrix %s (%s) must have %s precision",
			OBJ_NAME(dich2opp_mat),PREC_NAME(OBJ_PREC_PTR(dich2opp_mat)),
			PREC_NAME(PREC_FOR_CODE(PREC_SP)));
		error1(ERROR_STRING);
	}
	if( OBJ_COMPS(dich2opp_mat) != 1 ){
		sprintf(ERROR_STRING,"Matrix %s (%d) must have component dimension = 1",
			OBJ_NAME(dich2opp_mat),OBJ_COMPS(dich2opp_mat));
		error1(ERROR_STRING);
	}
	if( OBJ_PXL_INC(dich2opp_mat) != 1 ){
		sprintf(ERROR_STRING,"Matrix %s (%d) must have pixel increment = 1",
			OBJ_NAME(dich2opp_mat),OBJ_PXL_INC(dich2opp_mat));
		error1(ERROR_STRING);
	}
	if( OBJ_COLS(dich2opp_mat) != 2 || OBJ_ROWS(dich2opp_mat) != 2 ){
		sprintf(ERROR_STRING,"Matrix %s (%d x %d) must be 2x2",
			OBJ_NAME(dich2opp_mat),OBJ_ROWS(dich2opp_mat),OBJ_COLS(dich2opp_mat));
		error1(ERROR_STRING);
	}
	fptr = (float *) OBJ_DATA_PTR(dich2opp_mat);
	for(mask=0;mask<4;mask++){
		if( mask & RED_BIT ) r = ON_LEVEL;
		else r = OFF_LEVEL;
		if( mask & GREEN_BIT ) g = ON_LEVEL;
		else g = OFF_LEVEL;

		act_lum[mask]  = *(fptr  ) * r;
		act_lum[mask] += *(fptr+1) * g;
		act_rg[mask]   = *(fptr+2) * r;
		act_rg[mask]  += *(fptr+3) * g;
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(ERROR_STRING,"act_init:\t%d \t\t%g %g",mask,act_lum[mask],act_rg[mask]);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	}
	act_ready=1;
}

/* recalc_error fetches the desired values, and updates the error images based
 * on the value of mask (which encodes 2 bits of rg, and is used to index tables
 * of corresponding luma and chroma.
 * Called from:
 *	calc_errors
 *	init_dich_requant
 *	try_it
 */

static void recalc_error(QSP_ARG_DECL  dimension_t x,dimension_t y,int mask)
{
	double lum, rg;
	double deslum, desrg;
	float *fptr;


	/* assume the matrix is 2x2 */

	if( !act_ready ) act_init(SINGLE_QSP_ARG);

	lum = act_lum[mask];
	rg = act_rg[mask];

	fptr = (float *) OBJ_DATA_PTR(deslum_dp);
	deslum = *(fptr + (x+y*OBJ_ROW_INC(deslum_dp))*OBJ_PXL_INC(deslum_dp) );
	fptr = (float *) OBJ_DATA_PTR(lum_err_dp);
	*(fptr + (x+y*OBJ_ROW_INC(lum_err_dp))*OBJ_PXL_INC(lum_err_dp) ) = (float)(lum - deslum);

	fptr = (float *) OBJ_DATA_PTR(desrg_dp);
	desrg = *(fptr + (x+y*OBJ_ROW_INC(desrg_dp))*OBJ_PXL_INC(desrg_dp) );
	fptr = (float *) OBJ_DATA_PTR(rg_err_dp);
	*(fptr + (x+y*OBJ_ROW_INC(rg_err_dp))*OBJ_PXL_INC(rg_err_dp) ) = (float)(rg - desrg);

#ifdef QUIP_DEBUG
/*
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"recalc_error %d %d  %d:  act %f %f",
x,y,mask,lum,rg);
advise(DEFAULT_ERROR_STRING);
sprintf(DEFAULT_ERROR_STRING,"recalc_error %d %d  %d:  des %f %f",
x,y,mask,deslum,desrg);
advise(DEFAULT_ERROR_STRING);
}
*/
#endif /* QUIP_DEBUG */

}

static void calc_errors(SINGLE_QSP_ARG_DECL)
{
	dimension_t x,y;
	int mask;
	char *bptr;

	advise("calculating initial error image");
	for(y=0;y<OBJ_ROWS(halftone_dp);y++){
		for(x=0;x<OBJ_COLS(halftone_dp);x++){
			bptr = (char *) OBJ_DATA_PTR(halftone_dp);
			bptr += (x + y*OBJ_ROW_INC(halftone_dp))
				*OBJ_PXL_INC(halftone_dp);
			mask = (*bptr);

			recalc_error(QSP_ARG  x,y,mask);
		}
	}
	advise("finished calculating initial error image");
}


static int setup_dich_requantize(SINGLE_QSP_ARG_DECL)
{
	long tvec;
	Precision *prec_p;

	if( halftone_dp == NULL ){
		WARN("output image not specified");
		return(ERROR_RETURN);
	}
	if( deslum_dp == NULL || desrg_dp == NULL ){
		WARN("input images not specified");
		return(ERROR_RETURN);
	}

	if( lum_filt_dp == NULL ){
		WARN("filters not specified");
		return(ERROR_RETURN);
	}

	if( OBJ_ROWS(halftone_dp) != OBJ_ROWS(deslum_dp) ||
		OBJ_COLS(halftone_dp) != OBJ_COLS(deslum_dp) ){
		WARN("input/output size mismatch");
		return(ERROR_RETURN);
	}

	if( (OBJ_ROWS(deslum_dp) != OBJ_COLS(deslum_dp)) &&
		( scan_func == get_xy_scattered_point) ){

		WARN("input image must be square for scattered scanning");
		return(ERROR_RETURN);
	}

	_npixels = OBJ_ROWS(halftone_dp) * OBJ_COLS(halftone_dp);

	if( lum_err_dp != NULL ){
		delvec(lum_err_dp);
		delvec(rg_err_dp);
		delvec(lum_ferr_dp);
		delvec(rg_ferr_dp);
	}
	prec_p = PREC_FOR_CODE(PREC_SP);
	lum_ferr_dp = mk_img(QSP_ARG  "lum_ferror",
		OBJ_ROWS(halftone_dp),OBJ_COLS(halftone_dp),1,prec_p);
	rg_ferr_dp = mk_img(QSP_ARG  "rg_ferror",
		OBJ_ROWS(halftone_dp),OBJ_COLS(halftone_dp),1,prec_p);
	if( lum_ferr_dp == NULL || rg_ferr_dp == NULL ){
		WARN("couldn't create filtered error images");
		return(ERROR_RETURN);
	}

	lum_err_dp = mk_img(QSP_ARG  "lum_error",
		OBJ_ROWS(halftone_dp),OBJ_COLS(halftone_dp),1,prec_p);
	rg_err_dp = mk_img(QSP_ARG  "rg_error",
		OBJ_ROWS(halftone_dp),OBJ_COLS(halftone_dp),1,prec_p);
	if( lum_err_dp == NULL || rg_err_dp == NULL ){
		WARN("couldn't create error images");
		return(ERROR_RETURN);
	}

	/* now need to initialize the errors ! */
	calc_errors(SINGLE_QSP_ARG);

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

COMMAND_FUNC( init_dich_requant )
{
	dimension_t x,y;
	int mask;

	if( setup_dich_requantize(SINGLE_QSP_ARG) == ERROR_RETURN ) return;

	/* initialize error images */

	if( halftone_dp == NULL ){
		NWARN("init_dich_requant:  no halftone image specified");
		return;
	}
	advise("init_dich_requant:  calculating error");
	for(y=0;y<OBJ_ROWS(halftone_dp);y++){
		for(x=0;x<OBJ_COLS(halftone_dp);x++){
			mask = get_ht( x, y );
			recalc_error(QSP_ARG  x,y,mask);
		}
	}
	/* filter the error */
	advise("filtering error...");
	init_ferror();
	advise("done");
	the_sos = NO_VALUE;
	lum_sos = NO_VALUE;
	rg_sos  = NO_VALUE;

	initialized=1;
}

static double dich_sos(void)
{
	/*
	lum_sos = get_sos( lum_ferr_dp );
	rg_sos = get_sos( rg_ferr_dp );
	*/
	lum_sos = get_sos( lum_err_dp, lum_filt_dp  );
	rg_sos = get_sos( rg_err_dp, rg_filt_dp );
	return( _w_lum*lum_sos + _w_rg*rg_sos );
}

static void adjust_dich_sos(dimension_t x,dimension_t y,double factor)
{
	if( the_sos == NO_VALUE )
		the_sos = dich_sos();
	lum_sos += adjust_sos(x,y,lum_ferr_dp,lum_filt_dp,factor);
	rg_sos += adjust_sos(x,y,rg_ferr_dp,rg_filt_dp,factor);
	the_sos = _w_lum*lum_sos + _w_rg*rg_sos ;
#ifdef QUIP_DEBUG
/*
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"adjust_dich_sos %d %d %g:  lum_sos = %g, rg_sos = %g,   total = %g",
x,y,factor,lum_sos,rg_sos,the_sos);
advise(DEFAULT_ERROR_STRING);
}
*/
#endif /* QUIP_DEBUG */

}

static void try_it(QSP_ARG_DECL  int mask,dimension_t x,dimension_t y)
{
#ifdef QUIP_DEBUG
/*
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"try_it %d  %d %d",mask,x,y);
advise(DEFAULT_ERROR_STRING);
}
*/
#endif /* QUIP_DEBUG */

	/* subtract contribution from this point */
	adjust_dich_sos(x,y,-1.0);
	adjust_dich_ferror(x,y,-1.0);

	/* update error */
	recalc_error(QSP_ARG  x,y,mask);

	/* add new contribution from this point */
	adjust_dich_ferror(x,y,1.0);
	adjust_dich_sos(x,y,1.0);
}

/* Set the halftone image at x,y to the value of mask */

static void set_ht(int mask,dimension_t x,dimension_t y)
{
	char *bptr;

	bptr = (char *) OBJ_DATA_PTR(halftone_dp);
	bptr += (x + y*OBJ_ROW_INC(halftone_dp))*OBJ_PXL_INC(halftone_dp);
	*bptr = (char) mask;
}

/* dich_migrate_pixel
 * instead of trying all possibilities for this pixel, we only examine
 * transformations that preserve the number of r and g bits...
 * (like "tunnelling" in the achromatic case).
 * For each component (rg), we try flipping the bit here, and in
 * all the neighbors where it has the opposite initial state.
 *
 *  0  1  2
 *  3     4
 *  5  6  7
 *
 * dich_migrate_pixel2 is similar in that it allows tunneling, but we
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
#define N_MASKS	4

static int _dx[8]={ -1, 0, 1, -1, 1, -1, 0, 1 };
static int _dy[8]={ 1, 1, 1, 0, 0, -1, -1, -1 };

#define NO_GO	100000.0

static void dich_migrate_pixel2(QSP_ARG_DECL  incr_t x, incr_t y)
{
	double delta_arr[N_MASKS][N_DIRS][N_MASKS];
	double orig_sos, min_delta;
	int init_state, neighbor;
	int min_dir;
	int mask1,mask2;
	int min_mask1,min_mask2;
	int dir;

	if( the_sos == NO_VALUE )
		the_sos = dich_sos();
	orig_sos = the_sos;

	min_mask1 = min_mask2 = (-1);	/* quiet compiler */

	init_state = get_ht(x,y);

#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(ERROR_STRING,"dich_migrate %d %d:  orig_state = %d, orig_sos = %g",x,y,init_state,orig_sos);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	for(mask1=0;mask1<N_MASKS;mask1++){
		try_it(QSP_ARG  mask1,x,y);
		/* now for this component, try to migrate in all directions */
		for(dir=0;dir<N_DIRS;dir++){
			if( (x+_dx[dir]) < 0 || (x+_dx[dir]) >= OBJ_COLS(halftone_dp) ||
			    (y+_dy[dir]) < 0 || (y+_dy[dir]) >= OBJ_ROWS(halftone_dp) ){
				for(mask2=0;mask2<N_MASKS;mask2++)
					delta_arr[mask1][dir][mask2] = NO_GO;
				continue;
			}
			neighbor = get_ht(x+_dx[dir],y+_dy[dir]);
			for(mask2=0;mask2<N_MASKS;mask2++){
				try_it(QSP_ARG  mask2,x+_dx[dir],y+_dy[dir]);
				delta_arr[mask1][dir][mask2] = the_sos - orig_sos;
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(ERROR_STRING,"dich_migrate2 %d %d %d:  the_sos = %g, delta = %g",
mask1,dir,mask2,the_sos,delta_arr[mask1][dir][mask2]);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			}
			/* restore former state! */
			try_it(QSP_ARG  neighbor,x+_dx[dir],y+_dy[dir]);
		}
	}
	try_it(QSP_ARG  init_state,x,y);
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
sprintf(ERROR_STRING,"dich_migrate2 %d %d, no improvement possible",x,y);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		return;	/* no improvement anywhere */
	}

	n_pixels_changed += 2;

#ifdef QUIP_DEBUG
#ifdef ALLOW_WARNINGFUL_STATEMENTS
if( debug & spread_debug ){
sprintf(ERROR_STRING,"FINAL dich_migrate2 %d %d %d (was %d), %d %d %d (was %d)",x,y,min_mask1,
init_state,
x+_dx[min_dir],y+_dy[min_dir],min_mask2,neighbor);
advise(ERROR_STRING);
}
#endif /* ALLOW_WARNINGFUL_STATEMENTS */
#endif /* QUIP_DEBUG */
	try_it(QSP_ARG  min_mask1,x,y);
	set_ht(min_mask1,x,y);
	try_it(QSP_ARG  min_mask2,x+_dx[min_dir],y+_dy[min_dir]);
	set_ht(min_mask2,x+_dx[min_dir],y+_dy[min_dir]);
}

void dich_migrate_pixel(QSP_ARG_DECL  incr_t x, incr_t y)
{
	double delta_arr[3][N_DIRS];
	double orig_sos, min_delta;
	int i,j;
	int bit;
	int init_state, neighbor, this_bit;
	int min_bit, min_dir;

	if( the_sos == NO_VALUE )
		the_sos = dich_sos();
	orig_sos = the_sos;

	init_state = get_ht(x,y);

#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(ERROR_STRING,"dich_migrate %d %d:  orig_sos = %g",x,y,orig_sos);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	bit=1;
	for(i=0;i<2 /* BUG use symbolic constant */ ;i++){
		try_it(QSP_ARG  init_state ^ bit,x,y);
		this_bit = init_state & bit;
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(ERROR_STRING,"dich_migrate %d:  bit = %d, init_state = %d, this_bit = %d",
i,bit,init_state,this_bit);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		/* now for this component, try to migrate in all directions */
		for(j=0;j<N_DIRS;j++){
			if( (x+_dx[j]) < 0 || (x+_dx[j]) >= OBJ_COLS(halftone_dp) ||
			    (y+_dy[j]) < 0 || (y+_dy[j]) >= OBJ_ROWS(halftone_dp) ){
				delta_arr[i][j] = NO_GO;
				continue;
			}
			neighbor = get_ht(x+_dx[j],y+_dy[j]);
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(ERROR_STRING,"dich_migrate %d %d:  neighbor = %d",i,j,neighbor);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			if( this_bit == (neighbor&bit) ){
				delta_arr[i][j] = NO_GO;
				continue;
			}
			/* Now we know that these two pixels have different states;
			 * Try flipping.
			 */
			try_it(QSP_ARG  neighbor ^ bit,x+_dx[j],y+_dy[j]);
			delta_arr[i][j] = the_sos - orig_sos;
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(ERROR_STRING,"dich_migrate %d %d:  the_sos = %g, delta = %g",i,j,the_sos,delta_arr[i][j]);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			/* restore former state! */
			try_it(QSP_ARG  neighbor,x+_dx[j],y+_dy[j]);
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(ERROR_STRING,"dich_migrate %d %d:  %d %d  %d, %d %d  %d, delta = %g",i,j,
x,y,init_state^bit,x+_dx[j],y+_dy[j],neighbor^bit,delta_arr[i][j]);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		}
		try_it(QSP_ARG  init_state,x,y);

		bit <<= 1;
	}
	bit=1;
	min_delta=NO_GO;
	min_bit=0;
	min_dir=(-1);
	for(i=0;i<2 /* BUG use symbolic constant */ ;i++){
		for(j=0;j<N_DIRS;j++){
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(ERROR_STRING,"dich_migrate %d %d, delta = %g",i,j,delta_arr[i][j]);
advise(ERROR_STRING);
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
sprintf(ERROR_STRING,"dich_migrate %d %d, no improvement possible",x,y);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		return;	/* no improvement anywhere */
	}

	n_pixels_changed += 2;

	try_it(QSP_ARG  init_state^min_bit,x,y);
	set_ht(init_state^min_bit,x,y);
	neighbor = get_ht(x+_dx[min_dir],y+_dy[min_dir]);
	try_it(QSP_ARG  neighbor^min_bit,x+_dx[min_dir],y+_dy[min_dir]);
	set_ht(neighbor^min_bit,x+_dx[min_dir],y+_dy[min_dir]);
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(ERROR_STRING,"dich_migrate %d %d:  %d %d, final setting %d",
i,j,x,y,init_state^min_bit);
advise(ERROR_STRING);
sprintf(ERROR_STRING,"dich_migrate %d %d:  %d %d, final setting %d",
i,j,x+_dx[min_dir],y+_dy[min_dir],neighbor^min_bit);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

}

/* dich_redo_pixel
 * We try all 4 possibilities for this pixel, and compute the net error for
 * each setting, settling on the lowest.
 *
 * This doesn't work very well; if we are trying to do a 50% gray, and it
 * is initialized with equal dithers in r,g, and b, then any single pixel
 * change will make the luminance error worse...  bit migration should work,
 * but we haven't been able to get it right yet.
 */

void dich_redo_pixel(QSP_ARG_DECL  incr_t x,incr_t y)
{

	/*
	 * The strategy here is to try all 4 rg combo's
	 * and pick the one with the lowest total SOS
	 */

	double delta, min_delta=0.0;
	int min_mask,mask;
	double orig_sos;
	double del_sos[N_MASKS];

	if( ! good_location(x,y) ) return;

	min_mask=(-1);	/* quiet compiler */

	if( the_sos == NO_VALUE )
		the_sos = dich_sos();
	orig_sos = the_sos;

#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(ERROR_STRING,"dich_redo_pixel %d %d:  orig_sos = %g",x,y,orig_sos);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	for(mask=0;mask<N_MASKS;mask++){
		try_it(QSP_ARG  mask,x,y);
		delta = the_sos - orig_sos;
		del_sos[mask]=delta;
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(ERROR_STRING,"dich_redo_pixel %d  %d %d:  the_sos = %g,  delta = %g",mask,x,y,the_sos,delta);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		if( mask==0 || delta <= min_delta ){
			min_delta = delta;
			min_mask=mask;
		}
	}
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
for(mask=0;mask<N_MASKS;mask++){
sprintf(ERROR_STRING,"%d, %d\tdelta SOS for mask %d is %g\n",x,y,mask,del_sos[mask]);
advise(ERROR_STRING);
}
sprintf(ERROR_STRING,"%d, %d\t\tfinal mask is %d\n",x,y,min_mask);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	/* set error image & sos */
	try_it(QSP_ARG  min_mask,x,y);
	set_ht(min_mask,x,y);
}

void dich_scan_requant(QSP_ARG_DECL  uint32_t ntimes)
{
	dimension_t i;
	uint32_t j;
	dimension_t x,y;


	if( _npixels == NO_PIXELS ){
		NWARN("have to tell me which images first!");
		return;
	}

	if( !initialized ) init_dich_requant(SGL_DEFAULT_QSP_ARG);

	/* scan image, updating y's */

	for(j=0;j<ntimes;j++){
		for(i=0;i<_npixels;i++){
			(*scan_func)(i,OBJ_COLS(halftone_dp),OBJ_ROWS(halftone_dp),&x,&y);
			dich_redo_pixel(QSP_ARG  x,y);
		}
	}
}

/* How should we lower temperature?
 * exponentially or linearly or ???
 *
 * Here we try linearly...
 */

void dich_scan_anneal(QSP_ARG_DECL  uint32_t ntimes,double start_temp, double end_temp)
{
	dimension_t i;
	uint32_t j;
	dimension_t x,y;
	double temp;
	double delta_t;


	if( _npixels == NO_PIXELS ){
		NWARN("have to tell me which images first!");
		return;
	}

	if( !initialized ) init_dich_requant(SGL_DEFAULT_QSP_ARG);

	/* scan image, updating y's */

	delta_t = end_temp - start_temp;
	delta_t /= _npixels * ntimes;
	temp = start_temp;

	for(j=0;j<ntimes;j++){
		for(i=0;i<_npixels;i++){
			(*scan_func)(i,OBJ_COLS(halftone_dp),OBJ_ROWS(halftone_dp),&x,&y);
			/*
			dich_anneal_pixel(x,y,temp);
			*/
			dich_anneal_migrate(QSP_ARG  x,y,temp);
			temp += delta_t;
#ifdef CAUTIOUS
			if( temp < 0 ){
				NWARN("CAUTIOUS:  dich_scan_anneal:  temperature is negative!?");
				temp=0.0;
			}
#endif /* CAUTIOUS */
		}
	}
}

void dich_scan_migrate(QSP_ARG_DECL  uint32_t n_times)
{
	dimension_t i;
	uint32_t j;
	dimension_t x,y;

	if( _npixels == NO_PIXELS ){
		NWARN("have to tell me which images first!");
		return;
	}

	if( !initialized ) init_dich_requant(SGL_DEFAULT_QSP_ARG);

	/* scan image, updating y's */

	n_pixels_changed=0;

	for(j=0;j<n_times;j++){
		for(i=0;i<_npixels;i++){
			(*scan_func)(i,OBJ_COLS(halftone_dp),OBJ_ROWS(halftone_dp),&x,&y);
			dich_migrate_pixel2(QSP_ARG  x,y);
		}
	}

	if( n_pixels_changed == 0 ){
		advise("No pixels changed.");
	} else {
		sprintf(DEFAULT_ERROR_STRING,"%d pixels changed.",n_pixels_changed);
		advise(DEFAULT_ERROR_STRING);
	}
}

COMMAND_FUNC( dich_tell_sos )
{
	char str[256];

	if( the_sos == NO_VALUE )
		the_sos = dich_sos();

	sprintf(msg_str,"\tlum_sos:  %f",lum_sos);
	prt_msg(msg_str);
	sprintf(msg_str,"\trg_sos:  %f",rg_sos);
	prt_msg(msg_str);
	sprintf(msg_str,"Composite:  %f",_w_lum*lum_sos+_w_rg*rg_sos);
	prt_msg(msg_str);

	sprintf(str,"%f",lum_sos);
	assign_var("lum_sos",str);
	sprintf(str,"%f",rg_sos);
	assign_var("rg_sos",str);

	/* now recompute... */
	the_sos = dich_sos();

	advise("RECOMPUTED:");

	sprintf(msg_str,"\tlum_sos:  %f",lum_sos);
	prt_msg(msg_str);
	sprintf(msg_str,"\trg_sos:  %f",rg_sos);
	prt_msg(msg_str);
	sprintf(msg_str,"Total:  %f",_w_lum*lum_sos+_w_rg*rg_sos);
	prt_msg(msg_str);

}

void set_dich_input(QSP_ARG_DECL  Data_Obj *lumdp,Data_Obj *rgdp)
{
	deslum_dp = check_not_temp(lumdp);
	desrg_dp  = check_not_temp(rgdp);

	/* BUG should check sizes match */
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(ERROR_STRING,"input images:  lum %s, rg %s",
OBJ_NAME(deslum_dp),OBJ_NAME(desrg_dp));
advise(ERROR_STRING);
longlist(deslum_dp);
}
#endif /* QUIP_DEBUG */

}

void set_dich_output(Data_Obj *hdp)
{
	halftone_dp = check_not_temp(hdp);
}

void set_dich_filter(Data_Obj *lumdp,Data_Obj *rgdp)
{
	lum_filt_dp = check_not_temp(lumdp);
	rg_filt_dp = check_not_temp(rgdp);
}

void set_dich_xform(Data_Obj *matrix)
{
#ifdef QUIP_DEBUG
	float *fptr;
if( debug & spread_debug ){
	fptr=(float *) OBJ_DATA_PTR(matrix);
	fprintf(stderr,"matrix:\n");
	fprintf(stderr,"\t%g\t%g\n",*fptr,*(fptr+1));
	fprintf(stderr,"\t%g\t%g\n",*(fptr+2),*(fptr+3));
	fflush(stderr);
}
#endif /* QUIP_DEBUG */
	dich2opp_mat = check_not_temp(matrix);
}

COMMAND_FUNC( dspread_tell )
{
	longlist(deslum_dp);
	longlist(desrg_dp);
	longlist(halftone_dp);
	longlist(lum_filt_dp);
	longlist(rg_filt_dp);
	longlist(dich2opp_mat);
	longlist(lum_err_dp);
	longlist(rg_err_dp);
}


/* simulated annealing for dichromatic images
 *
 * In the achromatic case, annealing is relatively straightforward:  in our implementation,
 * we flip a bit if it reduces the energy, and flip it probabilistically if it
 * increases the energy...  We could apply that approach to color (di- or tri-)
 * by considering one color bit at a time...  Otherwise, we have to come up with a new
 * scheme for annealing in a multi-way decision.
 *
 * Multi-way annealing
 *
 * Let's say that we have 4 possible states, each with an associated energy...
 * In strict descent, we would just pick that state with the lowest energy.
 * Here we want to do something based on T (temperature).  For high T, we would
 * like to pick each of the states with equal probablity, but as T is reduced
 * we want to go to the minimum energy state...  One way to do this would be to add
 * a random variable to each energy, scaled by T, and then pick the smallest.
 * We then have the problem of scaling the energies such that we can specify T in
 * meaningful units.  If we normalize the volumes of the filters, then the errors
 * are in the range -1:1, the squared error is in the range 0:1...  This is sort
 * of convenient in that we could add a uniform random number scaled by T/Tmax...
 *
 * The initial body of code was swiped from dich_redo_pixel().
 */

void dich_anneal_pixel( QSP_ARG_DECL  incr_t x, incr_t y, double temp )
{
	double delta, min_delta=0.0;
	int min_mask,mask;
	double orig_sos;
	double del_sos[N_MASKS];

	if( ! good_location(x,y) ) return;

	min_mask=(-1);	/* quiet compiler */

	if( the_sos == NO_VALUE ) the_sos = dich_sos();
	orig_sos = the_sos;
	min_delta = 1000000.0;		/* way big */

#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(ERROR_STRING,"dich_anneal_pixel %d %d:  orig_sos = %g",x,y,orig_sos);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	for(mask=0;mask<N_MASKS;mask++){
		try_it(QSP_ARG  mask,x,y);
		delta = the_sos - orig_sos;
		delta += THERMAL_NOISE( temp );
		del_sos[mask]=delta;
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(ERROR_STRING,"dich_redo_pixel %d  %d %d:  the_sos = %g,  delta = %g",mask,x,y,the_sos,delta);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		if( mask==0 || delta <= min_delta ){
			min_delta = delta;
			min_mask=mask;
		}
	}
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
for(mask=0;mask<N_MASKS;mask++)
sprintf(ERROR_STRING,"%d, %d\tdelta SOS for mask %d is %g\n",x,y,mask,del_sos[mask]);
advise(ERROR_STRING);
sprintf(ERROR_STRING,"%d, %d\t\tfinal mask is %d\n",x,y,min_mask);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	/* set error image & sos */
	try_it(QSP_ARG  min_mask,x,y);
	set_ht(min_mask,x,y);
}

void set_dich_weights( double w_lum, double w_rg )
{
	_w_lum = (float) w_lum;
	_w_rg = (float) w_rg;
}

void dich_anneal_migrate( QSP_ARG_DECL  incr_t x, incr_t y, double temp )
{
	double delta_arr[N_MASKS][N_DIRS][N_MASKS];
	double orig_sos, min_delta;
	int init_state, neighbor;
	int min_dir;
	int mask1,mask2;
	int min_mask1,min_mask2;
	int dir;

	min_mask1 = (-1);	/* just to quiet compiler */
	min_mask2 = (-1);	/* just to quiet compiler */

	if( the_sos == NO_VALUE )
		the_sos = dich_sos();
	orig_sos = the_sos;

	init_state = get_ht(x,y);

	for(mask1=0;mask1<N_MASKS;mask1++){
		try_it(QSP_ARG  mask1,x,y);
		/* now for this component, try to migrate in all directions */
		for(dir=0;dir<N_DIRS;dir++){
			if( (x+_dx[dir]) < 0 || (x+_dx[dir]) >= OBJ_COLS(halftone_dp) ||
			    (y+_dy[dir]) < 0 || (y+_dy[dir]) >= OBJ_ROWS(halftone_dp) ){
				for(mask2=0;mask2<N_MASKS;mask2++)
					delta_arr[mask1][dir][mask2] = NO_GO;
				continue;
			}
			neighbor = get_ht(x+_dx[dir],y+_dy[dir]);
			for(mask2=0;mask2<N_MASKS;mask2++){
				try_it(QSP_ARG  mask2,x+_dx[dir],y+_dy[dir]);
				delta_arr[mask1][dir][mask2] = the_sos - orig_sos;
				delta_arr[mask1][dir][mask2] -= THERMAL_NOISE( temp );
			}
			/* restore former state! */
			try_it(QSP_ARG  neighbor,x+_dx[dir],y+_dy[dir]);
		}
	}
	try_it(QSP_ARG  init_state,x,y);
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
		return;	/* no improvement anywhere */
	}

	n_pixels_changed += 2;

#ifdef QUIP_DEBUG
#ifdef ALLOW_WARNINGFUL_STATEMENTS
if( debug & spread_debug ){
sprintf(ERROR_STRING,
/* showing neighbor generates compiler warning */
"FINAL dich_migrate2 %d %d %d (was %d), %d %d %d (was %d)",
x,y,min_mask1, init_state,
x+_dx[min_dir],y+_dy[min_dir],min_mask2,neighbor);
advise(ERROR_STRING);
}
#endif /* ALLOW_WARNINGFUL_STATEMENTS */
#endif /* QUIP_DEBUG */

	try_it(QSP_ARG  min_mask1,x,y);
	set_ht(min_mask1,x,y);
	try_it(QSP_ARG  min_mask2,x+_dx[min_dir],y+_dy[min_dir]);
	set_ht(min_mask2,x+_dx[min_dir],y+_dy[min_dir]);
}
