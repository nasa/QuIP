#ifndef _VEC_UTIL_H_
#define _VEC_UTIL_H_

#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#include "typedefs.h"
#include "data_obj.h"
#include "debug.h"

#define posn_t incr_t

/* global variables */
extern debug_flag_t spread_debug;

extern void (*scan_func)(QSP_ARG_DECL  dimension_t index,dimension_t,dimension_t,
					dimension_t *,dimension_t *);
extern void (*scan_func3d)(QSP_ARG_DECL  dimension_t index,Dimension_Set *sizes,posn_t *posn);



/* integral.c */
extern void _cum_sum(QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);
#define cum_sum(dst_dp, src_dp) _cum_sum(QSP_ARG  dst_dp, src_dp)

/* scramble.c */
extern void _dp_scramble(QSP_ARG_DECL  Data_Obj *dp);
#define dp_scramble(dp) _dp_scramble(QSP_ARG  dp)

/* vinterp.c */
extern void _vinterp(QSP_ARG_DECL  Data_Obj *target, Data_Obj *source, Data_Obj *control );
#define vinterp(target, source, control ) _vinterp(QSP_ARG  target, source, control )

/* lutmap.c */

extern int _lutmap(QSP_ARG_DECL  Data_Obj *dest_dp, Data_Obj *src_dp, Data_Obj *lut_dp );
#define lutmap(dest_dp, src_dp, lut_dp ) _lutmap(QSP_ARG  dest_dp, src_dp, lut_dp )

/* project.c */
void _war_project(QSP_ARG_DECL  Data_Obj *,Data_Obj *);
#define war_project(dpto,dpfr) _war_project(QSP_ARG  dpto,dpfr)

/* cumsum.c */
extern void _war_cumsum(QSP_ARG_DECL  Data_Obj *dp_to,Data_Obj *dp_fr);
#define war_cumsum(dp_to,dp_fr) _war_cumsum(QSP_ARG  dp_to,dp_fr)


/* dct8.c */

extern void _compute_dct(QSP_ARG_DECL  Data_Obj *,int);
#define compute_dct(dp,code) _compute_dct(QSP_ARG  dp,code)

/* morph.c */

extern int pixelread(int,int);
extern void pixelwrite(int,int,int);

extern void _dilate(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr);
extern void _erode(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr);
extern void _ifl(QSP_ARG_DECL  Data_Obj *dp,index_t x,index_t y,double color,double tolerance);
extern void _thinzs(QSP_ARG_DECL  Data_Obj *x, double val);
extern void _morph_process(QSP_ARG_DECL  Data_Obj *to, Data_Obj *fr, Data_Obj *tbl);

#define dilate(dpto,dpfr) _dilate(QSP_ARG  dpto,dpfr)
#define erode(dpto,dpfr) _erode(QSP_ARG  dpto,dpfr)
#define ifl(dp,x,y,color,tolerance) _ifl(QSP_ARG  dp,x,y,color,tolerance)
#define thinzs(x, val) _thinzs(QSP_ARG  x, val)
#define morph_process(to, fr, tbl) _morph_process(QSP_ARG  to, fr, tbl)


/* sample.c */
extern void _render_samples(QSP_ARG_DECL  Data_Obj *img, Data_Obj *coord, Data_Obj *intens);
extern void _render_samples2(QSP_ARG_DECL  Data_Obj *image, Data_Obj *coord, Data_Obj *intens);
extern void _sample_image(QSP_ARG_DECL  Data_Obj *intens, Data_Obj *image, Data_Obj *coord);

#define render_samples(img, coord, intens) _render_samples(QSP_ARG  img, coord, intens)
#define render_samples2(image, coord, intens) _render_samples2(QSP_ARG  image, coord, intens)
#define sample_image(intens, image, coord) _sample_image(QSP_ARG  intens, image, coord)


/* File bessel.c */

extern int _bessel_of(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,int order);
extern int _acos_of(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr);
extern int _asin_of(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr);

#define bessel_of(dpto,dpfr,order) _bessel_of(QSP_ARG  dpto,dpfr,order)
#define acos_of(dpto,dpfr) _acos_of(QSP_ARG  dpto,dpfr)
#define asin_of(dpto,dpfr) _asin_of(QSP_ARG  dpto,dpfr)



/* dither.c */

extern void _odither(QSP_ARG_DECL  Data_Obj *dp,int size);

#define odither(dp,size) _odither(QSP_ARG  dp,size)

/* dpinvert.c */

extern double _dt_invert(QSP_ARG_DECL  Data_Obj *dp);
extern double _invert_sq_matrix(QSP_ARG_DECL  float *,dimension_t);
#define dt_invert(dp) _dt_invert(QSP_ARG  dp)
#define invert_sq_matrix(p,s) _invert_sq_matrix(QSP_ARG  p,s)


/* dptone.c */

extern void _dp_halftone(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,dimension_t n,float *levels);
#define dp_halftone(dpto,dpfr,n,levels) _dp_halftone(QSP_ARG  dpto,dpfr,n,levels)


/* File gj.c */


extern int _gauss_jordan(QSP_ARG_DECL  float *matrix,dimension_t size);
extern int _dp_gauss_jordan(QSP_ARG_DECL  double *matrix,dimension_t size);
#define gauss_jordan(matrix,size) _gauss_jordan(QSP_ARG  matrix,size)
#define dp_gauss_jordan(matrix,size) _dp_gauss_jordan(QSP_ARG  matrix,size)


/* File histo.c */

extern void _compute_histo(QSP_ARG_DECL  Data_Obj *histo_dp,Data_Obj *data_dp,double bin_width,double min);
extern void _multivariate_histo(QSP_ARG_DECL  Data_Obj *histo_dp,Data_Obj *data_dp,float *,float *);

#define compute_histo(histo_dp,data_dp,bin_width,min) _compute_histo(QSP_ARG  histo_dp,data_dp,bin_width,min)
#define multivariate_histo(histo_dp,data_dp,wp,mp) _multivariate_histo(QSP_ARG  histo_dp,data_dp,wp,mp)

/* file hough.c */
extern void _hough(QSP_ARG_DECL  Data_Obj *xform_dp, Data_Obj *src_dp, float threshold , float x0 , float y0);

#define hough(xform_dp, src_dp, threshold , x0 , y0) _hough(QSP_ARG  xform_dp, src_dp, threshold , x0 , y0)

/* file local_max.c */
extern long _local_maxima(QSP_ARG_DECL  Data_Obj *val_dp, Data_Obj *coord_dp, Data_Obj *src_dp);

#define local_maxima(val_dp, coord_dp, src_dp) _local_maxima(QSP_ARG  val_dp, coord_dp, src_dp)

/* File inittbls.c */

extern void inittbls(void);


/* File mkwheel.c */

extern int whchk(Data_Obj *dp);
extern void make_axle(Data_Obj *dp);
extern void mkwheel(Data_Obj *dp,int nspokes,double arg0);


/* File oddeven.c */



extern void mkodd(Data_Obj *dp);
extern void mkeven(Data_Obj *dp);


/* File quads.c */

extern void _make_all_quads(QSP_ARG_DECL  Data_Obj *target,Data_Obj *source);

#define make_all_quads(target,source) _make_all_quads(QSP_ARG  target,source)


/* File radavg.c */

extern int _rad_avg(QSP_ARG_DECL  Data_Obj *mean_dp,Data_Obj *var_dp,Data_Obj *count_dp,
			Data_Obj *img_dp);
extern int _ori_avg(QSP_ARG_DECL  Data_Obj *mean_dp,Data_Obj *var_dp,Data_Obj *count_dp,
			Data_Obj *img_dp);

#define rad_avg(mean_dp,var_dp,count_dp, img_dp) _rad_avg(QSP_ARG  mean_dp,var_dp,count_dp, img_dp)
#define ori_avg(mean_dp,var_dp,count_dp, img_dp) _ori_avg(QSP_ARG  mean_dp,var_dp,count_dp, img_dp)


/* File resample.c */

extern void set_resample_wrap(int flag);
extern void _resample(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *dpwarp);
extern void _bilinear_warp(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *dpwarp);
extern void _new_bilinear_warp(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *dpwarp);

#define resample(dpto,dpfr,dpwarp) _resample(QSP_ARG  dpto,dpfr,dpwarp)
#define bilinear_warp(dpto,dpfr,dpwarp) _bilinear_warp(QSP_ARG  dpto,dpfr,dpwarp)
#define new_bilinear_warp(dpto,dpfr,dpwarp) _new_bilinear_warp(QSP_ARG  dpto,dpfr,dpwarp)



/* File tspread.c */
// BUG?  these should go in a module-specific include file?

extern void _setup_ffilter3d(QSP_ARG_DECL  Data_Obj *fdp);
extern void _get_3d_scattered_point(QSP_ARG_DECL  dimension_t n,Dimension_Set *sizes,posn_t *posn );
extern void _get_3d_raster_point(QSP_ARG_DECL  dimension_t n,Dimension_Set *sizes,posn_t *posn );

#define setup_ffilter3d(fdp) _setup_ffilter3d(QSP_ARG  fdp)
#define get_3d_scattered_point(n,sizes,posn ) _get_3d_scattered_point(QSP_ARG  n,sizes,posn )
#define get_3d_raster_point(n,sizes,posn ) _get_3d_raster_point(QSP_ARG  n,sizes,posn )

/* File resample.c */

extern void set_resample_wrap(int flag);
extern void _resample(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *dpwarp);
extern void _bilinear_warp(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *dpwarp);
extern void _new_bilinear_warp(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *dpwarp);

#define resample(dpto,dpfr,dpwarp) _resample(QSP_ARG  dpto,dpfr,dpwarp)
#define bilinear_warp(dpto,dpfr,dpwarp) _bilinear_warp(QSP_ARG  dpto,dpfr,dpwarp)
#define new_bilinear_warp(dpto,dpfr,dpwarp) _new_bilinear_warp(QSP_ARG  dpto,dpfr,dpwarp)


/* File convolve.c */

extern void add_impulse(double amp,Data_Obj *image_dp,Data_Obj *ir_dp,incr_t x,incr_t y);
extern void img_clear(Data_Obj *dp);
extern void _convolve(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *dpfilt);

#define convolve(dpto,dpfr,dpfilt) _convolve(QSP_ARG  dpto,dpfr,dpfilt)


/* File conv3d.c */

extern void add_impulse3d(double amp,Data_Obj *image_dp,Data_Obj *ir_dp,posn_t *posns);
extern void img_clear3d(Data_Obj *dp);
extern void _convolve3d(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *dpfilt);
#define convolve3d(dpto,dpfr,dpfilt) _convolve3d(QSP_ARG  dpto,dpfr,dpfilt)


/* File tspread.c */
// BUG?  these should go in a module-specific include file?

extern void _setup_ffilter3d(QSP_ARG_DECL  Data_Obj *fdp);
extern void _get_3d_scattered_point(QSP_ARG_DECL  dimension_t n,Dimension_Set *sizes,posn_t *posn );
extern void _get_3d_raster_point(QSP_ARG_DECL  dimension_t n,Dimension_Set *sizes,posn_t *posn );
extern void _get_3d_random_point(QSP_ARG_DECL  dimension_t n,Dimension_Set *sizes,posn_t *posn );
extern int _redo_pixel3d(QSP_ARG_DECL  posn_t *posn);
extern void _redo_two_pixels3d(QSP_ARG_DECL  posn_t *posn);

extern double get_sos3d(Data_Obj *edp,Data_Obj *fdp);
extern float get_delta3d(posn_t *posn);
extern void set_filter3d(Data_Obj *fdp);
extern void set_grayscale3d(Data_Obj *gdp);
extern void set_halftone3d(Data_Obj *hdp);
extern void set_temp3d(double temp);

extern int _scan_requant3d(QSP_ARG_DECL  int ntimes);
extern void _scan2_requant3d(QSP_ARG_DECL  int ntimes);
extern void _scan_anneal3d(QSP_ARG_DECL  double temp,int ntimes);
extern void _normalize_filter3d(QSP_ARG_DECL  Data_Obj *fdp);
extern double _add_to_sos3d(QSP_ARG_DECL  posn_t *posn,Data_Obj *edp,Data_Obj *fdp,int factor);
extern int _setup_requantize3d(SINGLE_QSP_ARG_DECL);
extern void _init_requant3d(SINGLE_QSP_ARG_DECL);

#define get_3d_scattered_point(n,sizes,posn ) _get_3d_scattered_point(QSP_ARG  n,sizes,posn )
#define get_3d_raster_point(n,sizes,posn ) _get_3d_raster_point(QSP_ARG  n,sizes,posn )
#define get_3d_random_point(n,sizes,posn ) _get_3d_random_point(QSP_ARG  n,sizes,posn )
#define scan_requant3d(ntimes) _scan_requant3d(QSP_ARG  ntimes)
#define scan2_requant3d(ntimes) _scan2_requant3d(QSP_ARG  ntimes)
#define scan_anneal3d(temp,ntimes) _scan_anneal3d(QSP_ARG  temp,ntimes)
#define redo_pixel3d(posn) _redo_pixel3d(QSP_ARG  posn)
#define redo_two_pixels3d(posn) _redo_two_pixels3d(QSP_ARG  posn)
#define normalize_filter3d(fdp) _normalize_filter3d(QSP_ARG  fdp)
#define add_to_sos3d(posn,edp,fdp,factor) _add_to_sos3d(QSP_ARG  posn,edp,fdp,factor)
#define setup_requantize3d()	_setup_requantize3d(SINGLE_QSP_ARG)
#define init_requant3d()	_init_requant3d(SINGLE_QSP_ARG)

/* File spread.c */

extern double get_prob(double dele);
extern double get_ferror(Data_Obj *edp,Data_Obj *fdp,dimension_t x,dimension_t y);
extern double get_sos(Data_Obj *edp,Data_Obj *fdp);
extern void set_filter(Data_Obj *fdp);
extern void set_grayscale(Data_Obj *gdp);
extern void set_halftone(Data_Obj *hdp);
extern double get_volume(Data_Obj *dp);
extern float get_delta(dimension_t x,dimension_t y);
extern void set_temp(double temp);
extern void insist_pixel(dimension_t x,dimension_t y);

extern void _normalize_filter(QSP_ARG_DECL  Data_Obj *fdp);
extern void _init_requant(SINGLE_QSP_ARG_DECL);
extern int _scan_requant(QSP_ARG_DECL  int ntimes);
extern void _scan_anneal(QSP_ARG_DECL  double temp,int ntimes);
extern int _redo_pixel(QSP_ARG_DECL  dimension_t x,dimension_t y);
extern int _redo_two_pixels(QSP_ARG_DECL  dimension_t x,dimension_t y);
extern void _get_xy_scattered_point(QSP_ARG_DECL  dimension_t n,dimension_t xsize,dimension_t ysize,
							dimension_t *xp,dimension_t *yp);
extern void _get_xy_raster_point(QSP_ARG_DECL  dimension_t n,dimension_t xsize,dimension_t ysize,
							dimension_t *xp,dimension_t *yp);
extern void _get_xy_random_point(QSP_ARG_DECL  dimension_t n,dimension_t xsize,dimension_t ysize,
							dimension_t *xp,dimension_t *yp);

extern double _add_to_sos(QSP_ARG_DECL  dimension_t x,dimension_t y,Data_Obj *edp,Data_Obj *fdp,int factor);

#define normalize_filter(fdp) _normalize_filter(QSP_ARG  fdp)
#define init_requant() _init_requant(SINGLE_QSP_ARG)
#define scan_requant(ntimes) _scan_requant(QSP_ARG  ntimes)
#define scan_anneal(temp,ntimes) _scan_anneal(QSP_ARG  temp,ntimes)
#define redo_pixel(x,y) _redo_pixel(QSP_ARG  x,y)
#define redo_two_pixels(x,y) _redo_two_pixels(QSP_ARG  x,y)
#define add_to_sos(x,y,edp,fdp,factor) _add_to_sos(QSP_ARG  x,y,edp,fdp,factor)

#define get_xy_scattered_point(n,xsize,ysize,xp,yp) _get_xy_scattered_point(QSP_ARG  n,xsize,ysize,xp,yp)
#define get_xy_raster_point(n,xsize,ysize,xp,yp) _get_xy_raster_point(QSP_ARG  n,xsize,ysize,xp,yp)
#define get_xy_random_point(n,xsize,ysize,xp,yp) _get_xy_random_point(QSP_ARG  n,xsize,ysize,xp,yp)

/* File cspread.c */
extern double adjust_sos(dimension_t x,dimension_t y,Data_Obj *fedp,Data_Obj *fdp,double factor);
extern void adjust_ferror( Data_Obj *fedp, Data_Obj *edp, Data_Obj *fdp, dimension_t x, dimension_t y, double factor);
extern void filter_error( Data_Obj *dpto, Data_Obj *dpfr, Data_Obj *filtdp);
extern COMMAND_FUNC( init_clr_requant );
extern void set_rgb_output(Data_Obj *);
extern void set_rgb_filter(Data_Obj *,Data_Obj *,Data_Obj *);

extern void set_clr_xform(Data_Obj *);
extern COMMAND_FUNC( tell_sos );
extern COMMAND_FUNC( cspread_tell );
extern Data_Obj *check_not_temp(Data_Obj *);

extern void _set_rgb_input(QSP_ARG_DECL  Data_Obj *,Data_Obj *,Data_Obj *);
extern void _clr_redo_pixel(QSP_ARG_DECL  incr_t x, incr_t y);
extern void _clr_migrate_pixel(QSP_ARG_DECL  incr_t x, incr_t y);
extern void _clr_scan_requant(QSP_ARG_DECL  index_t ntimes);
extern void _clr_scan_migrate(QSP_ARG_DECL  index_t ntimes);
extern int _setup_requantize(SINGLE_QSP_ARG_DECL);
extern void _setup_ffilter(QSP_ARG_DECL  Data_Obj *fdp);
extern void _scan2_requant(QSP_ARG_DECL  int ntimes);

#define set_rgb_input(a,b,c)	_set_rgb_input(QSP_ARG  a,b,c)
#define clr_redo_pixel(x,y)	_clr_redo_pixel(QSP_ARG  x,y)
#define clr_migrate_pixel(x,y)	_clr_migrate_pixel(QSP_ARG  x,y)
#define clr_migrate_pixel2(x,y)	_clr_migrate_pixel2(QSP_ARG  x,y)
#define clr_scan_requant(n)	_clr_scan_requant(QSP_ARG  n)
#define clr_scan_migrate(n)	_clr_scan_migrate(QSP_ARG  n)
#define setup_requantize()	_setup_requantize(SINGLE_QSP_ARG)
#define setup_ffilter(fdp)	_setup_ffilter(QSP_ARG  fdp)
#define scan2_requant(n)	_scan2_requant(QSP_ARG  n)

/* file dspread.c */
extern COMMAND_FUNC( init_dich_requant );
extern void set_dich_output(Data_Obj *);
extern void set_dich_filter(Data_Obj *,Data_Obj *);
extern COMMAND_FUNC( dich_tell_sos );
extern void set_dich_xform(Data_Obj *);
extern COMMAND_FUNC( dspread_tell );
extern void set_dich_weights(double,double);

extern void _dich_anneal_migrate( QSP_ARG_DECL  incr_t x, incr_t y, double temp );
extern void _set_dich_input(QSP_ARG_DECL  Data_Obj *,Data_Obj *);
extern void _dich_scan_anneal(QSP_ARG_DECL  index_t ntimes, double temp1, double temp2);
extern void _dich_redo_pixel(QSP_ARG_DECL  incr_t x, incr_t y);
extern void _dich_anneal_pixel(QSP_ARG_DECL  incr_t x, incr_t y, double temp);
extern void _dich_migrate_pixel(QSP_ARG_DECL  incr_t x, incr_t y);
extern void _dich_scan_requant(QSP_ARG_DECL  index_t ntimes);
extern void _dich_scan_migrate(QSP_ARG_DECL  index_t ntimes);

#define dich_anneal_migrate(x,y,temp)	_dich_anneal_migrate(QSP_ARG  x,y,temp)
#define set_dich_input(dp1,dp2)		_set_dich_input(QSP_ARG  dp1,dp2)
#define dich_scan_anneal(ntimes, temp1, temp2)	_dich_scan_anneal(QSP_ARG  ntimes, temp1, temp2)
#define dich_redo_pixel(x, y)		_dich_redo_pixel(QSP_ARG  x, y)
#define dich_anneal_pixel(x, y, temp)	_dich_anneal_pixel(QSP_ARG  x, y, temp)
#define dich_migrate_pixel(x, y)	_dich_migrate_pixel(QSP_ARG  x, y)
#define dich_scan_requant(ntimes)	_dich_scan_requant(QSP_ARG  ntimes)
#define dich_scan_migrate(ntimes)	_dich_scan_migrate(QSP_ARG  ntimes)

/* File median.c */

extern void _median_1D(QSP_ARG_DECL  Data_Obj *,Data_Obj *,int radius);
extern void _median(QSP_ARG_DECL  Data_Obj *,Data_Obj *);
extern void _median_clip(QSP_ARG_DECL  Data_Obj *,Data_Obj *);
extern void _sort_data(QSP_ARG_DECL  Data_Obj *dp);
extern void _sort_indices(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2);

#define median_1D(dst,src,radius) _median_1D(QSP_ARG  dst,src,radius)
#define median(dst,src) _median(QSP_ARG  dst,src)
#define median_clip(dst,src) _median_clip(QSP_ARG  dst,src)
#define sort_data(dp) _sort_data(QSP_ARG  dp)
#define sort_indices(dp1,dp2) _sort_indices(QSP_ARG  dp1,dp2)

/* krast.c */

extern void _mk_krast(QSP_ARG_DECL  Data_Obj *);
#define mk_krast(dp) _mk_krast(QSP_ARG  dp)

/* graph_path.c */

extern void _extend_shortest_paths(QSP_ARG_DECL  Data_Obj *,Data_Obj *);
#define extend_shortest_paths(dst,src) _extend_shortest_paths(QSP_ARG  dst,src)


/* yuv2rgb.c */
extern void _yuv422_to_rgb24(QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);
extern void _yuv422_to_gray(QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp);
#define yuv422_to_rgb24(dst_dp, src_dp) _yuv422_to_rgb24(QSP_ARG  dst_dp, src_dp)
#define yuv422_to_gray(dst_dp, src_dp) _yuv422_to_gray(QSP_ARG  dst_dp, src_dp)

/* size.c */
extern int _reduce(QSP_ARG_DECL  Data_Obj *lil_dp,Data_Obj *big_dp);
extern int _enlarge(QSP_ARG_DECL  Data_Obj *big_dp,Data_Obj *lil_dp);
#define reduce(lil_dp,big_dp) _reduce(QSP_ARG  lil_dp,big_dp)
#define enlarge(big_dp,lil_dp) _enlarge(QSP_ARG  big_dp,lil_dp)

/* wrap.c */
extern void _dp_scroll(QSP_ARG_DECL  Data_Obj *dst_dp,Data_Obj *src_dp,incr_t dx,incr_t dy);
extern void _wrap(QSP_ARG_DECL  Data_Obj *dst_dp,Data_Obj *src_dp);
#define dp_scroll(dst_dp,src_dp,dx,dy) _dp_scroll(QSP_ARG  dst_dp,src_dp,dx,dy)
#define wrap(dst_dp,src_dp) _wrap(QSP_ARG  dst_dp,src_dp)

/* scale.c */
extern void _scale(QSP_ARG_DECL  Data_Obj *dp,double desmin,double desmax);
#define scale(dp,desmin,desmax) _scale(QSP_ARG  dp,desmin,desmax)


#endif /* _VEC_UTIL_H_ */

