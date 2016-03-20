
#ifndef _CMAPS_H_
#define _CMAPS_H_

#include "data_obj.h"
#include "display.h"

/* colormap allocation */

#define N_COLORS	256
#define N_COMPS		4	/* RGBA */
#define ALPHA_INDEX	3

/* on linux w/ kde, all sorts of colors get used...
 * when we started this stuff way-back-when on the suns,
 * only the first 8 or so entries were used for the window
 * border colors, etc...  the system w/ kde does have a reasonable
 * colormap, a sample rgb cube and a grayscale ramp...
 * we should figure this out some time!  BUG
 */

/*#define NC_SYSTEM	256 */
#define NC_SYSTEM	16

#define NC_GRAYSCALE	32


/* that makes 48 - 208 remain */

/* lum ratios are approx 1 - 3 - 6 */
/* number of levels:
 * 2 x 6 x 10 = 120
 * 3 x 6 x 10 = 180
 * 3 x 6 x 11 = 198
 */

#define N_BLUE_LEVELS	3
#define N_RED_LEVELS	6
#define N_GREEN_LEVELS	11
#define NC_COLOR	(N_BLUE_LEVELS*N_RED_LEVELS*N_GREEN_LEVELS)

#define SYSTEM_BASE	0
#define GRAYSCALE_BASE	(SYSTEM_BASE+NC_SYSTEM)		/* 16 */
#define COLOR_BASE	(GRAYSCALE_BASE+NC_GRAYSCALE)	/* 48 */

#ifdef HAVE_X11
/* globals */
extern Dpyable *current_dpyp;
#endif /* HAVE_X11 */

#ifdef HAVE_X11
#define CM_DATA(dp,component,index)	(*( ((u_char *)OBJ_DATA_PTR(dp)) +component*OBJ_COMP_INC(dp)+index*OBJ_PXL_INC(dp)))
//#define CHECK_DPY(func_name)	if( current_dpyp == NULL ){sprintf(error_string,"%s:  current_dpyp is null!?",func_name); NERROR1(error_string);}
#else
#define CM_DATA(dp,component,index)	0
//#define CHECK_DPY(func_name)
#endif

#define CM_ADDR(dp,component,index)	( ((u_char *)OBJ_DATA_PTR(dp)) +component*OBJ_COMP_INC(dp)+index*OBJ_PXL_INC(dp))


/* formerly in cmflags.h */


/* flags for component modulation */
#define NLEVELS			1	/* nlevels specified */
#define SETWHITE		2	/* white point specified */
#define SETBITVECS		4	/* vectors specified */

/* flags for linearization */
#define LININIT			8	/* linearization specified */
#define LINRD			16	/* calibration data read */

#define RANGE_ERR		128	/* out of range for current lut */

#define SET_CM_FLAG(code)	cm_flags |= code
#define CLR_CM_FLAG(code)	cm_flags &= ~(code)

#define CM_FLAG_IS_SET(code)	( cm_flags & (code) )
#define CM_FLAG_IS_CLR(code)	( !(cm_flags & (code)) )
#define CM_FLAG_MATCHES(code)	( (cm_flags & code) == code )

/* #define WANTING(code)	( (desired & code) && !(cm_flags & code) ) */

/* transient states */
#define IMMEDIATE		1	/* flush colors to HW immediately */

#define SET_CM_STATE(code)	cm_state |= code
#define CLR_CM_STATE(code)	cm_state &= ~(code)

#define CM_IS_IMMEDIATE		(cm_state & IMMEDIATE)

extern long cm_flags,cm_state;

#ifndef FALSE
#define FALSE	0
#endif /* FALSE */
#ifndef TRUE
#define TRUE	1
#endif /* TRUE */


/* prototypes */

/* cmfuncs.c */
extern void		push_cm_state(void);
extern void		pop_cm_state(void);
extern void		cm_immediate(long immediate);
extern void		update_all(void);
#define setcolor(c,r,g,b)	_setcolor(QSP_ARG  c,r,g,b)
extern void		_setcolor(QSP_ARG_DECL  int c, int r, int g, int b);
extern void		const_cmap(QSP_ARG_DECL  int base,int n,int r,int g,int b);
extern void		make_grayscale(QSP_ARG_DECL  int base,int n_colors);
extern void		make_rgb(QSP_ARG_DECL  int base, int nr,int ng, int nb);
extern void		poke_lut(QSP_ARG_DECL  int c, int r, int g, int b);
extern void		setmap(QSP_ARG_DECL  Data_Obj *);
extern void		getmap(Data_Obj *);
extern Data_Obj *	new_colormap(QSP_ARG_DECL  const char *);
extern void		set_colormap(Data_Obj *);
#ifdef HAVE_X11
extern void		default_cmap(QSP_ARG_DECL  Dpyable *);
extern void		select_cmap_display(Dpyable *);
#endif /* HAVE_X11 */
extern int		color_index_out_of_range(QSP_ARG_DECL  u_int index);
extern void		update_if(void);

/* funcvec.c */
extern void	dump_lut(Data_Obj *cm_dp);

/* linear.c */
extern void	lin_setup(QSP_ARG_DECL  Data_Obj *,double gam,double vz);


/* bplanes.c */

extern void	set_lvls_per_comp(int n);
extern void	set_c_amps(QSP_ARG_DECL  int index);
extern void	count(QSP_ARG_DECL  int digit,int offset);
extern void	set_ncomps(int n);
extern int	get_ncomps(void);
extern void	set_comp_amps(QSP_ARG_DECL  float *amps);
extern void	sine_mod_amp(QSP_ARG_DECL  int nframes, float *phases,
			int period, float*envelope, const char *lutstem);
extern void	set_base_index(int i);
extern int	get_base_index(void);
extern void	setwhite(float *white);
extern void	set_bit_vecs(float veclist[][3]);
extern void	set_bitplanes(QSP_ARG_DECL  int nplanes, float *amps);
extern void	set_bits_per_comp(int n);

/* alpha.c */

extern void	set_alpha(int index,int alpha);
extern void	index_alpha(int index,int lv,int hv);
extern void	const_alpha(QSP_ARG_DECL  int value);

/* verluts.c */
extern void	verluts(SINGLE_QSP_ARG_DECL);


#endif /* ! _CMAPS_H_ */

