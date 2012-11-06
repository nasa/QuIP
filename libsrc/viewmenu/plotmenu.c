#include "quip_config.h"

char VersionId_viewmenu_plotmenu[] = QUIP_VERSION_STRING;

#ifdef HAVE_X11

/* plot subroutines using pixrect library */
#include <stdio.h>

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#include "getbuf.h"
#include "viewer.h"
#include "debug.h"
#include "data_obj.h"
#include "view_cmds.h"
#include "view_util.h"

#include "get_viewer.h"

#define std_type	float

static std_type sx1,sx2,sy1,sy2;

static int lat_long_hack=0;	/* BUG for plotting the globe we don't want to wrap
				 * around, but now we have no way to set this...
				 * Need a better solution for plotting the globe.
				 */


#define XYPLOT_LOOP( this_type, init_statements, loop_statements )			\
											\
	init_statements;								\
											\
	for(k=0;k<dp->dt_frames;k++){							\
		for(i=0;i<dp->dt_rows;i++){						\
			p = (this_type *) dp->dt_data;					\
			p += i*dp->dt_rinc + k*dp->dt_finc;				\
			for(j=0;j<dp->dt_cols;j++){					\
											\
				loop_statements;					\
											\
				p += dp->dt_pinc;					\
											\
				if( j==0 ) xp_fmove(x,y);				\
											\
				/* remove the "else" here so that			\
				 * column vectors will draw as single points		\
				 */							\
											\
				xp_fcont(x,y);						\
			}								\
		}									\
	}

COMMAND_FUNC( do_xp_arc )
{
	float cx,cy,x1,y1,x2,y2;

	cx=HOW_MUCH("center x");
	cy=HOW_MUCH("center y");
	x1=HOW_MUCH("arc start x");
	y1=HOW_MUCH("arc start y");
	x2=HOW_MUCH("arc end x");
	y2=HOW_MUCH("arc end y");
	xp_farc(cx,cy,x1,y1,x2,y2);
}

COMMAND_FUNC( do_xp_fill_arc )
{
	float cx,cy,x1,y1,x2,y2;

	cx=HOW_MUCH("center x");
	cy=HOW_MUCH("center y");
	x1=HOW_MUCH("arc start x");
	y1=HOW_MUCH("arc start y");
	x2=HOW_MUCH("arc end x");
	y2=HOW_MUCH("arc end y");
	xp_ffill_arc(cx,cy,x1,y1,x2,y2);
}

COMMAND_FUNC( do_xp_fill_polygon )
{
	int* x_vals=NULL, *y_vals = NULL;
	int num_points;
	int i;

	num_points = HOW_MUCH("number of polygon points");
	x_vals = (int *) getbuf(sizeof(int) * num_points);
	y_vals = (int *) getbuf(sizeof(int) * num_points);
	
	for (i=0; i < num_points; i++) {
		char s[100];
		sprintf(s, "point %d x value", i+1);
		x_vals[i] = HOW_MUCH(s);
		sprintf(s, "point %d y value", i+1);
		y_vals[i] = HOW_MUCH(s);
	}

	xp_fill_polygon(num_points, x_vals, y_vals);

	givbuf(x_vals);
	givbuf(y_vals);
}

COMMAND_FUNC( do_xp_space )
{
	sx1=HOW_MUCH("minimum x coord");
	sy1=HOW_MUCH("minimum y coord");
	sx2=HOW_MUCH("maximum x coord");
	sy2=HOW_MUCH("maximum y coord");

	/* There is no real reason to require this, except that
	 * we assumed it must have been a mistake if the user did this...
	 * But we often deal with situations where the y coordinate needs
	 * to be turned upside down to make various conventions mesh...
	 */
	/*
	if( sx2 <= sx1 ){
	WARN("do_xp_space:  x max coord must be larger than x min coord");
		return;
	}
	if( sy2 <= sy1 ){
	WARN("do_xp_space:  y max coord must be larger than y min coord");
		return;
	}
	*/
	if( sx2 == sx1 ){
	WARN("do_xp_space:  x max coord must be different from x min coord");
		return;
	}
	if( sy2 == sy1 ){
	WARN("do_xp_space:  y max coord must be different from y min coord");
		return;
	}

	xp_fspace(sx1,sy1,sx2,sy2);
}

COMMAND_FUNC( do_xp_move )
{
	std_type x,y;

	x=HOW_MUCH("x coord");
	y=HOW_MUCH("y coord");
	xp_fmove(x,y);
}

COMMAND_FUNC( do_xp_cont )
{
	std_type x,y;

	x=HOW_MUCH("x coord");
	y=HOW_MUCH("y coord");
	xp_fcont(x,y);
}

COMMAND_FUNC( do_xp_point )
{
	std_type x,y;

	x=HOW_MUCH("x coord");
	y=HOW_MUCH("y coord");
	xp_fpoint(x,y);
}

COMMAND_FUNC( do_xp_line )
{
	std_type x1,x2,y1,y2;

	x1=HOW_MUCH("first x coord");
	y1=HOW_MUCH("first y coord");
	x2=HOW_MUCH("second x coord");
	y2=HOW_MUCH("second y coord");
	xp_fline(x1,y1,x2,y2);
}

COMMAND_FUNC( do_xp_select )
{
	int color;
	
	color=HOW_MANY("color index");
	xp_select(color);
}

COMMAND_FUNC( do_rdplot )
{
	FILE *fp;

	fp=TRY_OPEN( NAMEOF("filename"), "r") ;
	if( !fp ) return;
	rdplot(QSP_ARG  fp);
}

COMMAND_FUNC( do_xp_erase )
{
	xp_erase();
}

COMMAND_FUNC( do_dumpdraw )
{
	Viewer *vp;

	/* BUG dump drawlist is implemented in xsupp (where the draw
	 * commands are remembered for the redraw function), but
	 * plot_vp is static to view/xplot.c ...
	 *
	 * the user shouldn't have to specify the viewer twice,
	 * or this command should be in a different menu.
	 */

	GET_VIEWER("do_dumpdraw")
	if( vp == NO_VIEWER ) return;

	dump_drawlist(vp);
}

static int bad_plot_vec2(QSP_ARG_DECL Data_Obj *dp,dimension_t n_comps_expected,const char *funcname)
{
	if( dp==NO_OBJ ) return 1;

	if( dp->dt_prec != PREC_SP && dp->dt_prec != PREC_DP ){
		sprintf(error_string,
			"%s:  data vector %s (%s) should have float or double precision",
			funcname,
			dp->dt_name,prec_name[MACHINE_PREC(dp)]);
		WARN(error_string);
		return 1;
	}
	if( dp->dt_comps != n_comps_expected ){
		sprintf(error_string,"%s:  data vector %s (%d) should have %d components",
			funcname,dp->dt_name,dp->dt_comps,n_comps_expected);
		WARN(error_string);
		return 1;
	}
	return 0;
}

static int bad_plot_vec(QSP_ARG_DECL Data_Obj *dp,dimension_t n_comps_expected,const char *funcname)
{
	if( dp==NO_OBJ ) return 1;

	if( dp->dt_prec != PREC_SP ){
		sprintf(error_string,
			"%s:  data vector %s (%s) should have float precision",
			funcname,
			dp->dt_name,prec_name[MACHINE_PREC(dp)]);
		WARN(error_string);
		return 1;
	}
	if( dp->dt_comps != n_comps_expected ){
		sprintf(error_string,"%s:  data vector %s (%d) should have %d components",
			funcname,dp->dt_name,dp->dt_comps,n_comps_expected);
		WARN(error_string);
		return 1;
	}
	return 0;
}

#ifdef FOOBAR
static void get_data_params(Data_Obj *dp, u_long *np, long *incp)
{
	if( dp->dt_cols==1 ){	/* maybe this is a column vector? */
		*np=dp->dt_rows;
		*incp = dp->dt_rinc;
	} else {
		*np=dp->dt_cols;
		*incp = dp->dt_pinc;
	}
}
#endif /* FOOBAR */

COMMAND_FUNC( do_xplot )
{
	Data_Obj *dp;
	u_long i,j,k;

	std_type x,y,y0,dy,*p;

	dp=PICK_OBJ("data vector");

	if( bad_plot_vec(QSP_ARG dp,1,"xplot") ) return;

	INSIST_RAM(dp,"xplot")

	dy=1;

	XYPLOT_LOOP( std_type, y0=0, x = *p; y = y0; y0+=dy )

}

COMMAND_FUNC( do_yplot )
{
	Data_Obj *dp;
	u_long i,j,k;
	std_type x,y,dx,*p;
	std_type x0;

	dp=PICK_OBJ("data vector");

	if( bad_plot_vec(QSP_ARG dp,1,"yplot") ) return;

	INSIST_RAM(dp,"yplot")

	dx=1;

	XYPLOT_LOOP( std_type, x0=0, y = *p; x = x0; x0+=dx )

}

COMMAND_FUNC( do_cyplot )
{
	Data_Obj *dp;
	Data_Obj *cdp;
	u_long i, np;		/* number of points */
	long inc;
	long cinc;
	std_type x,y,dx,*yp;
	char *cp;

	dp=PICK_OBJ("data vector");
	cdp=PICK_OBJ("color vector");

	if( dp==NO_OBJ ) return;
	if( cdp==NO_OBJ ) return;

	INSIST_RAM(dp,"cyplot")
	INSIST_RAM(cdp,"cyplot")

	if( dp->dt_prec != PREC_SP ){
		WARN("do_cyplot:  data vector should be float");
		return;
	}
	if( cdp->dt_prec != PREC_BY ){
		WARN("color vector should be byte");
		return;
	}
	if( !dp_same_size(QSP_ARG  dp,cdp,"do_cyplot") ){
		sprintf(error_string,"data vector %s and color vector %s must have identical sizes",
			dp->dt_name,cdp->dt_name);
		WARN(error_string);
		return;
	}

	if( dp->dt_cols==1 ){	/* maybe this is a column vector? */
		np=dp->dt_rows;
		inc = dp->dt_rinc;
		cinc = cdp->dt_rinc;
	} else {
		np=dp->dt_cols;
		inc = dp->dt_pinc;
		cinc = cdp->dt_pinc;
	}


	x=0;
	dx=1;
	i=np-1;
	yp = (std_type *) dp->dt_data;
	cp = (char *) cdp->dt_data;

	xp_fmove(x,*yp);
	xp_select(*cp);
	xp_fcont(x,*yp);
	while(i--){
		yp += inc;
		cp += cinc;
		x += dx;
		y = *yp;
		xp_select(*cp);
		xp_fcont(x,y);
	}
}

static void double_xyplot(Data_Obj *dp)
{
	u_long i,j,k;
	double x,y,*p;

	XYPLOT_LOOP( double, , x = *p; y = *(p+dp->dt_cinc) )
}

static void float_xyplot(Data_Obj *dp)
{
	u_long i,j,k;
	float x,y,*p;

	XYPLOT_LOOP( float, , x = *p; y = *(p+dp->dt_cinc) )
}

COMMAND_FUNC( do_xyplot )
{
	Data_Obj *dp;

	dp=PICK_OBJ("data vector");
	if( dp==NO_OBJ ) return;

	INSIST_RAM(dp,"xyplot")

	if( bad_plot_vec2(QSP_ARG dp,2,"xyplot") ) return;

	switch( MACHINE_PREC(dp) ){
		case PREC_SP:
			float_xyplot(dp);
			break;
		case PREC_DP:
			double_xyplot(dp);
			break;
		default:
			sprintf(error_string,"do_xyplot:  unhandled precision %s (object %s)",
				name_for_prec(dp->dt_prec),dp->dt_name);
			WARN(error_string);
			break;
	}
}

/*
 * This is like xyplot, except that we don't draw stuff with negative z coords...
 */

COMMAND_FUNC( do_xyzplot )
{
	Data_Obj *dp;
	dimension_t i,j;
	std_type x,y,z,*p;
	std_type lastx,lasty;
#define DOWN	1
#define UP	2
	int pen_state=UP;

	dp=PICK_OBJ("data vector");
	if( dp==NO_OBJ ) return;

	INSIST_RAM(dp,"xyzplot")

	if( bad_plot_vec(QSP_ARG dp,3,"xyzplot") ) return;

	for(i=0;i<dp->dt_rows;i++){
		p = (std_type *) dp->dt_data;
		p += i*dp->dt_rinc;
		for(j=0;j<dp->dt_cols;j++){
			x = *p;
			y = *(p + dp->dt_cinc);
			z = *(p + 2*dp->dt_cinc);
			p += dp->dt_pinc;

			/* removed else so that a single column data set will plot the pt */
			if( z < 0 ){
				lastx = x;
				lasty = y;
				pen_state = UP;
			}

			if( z >= 0 ){	/* draw this point */

				/* THis is a hack for plotting lat/long without
				 * wrap-around...  BUG
				 */
				if( pen_state == UP ){
					xp_fmove(x,y);
				}
				if( lat_long_hack ){
					if( fabs(lastx-x) < 180 &&
						fabs(lasty-y) < 180 ){
						xp_fcont(x,y);
						lastx=x;
						lasty=y;
					} else {
						xp_move(x,y);
						lastx=x;
						lasty=y;
					}
				} else {
					xp_fcont(x,y);
					lastx=x;
					lasty=y;
				}
				pen_state=DOWN;
			}
		}
	}
}

static COMMAND_FUNC( do_xp_circ )
{
	std_type rad;

	rad = HOW_MUCH("radius");
	xp_circle(rad);
}

static COMMAND_FUNC( do_plot_string )
{
	const char *s;

	s=NAMEOF("string");
	xp_text(s);
}

static COMMAND_FUNC( do_tell_plot_space ){ tell_plot_space(); }

static Command xp_ctbl[]={
{ "space",	do_xp_space,	"define plotting space"			},
{ "move",	do_xp_move,	"move CAP"				},
{ "cont",	do_xp_cont,	"draw vector"				},
{ "line",	do_xp_line,	"draw line"				},
{ "arc",	do_xp_arc,	"draw arc"				},
{ "circle",	do_xp_circ,	"draw circle"				},
{ "fill_arc",   do_xp_fill_arc, "draw and fill an arc"                  },
{ "fill_polygon", do_xp_fill_polygon, "draw and fill a polygon"         },
{ "select",	do_xp_select,	"select drawing color"			},
{ "erase",	do_xp_erase,	"erase"					},
{ "yplot",	do_yplot,	"graph a vector of data values"		},
{ "xplot",	do_xplot,	"plot a function along the y axis"	},
{ "string",	do_plot_string,	"draw a text string"			},
{ "cyplot",	do_cyplot,	"graph a vector of data values, w/ colors"},
{ "xyplot",	do_xyplot,	"graph an image of x-y pairs"		},
{ "xyzplot",	do_xyzplot,	"graph an image of x-y pairs, for z>0"	},
	/*
{ "poly",	do_xp_poly,	"draw filled polygon"			},
{ "bitwise",	multiop,	"do bit operations between images"	},
	*/
{ "plot",	do_rdplot,	"interpret plot(5) file"		},
{ "dump",	do_dumpdraw,	"dump viewer drawlist to stdout"	},
{ "info",	do_tell_plot_space,"report plotting space"			},
{ "quit",	popcmd,		"quit"					},
{ NULL,		NULL,		NULL					}
};

COMMAND_FUNC( xp_menu )
{
	Viewer *vp;

	insure_x11_server(SINGLE_QSP_ARG);
	GET_VIEWER("xp_menu")
	if( vp != NO_VIEWER )
		xp_setup(vp);

	/*
	if( CANNOT_SHOW(vp) ){
		advise("waiting for viewer to be mapped or exposed");
		while( CANNOT_SHOW(vp) ) event_loop();
	}
	*/
	PUSHCMD(xp_ctbl,"xplot");
}


#endif /* HAVE_X11 */

