#include "quip_config.h"

char VersionId_viewer_xplot[] = QUIP_VERSION_STRING;

#ifdef HAVE_X11

/* plot subroutines using pixrect library */
#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#include "viewer.h"
#include "xsupp.h"
#include "debug.h"
#include "data_obj.h"


#define X0	0
#define Y0	0

static Viewer *plot_vp=NO_VIEWER;
static int _currx=0, _curry=0;		/* current posn, in screen (pixel) units */
/* static float _screenx, _screeny;
static float _x0,_y0,delx,dely; */

/* local prototypes */
static void unscalefxy(Viewer *vp,float *px,float *py);
static void remember_space(Viewer *vp,float x0,float y0,
				float xdel,float ydel);


void xp_space(int x1,int y1,int x2,int y2)
{
	float _x0,_y0,delx,dely;

	_x0=x1;
	_y0=y1;
	delx=x2-x1;
	dely=y2-y1;
	remember_space(plot_vp,_x0,_y0,delx,dely);
}

/** like libplot space, but takes float args */
void xp_fspace(float x1,float y1,float x2,float y2)
{
	float _x0,_y0,delx,dely;

	_x0=x1;
	_y0=y1;
	delx=x2-x1;
	dely=y2-y1;
	remember_space(plot_vp,_x0,_y0,delx,dely);
}

/* This routine takes a coordinate pair from screen coords to plotting units */

static void unscalefxy(Viewer *vp,float *px,float *py)
{
	float x,y;

	x = (*px) - X0;		y = Y0 + vp->vw_height - ((*py)+1) ;
	x /= (vp->vw_width-1);	y /= (vp->vw_height-1);
	x *= vp->vw_xdel;	y *= vp->vw_ydel;
	*px = vp->vw_xmin + x;	*py = y + vp->vw_ymin;
}

void tell_plot_space(void)
{
	sprintf(msg_str,"Viewer %s:",plot_vp->vw_name);
	prt_msg(msg_str);
	sprintf(msg_str,"\twidth = %d\theight = %d",plot_vp->vw_width,plot_vp->vw_height);
	prt_msg(msg_str);
	sprintf(msg_str,"\txmin = %g\txdel = %g",plot_vp->vw_xmin,plot_vp->vw_xdel);
	prt_msg(msg_str);
	sprintf(msg_str,"\tymin = %g\tydel = %g",plot_vp->vw_ymin,plot_vp->vw_ydel);
	prt_msg(msg_str);
}

void xp_circle(float radius)
{
	float cx,cy;

	if( plot_vp == NO_VIEWER ) return;

	/* current posn is remembered in screen units */
	cx=_currx; cy=_curry;

	/* convert from screen coords to plotting units */
	unscalefxy(plot_vp,&cx,&cy);

	xp_farc(cx,cy,cx+radius,cy,cx+radius,cy);
}

void xp_fill_polygon(int num_points, int* x_vals, int* y_vals)
{
	int *xp,*yp;
	int i;

	if( plot_vp == NO_VIEWER ) return;

	/* BUG should scale the points */
	xp = (int *) getbuf( num_points * sizeof(int) );
	yp = (int *) getbuf( num_points * sizeof(int) );

	for(i=0;i<num_points;i++){
		xp[i] = x_vals[i];
		yp[i] = y_vals[i];
		scalexy(plot_vp,&xp[i],&yp[i]);
	}

	_xp_fill_polygon(plot_vp, num_points, xp, yp);

	givbuf(xp);
	givbuf(yp);
}

void xp_ffill_arc(float cx,float cy,float x1,float y1,float x2,float y2)
{
	float _cx,_cy,_x1,_y1,_x2,_y2;
	int xl,yu,w,h,a1,a2;
	int start,delta;

	if( plot_vp == NO_VIEWER ) return;

	_cx=cx; _cy=cy; scale_fxy(plot_vp,&_cx,&_cy);
	_x1=x1; _y1=y1; scale_fxy(plot_vp,&_x1,&_y1);
	_x2=x2; _y2=y2; scale_fxy(plot_vp,&_x2,&_y2);


	/* compute the radius */

	w = (int) sqrt( (_cx-_x1)*(_cx-_x1) + (_cy-_y1)*(_cy-_y1) );
	xl= _cx - w;
	yu= _cy - w;
	w *= 2;
	h=w;

/*
printf("xp_farc:  ctr %f %f   p1 %f %f   p2 %f %f\n",
_cx,_cy,_x1,_y1,_x2,_y2);
*/

	a1=64.0 * 45.0 * atan2(_cy-_y1,_x1-_cx) / atan(1.0);
	a2=64.0 * 45.0 * atan2(_cy-_y2,_x2-_cx) / atan(1.0);

	while( a1 < 0 ) a1 += 64*360;
	while( a2 < 0 ) a2 += 64*360;

	/* we want to draw CCW from first pt... */
	delta = a2 - a1;
	start = a1;
	while( delta <= 0 ) delta += 64*360;

	_xp_fill_arc(plot_vp,xl,yu,w,h,start,delta);  
}

void xp_arc(int cx,int cy,int x1,int y1,int x2,int y2)
{
	xp_farc((float)cx,(float)cy,
		(float)x1,(float)y1,
		(float)x2,(float)y2 );
}

void xp_farc(float cx,float cy,float x1,float y1,float x2,float y2)
{
	float _cx,_cy,_x1,_y1,_x2,_y2;
	int xl,yu,w,h,a1,a2;
	int start,delta;

	if( plot_vp == NO_VIEWER ) return;

	_cx=cx; _cy=cy; scale_fxy(plot_vp,&_cx,&_cy);
	_x1=x1; _y1=y1; scale_fxy(plot_vp,&_x1,&_y1);
	_x2=x2; _y2=y2; scale_fxy(plot_vp,&_x2,&_y2);


	/* compute the radius */

	w = (int) sqrt( (_cx-_x1)*(_cx-_x1) + (_cy-_y1)*(_cy-_y1) );
	xl= _cx - w;
	yu= _cy - w;
	w *= 2;
	h=w;

/*
printf("xp_farc:  ctr %f %f   p1 %f %f   p2 %f %f\n",
_cx,_cy,_x1,_y1,_x2,_y2);
*/

	a1=64.0 * 45.0 * atan2(_cy-_y1,_x1-_cx) / atan(1.0);
	a2=64.0 * 45.0 * atan2(_cy-_y2,_x2-_cx) / atan(1.0);

	while( a1 < 0 ) a1 += 64*360;
	while( a2 < 0 ) a2 += 64*360;

	/* we want to draw CCW from first pt... */
	delta = a2 - a1;
	start = a1;
	while( delta <= 0 ) delta += 64*360;

	_xp_arc(plot_vp,xl,yu,w,h,start,delta);
}

void scale_fxy(Viewer *vp,float *px,float *py)
{
	float x,y;

	x = (*px);		y = (*py);
	x -= vp->vw_xmin;	y -= vp->vw_ymin;
	x /= vp->vw_xdel;	y /= vp->vw_ydel;
	x *= (vp->vw_width-1);	y *= (vp->vw_height-1);
	*px = X0 + x;		*py = Y0 + vp->vw_height - (y+1);
}

void scalexy(Viewer *vp,int *px,int *py)
{
	float x,y;

	x = (*px);		y = (*py);
	x -= vp->vw_xmin;	y -= vp->vw_ymin;
	x /= vp->vw_xdel;	y /= vp->vw_ydel;
	x *= (vp->vw_width-1);	y *= (vp->vw_height-1);
	*px = X0+ x;		*py = Y0 + vp->vw_height - (y+1);
}

void xp_fmove(float x,float y)
{
	float fx,fy;

	if( plot_vp == NO_VIEWER ) return;

	fx=x; fy=y;
	scale_fxy(plot_vp,&fx,&fy);

	_currx=(int) floor(fx+0.5);
	_curry=(int) floor(fy+0.5);
	_xp_move(plot_vp,_currx,_curry);
}

void xp_move(int x,int y)
{
	if( plot_vp == NO_VIEWER ) return;

	scalexy(plot_vp,&x,&y);
	_currx=x;
	_curry=y;
	_xp_move(plot_vp,_currx,_curry);
}

void xp_fcont(float x,float y)
{
	int ix,iy;
	float fx,fy;

	if( plot_vp == NO_VIEWER ){
		return;
	}

	fx=x; fy=y;
	scale_fxy(plot_vp,&fx,&fy);
	ix=(int)floor(fx+0.5);
	iy=(int)floor(fy+0.5);

	_xp_line(plot_vp,_currx,_curry,ix,iy);

	_currx=ix;
	_curry=iy;
}

void xp_point(int x,int y)
{
	xp_move(x,y);
	xp_cont(x,y);
}

void xp_fpoint(float x,float y)
{
	xp_fmove(x,y);
	xp_fcont(x,y);
}

void xp_cont(int x,int y)
{
	if( plot_vp == NO_VIEWER ) return;

	scalexy(plot_vp,&x,&y);
	_xp_line(plot_vp,_currx,_curry,x,y);
	/* why don't we use cont here??? something to do with scaling??? */
	/* _xp_cont(plot_vp,x,y); */

	_currx=x;
	_curry=y;
}

void xp_line(int x1,int y1,int x2,int y2)
{
	xp_move(x1,y1);
	xp_cont(x2,y2);
}

void xp_fline(float x1,float y1,float x2,float y2)
{
	xp_fmove(x1,y1);
	xp_fcont(x2,y2);
}

#define DEFAULT_FONT_WIDTH	10

/* a hack so that labels produced by graph(1) don't overlap the frame */

/* BUG this should be determined by the font... */
#define PLOT_TEXT_OFFSET	0
#define PIXELS_PER_CHAR		8

void xp_text(const char *s)
{
	float delta;

	if( plot_vp != NO_VIEWER ){
		_xp_text(plot_vp,_currx,_curry+PLOT_TEXT_OFFSET,s);
	}

	/* Probably graph(1) expects that this will move the pointer... */
	/* this is a hack... */
	/* BUG need to get the font size (string length) */
	delta = strlen(s)*PIXELS_PER_CHAR;	/* screen units */

	_currx += delta;
}

void xp_setup(Viewer *vp)
{
	plot_vp = vp;

	if( plot_vp != NO_VIEWER ){
		/* this is just a default... */
		if( vp->vw_xdel != 0.0 ){	/* params already set */
			xp_fspace(vp->vw_xmin,vp->vw_ymin,
				vp->vw_xmin+vp->vw_xdel,vp->vw_ymin+vp->vw_ydel);
		} else {			/* use default */
			xp_space(0,0,vp->vw_width-1,vp->vw_height-1);
		}
	}
}

void xp_bgselect(u_long color)
{
	if( plot_vp == NO_VIEWER ) {
		return;
	}

	_xp_bgselect(plot_vp,color);
}

void xp_select(u_long color)
{
	if( plot_vp == NO_VIEWER ) {
		return;
	}

	_xp_select(plot_vp,color);
}

void xp_erase(void)
{
	if( plot_vp != NO_VIEWER ){
		_xp_erase(plot_vp);
	}
}


static void remember_space(Viewer *vp,float _x0,float _y0,float delx,float dely)
{
	if( vp == NO_VIEWER ){
		return;
	}

	vp->vw_xmin = _x0;
	vp->vw_ymin = _y0;
	vp->vw_xdel = delx;
	vp->vw_ydel = dely;
}

#endif /* HAVE_X11 */

