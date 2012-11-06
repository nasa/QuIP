#ifndef NO_POINT

/* utilities for 2D geometry */

typedef struct point {
	atc_type	p_c[2];
} Point;

#define NO_POINT	((Point *)NULL)

#define p_x	p_c[0]
#define p_y	p_c[1]

typedef struct vector {
	atc_type	v_c[2];
} Vector;

#define v_x	v_c[0]
#define v_y	v_c[1]

#define PERPENDICULAR( vp1 , vp2 )				\
	{							\
		(vp1)->v_y =   (vp2)->v_x;			\
		(vp1)->v_x = - (vp2)->v_y;			\
	}

#define DOT_PROD( vp1, vp2 )	( (vp1)->v_x * (vp2)->v_x + (vp1)->v_y * (vp2)->v_y )
#define SCALE_VECTOR(vp,factor) { (vp)->v_x *= factor; (vp)->v_y *= factor; }
#define DISPLACE_POINT(ptp,vecp)	{ (ptp)->p_x += (vecp)->v_x; (ptp)->p_y += (vecp)->v_y; }

#define DELTA_X(p1p,p2p)	( (p1p)->p_x - (p2p)->p_x )
#define DELTA_Y(p1p,p2p)	( (p1p)->p_y - (p2p)->p_y )

#define DIST( p1p, p2p )					\
	sqrt( DELTA_X(p1p,p2p)*DELTA_X(p1p,p2p) + DELTA_Y(p1p,p2p)*DELTA_Y(p1p,p2p) )

#define MAG_SQ( vecp )		( (vecp)->v_x*(vecp)->v_x + (vecp)->v_y*(vecp)->v_y )

#endif /* ! NO_POINT */

