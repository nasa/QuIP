#include <math.h>
#include "conflict.h"

atc_type dist(Point *p1p,Point *p2p)
{
	atc_type dx,dy;

	dx = p1p->p_x - p2p->p_x;
	dy = p1p->p_y - p2p->p_y;

	return( sqrt( dx*dx + dy*dy ) );
}

