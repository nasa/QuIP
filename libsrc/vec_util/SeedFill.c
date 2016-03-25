#include "quip_config.h"

#include "quip_prot.h"
#include "ggem.h"
#include "vec_util.h"

/*
 * A Seed Fill Algorithm
 * by Paul Heckbert
 * from "Graphics Gems", Academic Press, 1990
 *
 * user provides pixelread() and pixelwrite() routines
 */

/*
 * fill.c : simple seed fill program
 * Calls pixelread() to read pixels, pixelwrite() to write pixels.
 *
 * Paul Heckbert	13 Sept 1982, 28 Jan 1987
 */

typedef struct {		/* window: a discrete 2-D rectangle */
	int x0, y0;			/* xmin and ymin */
	int x1, y1;			/* xmax and ymax (inclusive) */
} Window;

typedef int Pixel;		/* 1-channel frame buffer assumed */

typedef struct {short y, xl, xr, dy;} Segment;
/*
 * Filled horizontal segment of scanline y for xl<=x<=xr.
 * Parent segment was on line y-dy.  dy=1 or -1
 */

#define MAX_STACK_DEPTH 10000		/* max depth of stack */

#define PUSH(Y, XL, XR, DY)	/* push new segment on stack */ \
								\
	if (		sp<stack+MAX_STACK_DEPTH &&		\
			Y+(DY)>=0 &&				\
			Y+(DY)<height ){ 			\
		sp->y = (short) Y;				\
		sp->xl = (short) XL;				\
		sp->xr = (short) XR;				\
		sp->dy = (short) DY;				\
		sp++;						\
	}

#define POP(Y, XL, XR, DY)	/* pop segment off stack */ 	\
	{							\
		sp--;						\
		Y = (short) sp->y+(DY = (short) sp->dy);	\
		XL = (short) sp->xl;				\
		XR = (short) sp->xr;				\
	}

/*
 * fill: set the pixel at (x,y) and all of its 4-connected neighbors
 * with the same pixel value to the new pixel value nv.
 * A 4-connected neighbor is a pixel above, below, left, or right of a pixel.
 */

#define IN_FILL_REGION		(abs(pixelread(x, y)-ov) <= tolerance)
#define OUTSIDE_FILL_REGION	(abs(pixelread(x, y)-ov) >  tolerance)

/*
 * JBM modifications for floating point:
 *
 * Fill if the current (test) pixel value is within tolerance of the
 * first (original) pixel value...
 */

/* x, y = seed point */
/* nv = new pixel value */
void ggem_fill(QSP_ARG_DECL  int x, int y, int width, int height, Pixel nv, int tolerance)
{
	int l, x1, x2, dy;
	Pixel ov;	/* old pixel value */
	Segment stack[MAX_STACK_DEPTH], *sp = stack;	/* stack of filled segments */

	ov = pixelread(x, y);		/* read pv at seed point */


	/* why is this a problem??? */

	if( abs(nv-ov) <= tolerance ){
		sprintf(ERROR_STRING,
	"ggem_fill:  Value %d at seed point %d, %d is within tolerance %d of new value %d",
			ov,x,y,tolerance,nv);
		WARN(ERROR_STRING);
		return;
	}

	/* originally this returned here if ov==nv */

	if (x<0 || x>=width || y<0 || y>=height) return;
	PUSH(y, x, x, 1);			/* needed in some cases */
	PUSH(y+1, x, x, -1);		/* seed segment (popped 1st) */

	while (sp>stack) {
		/* pop segment off stack and fill a neighboring scan line */
		POP(y, x1, x2, dy);

		/*
		 * segment of scan line y-dy for x1<=x<=x2 was previously filled,
		 * now explore adjacent pixels in scan line y
		 */

		/* fill to the left */
		for (x=x1; x>=0 && IN_FILL_REGION ; x--)
			pixelwrite(x, y, nv);

		if (x>=x1)		/* did anything happen? */
			goto skip;	/* no */

		l = x+1;
		if (l<x1)				/* leak on left? */
			PUSH(y, l, x1-1, -dy);		/* reverse y direction */

		x = x1+1;
		do {
			/* fill to the right */
			for (; x<width && IN_FILL_REGION ; x++)
				pixelwrite(x, y, nv);
	
			PUSH(y, l, x-1, dy);	/* continue */

			if (x>x2+1)		/* leak on right? */
				PUSH(y, x2+1, x-1, -dy);
skip:			for (x++; x<=x2 && OUTSIDE_FILL_REGION ; x++)
				;
			l = x;
		} while (x<=x2);
	}
}

void gen_fill(incr_t x, incr_t y, Data_Obj *dp, int (*inside_func)(long,long), void (*fill_func)(long,long) )
{
	incr_t width, height;
	incr_t l;
	long dy;
	incr_t x1, x2;
	Segment stack[MAX_STACK_DEPTH], *sp = stack;	/* stack of filled segments */

	width = OBJ_COLS(dp);
	height = OBJ_ROWS(dp);

	/* originally this returned here if ov==nv */

	if (x<0 || x>=width || y<0 || y>=height) return;

	PUSH(y, x, x, 1);			/* needed in some cases */
	PUSH(y+1, x, x, -1);		/* seed segment (popped 1st) */

	while (sp>stack) {
		/* pop segment off stack and fill a neighboring scan line */
		POP(y, x1, x2, dy);
//sprintf(ERROR_STRING,"Popped y = %ld   x1 = %ld   x2 = %ld    dy = %ld",y,x1,x2,dy);
//advise(ERROR_STRING);

		/*
		 * segment of scan line y-dy for x1<=x<=x2 was previously filled,
		 * now explore adjacent pixels in scan line y
		 */

		/* fill to the left */
		for (x=x1; (x>=0) && inside_func(x,y) ; x--){
//sprintf(ERROR_STRING,"Filling to the left, x = %ld   y = %ld",x,y);
//advise(ERROR_STRING);
			fill_func(x, y);
		}

		if (x>=x1){		/* did anything happen? */
//sprintf(ERROR_STRING,"nothing happened, skipping, x = %ld   x1 = %ld...",x,x1);
//advise(ERROR_STRING);
			goto skip;	/* no */
		}

		l = x+1;
		if (l<x1){				/* leak on left? */
//sprintf(ERROR_STRING,"Leak on left, Pushing y = %ld   l = %ld   x1-1 = %ld    -dy = %ld",y,l,x1-1,-dy);
//advise(ERROR_STRING);
			PUSH(y, l, x1-1, -dy);		/* reverse y direction */
		}

		x = x1+1;
		do {
			/* fill to the right */
			for (; x<width && inside_func(x,y) ; x++){
//sprintf(ERROR_STRING,"Filling to the right, x = %ld   y = %ld",x,y);
//advise(ERROR_STRING);
				fill_func(x, y);
			}
	
//sprintf(ERROR_STRING,"Pushing y = %ld   l = %ld   x-1 = %ld    dy = %ld",y,l,x-1,dy);
//advise(ERROR_STRING);
			PUSH(y, l, x-1, dy);	/* continue */

			if (x>x2+1){		/* leak on right? */
//sprintf(ERROR_STRING,"Leak on right, Pushing y = %ld   x2+1 = %ld   x-1 = %ld    -dy = %ld",y,x2+1,x-1,-dy);
//advise(ERROR_STRING);
				PUSH(y, x2+1, x-1, -dy);
			}
skip:			
//sprintf(ERROR_STRING,"after skip label, x = %ld, x2 = %ld",x,x2);
//advise(ERROR_STRING);
			for (x++; x<=x2 && (!inside_func(x,y)) ; x++)
				;
//sprintf(ERROR_STRING,"x advanced to %ld, x2 = %ld",x,x2);
//advise(ERROR_STRING);
			l = x;
		} while (x<=x2);
	}
}

