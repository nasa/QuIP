
#ifndef _VISTAHDR_H_

/* the vista file header consists of 9 short words */

typedef struct vista_hdr {
	short	dummy1;		/* 0 don't know what this is for */
	short	always_2;	/* 1 don't know, but is always 2!? */
	short	dummies[4];	/* 2-5 */
	short	ncols;		/* 6 */
	short	nrows;		/* 7 */

				/* this may not be just pixel size... */
				/* when doing putpic we get 2 bytes w/ 8 !? */
	short	pixel_size;	/* 8 */
} Vista_Hdr;

#define N_VH_WORDS	(sizeof(Vista_Hdr)/sizeof(short))

#define _VISTAHDR_H_
#endif /* _VISTAHDR_H_ */

