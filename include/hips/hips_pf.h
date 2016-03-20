/*
 * Copyright (c) 1991 Michael Landy
 *
 * Disclaimer:  No guarantees of performance accompany this software,
 * nor is any responsibility assumed on the part of the authors.  All the
 * software has been tested extensively and every effort has been made to
 * insure its reliability.
 */

/*
 * hips_pf.h - pixel format codes for hips
 *
 * Michael Landy - 12/28/90
 *
 * this file shared by hips2 & hips1, although not all
 * of the codes are used or recognized by hips1
 * jbm 8/5/96
 */

#ifndef	PFBYTE

/*
 * Pixel Format Codes
 */

#define	PFBYTE		0	/* Bytes interpreted as unsigned integers */
#define PFSHORT		1	/* Short integers (2 bytes) */
#define PFINT		2	/* Integers (4 bytes) */
#define	PFFLOAT		3	/* Float's (4 bytes)*/
#define	PFCOMPLEX 	4	/* 2 Float's interpreted as (real,imaginary) */
#define PFASCII		5	/* ASCII rep, with linefeeds after each row */
#define	PFDOUBLE 	6	/* Double's (8 byte floats) */
#define	PFDBLCOM 	7	/* Double complex's (2 Double's) */
#define PFQUAD		10	/* quad-tree encoding (Mimaging) */
#define PFQUAD1		11	/* quad-tree encoding */
#define PFHIST		12	/* histogram of an image (using ints) */
#define PFSPAN		13	/* spanning tree format */
#define PLOT3D		24	/* plot-3d format */
#define	PFMSBF		30	/* packed, most-significant-bit first */
#define	PFLSBF		31	/* packed, least-significant-bit first */
#define	PFSBYTE		32	/* signed bytes */
#define	PFUSHORT	33	/* unsigned shorts */
#define	PFUINT		34	/* unsigned ints */
#define	PFRGB		35	/* RGB RGB RGB bytes */
#define	PFRGBZ		36	/* RGB0 RGB0 RGB0 bytes */
#define	PFZRGB		37	/* 0RGB 0RGB 0RGB bytes */
#define	PFMIXED		40	/* multiple frames in different pixel formats */
#define	PFBGR		41	/* BGR BGR BGR bytes */
#define	PFBGRZ		42	/* BGR0 BGR0 BGR0 bytes */
#define	PFZBGR		43	/* 0BGR 0BGR 0BGR bytes */
#define PFINTPYR	50	/* integer pyramid */
#define PFFLOATPYR	51	/* float pyramid */
#define PFPOLYLINE	100	/* 2D points */
#define PFCOLVEC	101	/* Set of RGB triplets defining colours */
#define PFUKOOA		102	/* Data in standard UKOOA format */
#define PFTRAINING	104	/* Set of colour vector training examples */
#define PFTOSPACE	105	/* TOspace world model data structure */
#define PFSTEREO	106	/* Stereo sequence (l, r, l, r, ...) */
#define PFRGPLINE	107	/* 2D points with regions */
#define PFRGISPLINE	108	/* 2D points with regions and interfaces */
#define PFCHAIN		200	/* Chain code encoding (Mimaging) */
#define PFLUT		300	/* LUT format (uses Ints) (Mimaging) */
#define PFAHC		400	/* adaptive hierarchical encoding */
#define PFOCT		401	/* oct-tree encoding */
#define	PFBT		402	/* binary tree encoding */
#define PFAHC3		403	/* 3-d adaptive hierarchical encoding */
#define PFBQ		404	/* binquad encoding */
#define PFRLED		500	/* run-length encoding */
#define PFRLEB		501	/* run-length encoding, line begins black */
#define PFRLEW		502	/* run-length encoding, line begins white */
#define PFPOLAR		600	/* rho-theta format (Mimaging) */
#define PFGRLE		601	/* gray scale run-length encoding */
#define PFSRLE		602	/* monochrome run-scale encoding */
#define	PFVFFT3D	701	/* float complex 3D virtual-very fast FT */
#define	PFVFFT2D	702	/* float complex 2D virtual-very fast FT */
#define	PFDVFFT3D	703	/* double complex 3D VFFT */
#define	PFDVFFT2D	704	/* double complex 2D VFFT */
#define	PFVVFFT3D	705	/* float 3D VFFT in separated planes */
#define	PFDVVFFT3D	706	/* double 3D VVFFT in separated planes */

#endif /* PFBYTE */

