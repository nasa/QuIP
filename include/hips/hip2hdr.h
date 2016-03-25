/*
 * Copyright (c) 1991 Michael Landy
 *
 * Disclaimer:  No guarantees of performance accompany this software,
 * nor is any responsibility assumed on the part of the authors.  All the
 * software has been tested extensively and every effort has been made to
 * insure its reliability.
 */

/*
 * hips_header.h - definitions related to the HIPS header
 *
 * Michael Landy - 12/28/90
 */

#ifndef HIPS2_HEADER

#include "hipbasic.h"

/* The HIPS header as stored in memory */

typedef struct hips2_header {
	const char *	orig_name;	/* The originator of this sequence */
	const char *	seq_name;	/* The name of this sequence */
	int		num_frame;	/* The number of frames in this sequence */
	const char *	orig_date;	/* The date the sequence was originated */
	int		orows;		/* The number of rows in each stored image */
	int		ocols;		/* The number of columns in each stored image */
	int		rows;		/* The number of rows in each image (ROI) */
	int		cols;		/* The number of columns in each image (ROI) */
	int		frow;		/* The first ROI row */
	int		fcol;		/* The first ROI col */
	int		pixel_format;	/* The format of each pixel */
	int		numcolor;	/* The number of color frames per image */
	int		numpix;		/* The number of pixels per stored frame */
	hsize_t		sizepix;	/* The number of bytes per pixel */
	hsize_t		sizeimage;	/* The number of bytes per stored frame */
	h_byte *	image;		/* The image itself */
	Bool		imdealloc;	/* if nonzero, free image when requested */
	h_byte *	firstpix;	/* Pointer to first pixel (for ROI) */
	int		sizehist;	/* Number of bytes in history (excluding
					null, including <newline>) */
	const char *	seq_history;	/* The sequence's history of transformations */
	Bool		histdealloc;	/* If nonzero, free history when requested */
	int		sizedesc;	/* Number of bytes in description (excluding
					null, including <newline>) */
	const char *	seq_desc;	/* Descriptive information */
	Bool		seqddealloc;	/* If nonzero, free desc when requested */
	int		numparam;	/* Count of additional parameters */
	Bool		paramdealloc;	/* If nonzero, free param structures and/or
					param values when requested */
	struct extpar *	params;	/* Additional parameters */
} Hips2_Header;

#define HIPS2_HEADER		((Hips2_Header *)NULL)

#include "hips_pf.h"		/* pixel format codes */

struct hips_roi {
	int	rows;		/* The number of rows in the ROI */
	int	cols;		/* The number of columns in the ROI */
	int	frow;		/* The first ROI row */
	int	fcol;		/* The first ROI col */
};

/* The Extended Parameters Structure */

struct extpar {
	const char *name;		/* name of this variable */
	int format;		/* format of values (PFBYTE, PFINT, etc.) */
	int count;		/* number of values */
	union {
		h_byte v_b;	/* PFBYTE/PFASCII, count = 1 */
		int v_i;	/* PFINT, count = 1 */
		short v_s;	/* PFSHORT, count = 1 */
		float v_f;	/* PFFLOAT, count = 1 */
		h_byte *v_pb;	/* PFBYT/PFASCIIE, count > 1 */
		int *v_pi;	/* PFINT, count > 1 */
		short *v_ps;	/* PFSHORT, count > 1 */
		float *v_pf;	/* PFFLOAT, count > 1 */
	} val;
	Bool dealloc;	/* if nonzero, free memory for val */
	struct extpar *nextp;	/* next parameter in list */
};

#define FBUFLIMIT 30000		/* increase this if you use large PLOT3D
					files */
#define	LINELENGTH 400		/* max characters per line in header vars */
#define	NULLPAR	((struct extpar *) 0)


#endif /* HIPS2_HEADER */

