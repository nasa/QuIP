
/*
 * HIPL Picture Header Format Standard
 *
 * Michael Landy - 2/1/82
 */

#ifndef NO_HIPL_FMT
#define NO_HIPL_FMT

typedef struct header {
	const char *	orig_name;	/* The originator of this sequence */
	const char *	seq_name;	/* The name of this sequence */
	int		num_frame;	/* The number of frames in this sequence */
	const char *	orig_date;	/* The date the sequence was originated */
	int		rows;		/* The number of rows in each image */
	int		cols;		/* The number of columns in each image */
	int		bits_per_pixel;	/* The number of significant bits per pixel */
	int		bit_packing;	/* Nonzero if bits were packed contiguously */
	int		pixel_format;	/* The format of each pixel, see below */
	const char *	seq_history;	/* The sequence's history of transformations */
	const char *	seq_desc;	/* Descriptive information */
} Header ;

/*
 * Pixel Format Codes
 */

#define	PFBYTE	0		/* Bytes interpreted as integers */
#define PFSHORT	1		/* Short int's interpreted as integers */
#define PFINT	2		/* Int's */
#define	PFFLOAT	3		/* Float's */
#define	PFCOMPLEX 4		/* 2 Float's interpreted as (real,imaginary) */
#define PFASCII	5		/* Ascii representation, with linefeeds after each row */
#define PFDBL	6		/* double prec, jbm addition 7/31/96 */
#define PFQUAD1	11		/* quad-tree encoding */
#define PFBHIST	12		/* histogram of byte image */
#define PFSPAN	13		/* spanning tree format */
#define PLOT3D	24		/* plot-3d format */
#define PFAHC	400		/* adaptive hierarchical encoding */
#define PFOCT	401		/* oct-tree encoding */
#define	PFBT	402		/* binary tree encoding */
#define PFAHC3	403		/* 3-d adaptive hierarchical encoding */
#define PFBQ	404		/* binquad encoding */
#define PFRLED	500		/* run-length encoding */
#define PFRLEB	501		/* run-length encoding, line begins black */
#define PFRLEW	502		/* run-length encoding, line begins white */

/*
 * Bit packing formats
 */

#define	MSBFIRST 1		/* bit packing - most significant bit first */
#define	LSBFIRST 2		/* bit packing - least significant bit first */

#define FBUFLIMIT 30000

/*
 * For general readability
 */

#define	TRUE	1
#define	FALSE	0

#endif /* NO_HIPL_FMT */
