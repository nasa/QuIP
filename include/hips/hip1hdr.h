/*
 * HIPL Picture Header Format Standard
 *
 * Michael Landy - 2/1/82
 */

#ifndef NO_HIP1HDR

typedef struct hips1_header {
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
} Hips1_Header ;

#define NO_HIP1HDR	((Hips1_Header *)NULL)

#include "hips_pf.h"		/* pixel format codes */

/*
 * Bit packing formats
 */

#define	MSBFIRST 1		/* bit packing - most significant bit first */
#define	LSBFIRST 2		/* bit packing - least significant bit first */

#define FBUFLIMIT 30000

/*
 * For general readability
 */

#ifndef TRUE
#define	TRUE	1
#define	FALSE	0
#endif /* TRUE */

#endif /* NO_HIP1HDR */

