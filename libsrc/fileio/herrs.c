#include "quip_config.h"

char VersionId_fio_herrs[] = QUIP_VERSION_STRING;

/*
 * Copyright (c) 1991 Michael Landy
 *
 * Disclaimer:  No guarantees of performance accompany this software,
 * nor is any responsibility assumed on the part of the authors.  All the
 * software has been tested extensively and every effort has been made to
 * insure its reliability.
 */

/*
 * herrs.c - standard definitions for HIPS error strings
 *
 * Michael Landy - 1/3/91
 */

#include "fio_prot.h"
#include "hips2.h"

struct h_errstruct h_errors[] = {
{"can't allocate memory",HEP_N,HEL_ERROR},		/* HE_ALLOC 1 */
{"%s: can't allocate memory",HEP_S,HEL_ERROR},		/* HE_ALLOCSUBR 2 */
{"can't free memory",HEP_N,HEL_ERROR},			/* HE_FREE 3 */
{"error reading frame %d",HEP_D,HEL_ERROR},		/* HE_READFR 4 */
{"error reading frame %d from file %s",HEP_DS,HEL_ERROR},/*HE_READFRFILE 5 */
{"error reading from file %s",HEP_S,HEL_ERROR},		/* HE_READFILE 6 */
{"error during read",HEP_N,HEL_ERROR},			/* HE_READ 7 */
{"error writing frame %d",HEP_D,HEL_ERROR},		/* HE_WRITEFR 8 */
{"error writing frame %d to file %s",HEP_DS,HEL_ERROR},	/* HE_WRITEFRFILE 9 */
{"can't open file: `%s'",HEP_S,HEL_ERROR},		/* HE_OPEN 10 */
{"can't perform seek",HEP_N,HEL_ERROR},			/* HE_SEEK 11 */
{"pixel format mismatch",HEP_N,HEL_ERROR},		/* HE_FMTMATCH 12 */
{"pixel format mismatch, file %s",HEP_S,HEL_ERROR},	/* HE_FMTMATCHFILE 13 */
{"can't handle pixel format %s",HEP_S,HEL_ERROR},	/* HE_FMT 14 */
{"can't handle pixel format %s, file: %s",HEP_SS,HEL_ERROR},/* HE_FMTFILE 15 */
{"image dimensions must be powers of 2",HEP_N,HEL_ERROR},/*HE_POW2 16 */
{"number of frames must be zero or positive",HEP_N,HEL_ERROR},/* HE_SNEG 17 */
{"size mismatch",HEP_N,HEL_ERROR},			/* HE_SMATCH 18 */
{"mismatch of number of frames",HEP_N,HEL_ERROR},	/* HE_FRMATCH 19 */
{"frame dimensions are too large",HEP_N,HEL_ERROR},	/* HE_LARGE 20 */
{"zero or negative dimension",HEP_N,HEL_ERROR},		/* HE_ZNEG 21 */
{"%s: buffer limit exceeded",HEP_S,HEL_ERROR},		/* HE_BUF 22 */
{"%s: unknown op-code",HEP_S,HEL_ERROR},		/* HE_CODE 23 */
{"cut_frame: 1 intersection",HEP_N,HEL_ERROR},		/* HE_CUT1 24 */
{"cut_frame: intersections not between?",HEP_N,HEL_ERROR},/* HE_CUTI 25 */
{"%s: strange index=%d",HEP_SD,HEL_ERROR},		/* HE_FFTI 26 */
{"pyrnumpix: toplev less than zero?",HEP_N,HEL_ERROR},	/* HE_PYRTLZ 27 */
{"pyrnumpix: toplev too large?",HEP_N,HEL_ERROR},	/* HE_PYRTL 28 */
{"%s: Invalid reflection type %d",HEP_SD,HEL_ERROR},	/* HE_REFL 29 */
{"read_frame: did not find frame-end",HEP_N,HEL_ERROR},	/* HE_FRMEND 30 */
{"error while reading header, file %s",HEP_S,HEL_ERROR},/* HE_HDRREAD 31 */
{"error reading extended parameters, file %s",HEP_S,HEL_ERROR},
							/* HE_HDRPREAD 32 */
{"error reading binary parameters, file %s",HEP_S,HEL_ERROR},
							/* HE_HDRBREAD 33 */
{"invalid extended parameter format %d, file %s",HEP_DS,HEL_ERROR},
							/* HE_HDRPTYPE 34 */
{"header parameters overflow, file: %s, size should be %d and was %d",
		HEP_SDD,HEL_ERROR},			/* HE_HDRXOV 35 */
{"error while writing header, file %s",HEP_S,HEL_ERROR},/* HE_HDRWRT 36 */
{"error while writing header parameters, file %s",HEP_S,HEL_ERROR},
							/* HE_HDRPWRT 37 */
{"error while writing header binary area, file %s",HEP_S,HEL_ERROR},
							/* HE_HDRBWRT 38 */
{"invalid extended parameter format %d during write, file %s",
		HEP_DS,HEL_ERROR},			/* HE_HDWPTYPE 39 */
{"%s: reqested %d, got %d",HEP_SDD,HEL_ERROR},		/* HE_REQ 40 */
{"invalid format code %d",HEP_D,HEL_ERROR},		/* HE_BADFMT 41 */
{"%s: invalid extended parameter format %d",HEP_SD,HEL_ERROR},
							/* HE_HDPTYPE 42 */
{"%s: can't find extended parameter %s",HEP_SS,HEL_ERROR},/* HE_MISSPAR 43 */
{"mismatched number of rows, file: %s",HEP_S,HEL_ERROR},/* HE_C_ROW 44 */
{"mismatched number of columns, file: %s",HEP_S,HEL_ERROR},/* HE_C_COL 45 */
{"mismatched number of frames, file: %s",HEP_S,HEL_ERROR},/* HE_C_FRM 46 */
{"mismatched pixel format (%s), file: %s",HEP_SS,HEL_ERROR},/* HE_C_FMT 47 */
{"mismatched number of colors, file: %s",HEP_S,HEL_ERROR},/* HE_C_NCL 48 */
{"mismatched number of pyramid levels, file: %s",HEP_S,HEL_ERROR},
							/* HE_C_NLV 49 */
{"%s: can't handle pixel format %s",HEP_SS,HEL_ERROR},	/* HE_FMTSUBR 50 */
{"%s: unknown complex-to-real conversion: %d",HEP_SD,HEL_ERROR},
							/* HE_CTORTP 51 */
{"%s: unknown method (%d), file: %s",HEP_SDS,HEL_ERROR},/* HE_METH 52 */
{"%s: can't handle pixel format %s, file: %s",HEP_SSS,HEL_ERROR},
							/* HE_FMTSUBRFILE 53 */
{"setformat: can't handle pyramid formats",HEP_N,HEL_ERROR},/* HE_SETFP 54 */
{"setpyrformat: can only handle pyramid formats",HEP_N,HEL_ERROR},
							/* HE_SETPF 55 */
{"%s: unknown real-to-complex conversion: %d",HEP_SD,HEL_ERROR},
							/* HE_RTOCTP 56 */
{"%s",HEP_S,HEL_ERROR},					/* HE_MSG 57 */
{"setroi: ROI out of bounds, first=(%d,%d), size=(%d,%d)",HEP_DDDD,HEL_ERROR},
							/* HE_ROI 58 */
{"converting from %s to %s, file: %s",HEP_SSS,HEL_WARN},/* HE_CONV 59 */
{"converting from %s to %s via integer, file: %s",HEP_SSS,HEL_WARN},
							/* HE_CONVI 60 */
{"frame dimensions are too small",HEP_N,HEL_ERROR},	/* HE_SMALL 61 */
{"%s: invalid range",HEP_S,HEL_ERROR},			/* HE_RNG 62 */
{"header parameters inconsistency for (%s %d %d %d) offset is %d, file: %s",
		HEP_SDDDDS,HEL_ERROR},			/* HE_XINC 63 */
{"%s: packed image ROI - columns not multiple of 8",HEP_S,HEL_ERROR},
							/* HE_ROI8 64 */
{"%s: packed image ROI - columns not multiple of 8, file: %s",HEP_SS,
		HEL_ERROR},				/* HE_ROI8F 65 */
{"%s: packed image ROI - columns not multiple of 8, clearing ROI",HEP_S,
		HEL_WARN},				/* HE_ROI8C 66 */
{"%s",HEP_S,HEL_INFORM},				/* HE_IMSG 67 */
{"unrecognised flag option %s\n%s",HEP_SS,HEL_ERROR},	/* HE_UNKFLG 68 */
{"flags are mutually exclusive\n%s",HEP_S,HEL_ERROR},	/* HE_MUTEX 69 */
{"missing parameter for flag -%s\n%s",HEP_SS,HEL_ERROR},
							/* HE_MISSFPAR 70 */
{"invalid parameter \"%s\" for flag -%s\n%s",HEP_SSS,HEL_ERROR},
							/* HE_INVFPAR 71 */
{"invalid syntax\n%s",HEP_S,HEL_ERROR},			/* HE_SYNTAX 72 */
{"too many image filenames\n%s",HEP_S,HEL_ERROR},	/* HE_FILECNT 73 */
{"missing image filename(s)\n%s",HEP_S,HEL_ERROR},	/* HE_MISSFILE 74 */
{"invalid image filename %s",HEP_S,HEL_ERROR},		/* HE_INVFILE 75 */
{"can't open the standard input twice",HEP_N,HEL_ERROR},/* HE_STDIN 76 */
{"%s: can't handle pixel format combination %s/%s/%s",HEP_SSSS,HEL_ERROR},
							/* HE_FMT3SUBR 77 */
{"%s: can't handle pixel format combination %s/%s",HEP_SSS,HEL_ERROR},
							/* HE_FMT2SUBR 78 */
{"%s: row/column selection out of range",HEP_S,HEL_ERROR}, /* HE_RCSELN 79 */
{"mismatched total number of rows, file: %s",HEP_S,HEL_ERROR},/* HE_C_OROW 80 */
{"mismatched total number of columns, file: %s",HEP_S,HEL_ERROR},
							/* HE_C_OCOL 81 */
{"mismatched number of frames and numcolor>1, file: %s",HEP_S,HEL_ERROR},
							/* HE_C_FRMC 82 */
{"image dimensions must be multiples of 2",HEP_N,HEL_ERROR},/* HE_MULT2 83 */
{"bad mask function number %d, file: %s",HEP_DS,HEL_ERROR},
							/* HE_MSKFUNFILE 84 */
{"%s: bad mask function number %d",HEP_SD,HEL_ERROR},	/* HE_MSKFUNSUBR 85 */
{"%s: mask function %d, bad mask count %d",HEP_SDD,HEL_ERROR},
							/* HE_MSKCNT 86 */
{"%s: sigma less than or equal to zero",HEP_S,HEL_ERROR}, /* HE_SIGMA 87 */
{"flag `-%s' specified more than once",HEP_S,HEL_ERROR},/* HE_PTWICE 88	*/
{"%s: invalid window size (%d)",HEP_SD,HEL_ERROR},	/* HE_WINDSZ 89 */
{"component table overflow",HEP_N,HEL_ERROR},		/* HE_COMPOV 90	*/
{"point table overflow",HEP_N,HEL_ERROR},		/* HE_PNTOV 91 */
{"bad filter parameter",HEP_N,HEL_ERROR},		/* HE_FILTPAR 92 */
{"invalid extended parameter format %s, file %s",HEP_SS,HEL_ERROR},
							/* HE_HDRPTYPES 93 */
{"%s: invalid color specification: `%s'",HEP_SS,HEL_ERROR},
							/* HE_COLSPEC 94 */
{"%s: parameter name contains white space: `%s'",HEP_SS,HEL_ERROR},
							/* HE_PNAME 95 */
{"%s: supplied count doesn't match that of `%s'",HEP_SS,HEL_ERROR},
							/* HE_PCOUNT 96	*/
{"%s: colormap table overflow, file: %s",HEP_SS,HEL_ERROR},
							/* HE_COLOVF 97 */
{"length of formats array mismatch, file: %s",HEP_S,HEL_ERROR},
							/* HE_FMTSLEN 98 */
{"input 3-color image has numcolor>1, file %s",HEP_S,HEL_ERROR},
							/* HE_COL1 99 */
{"can't convert 1-color image with numcolor!=3 to 3-color, file %s",HEP_S,
		HEL_ERROR},				/* HE_COL3 100 */
{"perr: unknown error code %d",HEP_D,HEL_SEVERE}	/* Unknown error code */
};
