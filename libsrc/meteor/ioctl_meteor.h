/*
 * Copyright (c) 1995 Mark Tinguely and Jim Lowe
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by Mark Tinguely and Jim Lowe
 * 4. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * changes:
 *
 * 20-Mar-1999 mjw
 *
 *	ioctl constants for Matrox Meteor Capture card.
 */

#ifndef _MACHINE_IOCTL_METEOR_H
#define _MACHINE_IOCTL_METEOR_H


#define MAX_NUM_FRAMES 256	/* should be the same as meteor.h */
#define MAGIC_FRAME_INDEX	(2*MAX_NUM_FRAMES)


#ifndef __KERNEL__
#ifdef HAVE_LINUX_IOCTL_H
#include <linux/ioctl.h>
#endif // HAVE_LINUX_IOCTL_H
#ifdef HAVE_STDINT_H
#include <stdint.h>
#endif // HAVE_STDINT_H
#endif /* ! __KERNEL__ */

struct meteor_capframe {
	short	command;	/* see below for valid METEORCAPFRM commands */
	short	lowat;		/* start transfer if < this number */
	short	hiwat;		/* stop transfer if > this number */
} ;

/* structure for METEOR[GS]WIN - get/set capture window */
struct meteor_window {
	u_short		xstart;
	u_short		xsize;
	u_short		ystart;
	u_short		ysize;
} ;

/* structure for METEOR[GS]ETGEO - get/set geometry */
struct meteor_geomet {
	u_short		rows;
	u_short		columns;
	u_short		frames;
	uint32_t	oformat;
	uint32_t	frame_size;
} ;

/* structure for METEORGCOUNT-get count of frames, fifo errors and dma errors */
struct meteor_counts {
	uint32_t fifo_errors;	/* count of fifo errors since open */
	uint32_t dma_errors;	/* count of dma errors since open */
	uint32_t n_frames_captured;	/* count of frames captured since open */
	uint32_t even_fields_captured;	/* count of even fields captured */
	uint32_t odd_fields_captured;	/* count of odd fields captured */
} ;

/* structure to get the frame offsets in memory */
struct meteor_frame_offset {
	/* shouldnt' these be long??? */
	uint32_t frame_offset[MAX_NUM_FRAMES];
	uint32_t fb_size;
	uint32_t mem_off;
} ;

#ifdef METEOR_EXT_FRAME_BUF
/* structure for getting and setting direct transfers to vram */
struct meteor_fbuf {
	/*uint32_t	addr;*/		/* Address of location to dma to */
	void *		addr;		/* Address of location to dma to */
	uint32_t	width;		/* Width of memory area */
	uint32_t	banksize;	/* Size of Vram bank */
	uint32_t	ramsize;	/* Size of Vram */
};
#endif // METEOR_EXT_FRAME_BUF


#define METEORCAPTUR	_IOW('x', 1, int)			/* capture a frame */
#define METEORCAPFRM	_IOW('x', 2, struct meteor_capframe)	/* sync capture */
#define METEORSETGEO	_IOW('x', 3, struct meteor_geomet)	/* set geometry */
#define METEORGETGEO	_IOR('x', 4, struct meteor_geomet)	/* get geometry */
#define METEORSTATUS	_IOR('x', 5, unsigned short)		/* get status */
#define METEORSHUE	_IOW('x', 6, signed char)		/* set hue */
#define METEORGHUE	_IOR('x', 6, signed char)		/* get hue */
#define METEORSFMT	_IOW('x', 7, uint32_t)			/* set format */
#define METEORGFMT	_IOR('x', 7, uint32_t)			/* get format */
#define METEORSINPUT	_IOW('x', 8, uint32_t)			/* set input dev */
#define METEORGINPUT	_IOR('x', 8, uint32_t)			/* get input dev */
#define	METEORSCHCV	_IOW('x', 9, unsigned char)		/* set uv gain */
#define	METEORGCHCV	_IOR('x', 9, unsigned char)		/* get uv gain */
#define	METEORSCOUNT	_IOW('x',10, struct meteor_counts)
#define	METEORGCOUNT	_IOR('x',10, struct meteor_counts)
#define METEORSFPS	_IOW('x',11, unsigned short)		/* set fps */
#define METEORGFPS	_IOR('x',11, unsigned short)		/* get fps */
#define METEORSSIGNAL	_IOW('x', 12, unsigned int)		/* set signal */
#define METEORGSIGNAL	_IOR('x', 12, unsigned int)		/* get signal */
#ifdef METEOR_EXT_FRAME_BUF
#define	METEORSVIDEO	_IOW('x', 13, struct meteor_fbuf)	/* set video */
#define	METEORGVIDEO	_IOR('x', 13, struct meteor_fbuf)	/* get video */
#endif // METEOR_EXT_FRAME_BUF
#define	METEORSBRIG	_IOW('x', 14, unsigned char)		/* set brightness */
#define METEORGBRIG	_IOR('x', 14, unsigned char)		/* get brightness */
#define	METEORSCSAT	_IOW('x', 15, unsigned char)		/* set chroma sat */
#define METEORGCSAT	_IOR('x', 15, unsigned char)		/* get uv saturation */
#define	METEORSCONT	_IOW('x', 16, unsigned char)		/* set contrast */
#define	METEORGCONT	_IOR('x', 16, unsigned char)		/* get contrast */
#define METEORSBT254	_IOW('x', 17, unsigned short)		/* set Bt254 reg */
#define METEORGBT254	_IOR('x', 17, unsigned short)		/* get Bt254 reg */
#define METEORSHWS	_IOW('x', 18, unsigned char)		/* set hor start reg */
#define METEORGHWS	_IOR('x', 18, unsigned char)		/* get hor start reg */
#define METEORSVWS	_IOW('x', 19, unsigned char)		/* set vert start reg*/
#define METEORGVWS	_IOR('x', 19, unsigned char)		/* get vert start reg*/
#define	METEORSTS	_IOW('x', 20, uint32_t)			/* set time stamp */
#define	METEORGTS	_IOR('x', 20, uint32_t)			/* get time stamp */
#define	METEORGFROFF	_IOR('x', 21, struct meteor_frame_offset )	/* get frame offsets */
#define	METEORGCAPT	_IOR('x', 22, uint32_t) 			/* get saa7116 capt_ctl (0x40) reg */
#define METEORSGAMMA	_IOW('x', 23, int)			/* set/get gamma by-pass ROM bit */
#define METEORGGAMMA	_IOR('x', 23, int)			/* 1=by-pass, 0=raise RGB to 1.2 pwr */
#define METEORSWIN	_IOW('x', 24, struct meteor_window)	/* set captur window*/
#define METEORGWIN	_IOR('x', 24, struct meteor_window)	/* get captur window*/

/* jbm added ioctl's */
struct drop_info {
	uint32_t	n_total;
	uint32_t	n_saved;
};

struct error_info {
	struct drop_info ei[3];
};


#define METEORGNERR	_IOR('x', 25, struct error_info)	/* get the number of error frames */
#define METEORGERR0FRMS	_IOR('x', 26, uint32_t)			/* get error frames,var # */
#define METEORGERR1FRMS	_IOR('x', 27, uint32_t)			/* get error frames, var # */
#define METEORGERR2FRMS	_IOR('x', 28, uint32_t)			/* get error frames,var # */
#define METEORGNDROP	_IOR('x', 29, struct drop_info)		/* get # of dropeed frames */
#define METEORGDRPFRMS	_IOR('x', 30, uint32_t)

#define METEORGNBUFFRMS	_IOR('x', 31, uint32_t)			/* get # frames in ring buffer */
#define METEORSFRMIDX	_IOW('x', 32, uint32_t)			/* set current frame index for mmap */

#define	METEOR_STATUS_ID_MASK	0xf000		/* ID of 7196 */
#define	METEOR_STATUS_DIR	0x0800		/* Direction of Expansion port YUV */
#define	METEOR_STATUS_OEF	0x0200		/* Field detected: Even/Odd */
#define	METEOR_STATUS_SVP	0x0100		/* State of VRAM Port:inactive/active */
#define	METEOR_STATUS_STTC	0x0080		/* Time Constant: TV/VCR */
#define	METEOR_STATUS_HCLK	0x0040		/* Horiz PLL: locked/unlocked */
#define	METEOR_STATUS_FIDT	0x0020		/* Field detect: 50/60hz */
#define	METEOR_STATUS_ALTD	0x0002		/* Line alt: no line alt/line alt */
#define METEOR_STATUS_CODE	0x0001		/* Colour info: no colour/colour */

					/* METEORCAPTUR capture options */
#define METEOR_CAP_SINGLE	0x0001		/* capture one frame */
#define METEOR_CAP_CONTINOUS	0x0002		/* continuously capture */
#define METEOR_CAP_STOP_CONT	0x0004		/* stop the continuous capture */
#define METEOR_CAP_CONT_ONCE	0x0008		/* capture all allocated frames once*/

					/* METEORCAPFRM capture commands */
#define METEOR_CAP_N_FRAMES	0x0001		/* capture N frames */
#define METEOR_CAP_STOP_FRAMES	0x0002		/* stop capture N frames */
#define	METEOR_HALT_N_FRAMES	0x0003		/* halt of capture N frames */
#define METEOR_CONT_N_FRAMES	0x0004		/* continue after above halt */

					/* valid video input formats: */
#define METEOR_FMT_NTSC		0x00100		/* NTSC -- initialized default */
#define METEOR_FMT_PAL		0x00200		/* PAL */
#define METEOR_FMT_SECAM	0x00400		/* SECAM */
#define METEOR_FMT_AUTOMODE	0x00800		/* auto-mode */
#define METEOR_INPUT_DEV0	0x01000		/* camera input 0 -- default */
#define METEOR_INPUT_DEV_RCA	METEOR_INPUT_DEV0
#define METEOR_INPUT_DEV1	0x02000		/* camera input 1 */
#define METEOR_INPUT_DEV2	0x04000		/* camera input 2 */
#define METEOR_INPUT_DEV3	0x08000		/* camera input 3 */
#define METEOR_INPUT_DEV_RGB	0x0a000		/* for rgb version of meteor */
#define METEOR_INPUT_DEV_SVIDEO	0x06000		/* S-video input port */

					/* valid video output formats: */
#define METEOR_GEO_RGB16	0x00010000	/* packed -- initialized default */
#define METEOR_GEO_RGB24	0x00020000	/* RBG 24 bits packed */
#define METEOR_GEO_RGB24_2FLD	0x00030000	/* RBG 24 - fields separated */
#define METEOR_GEO_YUV_PACKED	0x00040000	/* 4-2-2 YUV 16 bits packed */
#define METEOR_GEO_YUV_2FLD	0x00060000	/* 422 YUV 16 bits pcked, flds sep */
#define METEOR_GEO_YUV_PLANAR	0x00080000	/* 4-2-2 YUV 16 bits planar */
#define METEOR_GEO_UNSIGNED	0x00400000	/* unsigned uv outputs */
#define METEOR_GEO_EVEN_ONLY	0x01000000	/* set for even only field capture */
#define METEOR_GEO_ODD_ONLY	0x02000000	/* set for odd only field capture */
#define METEOR_GEO_FIELD_MASK	0x03000000
#define METEOR_GEO_YUV_422	0x04000000	/* 4-2-2 YUV in Y-U-V combined */
#define METEOR_GEO_OUTPUT_MASK	0x040f0000
#define METEOR_GEO_YUV_12	0x10000000	/* YUV 12 format */
#define METEOR_GEO_YUV_9	0x40000000	/* YUV 9 format */

#define METEOR_GEO_FIELD_MODE	0x80000000

#define METEOR_SIG_MODE_MASK	0xffff0000
#define METEOR_SIG_FRAME	0x00000000	/* signal every frame */
#define METEOR_SIG_FIELD	0x00010000	/* signal every field */


/* following structure is used to coordinate the synchronous */

struct meteor_mem {
		/* kernel write only */
	int	frame_size;		/* row*columns*depth */
	int	num_bufs;		/* number of frames in buffer (1-32) */
	u_int	n_frames_captured;	/* total frames captured */
	int16_t	cur_frame;		/* index of most recent acquisition */
	int16_t	cur_field;		/* 0=even, 1=odd */
		/* user and kernel change these */
	int16_t	lowat;			/* kernel starts capture if < this number */
	int16_t	hiwat;			/* kernel stops capture if > this number.
					   hiwat <= numbufs */
	u_long	active;			/* bit mask of active frame buffers
					    kernel sets, user clears */
	int	num_active_bufs;	/* count of active frame buffer
					    kernel increments, user decrements */
		/* reference to mmapped data */
	caddr_t	buf;			/* The real space (virtual addr) */
} ;

#endif	/* ifndef _MACHINE_IOCTL_METEOR_H */

