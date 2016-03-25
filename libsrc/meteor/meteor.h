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
 */

#ifndef _METEOR_H
#define _METEOR_H

#include "ioctl_meteor.h"

/* Debug related defines */
/* Uncomment to print debug info to kernel message log  */
/*#define DEBUG_METEOR */
/* Uncomment if you want FIFO overflow and DMA address error messages. */
/*#define SHOW_CAPT_ERRORS  */

/* Frame buffer RAM source defines */
/* Uncomment if you wish to use external RAM other than bigphusarea */
#define METEOR_EXT_FRAME_BUF
/* Uncomment if you wish to include bigphysarea code */
/* jbm:  himemfb works so well, I removed the ifdef BIGPHYSAREA code... */
/*#define METEOR_BIGPHYSAREA */

#define METEOR_VERSION "2.2"

#define METEOR_MAJOR 40

#define MAXMETEORS 2

#define METEOR_NUM(mtr) (mtr->unit)

#define WAIT_JIFFS 3		/* Should be about 20 milliseconds */


#ifndef METEOR_BIGPHYSAREA
# define bigphysarea_alloc(a) 0
# define bigphysarea_free(a, b) 
# define bigphysarea_alloc_pages(a)  0
# define bigphysarea_free_pages(a, b)
#endif

/* PCI-related #defines */

/* Jim Lowe suggests 32, or 16 if bus master slots > 4 */
/* make this bigger when there are fifo errors */
/* 80 helped on donders, but still 4 out of 600... */
#define METEOR_LATENCY 250

/*
 * Definitions for the Philips SAA7196 digital video decoder,
 * scalar, and clock generator circuit (DESCpro).
 */
#define NUM_SAA7196_I2C_REGS	49
#define	SAA7196_I2C_ADDR_W	0x40
#define	SAA7196_I2C_ADDR_R	0x41

#define SAA7196_WRITE(mtr, reg, data) \
 talk_i2c(mtr, SAA7196_I2C_ADDR_W, reg, data), \
  mtr->saa7196_i2c[reg] = data; /* Update the saved image */
#define	SAA7196_READ(mtr) \
 talk_i2c(mtr, SAA7196_I2C_ADDR_R, 0x0, 0x0)

#define SAA7196_REG(mtr, reg) mtr->saa7196_i2c[reg]


#define SAA7196_IDEL	0x00	/* Increment delay */
#define SAA7196_HSB5	0x01	/* H-sync begin; 50 hz */
#define SAA7196_HSS5	0x02	/* H-sync stop; 50 hz */
#define SAA7196_HCB5	0x03	/* H-clamp begin; 50 hz */
#define SAA7196_HCS5	0x04	/* H-clamp stop; 50 hz */
#define SAA7196_HSP5	0x05	/* H-sync after PHI1; 50 hz */
#define SAA7196_LUMC	0x06	/* Luminance control */
#define SAA7196_HUEC	0x07	/* Hue control */
#define SAA7196_CKTQ	0x08	/* Colour Killer Threshold QAM (PAL, NTSC) */
#define SAA7196_CKTS	0x09	/* Colour Killer Threshold SECAM */
#define SAA7196_PALS	0x0a	/* PAL switch sensitivity */
#define SAA7196_SECAMS	0x0b	/* SECAM switch sensitivity */
#define SAA7196_CGAINC	0x0c	/* Chroma gain control */
#define SAA7196_STDC	0x0d	/* Standard/Mode control */
#define SAA7196_IOCC	0x0e	/* I/O and Clock Control */
#define SAA7196_CTRL1	0x0f	/* Control #1 */
#define SAA7196_CTRL2	0x10	/* Control #2 */
#define SAA7196_CGAINR	0x11	/* Chroma Gain Reference */
#define SAA7196_CSAT	0x12	/* Chroma Saturation */
#define SAA7196_CONT	0x13	/* Luminance Contrast */
#define SAA7196_HSB6	0x14	/* H-sync begin; 60 hz */
#define SAA7196_HSS6	0x15	/* H-sync stop; 60 hz */
#define SAA7196_HCB6	0x16	/* H-clamp begin; 60 hz */
#define SAA7196_HCS6	0x17	/* H-clamp stop; 60 hz */
#define SAA7196_HSP6	0x18	/* H-sync after PHI1; 60 hz */
#define SAA7196_BRIG	0x19	/* Luminance Brightness */
#define SAA7196_FMTS	0x20	/* Formats and sequence */
#define SAA7196_OUTPIX	0x21	/* Output data pixel/line */
#define SAA7196_INPIX	0x22	/* Input data pixel/line */
#define SAA7196_HWS	0x23	/* Horiz. window start */
#define SAA7196_HFILT	0x24	/* Horiz. filter */
#define SAA7196_OUTLINE	0x25	/* Output data lines/field */
#define SAA7196_INLINE	0x26	/* Input data lines/field */
#define SAA7196_VWS	0x27	/* Vertical window start */
#define SAA7196_VYP	0x28	/* AFS/vertical Y processing */
#define SAA7196_VBS	0x29	/* Vertical Bypass start */
#define SAA7196_VBCNT	0x2a	/* Vertical Bypass count */
#define SAA7196_VBP	0x2b	/* veritcal Bypass Polarity */
#define SAA7196_VLOW	0x2c	/* Colour-keying lower V limit */
#define SAA7196_VHIGH	0x2d	/* Colour-keying upper V limit */
#define SAA7196_ULOW	0x2e	/* Colour-keying lower U limit */
#define SAA7196_UHIGH	0x2f	/* Colour-keying upper U limit */
#define SAA7196_DPATH	0x30	/* Data path setting  */

/*
 * Defines for the PCF8574.
 *
 * There are two PCF8574As on the board.  One is at address 0x70 
 * (W: 0x70, R: 0x71) and performs various control functions to do with 
 * the RGB section:
 *
 * P7: saa7196 enable (0=en, 1=dis)
 * P6: bt254 enable (0=en, 1=dis)
 * P5: ntsc (1=ntsc, 0=pal/secam)
 * P4: BT254 WR*
 * P3: ?? (probably BT254 RD*)
 * P2: BT254 A2
 * P1: BT254 A1
 * P0: BT254 A0
 *
 * The other is the data for (or from) the Bt254 at address 0x72 (W: 0x72, R: 0x73)
 *
 */
#define NUM_PCF8574_I2C_REGS	2
#define	PCF8574_CTRL_I2C_ADDR_W	0x70
#define	PCF8574_CTRL_I2C_ADDR_R	0x71
#define PCF8574_DATA_I2C_ADDR_W	0x72
#define PCF8574_DATA_I2C_ADDR_R	0x73
#define	PCF8574_CTRL_WRITE(mtr, data) \
talk_i2c(mtr,  PCF8574_CTRL_I2C_ADDR_W, data, data), \
	mtr->pcf_i2c[0] = data
#define	PCF8574_DATA_WRITE(mtr, data) \
 talk_i2c(mtr,  PCF8574_DATA_I2C_ADDR_W, data, data), \
	mtr->pcf_i2c[1] = data
#define PCF8574_CTRL_REG(mtr) mtr->pcf_i2c[0]
#define PCF8574_DATA_REG(mtr) mtr->pcf_i2c[1]


/*
 * Defines for the BT254.
 */
#define	NUM_BT254_REGS	7

#define BT254_COMMAND	0
#define	BT254_IOUT1	1
#define	BT254_IOUT2	2
#define	BT254_IOUT3	3
#define BT254_IOUT4	4
#define	BT254_IOUT5	5
#define	BT254_IOUT6	6

#define	METEOR_INITALIZED	0x00000001
#define	METEOR_OPEN		0x00000002 
#define	METEOR_MMAP		0x00000004
#define	METEOR_INTR		0x00000008
#define	METEOR_ONCE		0x00000010	/* capture all frames once */
#define	METEOR_SINGLE		0x00000020	/* get single frame */
#define	METEOR_CONTIN		0x00000040	/* continuously get frames */
#define	METEOR_SYNCAP		0x00000080	/* synchronously get frames */
#define	METEOR_CAP_MASK		0x000000f0
#define	METEOR_NTSC		0x00000100
#define	METEOR_PAL		0x00000200
#define	METEOR_SECAM		0x00000400
#define	METEOR_AUTOMODE		0x00000800
#define	METEOR_FORM_MASK	0x00000f00
#define	METEOR_DEV0		0x00001000
#define	METEOR_DEV1		0x00002000
#define	METEOR_DEV2		0x00004000
#define	METEOR_DEV3		0x00008000
#define METEOR_DEV_SVIDEO	0x00006000
#define METEOR_DEV_RGB		0x0000a000
#define	METEOR_DEV_MASK		0x0000f000
#define	METEOR_WANT_EVEN	0x00100000
#define	METEOR_WANT_ODD		0x00200000
#define	METEOR_WANT_MASK	0x00300000
#define METEOR_YUV_422		0x04000000
#define	METEOR_OUTPUT_FMT_MASK	0x040f0000
#define	METEOR_WANT_TS		0x08000000	  /* time-stamp a frame */
#define METEOR_RGB		0x20000000	/* meteor rgb unit */
#define METEOR_FIELD_MODE	0x80000000

#define MAX_NUM_FRAMES		256 /* or what YOU think is reasonable */
 struct meteor {
   struct saa7116 * s7116;	/* saa7116 register virtual address */
   uint32_t maxRange;		/* maximum range-checkable size */
   struct pci_dev *dev;		/* PCI dev, for doing PCI commands */
   u_char irq;			/* IRQ */
   short unit;			/* Unit number */
   void *fb_phys_addr;		/* frame buffer as seen by device */
   void *fb_remap_addr;		/* frame buffer as seen by kernel */
   void *bigphysaddr;		/* address of allocated bigphysarea */
   u_int bigphyssize;		/* size of allocated bigphysarea */
   u_int fb_size;		/* Size of buffer */
   struct task_struct *proc;	/* process to receive raised signal */
   int pid;			/* PID of said process */
   struct wait_queue *waitq;	/* Queue of sleeping procs */
   int		signal;		/* signal to send to process */
   int		sigmode;	/* when to signal */
   struct meteor_mem *mem;	/* used to control sync. multi-frame output */
   uint32_t	synch_wait;	/* wait for free buffer before continuing */
   short	cur_frame;	/* frame number in buffer (1-frames) */
   short	rows;		/* number of rows in a frame */
   short	cols;		/* number of columns in a frame */
   short	frames;		/* number of frames allocated */
   uint32_t	oformat;	/* output format */
   short	depth;		/* number of bits per pixel */
   int		frame_size;	/* number of bytes in a frame */
   uint32_t	fifo_errors;	/* number of fifo capture errors since open */
   uint32_t	dma_errors;	/* number of DMA capture errors since open */
   uint32_t	frames_captured;/* number of frames captured since open */
   uint32_t	even_fields_captured; /* number of even fields captured */
   uint32_t	odd_fields_captured; /* number of odd fields captured */
   uint32_t	flags;
   u_char	saa7196_i2c[NUM_SAA7196_I2C_REGS]; /* saa7196 register values */
   u_char	pcf_i2c[NUM_PCF8574_I2C_REGS];	/* PCF8574 register values */
   u_char	bt254_reg[NUM_BT254_REGS];	/* BT254 register values */
   u_short	fps;		/* frames per second */
   uint32_t	dma_add_e[3];
   uint32_t	dma_add_o[3];
   uint32_t	dma_str_e[3];
   uint32_t	dma_str_o[3];
   struct meteor_fbuf	fbuf;	/* used to describe extern RAM if used as fb */ 
   uint32_t	frame_offset[MAX_NUM_FRAMES];
 };

#endif
