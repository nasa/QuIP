/* nvstusb.c 
 * Copyright (C) 2010 Bjoern Paetzel
 * Copyright (C) 2010 Johann Baudy
 *
 * This program comes with ABSOLUTELY NO WARRANTY.
 * This is free software, and you are welcome to redistribute it
 * under certain conditions. See the file COPYING for details
 * */

#include "quip_config.h"

#ifdef HAVE_OPENGL
#ifdef HAVE_LIBUSB

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif // HAVE_MALLOC_H
#include <time.h>
#include <assert.h>
#include <math.h>
#include <pthread.h>

#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glxext.h>
#include <GL/glext.h>

#include "nvstusb.h"
#include "usb.h"

// These types come up undefined when we compile
// with all of our -I options!? - jbm
static PFNGLXGETVIDEOSYNCSGIPROC glXGetVideoSyncSGI = NULL;
static PFNGLXWAITVIDEOSYNCSGIPROC glXWaitVideoSyncSGI = NULL;
static PFNGLXSWAPINTERVALSGIPROC glXSwapIntervalSGI = NULL;
//static PFNGLXSWAPINTERVALEXTPROC glXSwapIntervalEXT = NULL;

/* Static functions */
//static void * nvstusb_stereo_thread(void * in_pv_arg);

/* cpu clock */
#define NVSTUSB_CLOCK           48000000LL

/* T0 runs at 4 MHz */
#define NVSTUSB_T0_CLOCK        (NVSTUSB_CLOCK/12LL)
#define NVSTUSB_T0_COUNT(us)    (-(us)*(NVSTUSB_T0_CLOCK/1000000)+1)
#define NVSTUSB_T0_US(count)    (-(count-1)/(NVSTUSB_T0_CLOCK/1000000))

/* T2 runs at 12 MHz */
#define NVSTUSB_T2_CLOCK        (NVSTUSB_CLOCK/ 4LL)
#define NVSTUSB_T2_COUNT(us)    (-(us)*(NVSTUSB_T2_CLOCK/1000000)+1)
#define NVSTUSB_T2_US(count)    (-(count-1)/(NVSTUSB_T2_CLOCK/1000000))

#define NVSTUSB_CMD_WRITE       (0x01)  /* write data */
#define NVSTUSB_CMD_READ        (0x02)  /* read data */
#define NVSTUSB_CMD_CLEAR       (0x40)  /* set data to 0 */

#define NVSTUSB_CMD_SET_EYE     (0xAA)  /* set current eye */
#define NVSTUSB_CMD_CALL_X0199  (0xBE)  /* call routine at 0x0199 */

/* state of the controller */
struct nvstusb_context {
	/* currently selected refresh rate */
	float rate;

	/* currently active eye */
	/*enum*/ nvstusb_eye eye;

	/* device handle */
	struct nvstusb_usb_device *device;

	/* Toggled state */
	int toggled3D;

	/* Vblank method */
	int vblank_method;

	/* Invert eyes command status */
	int invert_eyes;

	/* Stereo Thread handler */
	pthread_t s_thread;

	/* Stereo thread state */
	char b_thread_running;
};

/* initialize controller */
struct nvstusb_context * nvstusb_init(void) 
{

	/* initialize usb */
	if (!nvstusb_usb_init()) return 0;

	/* open device */
	struct nvstusb_usb_device *dev = nvstusb_usb_open_device("nvstusb.fw");
	if (0 == dev) return 0;

	/* allocate context */
	struct nvstusb_context *ctx = malloc(sizeof(*ctx));
	if (0 == ctx) {
		fprintf(stderr, "nvstusb: Could not allocate %d bytes for nvstusb_context...\n", (int)sizeof(*ctx));
		nvstusb_usb_close_device(dev);
		nvstusb_usb_deinit();
		return 0;
	}
	ctx->rate = 0.0;
	ctx->eye = 0;
	ctx->device = dev;
	ctx->vblank_method = 0;
	ctx->toggled3D = 0;
	ctx->invert_eyes = 0;
	ctx->b_thread_running = 0;


	/* Vblank init */
	/* NVIDIA VBlank syncing environment variable defined, signal it and disable
	 * any attempt to application side method */
	if (getenv ("__GL_SYNC_TO_VBLANK"))
	{
		fprintf (stderr, "__GL_SYNC_TO_VBLANK defined in environment\n");
		ctx->vblank_method = 2;
		goto out_err;
	}

	/* Swap interval */
	glXSwapIntervalSGI = (PFNGLXSWAPINTERVALSGIPROC)glXGetProcAddress((const GLubyte *)"glXSwapIntervalSGI");

	if (NULL != glXSwapIntervalSGI) {
		fprintf(stderr, "nvstusb: forcing vsync\n");
		ctx->vblank_method = 3;
	}

	/* Sync Video */
	glXGetVideoSyncSGI = (PFNGLXGETVIDEOSYNCSGIPROC)glXGetProcAddress((const GLubyte *)"glXGetVideoSyncSGI");
	glXWaitVideoSyncSGI = (PFNGLXWAITVIDEOSYNCSGIPROC)glXGetProcAddress((const GLubyte *)"glXWaitVideoSyncSGI");
	if (NULL == glXWaitVideoSyncSGI) {
		glXGetVideoSyncSGI = 0;
	} else {
		ctx->vblank_method = 1;
	}

	if (NULL != glXGetVideoSyncSGI ) {
		fprintf(stderr, "nvstusb: GLX_SGI_video_sync supported!\n");
	}

	fprintf(stderr, "nvstusb:selected vblank method: %d\n", ctx->vblank_method);
out_err:
	return ctx;
}

/* deinitialize controller */
void
nvstusb_deinit(
		struct nvstusb_context *ctx
		) {
	if (0 == ctx) return;

#ifdef NOT_USED
	/* Close thread if running */
	if(ctx->b_thread_running) {
		nvstusb_stop_stereo_thread(ctx);
	}
#endif // NOT_USED

	/* close device */
	if (0 != ctx->device) nvstusb_usb_close_device(ctx->device);
	ctx->device = 0;

	/* close usb */
	nvstusb_usb_deinit();

	/* free context */
	memset(ctx, 0, sizeof(*ctx));
	free(ctx);
}

/* set controller refresh rate (should be monitor refresh rate) */
void
nvstusb_set_rate(
		struct nvstusb_context *ctx,
		float rate
		) {
	assert(ctx != 0);
	assert(ctx->device != 0);
	assert(rate > 60);

	/* send some magic data to device, this function is mainly black magic */

	/* some timing voodoo */
	int32_t frameTime   = (1000000.0/rate);     /* 8.33333 ms if 120 Hz */
	int32_t activeTime  = 2080;                 /* 2.08000 ms time each eye is on*/

	int32_t w = NVSTUSB_T2_COUNT(4568.50);      /* 4.56800 ms */
	int32_t x = NVSTUSB_T0_COUNT(4774.25);      /* 4.77425 ms */
	int32_t y = NVSTUSB_T0_COUNT(activeTime);
	int32_t z = NVSTUSB_T2_COUNT(frameTime);    

	uint8_t cmdTimings[] = { 
		NVSTUSB_CMD_WRITE,      /* write data */
		0x00,                   /* to address 0x2007 (0x2007+0x00) = ?? */
		0x18, 0x00,             /* 24 bytes follow */

		/* original: e1 29 ff ff (-54815; -55835) */
		w, w>>8, w>>16, w>>24,    /* 2007: ?? some timer 2 counter, 1020 is subtracted from this
		                           *       loaded at startup with:
		                           *       0x44 0xEC 0xFE 0xFF (-70588(-1020)) */ 
		/* original: 68 b5 ff ff (-19096), 4.774 ms */
		x, x>>8, x>>16, x>>24,    /* 200b: ?? counter saved at long at address 0x4f
		                           *       increased at timer 0 interrupt if bit 20h.1 
		                           *       is cleared, on overflow
		                           *       to 0 the code at 0x03c8 is executed.
		                           *       timer 0 will be started with this value
		                           *       by timer2 */

		/* original: 81 df ff ff (-8319), 2.08 ms */
		y, y>>8, y>>16, y>>24,    /* 200f: ?? counter saved at long at address 0x4f, 784 is added to this
		                           *       if PD1 is set, delay until turning eye off? */

		/* wave forms to send via IR: */
		0x30,                     /* 2013: 110000 PD1=0, PD2=0: left eye off  */
		0x28,                     /* 2014: 101000 PD1=1, PD2=0: left eye on   */
		0x24,                     /* 2015: 100100 PD1=0, PD2=1: right eye off */
		0x22,                     /* 2016: 100010 PD1=1, PD2=1: right eye on  */

		/* ?? used when frameState is != 2, for toggling bits in Port B,
		 * values seem to have no influence on the glasses or infrared signals */
		0x0a,                     /* 2017: 1010 */
		0x08,                     /* 2018: 1000 */
		0x05,                     /* 2019: 0101 */
		0x04,                     /* 201a: 0100 */

		z, z>>8, z>>16, z>>24     /* 201b: timer 2 reload value */
	}; 
	nvstusb_usb_write_bulk(ctx->device, 2, cmdTimings, sizeof(cmdTimings));

	uint8_t cmd0x1c[] = {
		NVSTUSB_CMD_WRITE,      /* write data */
		0x1c,                   /* to address 0x2023 (0x2007+0x1c) = ?? */
		0x02, 0x00,             /* 2 bytes follow */

		0x02, 0x00              /* ?? seems to be the start value of some 
		                           counter. runs up to 6, some things happen
		                           when it is lower, that will stop if when
		                           it reaches 6. could be the index to 6 byte values 
		                           at 0x17ce that are loaded into TH0*/
	};
	nvstusb_usb_write_bulk(ctx->device, 2, cmd0x1c, sizeof(cmd0x1c));

	/* wait at most 2 seconds before going into idle */
	uint16_t timeout = rate * 4;  

	uint8_t cmdTimeout[] = {
		NVSTUSB_CMD_WRITE,      /* write data */
		0x1e,                   /* to address 0x2025 (0x2007+0x1e) = timeout */
		0x02, 0x00,             /* 2 bytes follow */

		timeout, timeout>>8     /* idle timeout (number of frames) */
	};
	nvstusb_usb_write_bulk(ctx->device, 2, cmdTimeout, sizeof(cmdTimeout));

	uint8_t cmd0x1b[] = {
		NVSTUSB_CMD_WRITE,      /* write data */
		0x1b,                   /* to address 0x2022 (0x2007+0x1b) = ?? */
		0x01, 0x00,             /* 1 byte follows */

		0x07                    /* ?? compared with byte at 0x29 in TD_Poll()
		                           bit 0-1: index to a table of 4 bytes at 0x17d4 (0x00,0x08,0x04,0x0C),
		                           PB1 is set in TD_Poll() if this index is 0, cleared otherwise
		                           bit 2:   set bool21_4, start timer 1, enable ext. int. 5
		                           bit 3:   PC1 is set to the inverted value of this bit in TD_Poll()
		                           bit 4-5: index to a table of 4 bytes at 0x2a 
		                           bit 6:   restart t0 on some conditions in TD_Poll()
		                         */
	};
	nvstusb_usb_write_bulk(ctx->device, 2, cmd0x1b, sizeof(cmd0x1b));

	ctx->rate = rate;
}

void
nvstusb_invert_eyes(
		struct nvstusb_context *ctx
		) {
	ctx->invert_eyes = !ctx->invert_eyes;
}

/* set currently open eye */
static void
nvstusb_set_eye(
		struct nvstusb_context *ctx,
		nvstusb_eye eye
		) {
	assert(ctx != 0);
	assert(ctx->device != 0);
	assert(eye == nvstusb_left || eye == nvstusb_right || eye == nvstusb_quad);
	uint32_t r;

	//#define FF_TEST_R
#ifdef FF_TEST_R
	static int i = 0;
	i++;
	static int j = 0;
	static uint32_t r_tmp = NVSTUSB_T2_COUNT(0);;

	if(ctx->toggled3D) {
		r = r_tmp;
	} else {
		r = NVSTUSB_T2_COUNT((1e6/ctx->rate)/1.8);
	}

	if(i%32 == 0) {
		if(ctx->toggled3D) {
		  r_tmp -= 500;
		}

		if(((int)r_tmp) < NVSTUSB_T2_COUNT((1e6/ctx->rate))) {
		  r_tmp = NVSTUSB_T2_COUNT(0);
		}
		printf("r:%08x %d %lld %lld\n",r, r,NVSTUSB_T0_US(r), NVSTUSB_T2_US(r));
	}
#else
	r = NVSTUSB_T2_COUNT((1e6/ctx->rate)/1.8);
#endif

	switch(eye) {
	case nvstusb_right:
	case nvstusb_left:
fprintf(stderr,"nvstusb_set_eye, eye = %d\n",eye);
		{
		  uint8_t buf[8] = { 
		    NVSTUSB_CMD_SET_EYE,      /* set shutter state */
		    ((eye==nvstusb_right)^(ctx->invert_eyes))?0xFE:0xFF,        /* eye selection */
		    0x00, 0x00,               /* unused */
		    r, r>>8, r>>16, r>>24
		  };
		  nvstusb_usb_write_bulk(ctx->device, 1, buf, 8);		
		}
		break;
	case nvstusb_quad:
		{
		  nvstusb_set_eye(ctx, nvstusb_right);
		  nvstusb_set_eye(ctx, nvstusb_left);
		}
		break;
	}
}


/* perform swap and toggle eyes hopefully with correct timing */
void
nvstusb_swap(
		struct nvstusb_context *ctx,
		nvstusb_eye eye,
		void (*swapfunc)()
		) {
	assert(ctx != 0);
	assert(ctx->device != 0);
	assert(eye == nvstusb_left || eye == nvstusb_right || eye == nvstusb_quad);

	/* if we have the GLX_SGI_video_sync extension, we just wait
	 * for vertical blanking, then issue swap. */
//fprintf(stderr,"nvstusb_swap:  method %d\n",ctx->vblank_method);
	switch(ctx->vblank_method) {
	case 0:
		{
		  /* Swap buffers */
		  if(swapfunc) {
		    swapfunc();
		  }

		  /* Sw Vsync method: read from front buffer.
		   * this operation can only finish after swapping is complete. 
		   * (seems like it won't work if page flipping is disabled) */
		  uint8_t pixels[4] = { 255, 0, 255, 255 };
		  glReadBuffer(GL_FRONT);
		  glReadPixels(1,1,1,1,GL_RGB, GL_UNSIGNED_BYTE, pixels);
		  nvstusb_set_eye(ctx, eye);
		}
		break;
	case 1:
		{
		  unsigned int count;


		  if(eye == nvstusb_quad) {
		    //int before_count;
		    /* Waiting OpenGL sync, Do not use current count to */
		    /* prevent eyes from being inverted */
		    glXGetVideoSyncSGI(&count);
		    glXWaitVideoSyncSGI(2, 0, &count);

		  } else {
		    /* Waiting OpenGL sync */
//fprintf(stderr,"using glXGetVideoSyncSGI...\n");
			// This path IS taken on craik...
		    glXGetVideoSyncSGI(&count);
		    glXWaitVideoSyncSGI(2, (count+1)%2, &count);
		  }

		  /* Change eye */
		  nvstusb_set_eye(ctx, eye);

		  /* Swap buffers */
		  if(swapfunc) {
		    swapfunc();
		  }
		}
		break;
	case 2:
		{
		  /* case __GL_SYNC_TO_VBLANK is defined */

		  /* Swap buffers */
		  if(swapfunc) {
		    swapfunc();
		  }

		  /* Change eye */
		  nvstusb_set_eye(ctx, eye);
		}
		break;
	case 3:
		{
		  static int i_current_interval = -1;
		  int i_interval;
		  if(eye == nvstusb_quad) {
		    i_interval = 2;
		  } else {
		    i_interval = 1;
		  }

		  /* Swap interval */
		  if(i_current_interval != i_interval) {
		    glXSwapIntervalSGI(i_interval);
		    i_current_interval = i_interval;
		  }

		  /* Swap buffers */
		  if(swapfunc) {
		    swapfunc();
		  }

		  /* Change eye */
		  nvstusb_set_eye(ctx, eye);

		}
		break;
	default:
		fprintf(stderr, "nvstusb: unknown vblank method\n");
	}
}

/* get key status from controller */
void
nvstusb_get_keys(
		struct nvstusb_context *ctx,
		struct nvstusb_keys *keys
		) {
	assert(ctx  != 0);
	assert(keys != 0);

	uint8_t cmd1[] = { 
		NVSTUSB_CMD_READ |      /* read and clear data */
		  NVSTUSB_CMD_CLEAR,

		0x18,                   /* from address 0x201F (0x2007+0x18) = status? */
		0x03, 0x00              /* read/clear 3 bytes */
	};
	nvstusb_usb_write_bulk(ctx->device, 2, cmd1, sizeof(cmd1));

	uint8_t readBuf[4+cmd1[2]];
	nvstusb_usb_read_bulk(ctx->device, 4, readBuf, sizeof(readBuf));

	/* readBuf[0] contains the offset (0x18),
	 * readBuf[1] contains the number of read bytes (0x03),
	 * readBuf[2] (msb) and readBuf[3] (lsb) of the bytes sent (sizeof(cmd1)) 
	 * readBuf[4] and following contain the requested data */

	/* from address 0x201F:
	 * signed 8 bit integer: amount the wheel was turned without the button pressed
	 */
	keys->deltaWheel = readBuf[4];

	/* from address 0x2020:
	 * signed 8 bit integer: amount the wheel was turned with the button pressed
	 */
	keys->pressedDeltaWheel = readBuf[5];

	/* from address 0x2021:
	 * bit 0: front button was pressed since last time (presumably fom pin 4 on port C)
	 * bit 1: logic state of pin 7 on port E
	 * bit 2: logic state of pin 2 on port C
	 */
	keys->toggled3D  = readBuf[6] & 0x01; 

	if(keys->toggled3D) {
		ctx->toggled3D = !ctx->toggled3D;
	} 
}

#ifdef NOT_USED
/* Start Stereo Thread - For GL_STEREO */
void nvstusb_start_stereo_thread(struct nvstusb_context *ctx) 
{
	assert(ctx != 0);
	assert(ctx->device != 0);

	ctx->b_thread_running = true;
	if ( pthread_create(&ctx->s_thread, NULL, nvstusb_stereo_thread, (void *)ctx) != 0 ) {
		fprintf(stderr, "nvstusb: Unable to start stereo stread");
	}
}

/* End Stereo Thread - For GL_STEREO  */
void nvstusb_stop_stereo_thread(struct nvstusb_context *ctx) 
{
	assert(ctx != 0);
	assert(ctx->device != 0);

	if(!ctx->b_thread_running) return;

	ctx->b_thread_running = false;
	if ( pthread_join(ctx->s_thread, NULL) != 0 ) {
		fprintf(stderr, "nvstusb: Unable to wait end of stereo stread");
	}
}

/* Stereo thread - For GL_STEREO  */
static void * nvstusb_stereo_thread(void * in_pv_arg)
{
	struct nvstusb_context *ctx = (struct nvstusb_context *) in_pv_arg;
	Display *dpy;
	Window win;

	/* Openning X display */
	dpy = XOpenDisplay(0);

	/* Preparing new X window */
	Window s_window;
	static int attributeList[] =
	{ GLX_RGBA,
		GLX_DOUBLEBUFFER,
		GLX_RED_SIZE,
		1,
		GLX_GREEN_SIZE,
		1,
		GLX_BLUE_SIZE,
		1,
		None };
	XVisualInfo *vi = glXChooseVisual(dpy, DefaultScreen(dpy), attributeList);
	s_window = RootWindow(dpy, vi->screen);
	XSetWindowAttributes swa;
	swa.colormap = XCreateColormap(dpy, s_window, vi->visual, AllocNone);
	swa.override_redirect = true;

	/* Create X window 1x1 top left of screen */
	win = XCreateWindow(dpy,
		  s_window ,
		  0,
		  0,
		  1,
		  1,
		  0,
		  vi->depth,
		  InputOutput,
		  vi->visual,
		  CWColormap|CWOverrideRedirect,
		  &swa);

	XMapWindow(dpy, win);

	/* Create glX context */
	GLXContext glx_ctx = glXCreateContext(dpy, vi, 0, true);
	glXMakeCurrent(dpy, win, glx_ctx);

	/* Loop until stop */
	while (ctx->b_thread_running) {
		/* Send swap to usb controler */
		nvstusb_swap(ctx, nvstusb_quad, NULL /*f_swap*/);

		/* Read status from usb controler */
		struct nvstusb_keys k;
		nvstusb_get_keys(ctx, &k);
		if (k.toggled3D) {
		  nvstusb_invert_eyes(ctx);
		}
	}
	/* Destroy context */
	glx_ctx = glXGetCurrentContext();
	glXDestroyContext(dpy, glx_ctx);

	return NULL;
}

/* Refresh rate calculation */
void nvstusb_print_refresh_rate(void)
{
	static int i_it = 0;
	static uint64_t i_last = 0;
	static uint64_t i_first = 0;
	static uint64_t i_acc_var = 0;
	double f_mean, f_var;

	/* First frame */
	if(i_it == 0) {
		struct timespec s_tmp;
		clock_gettime(CLOCK_REALTIME, &s_tmp);
		i_first = (uint64_t)s_tmp.tv_sec*1000000+(uint64_t)s_tmp.tv_nsec/1000;
		i_last = i_first;
		f_mean = 0;
		f_var = 0;

	} else {
		struct timespec s_tmp;
		clock_gettime(CLOCK_REALTIME, &s_tmp);
		uint64_t i_new = (uint64_t)s_tmp.tv_sec*1000000+(uint64_t)s_tmp.tv_nsec/1000;
		/* Update average */
		f_mean = (double)(i_new-i_first)/(i_it);
		/* Calculate variance */
		i_acc_var += pow((double)(i_new-i_last)-f_mean, 2);

		/* std dev */
		f_var = (double)sqrt(i_acc_var/i_it);
		i_last = i_new;

		/* Display each 512 frame */
		if(i_it % 512 == 0) {
		  fprintf(stderr,
"nvstusb: frame:%d (%0.2f s) mean: %f Hz (%0.2f us) sqrt(var): %0.2f us (%0.1f %%)\n",
			i_it,f_mean*i_it/1000000.0, 1000000/f_mean, f_mean, f_var, 100.0*f_var/f_mean);
		}
	}
	/* Increment frame counter */
	i_it++;
}
#endif // NOT_USED

#endif // HAVE_LIBUSB
#endif // HAVE_OPENGL
