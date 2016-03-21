// stereo.c - support for nVidia shutter glasses, aka nvstusb

#include "quip_config.h"

//#include <stdio.h>
//#include <stdlib.h>

#ifdef HAVE_GL_GLUT_H
#include <GL/glut.h>
#endif // HAVE_GL_GLUT_H

#include "quip_prot.h"
#include "glut_supp.h"

#include "nvstusb.h"

//#include "stereo_helper.h"

static int current_eye=0;

#ifdef HAVE_LIBUSB


// context for dealing with the stereo usb IR emitter
static struct nvstusb_context *nv_ctx = NULL;


// 3D camera from stereo helper
//StereoHelper::Camera cam;

// forces a particular eye to be displayed (for debugging)
// 0 = normal swapping, 1 = left always, 2 = right always
#ifdef FOOBAR
static int force_eye = 0;
#endif // FOOBAR

#define INSURE_NVST( whence )					\
								\
	if( nv_ctx == NULL ){					\
		sprintf(ERROR_STRING,				\
"%s:  nVidia shutter glasses system is not present, or has not been initialized.",	\
			#whence );				\
		WARN(ERROR_STRING);				\
	}

#else // ! HAVE_LIBUSB

#define INSURE_NVST( whence )					\
								\
	sprintf(ERROR_STRING,					\
	"%s:  Sorry, no libusb-1.0 support in this build, no nvidia stereo!?",	\
		#whence);					\
	WARN(ERROR_STRING);					\
	return;

#endif // ! HAVE_LIBUSB

#ifdef HAVE_LIBUSB

static void read_emitter(void)
{
	// get the status of the button/wheel on the emitter (you MUST do this,
	// otherwise the whole system will stall out after just a couple of frames)
	struct nvstusb_keys k;

	nvstusb_get_keys(nv_ctx, &k);

	// The sample program used these button events to control
	// the program...  We should add these things to the general
	// event library???

#ifdef FOOBAR
	// the 3D button on the IR emitter controls toggling the rotation on and
	// off
	if (k.toggled3D) {
		rotation = !rotation;
		printf("Toggled rotation.\n");
	}

	// the wheel on the back adjusts the focal length of the camera (and
	// interoccular distance, since we want to maintain IOD = 1/30th of the
	// focal length)
	if (k.deltaWheel != 0) {
		cam.focal += k.deltaWheel;
		cam.iod = cam.focal / 30.0f;
		printf("Set camera focal length to %f.\n", cam.focal);
	}

	// you can also use k.pressedDeltaWheel, which reports the amount the wheel
	// moves while the 3D button is pressed
#endif // FOOBAR
}

#endif // HAVE_LIBUSB

#ifdef FOOBAR
		case 'c': case 'C': // switch camera type
				if (cam.type == StereoHelper::TOE_IN) {
					cam.type = StereoHelper::PARALLEL_AXIS_ASYMMETRIC;
					printf("Using parallel axis asymmetric frusta camera.\n");
				} else {
					cam.type = StereoHelper::TOE_IN;
					printf("Using toe-in stereo camera.\n");
				}
				break;

		case 'f': case 'F': // force eye
				force_eye = (force_eye + 1) % 3;
				if (force_eye == 0) {
					printf("Swapping eyes normally.\n");
				} else if (force_eye == 1) {
					printf("Forcing left eye always.\n");
				} else {
					printf("Forcing right eye always.\n");
				}
				break;
#endif // FOOBAR


static COMMAND_FUNC( init_stereo )
{
	double frame_rate;

	frame_rate = HOW_MUCH("frame rate");

#ifdef HAVE_LIBUSB
	// initialize communications with the usb emitter
	if (nv_ctx == NULL) {
		nv_ctx = nvstusb_init();
		if (nv_ctx == NULL) {
			sprintf(ERROR_STRING, "init_stereo:  Could not initialize NVIDIA 3D Vision IR emitter!?");
			WARN(ERROR_STRING);
		}
		return;
	} else {
		sprintf(ERROR_STRING, "init_stereo:  nVidia 3D Vision IR emitter already initialized!?");
		WARN(ERROR_STRING);
	}

	// auto-config the vsync rate
//	StereoHelper::ConfigRefreshRate(nv_ctx);
//
//	// set up our 3D camera (see stereohelper.h for more documentation)
//	cam.type = StereoHelper::PARALLEL_AXIS_ASYMMETRIC;
//	cam.eye = StereoHelper::Vec3(39.0f, 53.0f, 22.0f);
//	cam.look = StereoHelper::Vec3(0.0f, 0.0f, 0.0f);
//	cam.up = StereoHelper::Vec3(0.0f, 1.0f, 0.0f);
//	cam.focal = 70.0f;
//	cam.fov = 50.0f;
//	cam.iod = cam.focal / 30.0f;
//	cam.near = 1.0f;
//	cam.far = 200.0f;

#ifdef NOT_YET
	// we'd like to read this modeline stuff, but
	// need to use the display we already have...

        //Display *display = XOpenDisplay(0);
        double display_num = DefaultScreen(display);
        XF86VidModeModeLine mode_line;
        int pixel_clk = 0;
        XF86VidModeGetModeLine(display, display_num, &pixel_clk, &mode_line);
        double frame_rate = (double) pixel_clk * 1000.0 / mode_line.htotal / mode_line.vtotal;
        printf("Detected refresh rate of %f Hz.\n", frame_rate);

        nvstusb_set_rate(ctx, frame_rate);
#endif // NOT_YET
        nvstusb_set_rate(nv_ctx, /*119.982181*/ frame_rate );



#else // ! HAVE_LIBUSB

	WARN("Sorry, no support for nVidia stereo (libusb-1.0 missing)");

#endif // ! HAVE_LIBUSB

}


static COMMAND_FUNC( cleanup_stereo )
{
	INSURE_NVST(cleanup_stereo)

#ifdef HAVE_LIBUSB
	// clean up usb emitter
	nvstusb_deinit(nv_ctx);
	nv_ctx = NULL;
#endif // HAVE_LIBUSB
}

static COMMAND_FUNC( swap_stereo )
{
	INSURE_NVST(swap_stereo)

#ifdef HAVE_LIBUSB
	// image for current eye should have been drawn already...
	nvstusb_swap(nv_ctx, (nvstusb_eye) current_eye, /*glutSwapBuffers*/ swap_buffers );
	current_eye = (current_eye + 1) % 2;

	read_emitter();	// they say we have to do this...
#endif // HAVE_LIBUSB
}

#define N_EYE_CHOICES	2
static const char *eye_choices[N_EYE_CHOICES]={"left","right"};

static COMMAND_FUNC( do_set_eye )
{
	int n;

	n=WHICH_ONE("eye",N_EYE_CHOICES,eye_choices);
	if( n < 0 ) return;
	current_eye = n;
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(stereo_menu,s,f,h)

MENU_BEGIN(stereo)
ADD_CMD( init,		init_stereo,	initialize nVidia shutter glasses system	)
ADD_CMD( cleanup,	cleanup_stereo,	shut down nVidia shutter glasses system		)
ADD_CMD( swap_buffers,	swap_stereo,	display the next eye image			)
ADD_CMD( set_eye,	do_set_eye,	select current eye				)
MENU_END(stereo)

COMMAND_FUNC( do_stereo_menu )
{
	// static int inited=0;

	// auto init?

	PUSH_MENU(stereo);
}

