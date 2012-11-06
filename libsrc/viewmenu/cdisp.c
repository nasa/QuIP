/* display of data using ncurses library */

#include "quip_config.h"

char VersionID_viewmenu_cdisp[] = QUIP_VERSION_STRING;

#ifdef HAVE_FB_DEV
#ifdef HAVE_NCURSES

#ifdef HAVE_NCURSES_H
#include <ncurses.h>
#endif

#include "my_fb.h"

static int inited=0;
static int n_updates=0;

static void nc_init()
{
	initscr();
	inited=1;
}


#define CFBVAR(f)	((u_long)fbip->fbi_var_info.f)
#define FBVAR(f)	(fbip->fbi_var_info.f)

#define ADDIT	addstr(msg_str); clrtoeol();

void nc_show_var_info(FB_Info *fbip)
{
	int x,y;

	if(!inited) nc_init();

	move(y=0,x=0);

	/* Now display the contents */
	sprintf(msg_str,"Frame buffer %s:",fbip->fbi_name);		ADDIT 
	x=10;
	y++;
	move(y++,x);
	sprintf(msg_str,"\tResolution:\t%ld\tx\t%ld",
		CFBVAR(xres),CFBVAR(yres));				ADDIT
	move(y++,x);
	sprintf(msg_str,"\tVirtual:\t%ld\tx\t%ld",
		CFBVAR(xres_virtual),CFBVAR(yres_virtual));		ADDIT
	move(y++,x);
	sprintf(msg_str,"\tOffset:\t%ld\t\t%ld",
		CFBVAR(xoffset),CFBVAR(yoffset));			ADDIT
	move(y++,x);
	sprintf(msg_str,"\tBitsPerPixel:\t%ld", CFBVAR(bits_per_pixel));	ADDIT
	move(y++,x);
	sprintf(msg_str,"\tGrayscale:\t%ld", CFBVAR(grayscale));		ADDIT
	move(y++,x);
	sprintf(msg_str,"\tred bitfield:\t%ld\t%ld\t%ld",
CFBVAR(red.offset),CFBVAR(red.length),CFBVAR(red.msb_right));		ADDIT
	move(y++,x);
	sprintf(msg_str,"\tgreen bitfield:\t%ld\t%ld\t%ld",
CFBVAR(green.offset),CFBVAR(green.length),CFBVAR(green.msb_right));	ADDIT
	move(y++,x);
	sprintf(msg_str,"\tblue bitfield:\t%ld\t%ld\t%ld",
CFBVAR(blue.offset),CFBVAR(blue.length),CFBVAR(blue.msb_right));	ADDIT
	move(y++,x);
	sprintf(msg_str,"\ttransp bitfield:\t%ld\t%ld\t%ld",
CFBVAR(transp.offset),CFBVAR(transp.length),CFBVAR(transp.msb_right));	ADDIT

	/* bitfields for red,green,blue,transp - ? */
	move(y++,x);
	sprintf(msg_str,"\tNon-standard pixel format:\t%ld",
		CFBVAR(nonstd));					ADDIT
	move(y++,x);
	sprintf(msg_str,"\tActivate:\t%ld",
		CFBVAR(activate));					ADDIT
	move(y++,x);
	sprintf(msg_str,"\tSize (mm):\t%ld x %ld",
		CFBVAR(width),CFBVAR(height));				ADDIT
	move(y++,x);
	sprintf(msg_str,"\tPixclock:\t%ld",
		CFBVAR(pixclock));					ADDIT
	move(y++,x);
	sprintf(msg_str,"\tMargins:\t%ld\t%ld\t%ld\t%ld",
		CFBVAR(left_margin), CFBVAR(right_margin),
		CFBVAR(upper_margin), CFBVAR(lower_margin));		ADDIT
	move(y++,x);
	sprintf(msg_str,"\tHsync len:\t%ld",
		CFBVAR(hsync_len));					ADDIT
	move(y++,x);
	sprintf(msg_str,"\tVsync len:\t%ld",
		CFBVAR(vsync_len));					ADDIT
	move(y++,x);
	sprintf(msg_str,"\tSync:\t%ld",	CFBVAR(sync));			ADDIT
	move(y++,x);
	sprintf(msg_str,"\tVMode:\t%ld",	CFBVAR(vmode));			ADDIT
	move(y++,x);
	sprintf(msg_str,"\tReserved:\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld",
		CFBVAR(reserved[0]), CFBVAR(reserved[1]),
		CFBVAR(reserved[2]), CFBVAR(reserved[3]),
		CFBVAR(reserved[4]), CFBVAR(reserved[5]));		ADDIT
	/* rotate field not present on purkinje - different kernel version? */
	/*
	sprintf(msg_str,"\tRotate:\t%d",		CFBVAR(rotate));			ADDIT
	*/
	y++;
	move(y++,x);
	sprintf(msg_str,"\tn_screen_updates:\t%d",++n_updates);		ADDIT
	refresh();
}

#endif /* HAVE_NCURSES */
#endif /* HAVE_FB_DEV */

