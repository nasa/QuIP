/* display of data using ncurses library */

#include "quip_config.h"

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
		CFBVAR(fbip,xres),CFBVAR(fbip,yres));				ADDIT
	move(y++,x);
	sprintf(msg_str,"\tVirtual:\t%ld\tx\t%ld",
		CFBVAR(fbip,xres_virtual),CFBVAR(fbip,yres_virtual));		ADDIT
	move(y++,x);
	sprintf(msg_str,"\tOffset:\t%ld\t\t%ld",
		CFBVAR(fbip,xoffset),CFBVAR(fbip,yoffset));			ADDIT
	move(y++,x);
	sprintf(msg_str,"\tBitsPerPixel:\t%ld", CFBVAR(fbip,bits_per_pixel));	ADDIT
	move(y++,x);
	sprintf(msg_str,"\tGrayscale:\t%ld", CFBVAR(fbip,grayscale));		ADDIT
	move(y++,x);
	sprintf(msg_str,"\tred bitfield:\t%ld\t%ld\t%ld",
CFBVAR(fbip,red.offset),CFBVAR(fbip,red.length),CFBVAR(fbip,red.msb_right));		ADDIT
	move(y++,x);
	sprintf(msg_str,"\tgreen bitfield:\t%ld\t%ld\t%ld",
CFBVAR(fbip,green.offset),CFBVAR(fbip,green.length),CFBVAR(fbip,green.msb_right));	ADDIT
	move(y++,x);
	sprintf(msg_str,"\tblue bitfield:\t%ld\t%ld\t%ld",
CFBVAR(fbip,blue.offset),CFBVAR(fbip,blue.length),CFBVAR(fbip,blue.msb_right));	ADDIT
	move(y++,x);
	sprintf(msg_str,"\ttransp bitfield:\t%ld\t%ld\t%ld",
CFBVAR(fbip,transp.offset),CFBVAR(fbip,transp.length),CFBVAR(fbip,transp.msb_right));	ADDIT

	/* bitfields for red,green,blue,transp - ? */
	move(y++,x);
	sprintf(msg_str,"\tNon-standard pixel format:\t%ld",
		CFBVAR(fbip,nonstd));					ADDIT
	move(y++,x);
	sprintf(msg_str,"\tActivate:\t%ld",
		CFBVAR(fbip,activate));					ADDIT
	move(y++,x);
	sprintf(msg_str,"\tSize (mm):\t%ld x %ld",
		CFBVAR(fbip,width),CFBVAR(fbip,height));				ADDIT
	move(y++,x);
	sprintf(msg_str,"\tPixclock:\t%ld",
		CFBVAR(fbip,pixclock));					ADDIT
	move(y++,x);
	sprintf(msg_str,"\tMargins:\t%ld\t%ld\t%ld\t%ld",
		CFBVAR(fbip,left_margin), CFBVAR(fbip,right_margin),
		CFBVAR(fbip,upper_margin), CFBVAR(fbip,lower_margin));		ADDIT
	move(y++,x);
	sprintf(msg_str,"\tHsync len:\t%ld",
		CFBVAR(fbip,hsync_len));					ADDIT
	move(y++,x);
	sprintf(msg_str,"\tVsync len:\t%ld",
		CFBVAR(fbip,vsync_len));					ADDIT
	move(y++,x);
	sprintf(msg_str,"\tSync:\t%ld",	CFBVAR(fbip,sync));			ADDIT
	move(y++,x);
	sprintf(msg_str,"\tVMode:\t%ld",	CFBVAR(fbip,vmode));			ADDIT
	move(y++,x);
	sprintf(msg_str,"\tReserved:\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld",
		CFBVAR(fbip,reserved[0]), CFBVAR(fbip,reserved[1]),
		CFBVAR(fbip,reserved[2]), CFBVAR(fbip,reserved[3]),
		CFBVAR(fbip,reserved[4]), CFBVAR(fbip,reserved[5]));		ADDIT
	/* rotate field not present on purkinje - different kernel version? */
	/*
	sprintf(msg_str,"\tRotate:\t%d",		CFBVAR(fbip,rotate));			ADDIT
	*/
	y++;
	move(y++,x);
	sprintf(msg_str,"\tn_screen_updates:\t%d",++n_updates);		ADDIT
	refresh();
}

#endif /* HAVE_NCURSES */
#endif /* HAVE_FB_DEV */

