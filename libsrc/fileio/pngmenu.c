
#include "quip_config.h"

char VersionId_fio_pngmenu[] = QUIP_VERSION_STRING;

#ifdef HAVE_PNG

#include <stdio.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include "debug.h"
#include "data_obj.h"

#include "fio_prot.h"
#include "img_file.h"
#include "filetype.h"

#include "query.h"
#include "rv_api.h"

#include "raw.h"	/* set_raw_sizes() ? */
#include "menuname.h"

extern void set_bg_color(int bg_color);
extern void set_color_type(int given_color_type);
	
static COMMAND_FUNC( do_set_bg_color )
{
	int bg_color;
			
#define MIN_BG_COLOR		0x000000
#define MAX_BG_COLOR		0xFFFFFF	

	bg_color= HOW_MANY("background color 0x000000 - 0xFFFFFF");

	if( bg_color < MIN_BG_COLOR || bg_color > MAX_BG_COLOR ) {
		sprintf(error_string, "background color (%d) should be in range %d - %d",
			bg_color, MIN_BG_COLOR, MAX_BG_COLOR);

		WARN(error_string);
		return;
	}
	
	set_bg_color(bg_color);
}


#ifdef UNUSED
static COMMAND_FUNC( do_set_color_type )
{
	int color_type;
		
#define N_COLOR_TYPES    5

        static char *color_type_strs[N_COLOR_TYPES]={"gray","palette","rgb","rgb_alpha","gray_alpha"};
			
	color_type = which_one("color type", N_COLOR_TYPES, color_type_strs);
	if( color_type < 0 ) return;

	set_color_type(color_type);
}
#endif /* UNUSED */



static Command png_ctbl[]={
{ "bg_color",		do_set_bg_color,	"specify default background color"	},
#ifdef UNUSED	
{ "color_type",		do_set_color_type,	"specify color type for writing"	},
#endif /* UNUSED */
{ "quit",		popcmd,		"exit submenu"					},
{ NULL_COMMAND										}
};

COMMAND_FUNC( png_menu )
{
	PUSHCMD(png_ctbl,"png");
}




#endif /* HAVE_PNG */
