
#include "quip_config.h"

#ifdef HAVE_PNG

#include <stdio.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include "debug.h"
#include "data_obj.h"

#include "fio_prot.h"
#include "img_file.h"

#include "quip_prot.h"
#include "rv_api.h"

#include "img_file/raw.h"	/* set_raw_sizes() ? */
#include "img_file/fio_png.h"	/* set_raw_sizes() ? */

static COMMAND_FUNC( do_set_bg_color )
{
	int bg_color;
			
#define MIN_BG_COLOR		0x000000
#define MAX_BG_COLOR		0xFFFFFF	

	bg_color= HOW_MANY("background color 0x000000 - 0xFFFFFF");

	if( bg_color < MIN_BG_COLOR || bg_color > MAX_BG_COLOR ) {
		sprintf(ERROR_STRING, "background color (%d) should be in range %d - %d",
			bg_color, MIN_BG_COLOR, MAX_BG_COLOR);

		WARN(ERROR_STRING);
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


#define ADD_CMD(s,f,h)	ADD_COMMAND(png_menu,s,f,h)

MENU_BEGIN(png)
ADD_CMD( bg_color,	do_set_bg_color,	specify default background color )
#ifdef UNUSED	
ADD_CMD( color_type,	do_set_color_type,	specify color type for writing )
#endif /* UNUSED */
MENU_END(png)

COMMAND_FUNC( do_png_menu )
{
	PUSH_MENU(png);
}




#endif /* HAVE_PNG */
