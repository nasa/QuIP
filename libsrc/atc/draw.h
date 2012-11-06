#ifdef HAVE_X11

#ifndef SELECTED_PLANE_COLOR

#include "flight_path.h"
#include "viewer.h"

#define SELECTED_PLANE_COLOR	LIGHT_CYAN
#define REVEAL_PLANE_COLOR	WHITE
#define DRAG_PLANE_COLOR	WHITE
#define DEFAULT_PLANE_COLOR	MAGENTA
#define DEFAULT_HL_COLOR	BRT_MAGENTA
extern int default_plane_color;
#define DATA_TAG_COLOR		GRAY
#define TAG_LINE_COLOR		DARK_GRAY
#define MOUSE_COLOR		GREEN
#define CROSSING_COLOR		GRAY

#define PLANECOLOR_1	RED	/* Corresponding to ALTITUDE_1 (310) */
#define PLANECOLOR_2	YELLOW	/* Altitude_2 (330) */
#define PLANECOLOR_3	GREEN	/* Altitude_3 (350) */
#define PLANECOLOR_4	BLUE	/* Altitude_4 (370) */

#define HL_PLANECOLOR_1	BRT_RED	/* Corresponding to ALTITUDE_1 (310) */
#define HL_PLANECOLOR_2	BRT_YELLOW	/* Altitude_2 (330) */
#define HL_PLANECOLOR_3	BRT_GREEN	/* Altitude_3 (350) */
#define HL_PLANECOLOR_4	BRT_BLUE	/* Altitude_4 (370) */

#define PLANE_RADIUS	20	/* BUG get a better number */

#define KNOTS_PER_SCREEN	266.6
#define PIXELS_PER_KNOT		( max_y / KNOTS_PER_SCREEN )
#define KNOTS_TO_PIXELS(k)	( k * PIXELS_PER_KNOT )

/* vertical distance between text in data tag
 * This was 6, but that is too small w/ the default X font...
 * we might like to select a different font.
 */
#define DEFAULT_DATA_TAG_SPACING	6

/* globals */

extern Point max_point;

#define max_x	(max_point.p_x)
#define max_y	(max_point.p_y)

extern int tag_x_offset;
extern Point *center_p;

#define DRAW_LINE(ptp1,ptp2)	atc_line( ((int) ((ptp1)->p_x))+this_disparity, (int) ((ptp1)->p_y), \
					 ((int) ((ptp2)->p_x))+this_disparity, (int) ((ptp2)->p_y))

#define DRAW_TAG_LINE(fpp,color)						\
										\
	{									\
		xp_select(color);						\
		DRAW_LINE( &fpp->fp_vertices[0], &fpp->fp_tag_line );		\
	}



#define PSEUDOCOLOR

#ifdef PSEUDOCOLOR

#define BASE_COLOR_INDEX	COLOR_BASE	/* big enough so not protected... */
					/* and compatible w/ default grayscale */

/* We make an enum for the colors so that the compiler will tell
 * us if we forget to define one of them in a switch statement (see draw.c).
 * We make these offsets from the base color so we don't have to mess with
 * the window manager's colors.
 */

typedef enum {
	LIGHT_CYAN_OFFSET,	/* 48 */
	WHITE_OFFSET,		/* 49 */
	BLACK_OFFSET,		/* 50 */
	DARK_GRAY_OFFSET,	/* 51 */
	GRAY_OFFSET,		/* 52 */
	GREEN_OFFSET,		/* 53 */
	YELLOW_OFFSET,		/* 54 */
	BLUE_OFFSET,		/* 55 */
	RED_OFFSET,		/* 56 */
	BROWN_OFFSET,
	MAGENTA_OFFSET,
	BRT_MAGENTA_OFFSET,
	BRT_GREEN_OFFSET,
	BRT_YELLOW_OFFSET,
	BRT_BLUE_OFFSET,
	BRT_RED_OFFSET,

	N_DEFINED_COLORS	/* must be last */
} Color_Offset;

#define LIGHT_CYAN	(BASE_COLOR_INDEX+LIGHT_CYAN_OFFSET)
#define WHITE		(BASE_COLOR_INDEX+WHITE_OFFSET)
#define BLACK		(BASE_COLOR_INDEX+BLACK_OFFSET)
#define DARK_GRAY	(BASE_COLOR_INDEX+DARK_GRAY_OFFSET)
#define GRAY		(BASE_COLOR_INDEX+GRAY_OFFSET)
#define GREEN		(BASE_COLOR_INDEX+GREEN_OFFSET)
#define YELLOW		(BASE_COLOR_INDEX+YELLOW_OFFSET)
#define BLUE		(BASE_COLOR_INDEX+BLUE_OFFSET)
#define RED		(BASE_COLOR_INDEX+RED_OFFSET)
#define BROWN		(BASE_COLOR_INDEX+BROWN_OFFSET)
#define MAGENTA		(BASE_COLOR_INDEX+MAGENTA_OFFSET)
#define BRT_MAGENTA	(BASE_COLOR_INDEX+BRT_MAGENTA_OFFSET)
#define BRT_GREEN	(BASE_COLOR_INDEX+BRT_GREEN_OFFSET)
#define BRT_YELLOW	(BASE_COLOR_INDEX+BRT_YELLOW_OFFSET)
#define BRT_BLUE	(BASE_COLOR_INDEX+BRT_BLUE_OFFSET)
#define BRT_RED		(BASE_COLOR_INDEX+BRT_RED_OFFSET)

#else /* ! PSEUDOCOLOR */

#define PACK_COLOR(r,g,b)	( ((r>>3)<<10) | ((g>>3)<5) | (b>>3) )

#define BLACK		PACK_COLOR(  0,  0,  0)
#define WHITE		PACK_COLOR(255,255,255)
#define GRAY		PACK_COLOR(140,140,140)
#define DARK_GRAY	PACK_COLOR( 60, 60, 60)
#define MAGENTA		PACK_COLOR(200,  0,255)
#define BLUE		PACK_COLOR(100, 50,240)
#define YELLOW		PACK_COLOR(240,210,  0)
#define RED		PACK_COLOR(220,  0, 90)
#define GREEN		PACK_COLOR(  0,190, 30)
#define LIGHT_CYAN	PACK_COLOR(100,200,255)
#define BROWN		PACK_COLOR(150, 50,  0)

#define BRT_MAGENTA	PACK_COLOR(200,100,255)
#define BRT_BLUE	PACK_COLOR(150, 80,255)
#define BRT_YELLOW	PACK_COLOR(255,255,100)
#define BRT_RED		PACK_COLOR(255, 50,120)
#define BRT_GREEN	PACK_COLOR( 50,255, 70)

#endif /* ! PSEUDOCOLOR */

#define CIRCLE_COLOR		GRAY
#define PARTIAL_TAG_COLOR	BROWN	/* model representation of tags w/ speed only */

#define FLIGHT_INFO_SYSVAR_NAME "_flight_paths"

#endif /* ! SELECTED_PLANE_COLOR */

#endif /* HAVE_X11 */

