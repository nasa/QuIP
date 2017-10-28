#include "viewer.h"

// This file should contain prototypes for all functions used
// just within the module...

extern COMMAND_FUNC( do_draw_menu );

#ifdef BUILD_FOR_IOS
extern void set_font_by_name(Viewer *vp,const char *s);
extern void set_backlight(CGFloat l);
#endif /* BUILD_FOR_IOS */

extern void bring_image_to_front(QSP_ARG_DECL
	Viewer *vp, Data_Obj *dp, int x, int y );

#ifdef BUILD_FOR_MACOS
extern void _posn_genwin(QSP_ARG_DECL  IOS_Item *ip, int x, int y);
#define posn_genwin(ip,x,y) _posn_genwin(QSP_ARG  ip,x,y)
#endif // BUILD_FOR_MACOS

