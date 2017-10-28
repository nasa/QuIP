/* This file contains the ObjC interface... */

#ifndef _IOS_GUI_H_
#define _IOS_GUI_H_

#include "quip_config.h"

#ifdef BUILD_FOR_OBJC

#include "quipAppDelegate.h"
#include "quipController.h"
#include "ios_item.h"
#include "screen_obj.h"
#ifdef BUILD_FOR_MACOS
#include <AppKit/NSAlert.h>
#include <AppKit/NSSlider.h>
#include <AppKit/NSScreen.h>
#include <AppKit/NSMenu.h>
#include <AppKit/NSMenuitem.h>
#include <AppKit/NSDocumentController.h>
#include <AppKit/NSOpenPanel.h>
#endif // BUILD_FOR_MACOS

typedef enum {
	QUIP_ALERT_NONE,
	QUIP_ALERT_NORMAL,
	QUIP_ALERT_BUSY,
	QUIP_ALERT_CONFIRMATION,
	N_QUIP_ALERT_TYPES
} Quip_Alert_Type;

#define IS_VALID_ALERT(aip)	((aip.type == QUIP_ALERT_NORMAL) || \
				 (aip.type == QUIP_ALERT_BUSY)   || \
				 (aip.type == QUIP_ALERT_CONFIRMATION) )

@interface Alert_Info : NSObject

#ifdef BUILD_FOR_IOS
@property (retain) QUIP_ALERT_OBJ_TYPE *	the_alert_p;

+(void) rememberAlert:(QUIP_ALERT_OBJ_TYPE *)a withType:(Quip_Alert_Type)t;
+(Alert_Info *) alertInfoFor:(QUIP_ALERT_OBJ_TYPE *)a;

#define ALERT_INFO_OBJ(aip)		(aip).the_alert_p
#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
@property (retain) NSAlert *		alert;
+(void) rememberAlert:(NSAlert *)a withType:(Quip_Alert_Type)t;
+(Alert_Info *) alertInfoFor:(NSAlert *)a;
#define ALERT_INFO_OBJ(aip)		(aip).alert

#endif // BUILD_FOR_MACOS

@property Quip_Alert_Type		type;
@property int				qlevel;

-(void) forget;

@end

// global vars

//extern UIAlertView *fatal_alert_view;

extern void dismiss_quip_alert(QUIP_ALERT_OBJ_TYPE *av,NSInteger buttonIndex);
extern void quip_alert_shown(QUIP_ALERT_OBJ_TYPE *av);

/* ios.m */
extern void _simple_alert(QSP_ARG_DECL  const char *type, const char *msg);
extern void fatal_alert(QSP_ARG_DECL  const char *msg);
extern int check_deferred_alert(SINGLE_QSP_ARG_DECL);

#define simple_alert(type,msg) _simple_alert(QSP_ARG  type,msg)

extern void window_sys_init(SINGLE_QSP_ARG_DECL);	// a nop
extern void give_notice(const char **msg_array);
extern void set_std_cursor(void);
extern void set_busy_cursor(void);

IOS_ITEM_NEW_PROT(Screen_Obj,scrnobj)
IOS_ITEM_PICK_PROT(Screen_Obj,scrnobj)
IOS_ITEM_DEL_PROT(Screen_Obj,scrnobj)

#ifdef BUILD_FOR_IOS
#define SOB_CTL_TYPE	UIView
#endif // BUILD_FOR_IOS
#ifdef BUILD_FOR_MACOS
#define SOB_CTL_TYPE	NSView
#endif // BUILD_FOR_MACOS

extern Screen_Obj *find_scrnobj(SOB_CTL_TYPE *cp);
extern Screen_Obj *find_any_scrnobj(SOB_CTL_TYPE *cp);

#define WIDGET_VERTICAL_GAP 10
#define CONSOLE_DISPLAY_NAME "Console output"

// quipConsole.m
extern void clear_message(void);
extern void display_message(const char *s);
extern void dismiss_console(quipViewController *qvcp);
extern void enable_console_output(int yesno);
extern void init_ios_text_output(void);

// moved to screen_obj.h
//extern void get_device_dims(Screen_Obj *sop);

//extern void ios_warn(QSP_ARG_DECL  const char *msg);
//extern void ios_error(QSP_ARG_DECL  const char *msg);

extern void console_prt_msg_frag(const char *msg);
extern void install_initial_text(Screen_Obj *sop);

#endif /* BUILD_FOR_OBJC */

#endif /* ! _IOS_GUI_H_ */

