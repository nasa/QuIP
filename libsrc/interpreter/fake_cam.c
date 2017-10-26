#include "quip_config.h"
#include "quip_prot.h"
#include "dobj_prot.h"
#include "item_type.h"

#include "camera.h"
#include "camera_api.h"
#include "viewer.h"

static IOS_Item_Type *camera_itp;
IOS_ITEM_INIT_FUNC(Camera,camera,0)
IOS_ITEM_NEW_FUNC(Camera,camera)
IOS_ITEM_CHECK_FUNC(Camera,camera)
IOS_ITEM_PICK_FUNC(Camera,camera)
IOS_ITEM_LIST_FUNC(Camera,camera)


static COMMAND_FUNC( do_list_cams )
{
	prt_msg("A/V Capture Devices:");
	_list_cameras(QSP_ARG  tell_msgfile());
}

static COMMAND_FUNC( do_cam_info )
{
	Camera *cam;

	cam = _pick_camera(QSP_ARG  "");
	if( cam == NULL ) return;

	WARN("do_cam_info:  not implemented!?");
}

static COMMAND_FUNC( do_get_cams )
{
	Data_Obj *dp;

	dp = pick_obj("string table");
	if( dp == NULL ) return;

	WARN("do_get_cams:  not implemented!?");
}

static COMMAND_FUNC( do_mon_cam )
{
	Viewer *vp;

	vp = pick_vwr("");
	if( vp == NULL ) return;

	WARN("do_mon_cam:  not implemented!?");
}

static COMMAND_FUNC( do_stop_mon )
{
	WARN("do_stop_mon:  not implemented!?");
}


static COMMAND_FUNC( do_chk_session )
{
	WARN("do_chk_session:  not implemented!?");
}

static COMMAND_FUNC( do_stop_session )
{
	WARN("do_stop_session:  not implemented!?");
}

static COMMAND_FUNC( do_pause_session )
{
	WARN("do_pause_session:  not implemented!?");
}

static COMMAND_FUNC( do_restart_session )
{
	WARN("do_restart_session:  not implemented!?");
}

static COMMAND_FUNC( do_start_session )
{
	Camera *cam;

	cam = _pick_camera(QSP_ARG  "");
	if( cam == NULL ) return;

	WARN("do_start_session:  not implemented!?");
}

static COMMAND_FUNC( do_grab_cam )
{
	Data_Obj *dp;

	dp = pick_obj("target image object");
	if( dp == NULL ) return;

	WARN("do_grab_cam:  not implemented!?");
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(camera_menu,s,f,h)
MENU_BEGIN(camera)
ADD_CMD( list,		do_list_cams,		list available cameras )
ADD_CMD( info,		do_cam_info,		print info about a camera )
ADD_CMD( monitor,	do_mon_cam,		monitor capture )
ADD_CMD( stop_mon,	do_stop_mon,		stop monitoring capture )
ADD_CMD( grab,		do_grab_cam,		grab a frame )
ADD_CMD( get_cameras,	do_get_cams,		copy camera names to an array )
ADD_CMD( check,		do_chk_session,		check session state )
ADD_CMD( start,		do_start_session,	start capture )
ADD_CMD( pause,		do_pause_session,	pause capture )
ADD_CMD( restart,	do_restart_session,	restart capture )
ADD_CMD( stop,		do_stop_session,	stop capture )
MENU_END(camera)

COMMAND_FUNC(do_cam_menu)
{
	static int inited=0;

	if( ! inited ){
		set_script_var_from_int(QSP_ARG  "n_cameras",0);
		inited=1;
	}

	PUSH_MENU(camera);
}

