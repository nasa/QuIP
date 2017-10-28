#include "quip_config.h"

#ifdef BUILD_FOR_OBJC

#include <AVFoundation/AVCaptureDevice.h>
#include <AVFoundation/AVMediaFormat.h>

#include "ios_item.h"

@interface Camera : IOS_Item

@property (retain) AVCaptureDevice	*dev;

+(void) initClass;

@end

#else // ! BUILD_FOR_OBJC

#include "item_type.h"

typedef struct camera {
	Item *	cam_item;
} Camera;

#include "map_ios_item.h"

#endif

IOS_ITEM_INIT_PROT(Camera,camera)
IOS_ITEM_NEW_PROT(Camera,camera)
IOS_ITEM_CHECK_PROT(Camera,camera)
IOS_ITEM_PICK_PROT(Camera,camera)
IOS_ITEM_LIST_PROT(Camera,camera)

#define init_cameras()		_init_cameras(SINGLE_QSP_ARG)
#define new_camera(s)		_new_camera(QSP_ARG  s)
#define camera_of(s)		_camera_of(QSP_ARG  s)
#define pick_camera(p)		_pick_camera(QSP_ARG  p)
#define list_cameras(fp)	_list_cameras(QSP_ARG  fp)

