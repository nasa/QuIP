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

