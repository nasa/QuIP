//
//  quipImage.h
//
//  Created by Jeff Mulligan on 8/9/19.
//  Copyright 2019 NASA. All rights reserved.
//

#ifndef _QUIPIMAGE_H_
#define _QUIPIMAGE_H_

#include "data_obj.h"

#ifdef BUILD_FOR_IOS
#import <UIKit/UIImage.h>

@interface quipImage : UIImage

#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
#include <AppKit/NSImageView.h>
#include <AppKit/NSImage.h>

@interface quipImageView : NSImageView
#endif // BUILD_FOR_MACOS

@property Data_Obj *	qi_dp;

-(id) initWithDataObj:(Data_Obj *)dp;

@end

extern QUIP_IMAGE_TYPE *objc_img_for_dp(Data_Obj *dp,int little_endian_flag);

#endif /* ! _QUIPIMAGE_H_ */

