//
//  quipImageView.h
//
//  Created by Jeff Mulligan on 3/7/13.
//  Copyright 2013 NASA. All rights reserved.
//

#ifndef _QUIPIMAGEVIEW_H_
#define _QUIPIMAGEVIEW_H_

#include "data_obj.h"

#ifdef BUILD_FOR_IOS
#import <UIKit/UIImageView.h>

@interface quipImageView : UIImageView

#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
@interface quipImageView : NSImageView
#endif // BUILD_FOR_MACOS

@property Data_Obj *	qiv_dp;

-(id) initWithDataObj:(Data_Obj *)dp;

@end

extern QUIP_IMAGE_TYPE *objc_img_for_dp(Data_Obj *dp,int little_endian_flag);

#endif /* ! _QUIPIMAGEVIEW_H_ */

