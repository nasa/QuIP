#ifndef _QUIPWINDOWCONTROLLER_H_
#define _QUIPWINDOWCONTROLLER_H_

#include "quipImages.h"
#include "quipCanvas.h"

@interface quipWindowController: NSWindowController <NSWindowDelegate>

// these were taken from quipView...
@property (nonatomic, retain) quipImages *	images;		// for movies
@property (nonatomic, retain) quipCanvas *	canvas;		// for drawing

-(void) windowButtonAction:(id) obj;
-(void) windowSliderAction:(id) obj;
-(void) windowChooserAction:(id) obj;

#define QWC_IMAGES(qwc)		(qwc).images

@end

#endif // _QUIPWINDOWCONTROLLER_H_

