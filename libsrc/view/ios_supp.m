/* IOS support for viewers and images */

// Is this file used for MacOS also???

#include "quip_config.h"

#include "view_prot.h"
#include "view_cmds.h"
#include "linear.h"
#include "quipView.h"
//#include "quipImageView.h"
#include "quipCanvas.h"
#include "quipViewController.h"
#include "quipAppDelegate.h"
#include "panel_obj.h"

#ifdef BUILD_FOR_IOS
#include <UIKit/UIStringDrawing.h>
#include <UIKit/UIScreen.h>
#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
#include <AppKit/NSColor.h>
#include <AppKit/NSGraphicsContext.h>
#endif // BUILD_FOR_MACOS

typedef enum {
	DO_UNUSED,	// don't use 0
	DO_MOVE,
	DO_CONT,
	DO_TEXT,
	DO_SET_BG,
	DO_SELECT_PEN,
	DO_ERASE,
	DO_FONT,
	DO_FONT_SIZE,
	DO_CHAR_SPACING,
	DO_RJUST,
	DO_LJUST,
	DO_CJUST,
	DO_TEXT_ANGLE,
	DO_LINEWIDTH,
	N_DRAWOP_CODES
} Draw_Op_Code;

typedef struct pt_arg {
	CGFloat x;
	CGFloat y;
} Point_Arg;

@interface Draw_Op :NSObject
@property Draw_Op_Code		code;
@property (retain) QUIP_COLOR_TYPE *	color;
@property Point_Arg		point;
@property const char *		string;
@property int			drawopIntArg;
@property float			drawopFloatArg;

-(void) set_x:(float)x;
-(void) set_y:(float)y;
@end

@implementation Draw_Op
@synthesize code;
@synthesize color;
@synthesize point;
@synthesize string;
@synthesize drawopIntArg;
@synthesize drawopFloatArg;

-(void) set_x:(float)v
{
	point.x = v;
}

-(void) set_y:(float)v
{
	point.y = v;
}

-(id) init
{
	self=[super init];

	color=NULL;
	point.x = point.y = 0.0;
	string=NULL;
	return self;
}
@end

/* All drawing has to be done from the drawRect method (apparently?),
 *
 * So we implement the drawing commands by buffering to a display list,
 * then we call a function to execute the display list from drawRect.
 */


#define DOA_STR(do_p)			(do_p).string
#define DOA_X(do_p)			(do_p).point.x
#define DOA_Y(do_p)			(do_p).point.y
#define DOA_FONT_SIZE(do_p)		(do_p).drawopIntArg
#define DOA_CHAR_SPACING(do_p)		(do_p).drawopIntArg
#define DOA_LINEWIDTH(do_p)		(do_p).drawopFloatArg
#define DOA_ANGLE(do_p)			(do_p).drawopFloatArg
#define DOA_COLOR(do_p)			(do_p).color

#define SET_DOA_STR(do_p,v)		[do_p setString:v]
#define SET_DOA_X(do_p,v)		[do_p set_x:v]
#define SET_DOA_Y(do_p,v)		[do_p set_y:v]
#define SET_DOA_FONT_SIZE(do_p,v)	[do_p setDrawopIntArg:v]
#define SET_DOA_CHAR_SPACING(do_p,v)	[do_p setDrawopIntArg:v]
#define SET_DOA_LINEWIDTH(do_p,v)	[do_p setDrawopFloatArg:v]
#define SET_DOA_ANGLE(do_p,v)		[do_p setDrawopFloatArg:v]
#define SET_DOA_COLOR(do_p,v)		[do_p setColor:v]


/*
//fprintf(stderr,"ADD_DRAW_OP adding node to draw list 0x%lx for viewer %s\n",(long)VW_DRAW_LIST(vp),VW_NAME(vp));\
 */


// This craps out after more than around 10 swipes in the
// csf app...
//#define MAX_DRAWLIST_LEN	4096
// 0x100 256
// 0x400 1024
// 0x1000 4096
#define MAX_DRAWLIST_LEN	0x40000	// 256k

#ifdef BUILD_FOR_IOS

#ifdef CAUTIOUS
#define ADD_DRAW_OP(vp,do_p)					\
	if( VW_DRAW_LIST(vp) == NULL ){			\
		SET_VW_DRAW_LIST(vp,new_ios_list());		\
	}							\
	if( ios_eltcount(VW_DRAW_LIST(vp)) > MAX_DRAWLIST_LEN ){\
		static int warned=0;				\
		if( !warned ){					\
			warn("Too many stored draw ops!?");	\
			warned=1;				\
		}						\
	} else {						\
		IOS_Node *np = mk_ios_node(do_p);		\
		ios_addTail(VW_DRAW_LIST(vp),np);		\
	}

#else /* ! CAUTIOUS */

#define ADD_DRAW_OP(vp,do_p)					\
	if( VW_DRAW_LIST(vp) == NULL ){			\
		SET_VW_DRAW_LIST(vp,new_ios_list());		\
	}							\
	IOS_Node *np = mk_ios_node(do_p);			\
	ios_addTail(VW_DRAW_LIST(vp),np);

#endif /* ! CAUTIOUS */
#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
#define ADD_DRAW_OP(vp,do_p)	exec_drawop(vp,do_p)
#endif // BUILD_FOR_MACOS


#define CHECK_COLOR_INDEX(funcname,color)		\
	if( color > 255 ){				\
		sprintf(ERROR_STRING,		\
"%s:  color (%ld) must be in the range 0-255",		\
			#funcname,color);		\
		warn(ERROR_STRING);		\
		return;					\
	}

#define DEFAULT_FONT_SIZE	15
#define DEFAULT_CHAR_SPACING	3


// BUG should be per-qsp
// Or per-viewer??
static int font_size=DEFAULT_FONT_SIZE;
// char spacing is expressed in pixels, but probably should
// be a fraction of the font size???  BUG
static int char_spacing=DEFAULT_CHAR_SPACING;
// BUG the text transform should be a viewer property...
static CGAffineTransform myTextTransform; // 2

static CGAffineTransform myScaleTransform;
static CGAffineTransform myRotTransform;


#ifdef NOT_USED
// For debugging
static void dump_matrix( CGAffineTransform xf )
{
fprintf(stderr,"%g %g %g\n",xf.a,xf.b,xf.tx);
fprintf(stderr,"%g %g %g\n",xf.c,xf.d,xf.ty);
}
#endif // NOT_USED

void init_gw_lut(Gen_Win *gwp)
{
	SET_GW_CMAP(gwp, [[NSMutableArray alloc] init] );
	/* initialize 256 entries */
	int i;
	QUIP_COLOR_TYPE *c;
	for(i=0;i<256 /* BUG use symbolic constant */;i++){
		float l = (float)(0.5 + i * (0.5/256.0));
#ifdef BUILD_FOR_IOS
		c = [[QUIP_COLOR_TYPE alloc]
			initWithRed:	l
			green:		l
			blue:		l
			alpha:		1.0];
#endif // BUILD_FOR_IOS
#ifdef BUILD_FOR_MACOS
		c = [NSColor colorWithSRGBRed:l  green:l blue:l alpha:1.0 ];
#endif // BUILD_FOR_MACOS

//printf("Inserting color at 0x%lx at table index %d\n",(long)c,i);
		[GW_CMAP(gwp) insertObject: c atIndex:i];
	}
}

// BUG shouldn't be a function, maybe compiler will inline...

static QUIP_COLOR_TYPE *get_color_from_index(Viewer *vp,int idx)
{
	QUIP_COLOR_TYPE *c;

	INSURE_GW_CMAP(VW_GW(vp))
	c=[GW_CMAP(VW_GW(vp)) objectAtIndex: idx];

//	fprintf(stderr,"get_color_from_index %s %d:  0x%lx (cmap = 0x%lx)\n",
//			VW_NAME(vp),idx,(long)c,(long)VW_CMAP(vp));

	return c;
}

static Draw_Op* new_drawop(Draw_Op_Code c)
{
	Draw_Op *do_p;

	do_p = [[Draw_Op alloc] init];
	do_p.code = c;
	return(do_p);
}

static void report_drawop(Viewer *vp, Draw_Op *do_p)
{
	switch( do_p.code ){
		case DO_CONT:
			fprintf(stderr,"DO_CONT %s  %g %g\n",VW_NAME(vp),DOA_X(do_p),DOA_Y(do_p));
			break;
		case DO_LINEWIDTH:
			fprintf(stderr,"DO_LINEWIDTH %s  %g\n",VW_NAME(vp),DOA_LINEWIDTH(do_p));
			break;
		case DO_MOVE:
			fprintf(stderr,"DO_MOVE %s  %g %g\n",VW_NAME(vp),DOA_X(do_p),DOA_Y(do_p));
			break;
		case DO_FONT_SIZE:
			fprintf(stderr,"DO_FONT_SIZE %s  %d\n",VW_NAME(vp),DOA_FONT_SIZE(do_p));
			break;
		case DO_CHAR_SPACING:
			fprintf(stderr,"DO_CHAR_SPACING %s  %d\n",VW_NAME(vp),DOA_CHAR_SPACING(do_p));
			break;
		case DO_TEXT:
			fprintf(stderr,"DO_TEXT %s  \"%s\"\n",VW_NAME(vp),DOA_STR(do_p));
			break;
		case DO_FONT:
			fprintf(stderr,"DO_FONT %s  \"%s\"\n",VW_NAME(vp),DOA_STR(do_p));
			break;
		case DO_SELECT_PEN:
			fprintf(stderr,"DO_SELECT_PEN %s  0x%lx\n",VW_NAME(vp),(long)DOA_COLOR(do_p));
			break;
		case DO_ERASE:
			fprintf(stderr,"DO_ERASE %s\n",VW_NAME(vp));
			break;
		case DO_LJUST:
			fprintf(stderr,"DO_LJUST %s\n",VW_NAME(vp));
			break;
		case DO_CJUST:
			fprintf(stderr,"DO_CJUST %s\n",VW_NAME(vp));
			break;
		case DO_RJUST:
			fprintf(stderr,"DO_RJUST %s\n",VW_NAME(vp));
			break;
		case DO_TEXT_ANGLE:
			fprintf(stderr,"DO_TEXT_ANGLE %s %g\n",VW_NAME(vp),DOA_ANGLE(do_p));
			break;
		case DO_SET_BG:
			fprintf(stderr,"DO_SET_BG %s\n",VW_NAME(vp));
			break;
		case DO_UNUSED:
			fprintf(stderr,"DO_UNUSED %s  shouldn't happen!?\n",VW_NAME(vp));
NWARN("bad drawop #1!?");
			break;
		case N_DRAWOP_CODES:
			fprintf(stderr,"N_DRAWOP_CODES %s  shouldn't happen!?\n",VW_NAME(vp));
NWARN("bad drawop #2!?");
			break;
		default:
			fprintf(stderr,"report_drawop:  OOPS - unhandled code %d (0x%x)\n",
				do_p.code, do_p.code);
NWARN("bad drawop #3!?");
			break;
	}
//fprintf(stderr,"\t\ttext_mode = 0x%x\n",VW_FLAGS(vp)&VW_JUSTIFY_MASK);
}

#ifdef MAX_DEBUG
#define REPORT_DRAWOP(vp,do_p)	report_drawop(vp,do_p);
#else // ! MAX_DEBUG
#define REPORT_DRAWOP(vp,do_p)
#endif // ! MAX_DEBUG

static CGPoint get_string_offset(CGContextRef ctx, const char *str)
{
	//CGTextDrawingMode mode = CGContextGetTextDrawingMode(ctx);
//fprintf(stderr,"get_string_offset \"%s\"\n",str);
	CGContextSaveGState(ctx);
	// needed?
	//CGContextSetCharacterSpacing( ctx, char_spacing);
	CGContextSetTextDrawingMode(ctx, kCGTextInvisible);
	CGContextShowTextAtPoint(ctx, 0, 0, str, strlen(str));
	CGPoint pt = CGContextGetTextPosition(ctx);
	//CGContextSetTextDrawingMode(ctx, mode);
	CGContextRestoreGState(ctx);
//fprintf(stderr,"get_string_offset '%s' (%zd):  offset is %g %g\n",
//str,strlen(str),pt.x,pt.y);
	return pt;
}

static void init_text_font(Viewer *vp)
{
#ifdef CAUTIOUS
	if( VW_GFX_CTX(vp) == NULL ){
		warn("CAUTIOUS:  init_text_font:  viewer has null context!?");
		return;
	}
#endif /* CAUTIOUS */

// New drawing code using Core Text - where to put???
// [[UIColor blackColor] setFill]; // This is the default
// [@"Some text" drawAtPoint:CGPointMake(barX, barY)
//	withAttributes:
//		@{NSFontAttributeName:[UIFont fontWithName:@"Helvetica"
//		size:kBarLabelSize] }];

//fprintf(stderr,"Selecting Helvetica-Bold...\n");
	CGContextSelectFont (VW_GFX_CTX(vp),
		"Helvetica-Bold", font_size, kCGEncodingMacRoman);
	CGContextSetCharacterSpacing (VW_GFX_CTX(vp), char_spacing);
	CGContextSetTextDrawingMode (VW_GFX_CTX(vp), kCGTextFillStroke);

	if( ! VW_TXT_MTRX_READY(vp) ){
		// BUG global should be per-viewer?
		// or does system copy it?
//fprintf(stderr,"Initializing text transform...\n");
#ifdef BUILD_FOR_IOS
//advise("init_text_font:  making scale transform...");
		myTextTransform =  CGAffineTransformMakeScale( 1.0, -1.0  );
//fprintf(stderr,"transform at 0x%lx\n",(long)myTextTransform);
#else // ! BUILD_FOR_IOS
		myTextTransform =  CGAffineTransformMakeScale( 1.0, 1.0  );
#endif // ! BUILD_FOR_IOS
		SET_VW_FLAG_BITS(vp,VW_TXT_MTRX_INITED);
	}
//fprintf(stderr,"installing myTextTransform\n");
//dump_matrix(myTextTransform);
	CGContextSetTextMatrix (VW_GFX_CTX(vp), myTextTransform);
}

#define exec_drawop(vp,do_p) _exec_drawop(QSP_ARG  vp,do_p)

static int _exec_drawop(QSP_ARG_DECL  Viewer *vp, Draw_Op *do_p)
{
	QUIP_COLOR_TYPE *c;
	static CGFloat x=0.0;
	static CGFloat y=0.0;
	CGRect rect;

	REPORT_DRAWOP(vp,do_p)

#ifdef CAUTIOUS
	if( do_p == NULL ) NERROR1("CAUTIOUS:  exec_drawop:  null operation ptr!?");

	if( VW_GFX_CTX(vp) == NULL ){
		warn("CAUTIOUS:  exec_drawop:  null context!?");
		return -1;
	}
#endif /* CAUTIOUS */
	switch( do_p.code ){
		case DO_UNUSED:
			warn("invalid zero Draw_Op_Code!?");
			return 0;
			break;
		case DO_MOVE:
			CGContextBeginPath(VW_GFX_CTX(vp));
			CGContextMoveToPoint(VW_GFX_CTX(vp),(int)(x=DOA_X(do_p)),(int)(y=DOA_Y(do_p)));

			break;

		case DO_FONT:
//fprintf(stderr,"exec_drawop calling CGContextSelectFont %s\n",
//DOA_STR(do_p));
// BUG - CGContextSelectFont fails silently if the font name is bad...

			CGContextSelectFont (VW_GFX_CTX(vp), DOA_STR(do_p),font_size,kCGEncodingMacRoman);
			break;

		case DO_CHAR_SPACING:
			char_spacing = DOA_CHAR_SPACING(do_p);
//fprintf(stderr,"char_spacing reset to %d\n",char_spacing);
			break;

		case DO_FONT_SIZE:
			font_size = DOA_FONT_SIZE(do_p);
			break;


		case DO_LJUST:
			CLEAR_VW_FLAG_BITS(vp,VW_JUSTIFY_MASK);
			SET_VW_FLAG_BITS(vp,VW_LEFT_JUSTIFY);
			break;

		case DO_RJUST:
			CLEAR_VW_FLAG_BITS(vp,VW_JUSTIFY_MASK);
			SET_VW_FLAG_BITS(vp,VW_RIGHT_JUSTIFY);
			break;

		case DO_CJUST:
			CLEAR_VW_FLAG_BITS(vp,VW_JUSTIFY_MASK);
			SET_VW_FLAG_BITS(vp,VW_CENTER_TEXT);
			break;

		case DO_TEXT_ANGLE:
#ifdef BUILD_FOR_IOS
//fprintf(stderr,"DO_TEXT_ANGLE %f making scale transform...\n",DOA_ANGLE(do_p));
			myScaleTransform =  CGAffineTransformMakeScale( 1.0, -1.0  );
#else // ! BUILD_FOR_IOS
			myScaleTransform =  CGAffineTransformMakeScale( 1.0, 1.0  );
#endif // ! BUILD_FOR_IOS
			myRotTransform =  CGAffineTransformMakeRotation(
				DOA_ANGLE(do_p) );
			myTextTransform = CGAffineTransformConcat(
				myRotTransform,myScaleTransform);
			SET_VW_FLAG_BITS(vp,VW_TXT_MTRX_INITED);
//fprintf(stderr,"installing transform at 0x%lx for rotated text\n",(long)myTextTransform);
//fprintf(stderr,"DO_TEXT_ANGLE:  installing myTextTransform\n");
			CGContextSetTextMatrix (VW_GFX_CTX(vp), myTextTransform);
//dump_matrix(myTextTransform);
			break;
		case DO_TEXT:
			// BUG do this differently...
			// If we make this conditional (only do it
			// the first time if the flag is not set,
			// then nothing seems to display!?

	// no other text draw modes?

			// Make sure a default justification mode is set
			if( (VW_FLAGS(vp) & VW_JUSTIFY_MASK) == 0 ){
				SET_VW_FLAG_BITS(vp,VW_LEFT_JUSTIFY);
			}

			if( VW_TEXT_LJUST(vp) ){
//fprintf(stderr,"drawing left-justified text \"%s\"\n",DOA_STR(do_p));
				CGContextShowTextAtPoint (VW_GFX_CTX(vp),
					x, y, DOA_STR(do_p), strlen(DOA_STR(do_p)) );
			} else if( VW_TEXT_CENTER(vp) ){
//fprintf(stderr,"drawing centered text \"%s\"\n",DOA_STR(do_p));
				CGPoint pt = get_string_offset(VW_GFX_CTX(vp),DOA_STR(do_p));
#define OLD_TEXT_METHOD

#ifdef OLD_TEXT_METHOD
//fprintf(stderr,"x = %g   y = %g\n",x,y);
//fprintf(stderr,"string offset = %g %g\n",pt.x,pt.y);
				CGContextShowTextAtPoint (VW_GFX_CTX(vp),
					x-pt.x/2, y-pt.y/2,
					DOA_STR(do_p), strlen(DOA_STR(do_p)) );


#else // ! OLD_TEXT_METHOD
/* iOS 7 only:
[@(DOA_STR(do_p)) drawAtPoint:CGPointMake(x-pt.x/2, y-pt.y/2)
	withAttributes:
		@{NSFontAttributeName:[UIFont fontWithName:@"Helvetica"
		size:20] }];
*/

/* This doesn't seem to use the CG transformations??? */
CGSize drawn_size =
[@(DOA_STR(do_p)) drawAtPoint:CGPointMake(x-pt.x/2, y-pt.y/2)
	withFont: [UIFont fontWithName:@"Helvetica" size:20] ];

#endif // ! OLD_TEXT_METHOD
			} else if( VW_TEXT_RJUST(vp) ){
//fprintf(stderr,"drawing right-justified text \"%s\"\n",DOA_STR(do_p));
				CGPoint pt = get_string_offset(VW_GFX_CTX(vp),DOA_STR(do_p));
				CGContextShowTextAtPoint (VW_GFX_CTX(vp),
					x-pt.x, y-pt.y, DOA_STR(do_p), strlen(DOA_STR(do_p)) );
			} else {
				sprintf(ERROR_STRING,"Unexpected text justification mode 0x%x!?",VW_FLAGS(vp)&VW_JUSTIFY_MASK);
				warn(ERROR_STRING);
			}
			break;

		case DO_CONT:
			CGContextAddLineToPoint(VW_GFX_CTX(vp),x=DOA_X(do_p),y=DOA_Y(do_p));
			CGContextStrokePath(VW_GFX_CTX(vp));
			break;
		case DO_LINEWIDTH:
			// BUG - Apple specifies this in points, not pixels,
			// and uses a different scale factor on "retina" displays...
			CGContextSetLineWidth(VW_GFX_CTX(vp),DOA_LINEWIDTH(do_p));
			break;
		case DO_SET_BG:
#ifdef BUILD_FOR_IOS
			[VW_QV(vp) setBackgroundColor: DOA_COLOR(do_p)];
#endif // BUILD_FOR_IOS
			break;
		case DO_SELECT_PEN:
			c =  DOA_COLOR(do_p) ;
			CGContextSetStrokeColorWithColor( VW_GFX_CTX(vp), c.CGColor );
			CGContextSetFillColorWithColor( VW_GFX_CTX(vp), c.CGColor );
			break;

		case DO_ERASE:
//fprintf(stderr,"DO_ERASE:  clearing %s (w = %d, h = %d)\n",VW_NAME(vp),VW_WIDTH(vp),VW_HEIGHT(vp));

			rect = CGRectMake(0,0,VW_WIDTH(vp),VW_HEIGHT(vp));
			// what about CGContextClearRect???
			CGContextClearRect( VW_GFX_CTX(vp), rect );
			// ClearRect doesn't seem to work, try fill rect...

			//c = [UIColor clearColor]; // BUG - user should select!

			//CGContextSetAlpha( VW_GFX_CTX(vp), 0.0 );	// fully transparent
			/*
			CGContextSetAlpha( VW_GFX_CTX(vp), 0.1 );	// half transparent
			c = [UIColor whiteColor]; // BUG - user should select!
			CGContextSetFillColorWithColor( VW_GFX_CTX(vp), c.CGColor );
			CGContextFillRect( VW_GFX_CTX(vp), rect );
			CGContextSetAlpha( VW_GFX_CTX(vp), 1.0 );	// fully opaque

// this does nothing!?
//			c = [UIColor clearColor];
//			CGContextSetFillColorWithColor( VW_GFX_CTX(vp), c.CGColor );
//			CGContextFillRect( VW_GFX_CTX(vp), rect );

			// BUG drawing color is not reset to user selection...?
			c = [UIColor blackColor]; // BUG - user should be cached!
			CGContextSetFillColorWithColor( VW_GFX_CTX(vp), c.CGColor );
			*/


			break;
		default:
			warn("Unexpected code in exec_drawop!?");
			break;
	}
	return 0;
}

// BUG it would be a lot more efficient to do this when the erase command is issued!?

static IOS_Node *find_erase_node(Viewer *vp)
{
	IOS_Node *np;
	Draw_Op *do_p;

	np = IOS_LIST_HEAD( VW_DRAW_LIST(vp) );
	while( np != NULL ){
		do_p = IOS_NODE_DATA(np);
		if( do_p.code == DO_ERASE ){
			return np;
		}
		np = IOS_NODE_NEXT(np);
	}
	return NULL;
}

static int is_drawing_node( IOS_Node *np )
{
	Draw_Op *do_p;

	do_p = (Draw_Op *) IOS_NODE_DATA(np);

	switch( do_p.code ){
		case DO_CONT:
		case DO_TEXT:
		case DO_ERASE:
			return 1;
			break;

		case DO_LINEWIDTH:
		case DO_MOVE:
		case DO_FONT_SIZE:
		case DO_CHAR_SPACING:
		case DO_FONT:
		case DO_SELECT_PEN:
		case DO_LJUST:
		case DO_CJUST:
		case DO_RJUST:
		case DO_TEXT_ANGLE:
		case DO_SET_BG:
			return 0;
			break;

		case DO_UNUSED:
            fprintf(stderr,"is_drawing_node:  DO_UNUSED shouldn't occur!?\n");
			break;
		case N_DRAWOP_CODES:
            fprintf(stderr,"is_drawing_node:  N_DRAWOP_CODES shouldn't occur!?\n");
			break;
		default:
			fprintf(stderr,"report_drawop:  OOPS - unhandled code %d (0x%x)\n",
				do_p.code, do_p.code);
			break;
	}
	return 1;
}

static void erase_drawlist_to_node( Viewer *vp, IOS_Node *erase_np)
{
	IOS_Node *np, *next;

	np = IOS_LIST_HEAD( VW_DRAW_LIST(vp) );

	do {
		next = IOS_NODE_NEXT(np);
		if( np == erase_np ){
			ios_remNode(VW_DRAW_LIST(vp),np);
			return;
		} else if( is_drawing_node(np) ){
			ios_remNode(VW_DRAW_LIST(vp),np);
		}
		np = next;
	} while( np != NULL );

	NERROR1("erase_drawlist_to_node:  CAUTIOUS:  should never happen!?");
}

static void scan_drawlist_for_erasures(Viewer *vp)
{
	IOS_Node *np;
	int found_erasure=0;

	if( VW_DRAW_LIST(vp) == NULL ) return;

	do {
		np = find_erase_node(vp);
		if( np != NULL ){
			erase_drawlist_to_node(vp,np);
			found_erasure=1;
		}
	} while( np != NULL );

	if( found_erasure ){
		Draw_Op *do_p;
		// put an erase node at the beginning of what's left
		do_p = new_drawop(DO_ERASE);
		np = mk_ios_node(do_p);
		ios_addHead(VW_DRAW_LIST(vp),np);
	}
}

int _exec_drawlist(QSP_ARG_DECL  Viewer *vp)
{
	IOS_Node *np;
	int retval=0;

//fprintf(stderr,"exec_drawlist BEGIN\n");
//dump_drawlist(vp);

	scan_drawlist_for_erasures(vp);

//fprintf(stderr,"exec_drawlist: after scanning:\n");
//dump_drawlist(vp);

#ifdef BUILD_FOR_IOS
	if( VW_GFX_CTX(vp) != UIGraphicsGetCurrentContext() )
		warn("viewer context does not match UIGraphicsGetCurrentContext !?");
#endif // BUILD_FOR_IOS
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextSetStrokeColorSpace( VW_GFX_CTX(vp), colorSpace );
	CGColorSpaceRelease(colorSpace);

	// We do this in exec_draw_op too!?!?
	//if( ! VW_TXT_MTRX_READY(vp) ){
		init_text_font(vp);
	//}

	if( VW_DRAW_LIST(vp) == NULL ) {
		fprintf(stderr,"exec_drawlist %s:  null draw list!?\n",VW_NAME(vp));
		return 0;
	}
	np = IOS_LIST_HEAD(VW_DRAW_LIST(vp));

	while(np!=NULL){
		Draw_Op *do_p;
		do_p = (Draw_Op *) IOS_NODE_DATA(np);
		if( exec_drawop(vp,do_p) < 0 ){	// error
fprintf(stderr,"exec_drawlist returning after error\n");
			return retval;
		} else {
			np = IOS_NODE_NEXT(np);
		}
	}
//fprintf(stderr,"exec_drawlist %s:  DONE\n",VW_NAME(vp));
	return retval;
}

// When is loadView called?


// A "viewer" is just another view that we are going to load with images...

int _make_viewer(QSP_ARG_DECL  Viewer *vp,int width,int height)
{
	// For now we disregard what size the user has requested...
	// All viewers are full-screen

	// Do we already have a Gen_Win (from a panel)?
	Gen_Win *gwp=genwin_of( QSP_ARG  VW_NAME(vp) );
	if( gwp == NULL ){
//fprintf(stderr,"make_viewer calling make_genwin %s\n",VW_NAME(vp));
		gwp=make_genwin(VW_NAME(vp), width, height );
	}

	// BUG?  reference cycle?
	SET_VW_GW(vp,gwp);
	SET_GW_VW(gwp,vp);

	return 0;

}	// end make_viewer

void show_viewer(QSP_ARG_DECL  Viewer *vp)
{
	// in iOS, viewers can also function as panels...
	// Showing a panel is more complicated than showing
	// a viewer, because we have to push the name context
	// for the widgets.
	push_nav(VW_GW(vp));
}

void unshow_viewer(QSP_ARG_DECL  Viewer *vp)
{
	pop_nav(1);
}

int is_image_viewer(QSP_ARG_DECL  Viewer *vp)
{
#ifdef BUILD_FOR_IOS
	switch( GW_VC_TYPE(VW_GW(vp)) ){
		case GW_VC_QVC:		return 1;
		case GW_VC_QTVC:	return 0;
		default:
			warn("Unhandled view controller case in is_image_viewer!?");
			return 0;
	}
#else // ! BUILD_FOR_IOS
	warn("is_image_viewer:  need to implement!?");
	return 1;
#endif // ! BUILD_FOR_IOS
}

/* we want little-endian for images we synthesize on the iPad...
 * but when we write and then read a png file, the bytes are swapped!?
 * What a mess...
 */

static QUIP_IMAGE_TYPE *objc_img_for_dp(Data_Obj *dp, int little_endian_flag)
{
	QUIP_IMAGE_TYPE *theImage;
	CGImageRef myimg;
	CGColorSpaceRef colorSpace;

	if( OBJ_PREC(dp) != PREC_UBY ){
		sprintf(DEFAULT_ERROR_STRING,
			"cgimg_for_dp:  object %s (%s) must be u_byte",
			OBJ_NAME(dp),OBJ_PREC_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
		return NULL;
	}
	if( OBJ_COMPS(dp) != 4 ){
		sprintf(DEFAULT_ERROR_STRING,
			"cgimg_for_dp:  object %s (%d) must have 4 components",
			OBJ_NAME(dp),OBJ_COMPS(dp));
		NWARN(DEFAULT_ERROR_STRING);
		return NULL;
	}

	colorSpace = CGColorSpaceCreateDeviceRGB();
//fprintf(stderr,"objc_img_for_dp:  little_endian_flag = %d\n",little_endian_flag);
	CGContextRef cref = CGBitmapContextCreateWithData(OBJ_DATA_PTR(dp),
		OBJ_COLS(dp), OBJ_ROWS(dp), 8, 4* OBJ_COLS(dp) ,
		colorSpace,
		(
#ifdef BUILD_FOR_IOS
		(little_endian_flag==0?0:kCGBitmapByteOrder32Little) |
#endif // BUILD_FOR_IOS
		kCGImageAlphaPremultipliedFirst
		/*kCGImageAlphaPremultipliedLast*/
		/*kCGImageAlphaFirst*/	// not compatible with other flags???
		/*kCGImageAlphaLast*/	// docs say have to be premultiplied???
		),

		NULL,		// release callback
		NULL			// release callback data arg
		);
	CGColorSpaceRelease(colorSpace);


	if ( cref == NULL ){
		printf("error creating bitmap context!?\n");
		return NULL;
	}

	myimg = CGBitmapContextCreateImage(cref);
	CGContextRelease(cref);

#ifdef BUILD_FOR_IOS
	theImage = [QUIP_IMAGE_TYPE imageWithCGImage:myimg];
#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
	theImage = [[QUIP_IMAGE_TYPE alloc]
		initWithCGImage:myimg
		size:NSZeroSize ];
#endif // BUILD_FOR_MACOS
    
	// image is retained by the property setting above,
	// so we can release the original
	// jbm:  DO WE NEED THAT WITH ARC???
	
	// static analyzer complains, so we remove for now...
	// but that broke things!
	// It appears that CG stuff needs releases, even with ARC...
	CGImageRelease(myimg);

	return theImage;

}	// objc_img_for_dp


/*
 * We create a view the first time we display an image, and cache a ptr
 * in the data_obj structure.
 * We set the view size to match the viewer, though, so there is a potential
 * problem if we later display the image in a differently-sized viewer!?
 */

static quipImages * insure_viewer_images(Viewer *vp)
{
	quipImages *qip;
	CGSize size;

#ifdef BUILD_FOR_IOS
	if( VW_IMAGES(vp) != NULL ){
		return VW_IMAGES(vp);
	}
#endif // BUILD_FOR_IOS

	size.width = VW_WIDTH(vp);
	size.height = VW_HEIGHT(vp);

	qip=[[quipImages alloc]initWithSize:size];

#ifdef BUILD_FOR_IOS
	SET_QV_IMAGES(VW_QV(vp),qip);

	[VW_QV(vp) addSubview:qip];


	// We want the canvas to be in front of the images,
	// but behind the controls...
	// BUT this brings the images to the front?
	// Does this depend on order of initialization?

	// That comment says we want the controls in front - so why
	// are we bringing the images to the front???

	// this puts it just in front of the default background?
	[VW_QV(vp) insertSubview:qip aboveSubview:QV_BG_IMG(VW_QV(vp))];

	qip.backgroundColor = [UIColor clearColor];
#endif // BUILD_FOR_IOS

	return qip;
}

#ifdef FOOBAR
static quipImageView * insure_viewer_has_image( Viewer *vp, Data_Obj *dp, int x, int y )
{
	quipImageView *qiv_p;
	CGRect rect;

	rect = CGRectMake(x,y,VW_WIDTH(vp),VW_HEIGHT(vp));

	qiv_p = img_view_for_obj(dp,rect);


	// We add the new imageView as a subview of the viewer view...
	if( VW_IMAGES(vp) == NULL ){
		insure_viewer_images(vp);
	}

	if( ! [VW_IMAGES(vp) hasSubview:qiv_p] ){
		// addSubview does nothing if the view is already a subview,
		// so the above check doesn't really do much...
		// are new subviews added at the back or front??

fprintf(stderr,"insure_viewer_has_image:  %s has %d subviews before adding %s\n",
VW_NAME(vp),[VW_IMAGES(vp) subviewCount],OBJ_NAME(dp));

		[VW_IMAGES(vp) addSubview:qiv_p];
	}
	return qiv_p;
}
#endif // FOOBAR

static UIImage * insure_object_has_uiimage(Data_Obj *dp)
{
	UIImage *uii_p;
fprintf(stderr,"insure_object_has_uiimage:  checking %s\n",OBJ_NAME(dp));
	if( OBJ_UI_IMG(dp) == NULL ){
        fprintf(stderr,"insure_object_has_uiimage:  creating UIImage for %s\n",OBJ_NAME(dp));
		uii_p = objc_img_for_dp( dp, 1 /* little_endian flag */ );

		// Save a reference to the image view in the data obj
		SET_OBJ_UI_IMG(dp,uii_p);
        fprintf(stderr,"insure_object_has_uiimage:  set to 0x%lx\n",(long)uii_p);
	} else {
        fprintf(stderr,"insure_object_has_uiimage:  returning existing UIImage (0x%lx) for %s\n",(long)OBJ_UI_IMG(dp),OBJ_NAME(dp));
		uii_p = OBJ_UI_IMG(dp);
	}
	return uii_p;
}

/*
 * OLD COMMENT:
 * We would like to be able to bring an image to the front,
 * but in a script we only have the data object name.
 * We solve this by subclassing UIImageView, and adding
 * a Data_Obj reference.
 *
 * NEW COMMENT:
 * We turned this around, and give the data_obj's a pointer to
 * an associated imageView - but the imageView holds a copy of the
 * image at the time it was created, and won't be updated!?
 * We would like to repoint the data object to the view's data???
 * Or create a data object for a view???  FIXME LATER
 */

void embed_image(QSP_ARG_DECL Viewer *vp, Data_Obj *dp,int x,int y)
{
	//quipImageView *qiv_p;
	UIImage *uii_p;		// BUG really only for iOS...

	INSIST_IMAGE_VIEWER(embed_image)

	////qiv_p = insure_viewer_has_image(vp,dp,x,y);
	uii_p = insure_object_has_uiimage(dp);

#ifdef BUILD_FOR_IOS
	[VW_IMAGES(vp) setImage:uii_p];
#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
	[GW_WINDOW(VW_GW(vp)) setContentView: qiv_p ];
#endif // BUILD_FOR_MACOS
}

void _queue_frame( QSP_ARG_DECL  Viewer *vp, Data_Obj *dp )
{
	UIImage *uii_p;
	quipImages *qi_p;

	uii_p = insure_object_has_uiimage(dp);
	assert(uii_p!=NULL);

	qi_p = insure_viewer_images(vp);
	assert(qi_p!=NULL);

	[qi_p queueFrame:uii_p];
}

void _clear_queue( QSP_ARG_DECL  Viewer *vp )
{
	quipImages *qi_p;

	qi_p = VW_IMAGES(vp);
	if(qi_p!=NULL){
		// quipImages may not be created if we have never
		// queued any frames before...
		[qi_p clearQueue];
	}
}

#ifdef FOOBAR
quipImageView *image_view_for_viewer(Viewer *vp)
{
	quipImageView *qiv_p;
	CGRect frame;

	if( VW_QIV(vp) != NULL ){
fprintf(stderr,"image_view_for_viewer returing existing image view 0x%lx for viewer %s\n",(long)VW_QIV(vp),VW_NAME(vp));
		return VW_QIV(vp);
	}

	frame = CGRectMake(0,0,VW_WIDTH(vp),VW_HEIGHT(vp));
	qiv_p = [[quipImageView alloc] initWithFrame:frame];

#ifdef BUILD_FOR_IOS
	qiv_p.contentMode = UIViewContentModeTopLeft;
#endif // BUILD_FOR_IOS

	// We add the new imageView as a subview of the viewer view...
	if( VW_IMAGES(vp) == NULL ){
		insure_viewer_images(vp);
	}

fprintf(stderr,"adding subview 0x%lx to images 0x%lx, viewer = 0x%lx...\n",
(long)qiv_p,(long)VW_IMAGES(vp),(long)vp);
	[VW_IMAGES(vp) addSubview:qiv_p];

#ifdef BUILD_FOR_MACOS
	[GW_WINDOW(VW_GW(vp)) setContentView: qiv_p ];
#endif // BUILD_FOR_MACOS

	// The newest image is placed in front.
	// For a movie, if we render them in the order
	// we want them played, then we can move from back-to-front...
#ifdef BUILD_FOR_IOS
fprintf(stderr,"bringing subview to front...\n");
	[VW_IMAGES(vp) bringSubviewToFront:qiv_p];
#endif // BUILD_FOR_IOS

	VW_QIV(vp) = qiv_p;

	return qiv_p;
}
#endif // FOOBAR

#ifdef BUILD_FOR_IOS

//Set backlight function implemented by GJS
void set_backlight(CGFloat l)
{
	//CGFloat level = l;
fprintf(stderr,"Setting backlight to %g\n",l);
	[[UIScreen mainScreen] setBrightness:l];
}

#endif // BUILD_FOR_IOS

void zap_viewer(Viewer *vp)
{

}

void extra_viewer_info(QSP_ARG_DECL  Viewer *vp)
{

}

Disp_Obj *curr_dop(void)
{
	return NULL;
}

void unembed_image(QSP_ARG_DECL  Viewer *vp,Data_Obj *dp,int x,int y)
{

}

void set_remember_gfx(int flag)
{

}

void _xp_erase(QSP_ARG_DECL  Viewer *vp)
{
	// We used to release the drawlist here, but that is wrong
	// because it may contain non-drawing directives, like setting the font etc.
	// what would be best would be to make the viewer needy here,
	// and then wait until it has been redrawn, and THEN release
	// the list...  Or we could have a non-drawing scan of the list first?
	Draw_Op *do_p;
	do_p = new_drawop(DO_ERASE);
	ADD_DRAW_OP(vp,do_p);
	MAKE_NEEDY(vp);
}

void _xp_update(Viewer *vp)
{
	MAKE_NEEDY(vp);
}


void _xp_move(QSP_ARG_DECL  Viewer *vp,int x1,int y1)
{
	Draw_Op *do_p;
	do_p = new_drawop(DO_MOVE);
	SET_DOA_X(do_p,x1);
	SET_DOA_Y(do_p,y1);
	ADD_DRAW_OP(vp,do_p);
}

void _xp_select(QSP_ARG_DECL  Viewer *vp, u_long index)
{
	QUIP_COLOR_TYPE *c;

	SET_VW_FGCOLOR(vp,(int)index);
	CHECK_COLOR_INDEX(_xp_select,index)
	Draw_Op *do_p;
	do_p = new_drawop(DO_SELECT_PEN);
	c = get_color_from_index(vp,(int)index);
	//fprintf(stderr,"_xp_select %s:  setting drawing index %lu, c = 0x%lx\n",
 //		VW_NAME(vp),index,(long)c);
	DOA_COLOR(do_p) = c;
	ADD_DRAW_OP(vp,do_p);
}

void _xp_bgselect(QSP_ARG_DECL  Viewer *vp, u_long index)
{
	QUIP_COLOR_TYPE *c;

	SET_VW_BGCOLOR(vp,(int)index);
	CHECK_COLOR_INDEX(_xp_bgselect,index)
	Draw_Op *do_p;
	do_p = new_drawop(DO_SET_BG);
	c = get_color_from_index(vp,(int)index);
	DOA_COLOR(do_p) = c;
	ADD_DRAW_OP(vp,do_p);
}

void _xp_line(QSP_ARG_DECL  Viewer *vp,int x1,int y1,int x2,int y2)
{
	Draw_Op *do_p;

	_xp_move(vp,x1,y1);

	do_p = new_drawop(DO_CONT);
	SET_DOA_X(do_p, x2);
	SET_DOA_Y(do_p, y2);
	ADD_DRAW_OP(vp,do_p);
//fprintf(stderr,"line draw_op added\n");
	MAKE_NEEDY(vp);
}

void _xp_linewidth(QSP_ARG_DECL  Viewer *vp,CGFloat w)
{
	Draw_Op *do_p;

	do_p = new_drawop(DO_LINEWIDTH);
	SET_DOA_LINEWIDTH(do_p, (float)w);
	ADD_DRAW_OP(vp,do_p);
	// This doesn't draw anything, so by itself
	// it doesn't require a redraw?
	//MAKE_NEEDY(vp);
}

void _set_font_size(QSP_ARG_DECL  Viewer *vp,int sz)
{
	Draw_Op *do_p;

	do_p = new_drawop(DO_FONT_SIZE);
	SET_DOA_FONT_SIZE(do_p, sz);
	ADD_DRAW_OP(vp,do_p);
	font_size = sz;		// BUG?  in case the script needs it
				// e.g. get_string_offset
}

void _set_text_angle(QSP_ARG_DECL  Viewer *vp,float a)
{
	Draw_Op *do_p;

	do_p = new_drawop(DO_TEXT_ANGLE);
	SET_DOA_ANGLE(do_p, a);
	ADD_DRAW_OP(vp,do_p);
}

void _set_char_spacing(QSP_ARG_DECL  Viewer *vp,int sz)
{
	Draw_Op *do_p;

	do_p = new_drawop(DO_CHAR_SPACING);
	SET_DOA_CHAR_SPACING(do_p, sz);
	ADD_DRAW_OP(vp,do_p);
}


void _set_font_by_name(QSP_ARG_DECL  Viewer *vp,const char *s)
{
	Draw_Op *do_p;

	do_p = new_drawop(DO_FONT);
	DOA_STR(do_p) = savestr(s);
	ADD_DRAW_OP(vp,do_p);
	// We only need to make needy for things that actually draw!
	//MAKE_NEEDY(vp);
}

void _xp_text(QSP_ARG_DECL  Viewer *vp,int x, int y, const char *s)
{
	Draw_Op *do_p;
	_xp_move(vp,x,y);
	do_p = new_drawop(DO_TEXT);
	DOA_STR(do_p) = savestr(s);
	ADD_DRAW_OP(vp,do_p);
	MAKE_NEEDY(vp);
}

void _xp_arc(QSP_ARG_DECL  Viewer *vp,int p1,int p2,int p3,int p4,int p5,int p6)
{

	// Not sure what the args are???
	warn("_xp_arc:  UNIMPLEMENTED!?");
	//CGContextAddArc(VW_GFX_CTX(vp),x,y,radius,startAngle,endAngle,clockwise);
}

void _xp_fill_arc(QSP_ARG_DECL  Viewer *vp,int p1,int p2,int p3,int p4,int p5,int p6)
{
	warn("_xp_fill_arc:  UNIMPLEMENTED!?");

}

void _xp_fill_polygon(QSP_ARG_DECL  Viewer* vp, int num_points, int* px_vals, int* py_vals)
{
	warn("_xp_fill_polygon:  UNIMPLEMENTED!?");

}

// CGContextSetAllowsAntialiasing
// CGContextSetShouldAntialias

void _dump_drawlist(QSP_ARG_DECL  Viewer *vp)
{
	IOS_Node *np;
	Draw_Op *do_p;

	if( VW_DRAW_LIST(vp) == NULL ) {
		fprintf(stderr,"dump_drawlist %s:  null draw list!?\n",VW_NAME(vp));
		return;
	}

	np = IOS_LIST_HEAD(VW_DRAW_LIST(vp));
	while(np!=NULL){
		do_p = (Draw_Op *) IOS_NODE_DATA(np);
		report_drawop(vp,do_p);
		np = IOS_NODE_NEXT(np);
	}
}

void _left_justify(QSP_ARG_DECL  Viewer *vp)
{
	Draw_Op *do_p;
	do_p = new_drawop(DO_LJUST);
	ADD_DRAW_OP(vp,do_p);
}

void _right_justify(QSP_ARG_DECL  Viewer *vp)
{
	Draw_Op *do_p;
	do_p = new_drawop(DO_RJUST);
	ADD_DRAW_OP(vp,do_p);
}

void _center_text(QSP_ARG_DECL  Viewer *vp)
{
	Draw_Op *do_p;
	do_p = new_drawop(DO_CJUST);
	ADD_DRAW_OP(vp,do_p);
}

int make_2d_adjuster(QSP_ARG_DECL  Viewer *vp,int w, int h)
{
	return 0;
}

void posn_viewer(Viewer *vp, int x, int y)
{
#ifdef BUILD_FOR_IOS
	NADVISE("posn_viewer:  UNIMPLEMENTED!?");
#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
	_posn_genwin(DEFAULT_QSP_ARG  VW_GW(vp),x,y);
#endif // BUILD_FOR_MACOS
}

// A "button arena" is a window in X11 that processes mouse clicks;
// There are no buttons in IOS, and all windows accept touch events, so
// this is a normal viewer...

int make_button_arena(QSP_ARG_DECL  Viewer *vp,int w,int h)
{
	return make_viewer(vp,w,h);
}

int make_dragscape(QSP_ARG_DECL  Viewer *vp,int w,int h)
{
	// BUG?  do something special here???
	return make_viewer(vp,w,h);
}

int make_mousescape(QSP_ARG_DECL  Viewer *vp,int w,int h)
{
	// BUG?  do something special here???
	return make_viewer(vp,w,h);
}

void relabel_viewer(Viewer *vp,const char *s)
{
fprintf(stderr,"relabel_viewer not implemented for iOS\n");
}

void redraw_viewer(QSP_ARG_DECL  Viewer *vp)
{
fprintf(stderr,"redraw_viewer not implemented for iOS\n");
}

int event_loop(SINGLE_QSP_ARG_DECL)
{
	return 0;

}

void install_default_lintbl(QSP_ARG_DECL  Dpyable *dpyp)
{

}

int _display_depth(SINGLE_QSP_ARG_DECL)
{
	return 4;	// bits or bytes?
}

void list_disp_objs(QSP_ARG_DECL  FILE *fp)
{

}

Disp_Obj *pick_disp_obj(QSP_ARG_DECL  const char *pmpt)
{
	return NULL;
}

#define DEFAULT_PIXELS_PER_CHAR 9

int get_string_width(Viewer *vp, const char *s)
{
	// Do we really initialize here???
	if( VW_GFX_CTX(vp) == NULL ){
		sprintf(ERROR_STRING,
			"get_string_width '%s':  drawing context for viewer %s is NULL!?",
			s,VW_NAME(vp));
		NADVISE(ERROR_STRING);
		return( (int)strlen(s) * DEFAULT_PIXELS_PER_CHAR );
	}

	if( ! VW_TXT_MTRX_READY(vp) ){
fprintf(stderr,"get_string_width(%s) calling init_text_font\n",s);
		// this has to be here for get_string_offset to work...
		init_text_font(vp);
	}

	CGPoint pt = get_string_offset(VW_GFX_CTX(vp),s);
//fprintf(stderr,
//"get_string_width '%s':  offset %g %g, width is %d, char_spacing = %d\n",
//s,pt.x,pt.y,(int)(pt.x-char_spacing),char_spacing);
	return ((int)(pt.x - char_spacing));
}

void embed_draggable(Data_Obj *dp,Draggable *dgp)
{

}

void zap_image_list(Viewer *vp)
{

}

void init_viewer_canvas(Viewer *vp)
{
	quipCanvas *qc;
	CGSize size;

	size.width = VW_WIDTH(vp);
	size.height = VW_HEIGHT(vp);

	qc=[[quipCanvas alloc]initWithSize:size];
#ifdef BUILD_FOR_IOS
	//SET_QV_CANVAS(VW_QV(vp),qc);
	SET_VW_CANVAS(vp,qc);

	// YES is the default value - why is this not happening???
	qc.clearsContextBeforeDrawing = YES;	// doesn't work!?

	[VW_QV(vp) addSubview:qc];
	// We want the canvas to be in front of the images,
	// but behind the controls...
	[VW_QV(vp) bringSubviewToFront:qc];
// background color set to clear for canvas!
// But we are having trouble erasing non-clear stuff to clear!?
	qc.backgroundColor = [QUIP_COLOR_TYPE clearColor];
	qc.opaque = YES;	// drawn content should be opaque!?

#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
	SET_GW_CANVAS(VW_GW(vp),qc);

	// set context here?
	if( VW_GFX_CTX(vp) == NULL ){
		SET_VW_GFX_CTX(vp, (CGContextRef)[[NSGraphicsContext currentContext] graphicsPort ] );
	}
#endif // BUILD_FOR_MACOS

	// We can't do this here because
	// the context is set by drawRect
	//init_text_font(vp);
} // end init_viewer_canvas

#ifdef BUILD_FOR_IOS
void bring_image_to_front(QSP_ARG_DECL  Viewer *vp, Data_Obj *dp,int x,int y)
{
	warn("bring_image_to_front not implemented for iOS!?");
}
#endif // BUILD_FOR_IOS

