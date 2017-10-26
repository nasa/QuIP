#include "quip_config.h"

/* cursor package */
#include "quip_prot.h"
#include "viewer.h"
#include "xsupp.h"

ITEM_INTERFACE_DECLARATIONS(View_Cursor,cursor,0)

u_short busy_bitmap[]={
#include "busy.cursor"
};

u_short bullseye_bitmap[]={
#include "bullseye.cursor"
};

u_short glass_bitmap[]={
#include "glass.cursor"
};

u_short hglass_bitmap[]={
#include "hglass.cursor"
};

static void swap_bytes( u_short *arr, int n )
{
	char *s;
	int c1, c2;

	s=(char *)arr;
	while( n-- ){
		c1 = *s;
		c2 = *(s+1);
		*s++ = c2;
		*s++ = c1;
	}

}

void default_cursors(SINGLE_QSP_ARG_DECL)
{
	swap_bytes(bullseye_bitmap,16);
	mk_cursor(QSP_ARG  "bullseye",bullseye_bitmap,16,16,0,0);

	swap_bytes(glass_bitmap,16);
	mk_cursor(QSP_ARG  "glass",glass_bitmap,16,16,0,0);

	swap_bytes(hglass_bitmap,16);
	mk_cursor(QSP_ARG  "hglass",hglass_bitmap,16,16,0,0);

	swap_bytes(busy_bitmap,16);
	mk_cursor(QSP_ARG  "busy",busy_bitmap,16,16,0,0);
}

/* mess around with the cursor */

void make_cursor( QSP_ARG_DECL  const char *name, Data_Obj *bitmap_dp, int x, int y )
{
	if( OBJ_PREC(bitmap_dp) == PREC_DI )
		mk_cursor(QSP_ARG  name,(u_short *)OBJ_DATA_PTR(bitmap_dp),OBJ_COLS(bitmap_dp) * 32,
			OBJ_ROWS(bitmap_dp),x,y);
	else if( OBJ_PREC(bitmap_dp) == PREC_IN )
		mk_cursor(QSP_ARG  name,(u_short *)OBJ_DATA_PTR(bitmap_dp),OBJ_COLS(bitmap_dp) * 16,
			OBJ_ROWS(bitmap_dp),x,y);
	else {
		sprintf(ERROR_STRING,"make_cursor:  bitmap object %s (%s) should have %s or %s precision",
			OBJ_NAME(bitmap_dp),OBJ_PREC_NAME(bitmap_dp),
			PREC_NAME(PREC_FOR_CODE(PREC_IN)),
			PREC_NAME(PREC_FOR_CODE(PREC_DI)) );
		WARN(ERROR_STRING);
	}
}

void mk_cursor( QSP_ARG_DECL  const char *name, u_short *data, dimension_t dx,dimension_t dy,dimension_t x,dimension_t y )
{
#ifdef HAVE_X11
	Pixmap src_pixmap;
	XColor fg_col, bg_col;
#endif /* HAVE_X11 */
	unsigned long fg,bg;
	View_Cursor *vcp;
	Disp_Obj *dop;

	vcp = new_cursor(name);
	if( vcp == NULL ) return;

	fg=1;
	bg=0;

	/* we assume that the pixmap is a PREC_DI long image, thus
	 * we multiply the columns by 2
	 * BUG we should check that the image is the right type!
	 */

#define CURSOR_DEPTH	1

	dop = curr_dop();
#ifdef HAVE_X11
	src_pixmap = XCreatePixmapFromBitmapData(DO_DISPLAY(dop),
		DO_ROOTW(dop),
		(char *)data, dx, dy,fg,bg,CURSOR_DEPTH);

	fg_col.red=200;
	fg_col.green=0;
	fg_col.blue=0;

	bg_col.red=0;
	bg_col.green=0;
	bg_col.blue=200;

	vcp->vc_cursor = XCreatePixmapCursor(DO_DISPLAY(dop),src_pixmap,
		/*NULL*/src_pixmap/*mask*/,
		&fg_col,&bg_col, 0 /* x_hot */, 0 /* y_hot */ );
#endif /* HAVE_X11 */
    
	vcp->vc_xhot=x;
	vcp->vc_yhot=y;
}

void root_cursor( View_Cursor *vcp )
{
	Disp_Obj *dop;
	dop = curr_dop();
#ifdef HAVE_X11
	XDefineCursor(DO_DISPLAY(dop),DO_ROOTW(dop),vcp->vc_cursor);
#endif /* HAVE_X11 */
}

void assign_cursor( Viewer *vp, View_Cursor *vcp )
{
#ifdef HAVE_X11
	XDefineCursor(VW_DPY(vp),vp->vw_xwin,vcp->vc_cursor);
#endif /* HAVE_X11 */
}

