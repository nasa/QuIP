#include "quip_config.h"

char VersionId_viewer_cursor[] = QUIP_VERSION_STRING;

#ifdef HAVE_X11

/* cursor package */
#include "viewer.h"
#include "getbuf.h"
#include "savestr.h"
#include "items.h"
#include "xsupp.h"

/* local prototypes */
static void swap_bytes(u_short *arr,int n);

ITEM_INTERFACE_DECLARATIONS(View_Cursor,cursor)

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
	if( bitmap_dp->dt_prec == PREC_DI )
		mk_cursor(QSP_ARG  name,(u_short *)bitmap_dp->dt_data,bitmap_dp->dt_cols * 32,
			bitmap_dp->dt_rows,x,y);
	else if( bitmap_dp->dt_prec == PREC_IN )
		mk_cursor(QSP_ARG  name,(u_short *)bitmap_dp->dt_data,bitmap_dp->dt_cols * 16,
			bitmap_dp->dt_rows,x,y);
	else {
		sprintf(error_string,"make_cursor:  bitmap object %s (%s) should have %s or %s precision",
			bitmap_dp->dt_name,name_for_prec(bitmap_dp->dt_prec),
			name_for_prec(PREC_IN),name_for_prec(PREC_DI));
		WARN(error_string);
	}
}

void mk_cursor( QSP_ARG_DECL  const char *name, u_short *data, dimension_t dx,dimension_t dy,dimension_t x,dimension_t y )
{
	Pixmap src_pixmap;
	XColor fg_col, bg_col;
	unsigned long fg,bg;
	View_Cursor *vcp;
	Disp_Obj *dop;

	vcp = new_cursor(QSP_ARG  name);
	if( vcp == NO_CURSOR ) return;

	fg=1;
	bg=0;

	/* we assume that the pixmap is a PREC_DI long image, thus
	 * we multiply the columns by 2
	 * BUG we should check that the image is the right type!
	 */

#define CURSOR_DEPTH	1

	dop = curr_dop();
	src_pixmap = XCreatePixmapFromBitmapData(dop->do_dpy,
		dop->do_rootw,
		(char *)data, dx, dy,fg,bg,CURSOR_DEPTH);

	fg_col.red=200;
	fg_col.green=0;
	fg_col.blue=0;

	bg_col.red=0;
	bg_col.green=0;
	bg_col.blue=200;

	vcp->vc_cursor = XCreatePixmapCursor(dop->do_dpy,src_pixmap,
		/*NULL*/src_pixmap/*mask*/,
		&fg_col,&bg_col, 0 /* x_hot */, 0 /* y_hot */ );

	vcp->vc_xhot=x;
	vcp->vc_yhot=y;
}

void root_cursor( View_Cursor *vcp )
{
	Disp_Obj *dop;
	dop = curr_dop();
	XDefineCursor(dop->do_dpy,dop->do_rootw,vcp->vc_cursor);
}

void assign_cursor( Viewer *vp, View_Cursor *vcp )
{
	XDefineCursor(vp->vw_dpy,vp->vw_xwin,vcp->vc_cursor);
}

#endif /* HAVE_X11 */

