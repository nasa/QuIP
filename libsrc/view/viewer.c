#include "quip_config.h"

char VersionId_viewer_viewer[] = QUIP_VERSION_STRING;

#ifdef HAVE_X11

#include <stdio.h>

#include "xsupp.h"
#include "savestr.h"
#include "debug.h"
#include "function.h"		/* prototype for add_sizable */
#include "gen_win.h"		/* add_genwin */
#include "linear.h"		/* set_lintbl */

#include "viewer.h"
Viewer *curr_vp=NO_VIEWER;

#include "items.h"

static int siz_done=0;

static double get_vw_size(Item *,int);
static void rls_vw_lists(Viewer *);

ITEM_INTERFACE_DECLARATIONS(Viewer,vwr)

static void rls_vw_lists(Viewer *vp)
{
	rls_list(vp->vw_image_list);
	rls_list(vp->vw_draglist);
	if( vp->vw_drawlist != NO_LIST )
		rls_list(vp->vw_drawlist);
}

void zap_image_list(Viewer *vp)
{
	Node *np,*np2;

	np = vp->vw_image_list->l_head;
	while(np!=NO_NODE){
		np2=np->n_next;
		rls_node(np);
		np=np2;
	}
	vp->vw_image_list->l_head=vp->vw_image_list->l_tail=NO_NODE;
}

void select_viewer(QSP_ARG_DECL  Viewer *vp)
{
	if( vp == NO_VIEWER ) return;

	curr_vp = vp;

	/* when creating the viewer this hasn't been done yet... */
	if( vp->vw_cm_dp != NO_OBJ ){
		select_cmap_display(&vp->vw_top);
		set_colormap(vp->vw_cm_dp);
	} else {
sprintf(error_string,"select_viewer %s:  no colormap",vp->vw_name);
advise(error_string);
	}
	if( vp->vw_lt_dp != NO_OBJ )
		set_lintbl(QSP_ARG  vp->vw_lt_dp);
}

void release_image(QSP_ARG_DECL  Data_Obj *dp)
{
	dp->dt_refcount --;
//sprintf(error_string,"release_image %s:  refcount = %d",dp->dt_name,dp->dt_refcount);
//advise(error_string);
	if( dp->dt_refcount <= 0 /* && IS_ZOMBIE(dp) */ )
		delvec(QSP_ARG  dp);
}

void delete_viewer(QSP_ARG_DECL  Viewer *vp)
{
	zap_viewer(vp);		/* window sys specific, no calls to givbuf... */
	zap_image_list(vp);	/* release nodes of image_list */
	rls_vw_lists(vp);	/* release list heads */

	if( vp->vw_dp != NO_OBJ )
		release_image(QSP_ARG  vp->vw_dp);
	if( vp->vw_cm_dp != NO_OBJ )
//{
//advise("releasing colormap");
		release_image(QSP_ARG  vp->vw_cm_dp);
//}
	if( vp->vw_lt_dp != NO_OBJ )
		release_image(QSP_ARG  vp->vw_lt_dp);

	if( vp->vw_label != vp->vw_name )
		rls_str((char *)vp->vw_label);
	del_vwr(QSP_ARG  vp->vw_name);
	rls_str((char *)vp->vw_name);
	select_viewer(QSP_ARG  NO_VIEWER);
}

static Size_Functions view_sf={
	/*(double (*)(Item *,int))*/		get_vw_size,
	/*(Item * (*)(Item *,index_t))*/	NULL,
	/*(Item * (*)(Item *,index_t))*/	NULL,
	/*(double (*)(Item *))*/		NULL
};

Viewer *viewer_init(QSP_ARG_DECL  const char *name,int dx,int dy,int flags)
{
	Viewer *vp;
	char str[256];
	int stat;

	if( dx <= 0 || dy <= 0 ){
		sprintf(error_string,
	"Dimensions for viewer %s (%d,%d) must be positive",
			name,dx,dy);
		WARN(error_string);
		return(NO_VIEWER);
	}

	vp=new_vwr(QSP_ARG  name);
	if( vp == NO_VIEWER ) return(vp);

	/* this might be better done in a global init routine... */
	if( !siz_done ){
		add_sizable(QSP_ARG  vwr_itp,&view_sf, NULL );
		siz_done=1;
	}

	vp->vw_label = vp->vw_name;		/* default label */
	vp->vw_time = (time_t) 0;

	vp->vw_width = dx;
	vp->vw_height = dy;
	vp->vw_x = 0;
	vp->vw_y = 0;

	vp->vw_xmin = 0;
	vp->vw_ymin = 0;
	vp->vw_xdel = 0;
	vp->vw_ydel = 0;

	/* should adopt a consistent policy re initializing these lists!? */
	vp->vw_image_list = new_list();
	vp->vw_draglist = new_list();
	vp->vw_drawlist = NO_LIST;		/* BUG should be in view_xlib.c */

	vp->vw_ip = (XImage *) NULL;
	vp->vw_ip2 = (XImage *) NULL;
	vp->vw_flags = flags;
	vp->vw_frameno = 0;
	vp->vw_text = NULL;
	vp->vw_text1 = NULL;
	vp->vw_text2 = NULL;
	vp->vw_text3 = NULL;

	/* we used to do this only if the display depth was 8, but now we simulate
	 * lutbuffers on 24 bit displays...
	 *
	 * BUG we need to have some way to do this only when requested...
	 */
	window_sys_init(SINGLE_QSP_ARG);/* make sure that the_dop is not NULL */
	set_viewer_display(vp);		/* sets vw_dop... */
	cmap_setup(vp);		/* refers to vw_dop, but that's not set until later? */

	install_default_lintbl(QSP_ARG  &vp->vw_top);
	sprintf(str,"colormap.%s",name);
	vp->vw_cm_dp = new_colormap(QSP_ARG  str);	/* new_colormap() used to call set_colormap(),
						 * but that generated a warning because the window
						 * doesn't actually exist yet.
						 * Better to do it in select_viewer().
						 */
	vp->vw_cm_dp->dt_refcount++;

	vp->vw_flags |= VIEW_LUT_SIM;

	/* do window system specific stuff */

	if( flags & VIEW_ADJUSTER )
		stat=make_2d_adjuster(QSP_ARG  vp);
	else if( flags & VIEW_DRAGSCAPE )
		stat=make_dragscape(QSP_ARG  vp);
	else if( flags & VIEW_MOUSESCAPE )
		stat=make_mousescape(QSP_ARG  vp);
	else if( flags & VIEW_BUTTON_ARENA )
		stat=make_button_arena(QSP_ARG  vp);
#ifdef SGI_GL
	else if( flags & VIEW_GL )
		stat=make_gl_window(QSP_ARG  vp);
#endif /* SGI_GL */
	else
		stat=make_viewer(QSP_ARG  vp);

	if( stat < 0 ){		/* probably can't open DISPLAY */
		rls_vw_lists(vp);
		del_vwr(QSP_ARG  vp->vw_name);
		rls_str((char *)vp->vw_name);
		return(NO_VIEWER);
	}


	vp->vw_depth = display_depth(SINGLE_QSP_ARG);

	vp->vw_dp = NO_OBJ;

#ifdef HAVE_OPENGL
	vp->vw_ctx = NULL;
#endif /* HAVE_OPENGL */

	select_viewer(QSP_ARG  vp);

	return(vp);
}


Node *first_viewer_node(SINGLE_QSP_ARG_DECL)
{
	List *lp;

	lp=item_list(QSP_ARG  vwr_itp);
	if( lp==NO_LIST ) return(NO_NODE);
	else return(lp->l_head);
}

List *viewer_list(SINGLE_QSP_ARG_DECL)
{
	return( item_list(QSP_ARG  vwr_itp) );
}

void info_viewer(Viewer *vp)
{
	const char *vtyp;

	if( IS_ADJUSTER(vp) ) vtyp="Adjuster";
	else if( IS_DRAGSCAPE(vp) ) vtyp="Dragscape";
	else if( IS_TRACKING(vp) ) vtyp="Tracking_Viewer";
	else if( IS_BUTTON_ARENA(vp) ) vtyp="Click_Viewer";
	else vtyp="Viewer";

	sprintf(msg_str,"%s \"%s\", %d rows, %d columns, at %d %d",vtyp,vp->vw_name,
		vp->vw_height,vp->vw_width,vp->vw_x,vp->vw_y);
	prt_msg(msg_str);

	if( vp->vw_dp != NO_OBJ ){
		sprintf(msg_str,
			"\tassociated data object:  %s",vp->vw_dp->dt_name);
		prt_msg(msg_str);
	} else prt_msg("\tNo associated image");

	if( SIMULATING_LUTS(vp) ){
		prt_msg("\tSimulating LUT color mapping");
		sprintf(msg_str,"\t\tcolormap object:  %s",vp->vw_cm_dp->dt_name);
		prt_msg(msg_str);
		if( vp->vw_lt_dp != NO_OBJ ){
			sprintf(msg_str,"\t\tlinearization table object:  %s",vp->vw_lt_dp->dt_name);
			prt_msg(msg_str);
		}
	}

	extra_viewer_info(vp);
}

static double get_vw_size(Item *ip,int index)
{
	double d;
	Viewer *vp;

	vp = (Viewer *)ip;

	switch(index){
		case 0: d = vp->vw_depth/8; break;
		case 1:	d = vp->vw_width; break;
		case 2:	d = vp->vw_height; break;
		default: d=1.0; break;
	}
	return(d);
}

double viewer_exists(QSP_ARG_DECL  const char *name)
{
	Viewer *vp;

	vp=VWR_OF(name);
	if( vp==NO_VIEWER ) return(0.0);
	else return(1.0);
}

/* genwin support */

/* These don't seem like the right place to put these,
 * but we need the item_type in order for
 * add_genwin() to work...
 */

void do_genwin_viewer_show(QSP_ARG_DECL  const char *s)
{
	Viewer *vp;

	vp=GET_VWR(s);
	if( vp == NO_VIEWER ) return;
	show_viewer(QSP_ARG  vp);
	return;
}

void do_genwin_viewer_unshow(QSP_ARG_DECL  const char *s)
{
	Viewer *vp;

	vp=GET_VWR(s);
	if( vp == NO_VIEWER ) return;
	show_viewer(QSP_ARG  vp);
	return;
}

void do_genwin_viewer_posn(QSP_ARG_DECL  const char *s, int x, int y)
{
	Viewer *vp;

	vp=GET_VWR(s);
	if( vp == NO_VIEWER ) return;
	posn_viewer(vp, x, y);
	return;
}

void do_genwin_viewer_delete(QSP_ARG_DECL  const char *s)
{
	Viewer *vp;

	vp=GET_VWR(s);
	if( vp == NO_VIEWER ) return;
	delete_viewer(QSP_ARG  vp);
	return;
}

static Genwin_Functions gwfp={
	(void (*)(QSP_ARG_DECL  const char *, int , int))do_genwin_viewer_posn,
	(void (*)(QSP_ARG_DECL  const char *))do_genwin_viewer_show,
	(void (*)(QSP_ARG_DECL  const char *))do_genwin_viewer_unshow,
	(void (*)(QSP_ARG_DECL  const char *))do_genwin_viewer_delete
};	

void init_viewer_genwin(SINGLE_QSP_ARG_DECL)
{
	if( vwr_itp == NO_ITEM_TYPE ) vwr_init(SINGLE_QSP_ARG);
	add_genwin(QSP_ARG  vwr_itp, &gwfp, NULL);
	return;
}	

#endif /* HAVE_X11 */

