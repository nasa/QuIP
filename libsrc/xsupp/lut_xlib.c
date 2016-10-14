#include "quip_config.h"

#ifdef HAVE_X11

/* #define X_LUT_BUFFERS_ON */

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "cmaps.h"
#include "xsupp.h"
#include "xsupp_prot.h"
#include "viewer.h"
#include "debug.h"

#define NO_WINDOW ((Window)NULL)
static Window curr_window=NO_WINDOW;
static int n_to_protect = NC_SYSTEM;
int simulating_luts=0;

#ifdef X_LUT_BUFFERS_ON
static Window curr_win(void)
{
	return(curr_window);
}
#endif // X_LUT_BUFFERS_ON

void set_curr_win(Window win)
{
#ifdef QUIP_DEBUG
if( debug & xdebug ){
sprintf(DEFAULT_ERROR_STRING,"set_curr_win:  setting current window to 0x%lx",(u_long)win);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	curr_window = win;
}

void cmap_setup(Viewer *vp)
{
	int i;

#ifdef QUIP_DEBUG
if( debug & xdebug ){
sprintf(DEFAULT_ERROR_STRING,"cmap_setup: viewer %s", vp->vw_name);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	/* only create colormap if visual is psuedocolor */

	if( VW_DEPTH(vp) == 8 ){

/* NADVISE("xld_setup:  creating colormap..."); */

		/* we need to create a colormap to be able to write color table entries...
		 * But we might like to initialize to the system's default map, not
		 * our own peculiar map...
		 */
		vp->vw_cmap = XCreateColormap(VW_DPY(vp), VW_ROOTW(vp),
				VW_VISUAL(vp), AllocAll );

	} else {
		if( verbose )
			NADVISE("cmap_setup:  not creating X colormap for non-pseudocolor display");
		return;
	}

	SET_VW_XCTBL(vp, (XColor *) getbuf( N_COLORS * sizeof(XColor) ) );
	for(i=0;i<N_COLORS;i++)
		SET_VW_XCTBL_PIXEL(vp,i, (u_long)i );
	vp->vw_n_protected_colors = n_to_protect;
}

u_long simulate_lut_mapping(Viewer *vp, u_long color)
{
	int index;
	int r,g,b;

	/* need to get the color map for this viewer */
	if( VW_CMAP_OBJ(vp) == NO_OBJ ){
NWARN("simulate_lut_mapping:  no colormap!?");
		return(color);
	}

	index = color;

	if( index < 0 || index >= N_COLORS ){
		sprintf(DEFAULT_ERROR_STRING,"simulate_lut_mapping:  index %d (0x%x) must be in the range 0-%d",
			index,index,N_COLORS);
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}

	r = CM_DATA(VW_CMAP_OBJ(vp),0,index);
	g = CM_DATA(VW_CMAP_OBJ(vp),1,index);
	b = CM_DATA(VW_CMAP_OBJ(vp),2,index);

	/* This is the 24 bit case */
	if( vp->vw_depth == 24 ){
		color = r;
		color <<= 8;
		color |= g;
		color <<= 8;
		color |= b;
		/* no alpha for now */
	} else if( vp->vw_depth == 16 ){
		color  = ((r>>3)&0x1f)<<11;
		color |= ((g>>2)&0x3f)<<5;
		color |= ((b>>2)&0x1f);
	}

	return(color);
}

/* THis FOOBAR disables a whole section with other FOOBAR's inside it???
 * change to X_LUT_BUFFERS_ON until we figure out what's going on...
 */

#ifdef X_LUT_BUFFERS_ON
/*
 * Get some initial values for a new lut buffer
 * normally we just take the values which are currently loaded
 *
 * This version works with iview but not with guimenu,
 * because it is set up to work with viewer structures,
 * which are not part of guimenu...  Need to fix this...
 */

static int depth_warned=0;

void x_init_lb_data( Lutbuf *lbp )
{
	Disp_Obj *dop;

//#ifdef CAUTIOUS
//	if( lbp->lb_dp != NO_OBJ ){
//		sprintf(ERROR_STRING,
//	"lutbuffer \"%s\":  data already initialized (%s)",
//			lbp->lb_name,OBJ_NAME(lbp->lb_dp));
//		NWARN(ERROR_STRING);
//		return;
//	}
//#endif /* CAUTIOUS */
	assert( lbp->lb_dp == NO_OBJ );

	if( (dop=curr_dop()) == NO_DISP_OBJ )
		return;

	if( DO_DEPTH(dop) != 8 ){
		if( verbose && !depth_warned ){
			sprintf(ERROR_STRING,
				"Display depth (%d) must be 8 for colormaps, simulating...",
				DO_DEPTH(dop));
			advise(ERROR_STRING);
			depth_warned++;
		}
		/* BUG?  should this be on a per-window basis?? */
		simulating_luts=1;
	}
	if( simulating_luts ){
		/* the ld_data points to a cmap structure */
		char cmap_name[LLEN];
		Data_Obj *dp;
		sprintf(cmap_name,"lutbuf_data%d",lb_data_serial++);
		/* lbp->lb_data = new_colormap(cmap_name); */
sprintf(ERROR_STRING,"testing colormap %s",cmap_name);
advise(ERROR_STRING);
		dp=obj_of(cmap_name);
		if( dp == NO_OBJ ){
			lbp->lb_dp = new_colormap(cmap_name);
		} else {
			sprintf(ERROR_STRING,"x_init_lb_data:  colormap object %s already exists",OBJ_NAME(dp));
			advise(ERROR_STRING);
			lbp->lb_dp = dp;
		}
	} else {
		lbp->lb_xldp = (XlibData *) getbuf(sizeof(XlibData));
		xld_setup((XlibData *)lbp->lb_xldp,dop,curr_win());
	}
}

#ifdef FOOBAR
void set_xl_cmap(XlibData *xldp,Data_Obj *cm_dp)
{
	int i;

	for(i=xldp->xld_protected_colors;i<N_COLORS;i++){
		xldp->xld_xctbl[i].red = ((u_short) CM_DATA(cm_dp,0,i))<<8;
		xldp->xld_xctbl[i].green = ((u_short) CM_DATA(cm_dp,1,i))<<8;
		xldp->xld_xctbl[i].blue = ((u_short) CM_DATA(cm_dp,2,i))<<8;
		xldp->xld_xctbl[i].flags = DoRed | DoGreen | DoBlue;
	}
	xldp->xld_flags |= CMAP_UPDATE;
}
#endif /* FOOBAR */

/*
 * assign the lut buffer values from the scratch cmap
 */

void x_assign_lutbuf(Lutbuf *lbp,Data_Obj *cm_dp)
{
	static int warned_once=0;

#ifdef FOOBAR
	if( luts_disabled ){
		/* for now, there is no difference between luts_disabled and simulating_luts!? */
		return;
	}
#endif /* FOOBAR */

	if( lbp == NO_LUTBUF ){
		if( !warned_once ){
			NWARN("assign_lutbuf(): lbp = NULL");
			warned_once++;
		}
		return;
	}

	if( simulating_luts ){
//#ifdef CAUTIOUS
//		if( lbp->lb_dp == NO_OBJ ){
//			NWARN("lutbuffer data NOT initialized (simulated)");
//			init_lb_data(lbp);
//		}
//#endif /* CAUTIOUS */
		assert( lbp->lb_dp != NO_OBJ );

sprintf(ERROR_STRING,"x_assign_lutbuf:  copying data from %s to %s",OBJ_NAME(cm_dp),lbp->OBJ_NAME(lb_dp));
advise(ERROR_STRING);
		dp_copy(lbp->lb_dp,cm_dp);
	} else {
//#ifdef CAUTIOUS
//		if( lbp->lb_xldp == NO_XLIBDATA ){
//			NWARN("lutbuffer data NOT initialized (xlib)");
//			init_lb_data(lbp);
//		}
//#endif /* CAUTIOUS */
		assert( lbp->lb_xldp != NO_XLIBDATA );

		set_xl_cmap(lbp->lb_xldp,cm_dp);
		curr_xldp = lbp->lb_xldp;
	}
}

void x_set_n_protect(int n)
{
	n_to_protect = n;
	if( curr_xldp != NO_XLIBDATA )
		curr_xldp->xld_protected_colors = n;
}

/*
 * read_lutbuf() transfers data from a named buffer into a scratch array
 *
 * what about a function to read back the hardware?
 */

void x_read_lutbuf(Data_Obj *cm_dp,Lutbuf *lbp)
{
	u_long j;
	XlibData *xldp;
	int start;
//#ifdef CAUTIOUS
//	static int warned=0;
//
//	warned=0;
//#endif /* CAUTIOUS */

	if( luts_disabled ) return;

	if( lbp == NO_LUTBUF ){
		NWARN("read_lutbuf():  lbp = NULL");
		return;
	}
	if( simulating_luts ){
//#ifdef CAUTIOUS
//		if( lbp->lb_dp == NO_OBJ ){
//			error1("CAUTIOUS:  missing object in lutbuf");
//		}
//#endif /* CAUTIOUS */
		assert( lbp->lb_dp != NO_OBJ );

sprintf(ERROR_STRING,"x_read_lutbuf:  copying data from %s to %s",OBJ_NAME(lbp->lb_dp),OBJ_NAME(cm_dp));
advise(ERROR_STRING);
		dp_copy(cm_dp,lbp->lb_dp);
		return;
	}

	xldp = lbp->lb_xldp;
//#ifdef CAUTIOUS
//	if( xldp==NO_XLIBDATA ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  x_read_lutbuf:  missing xlib data");
//		NWARN(ERROR_STRING);
//		return;
//	}
//#endif /* CAUTIOUS */
	assert( xldp != NO_XLIBDATA );

	curr_xldp = xldp;

	if( VIEWER_HAS_SYSCOLS(vp) ) start=0;
	else start=xldp->xld_protected_colors;

	for(j=start;j<N_COLORS;j++){
//#ifdef CAUTIOUS
//		u_long index;
//
//		index = xldp->xld_xctbl[j].pixel;
//		if( index != j ){
//			if( !warned ){
//	sprintf(ERROR_STRING,"x_read_lutbuf  CAUTIOUS:  unexpected pixel index[%ld] = %ld",
//				j,index);
//				NWARN(ERROR_STRING);
//				warned=1;
//			} else {
//				warned++;
//			}
//		}
//#endif /* CAUTIOUS */
		assert( xldp->xld_xctbl[j].pixel == j );

		CM_DATA(cm_dp,0,j) = xldp->xld_xctbl[j].red >> 8 ;
		CM_DATA(cm_dp,1,j) = xldp->xld_xctbl[j].green >> 8 ;
		CM_DATA(cm_dp,2,j) = xldp->xld_xctbl[j].blue >> 8 ;
	}
//#ifdef CAUTIOUS
//	if( warned > 2 ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  %d total errors detected",warned);
//		NWARN(ERROR_STRING);
//	}
//#endif /* CAUTIOUS */
}

void x_show_lb_value( Lutbuf *lbp; int index )
{
	int r,g,b;

	if (index < 0 || index > 255) {
		warn ( "show_lb_value(): Illegal lut index");
		return;
	}

	if( simulating_luts ){
//#ifdef CAUTIOUS
//		if( lbp->lb_dp == NO_OBJ ){
//			error1("CAUTIOUS:  x_show_lb_value:  no lb_dp");
//		}
//#endif /* CAUTIOUS */
		assert( lbp->lb_dp != NO_OBJ );

		r=CM_DATA(lbp->lb_dp,0,index);
		g=CM_DATA(lbp->lb_dp,1,index);
		b=CM_DATA(lbp->lb_dp,2,index);
	} else {
		XlibData *xldp;

		xldp = lbp->lb_xldp;
//#ifdef CAUTIOUS
//		if( xldp == NO_XLIBDATA ){
//			error1("CAUTIOUS:  x_show_lb_value:  no xlib data");
//		}
//#endif /* CAUTIOUS */
		assert( xldp != NO_XLIBDATA );

		curr_xldp = xldp;
		r=xldp->xld_xctbl[index].red>>8;
		g=xldp->xld_xctbl[index].green>>8;
		b=xldp->xld_xctbl[index].blue>>8;
	}

	printf("%d:\t",index);
	printf("  %3d",r);
	printf("  %3d",g);
	printf("  %3d",b);
	printf("\n");
}
#endif /* X_LUT_BUFFERS_ON */

static void fetch_system_colors(Dpyable *dpyp)
{
	Colormap cm;

	if( DPA_N_PROT_CLRS(dpyp) == 0 ) return;	/* don't need to know */

	cm = DefaultColormap(DPA_DISPLAY(dpyp),DefaultScreen(DPA_DISPLAY(dpyp)));

#ifdef QUIP_DEBUG
if( debug & xdebug ) NADVISE("XQueryColors");
#endif /* QUIP_DEBUG */
	XQueryColors(DPA_DISPLAY(dpyp),cm,DPA_XCTBL(dpyp),NC_SYSTEM);

	SET_DPA_FLAG_BITS(dpyp, KNOW_SYSCOLORS);
}

/*
 * send some data out to the hardware
 */

void install_colors(Dpyable *dpyp)
{
	int i;

	if( ! HAS_COLORMAP(dpyp) ) return;

	if( ! HAS_SYSCOLORS(dpyp) ) {
		/* get_system_colors(xldp); */
		fetch_system_colors(dpyp);
	}

	for(i=NC_SYSTEM;i<N_COLORS;i++){
		SET_DPA_XCTBL_PIXEL(dpyp,i, (unsigned long)i );
		SET_DPA_XCTBL_RED(dpyp,i, ((unsigned short) CM_DATA(DPA_CMAP_OBJ(dpyp),0,i) )<<8 );
		SET_DPA_XCTBL_GREEN(dpyp,i, ((unsigned short) CM_DATA(DPA_CMAP_OBJ(dpyp),1,i) )<<8 );
		SET_DPA_XCTBL_BLUE(dpyp,i, ((unsigned short) CM_DATA(DPA_CMAP_OBJ(dpyp),2,i) )<<8 );
		SET_DPA_XCTBL_FLAGS(dpyp,i, DoRed | DoGreen | DoBlue );
	}

	XStoreColors(DPA_DISPLAY(dpyp), DPA_CMAP(dpyp), DPA_XCTBL(dpyp), N_COLORS);

	XSetWindowColormap(DPA_DISPLAY(dpyp),DPA_XWIN(dpyp),DPA_CMAP(dpyp));

#ifdef QUIP_DEBUG
if( debug & xdebug ) NADVISE("back from XSetWindowColormap");
#endif /* QUIP_DEBUG */

	CLEAR_DPA_FLAG_BITS(dpyp,CMAP_UPDATE);
}

/*
 * send the lutbuffer values to the hardware
 */

void x_dump_lut(Dpyable *dpyp)
{
	if( DPA_CMAP_OBJ(dpyp)==NO_OBJ ) return;

	if( simulating_luts ) return;

#ifdef QUIP_DEBUG
if( debug & xdebug ){
sprintf(DEFAULT_ERROR_STRING,"dumping colormap %s",OBJ_NAME(DPA_CMAP_OBJ(dpyp)));
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	install_colors(dpyp);
}

#endif /* HAVE_X11 */

