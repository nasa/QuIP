#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif


#include "data_obj.h"
#include "quip_prot.h"
#include "cmaps.h"
#include "linear.h"
#include "xsupp.h"		/* x_dump_lut() */
#ifdef BUILD_FOR_MACOS
#include <AppKit/NSColor.h>
#endif // BUILD_FOR_MACOS

#define MAX_CHAR		80

#define MAXIMMEDDEPTH 32
static long state_stack[MAXIMMEDDEPTH];
static int depth=0;

/* globals ? */
long cm_flags=0;
long cm_state=IMMEDIATE;
#ifdef HAVE_X11
Dpyable *current_dpyp;
#endif /* HAVE_X11 */
#ifdef BUILD_FOR_IOS
#include "gen_win.h"
#endif /* BUILD_FOR_IOS */

Data_Obj *_new_colormap(QSP_ARG_DECL  const char *name)
{
	Data_Obj *dp;

#ifdef HAVE_CUDA
	push_data_area(ram_area_p);
#endif

	dp=mk_vec(name,N_COLORS,N_COMPS,PREC_FOR_CODE(PREC_UBY));

#ifdef HAVE_CUDA
	pop_data_area();
#endif

	return(dp);
}

void _push_cm_state(SINGLE_QSP_ARG_DECL)
{
	while( depth >= MAXIMMEDDEPTH ){
		depth--;
		warn("push_cm_state:  too many pushes");
	}
	state_stack[depth++] = cm_state;
}

void _pop_cm_state(SINGLE_QSP_ARG_DECL)
{
	if( depth <= 0 ){
		warn("pop_cm_state:  nothing to pop");
		return;
	}
	cm_state = state_stack[--depth];
}

void cm_immediate (long immediate)
{
	if( immediate ){
		SET_CM_STATE(IMMEDIATE);
	} else {
		CLR_CM_STATE(IMMEDIATE);
	}
}

/* Under the old vista board, there were "LUT's" inside the STAGE API, and you called
 * the dump function to send it to the actual hardware.  Our set of data_obj lut's replaces
 * STAGE's cache of buffers...
 */

void update_all(void)
{
#ifdef HAVE_X11
	x_dump_lut(current_dpyp);
#endif /* HAVE_X11 */
}

void update_if(void)
{
	if( CM_IS_IMMEDIATE ) update_all();
}

void _setcolor(QSP_ARG_DECL  int c,int r,int g,int b)
{
	/*insure_linearization(); */

	if( color_index_out_of_range(c) )
		return;

	/* we've checked the index; now check the phosphor levels */

	if( r < 0 ) r=0;
	else if( r>phosmax ){
		sprintf(ERROR_STRING,"Clipping red value %d to maximum %d",
			r,phosmax);
		warn(ERROR_STRING);
		r=phosmax;
	}

	if( g < 0 ) g=0;
	else if( g>phosmax ){
		sprintf(ERROR_STRING,"Clipping green value %d to maximum %d",
			g,phosmax);
		warn(ERROR_STRING);
		g=phosmax;
	}

	if( b < 0 ) b=0;
	else if( b>phosmax ){
		sprintf(ERROR_STRING,"Clipping blue value %d to maximum %d",
			b,phosmax);
		warn(ERROR_STRING);
		b=phosmax;
	}

	/* Why is the color map associated with the display
	 * and not the window?
	 */

#ifdef HAVE_X11
	// why is the colormap part of the display
	// and not the window?
	CM_DATA( DPA_CMAP_OBJ(current_dpyp),0,c) =
		LT_DATA( DPA_LINTBL_OBJ(current_dpyp),0,r);
	CM_DATA( DPA_CMAP_OBJ(current_dpyp),1,c) =
		LT_DATA( DPA_LINTBL_OBJ(current_dpyp),1,g);
	CM_DATA( DPA_CMAP_OBJ(current_dpyp),2,c) =
		LT_DATA( DPA_LINTBL_OBJ(current_dpyp),2,b);

	update_if();
#endif /* HAVE_X11 */

#ifdef BUILD_FOR_IOS

#define MAX_RGB_SPEC	256.0	// BUG?  what should this really be?
	UIColor *uic;
	// BUG?  should we linearize here?
	// or do the mapping in the script?
	uic = [[UIColor alloc]
		initWithRed:	r / MAX_RGB_SPEC
		green:		g / MAX_RGB_SPEC
		blue:		b / MAX_RGB_SPEC
		alpha:		1.0];
	INSURE_GW_CMAP(curr_genwin)
//fprintf(stderr,"setcolor %s %d:  %d %d %d  0x%lx, cmap = 0x%lx\n",
//GW_NAME(curr_genwin),c,r,g,b,(long)uic,(long)GW_CMAP(curr_genwin));
	[GW_CMAP(curr_genwin) insertObject: uic atIndex:c];
#endif

#ifdef BUILD_FOR_MACOS
#define MAX_RGB_SPEC	256.0	// BUG?  what should this really be?
	NSColor *nsc;
	nsc = [NSColor colorWithCalibratedRed: r/MAX_RGB_SPEC
			green:	g/MAX_RGB_SPEC
			blue:	b/MAX_RGB_SPEC
			alpha:	1.0];
	INSURE_GW_CMAP(curr_genwin)
	[GW_CMAP(curr_genwin) insertObject: nsc atIndex:c];
#endif // BUILD_FOR_MACOS

}

void _const_cmap(QSP_ARG_DECL  int base,int n,int r,int g,int b)
{
	push_cm_state();
	CLR_CM_STATE(IMMEDIATE);
	while(n--)
		setcolor(base++,r,g,b);
	pop_cm_state();

	update_if();
}

void _make_grayscale(QSP_ARG_DECL  int base,int n_colors)
{
	int i;
	int inc,v;

	/*insure_linearization(); */

	inc = phosmax/(n_colors-1);
	v=0;
	push_cm_state();
	CLR_CM_STATE(IMMEDIATE);
	for(i=0;i<n_colors;i++){
		setcolor(base+i,v,v,v);
		v+=inc;
	}
	pop_cm_state();

	update_if();
}

void _make_rgb(QSP_ARG_DECL  int base,int nr,int ng,int nb)
{
	int ir,ig,ib;
	int rinc,ginc,binc;
	int saved_ncolors;
	int ncolors,c;
	char str[256];

	if( nr > 1 ) rinc = (phosmax/(nr-1));
	else { rinc = 0; nr = -nr; }
	if( ng > 1 ) ginc = (phosmax/(ng-1));
	else { ginc = 0; ng = -ng; }
	if( nb > 1 ) binc = (phosmax/(nb-1));
	else { binc = 0; nb = -nb; }

	ncolors = nr*ng*nb;
	saved_ncolors = ncolors;
	if( base < 0 ){
		warn("wacky base");
		return;
	}
	if( (base + ncolors) >= N_COLORS){
		warn("too many RGB levels requested");
		ncolors = N_COLORS - base - 1;
	}

	push_cm_state();
	CLR_CM_STATE(IMMEDIATE);
	for(c=0;c<ncolors;c++){
		ib = c % nb;
		ig = (c/nb) % ng;
		ir = (c/(nb*ng)) % nr;
		setcolor(base+c,ir * rinc, ig * ginc, ib * binc );
	}
	pop_cm_state();

	update_if();

	if( (base + saved_ncolors) >= N_COLORS){
		sprintf(str,"too many colors specified, used %d", ncolors);
		warn(str);
		return;
	}
}

void _poke_lut(QSP_ARG_DECL  int c,int r,int g,int b)
{
	if( color_index_out_of_range(c) ) return;

#ifdef HAVE_X11
	CM_DATA( DPA_CMAP_OBJ(current_dpyp),0,c)= (unsigned char)r;
	CM_DATA( DPA_CMAP_OBJ(current_dpyp),1,c)= (unsigned char)g;
	CM_DATA( DPA_CMAP_OBJ(current_dpyp),2,c)= (unsigned char)b;
#endif /* HAVE_X11 */

	update_if();
}

void _setmap(QSP_ARG_DECL  Data_Obj *dp)
{
	short nc,ci,pxl_inc;
	float *fptr;
	short i,r,g,b;
	char str[256];

	if( dp == NULL ) return;

	if( OBJ_PREC(dp) != PREC_SP ){
		warn("setmap(): precision must be float");
		return;
	}
	if( ! IS_ROWVEC(dp) ){
		warn("setmap(): should be a row vector");
		return;
	}
	if( (nc=(short)OBJ_COLS(dp)) > N_COLORS ){
		sprintf(str,"setmap(): too many vector elements, using %d",
			N_COLORS);
		warn(str);
		nc=N_COLORS;
	}
	if( OBJ_COMPS(dp) != 3 ){
		warn("setmap(): vector should be tridimensional");
		return;
	}

	fptr= (float *)OBJ_DATA_PTR(dp);
	ci = (short)OBJ_COMP_INC(dp);
	pxl_inc = (short)OBJ_PXL_INC(dp);

	push_cm_state();
	CLR_CM_STATE(IMMEDIATE);
	for(i=0;i<nc;i++){
		r = (short)*fptr;
		g = (short)*(fptr + ci);
		b = (short)*(fptr + ci + ci);
		setcolor(i,r,g,b);
		fptr += pxl_inc;
	}
	pop_cm_state();

	update_if();
}

void _getmap(QSP_ARG_DECL  Data_Obj *dp)
{
	short nc,ci,pxl_inc;
	float *fptr;
	short i;
	char str[256];

	if( dp == NULL ) return;

	if( OBJ_PREC(dp) != PREC_SP ){
		warn("getmap(): precision must be float");
		return;
	}
	if( ! IS_ROWVEC(dp) ){
		warn("getmap(): should be a row vector");
		return;
	}

	if( (nc=(short)OBJ_COLS(dp)) > N_COLORS ){
		sprintf(str,"getmap(): too many vector elements, using %d",
			N_COLORS);
		warn(str);
		nc=N_COLORS;
	}
	if( OBJ_COMPS(dp) != 3 ){
		warn("getmap(): vector should be tridimensional");
		return;
	}

	fptr= (float *)OBJ_DATA_PTR(dp);
	ci = (short)OBJ_COMP_INC(dp);
	pxl_inc = (short)OBJ_PXL_INC(dp);

	for(i=0;i<nc;i++){
		*fptr			= CM_DATA( DPA_CMAP_OBJ(current_dpyp),0,i);
		*(fptr + ci)		= CM_DATA( DPA_CMAP_OBJ(current_dpyp),1,i);
		*(fptr + ci + ci)	= CM_DATA( DPA_CMAP_OBJ(current_dpyp),2,i);
		fptr += pxl_inc;
	}
}

#ifdef NOT_USED
void print_cm(QSP_ARG_DECL  u_int from, u_int to)
{
	u_int i;

	advise("Printing colormap");
	for(i=from;i<N_COLORS && i<=to;i++){
		sprintf(msg_str,"%d:\t\t%d\t%d\t%d\t%d\n",i,CM_DATA( DPA_CMAP_OBJ(current_dpyp),0,i),
			CM_DATA( DPA_CMAP_OBJ(current_dpyp),1,i),CM_DATA( DPA_CMAP_OBJ(current_dpyp),2,i),CM_DATA( DPA_CMAP_OBJ(current_dpyp),3,i));
		prt_msg(msg_str);
	}
}
#endif /* NOT_USED */

#ifdef HAVE_X11
void _select_cmap_display(QSP_ARG_DECL  Dpyable *dpyp)
{
#ifdef CAUTIOUS
	if( dpyp == NULL ){
		sprintf(ERROR_STRING,"CAUTIOUS:  select_cmap_display:  null display!?");
		warn(ERROR_STRING);
		return;
	}
#endif /* CAUTIOUS */

	current_dpyp = dpyp;		/* default_cmap() */
}

void _default_cmap(QSP_ARG_DECL  Dpyable *dpyp)
{
	current_dpyp = dpyp;		/* default_cmap() */

	if( verbose )
		advise("Initializing default color map");

	/* grayscale */
	make_grayscale(GRAYSCALE_BASE,NC_GRAYSCALE);

	/* color */
	make_rgb(COLOR_BASE,N_RED_LEVELS,N_GREEN_LEVELS,N_BLUE_LEVELS);
}
#endif /* HAVE_X11 */

int _color_index_out_of_range(QSP_ARG_DECL  unsigned int index)
{
#ifdef HAVE_X11
	// This condition can happen if the window name has a space!?
	// BUG we should replace with underscore...
	if(  DPA_CMAP_OBJ(current_dpyp) == NULL )
		return 1;

	if( /* index < 0 || */	// index is unsigned...
			index >= OBJ_COLS( DPA_CMAP_OBJ(current_dpyp)) ){

		char str[256];

		sprintf(str,"color index %d out of range for colormap %s",
			index,OBJ_NAME( DPA_CMAP_OBJ(current_dpyp)));
		WARN(str);
		return(1);
	}
#endif /* HAVE_X11 */
	return(0);
}

void set_colormap(Data_Obj *dp)
{
	/* BUG verify here that this object is a valid colormap */

#ifdef HAVE_X11
	assert(current_dpyp!=NULL);
	DPA_CMAP_OBJ(current_dpyp)=dp;
#endif /* HAVE_X11 */

}


