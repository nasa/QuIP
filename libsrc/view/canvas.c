#include "quip_config.h"

char VersionId_viewer_canvas[] = QUIP_VERSION_STRING;

#ifdef HAVE_X11

#include <stdio.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include "xsupp.h"
#include "getbuf.h"
#include "savestr.h"
#include "debug.h"
#include "viewer.h"

void add_image( Viewer *vp, Data_Obj *dp, int x, int y )
{
	Window_Image *wip;
	Node *np;

	wip=(Window_Image *)getbuf(sizeof(*wip));
	wip->wi_dp = dp;
	wip->wi_x = x;
	wip->wi_y = y;
	np = mk_node(wip);
	addTail(vp->vw_image_list,np);
}

void old_load_viewer( QSP_ARG_DECL  Viewer *vp, Data_Obj *dp )
{
	dimension_t i;

	/* We don't need to care if the size isn't an exact match */
	/*
	if( dp->dt_cols != vp->vw_width ||
		dp->dt_rows != vp->vw_height ){
		sprintf(error_string,
		"Size mismatch between viewer %s and image %s",
			vp->vw_name, dp->dt_name);
		WARN(error_string);
		return;
	}
	*/

	/*
	if( (8*dp->dt_tdim) != vp->vw_depth ){
		sprintf(error_string,
		"Depth mismatch between image %s (%ld) and viewer %s (%d)",
			dp->dt_name,8*dp->dt_tdim,vp->vw_name,vp->vw_depth);
		WARN(error_string);
		return;
	}
	*/

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(error_string,
		"Can't display non-contiguous image %s (viewer %s)",
			dp->dt_name,vp->vw_name);
		WARN(error_string);
		LONGLIST(dp);
		return;
	}
	if( MACHINE_PREC(dp) != PREC_BY && MACHINE_PREC(dp) != PREC_UBY ){
		sprintf(error_string,"Bad pixel format (%s), image %s",
				name_for_prec(dp->dt_prec),dp->dt_name);
		WARN(error_string);
		advise("Only byte images may be displayed in viewers");
		return;
	}
	if( IS_DRAGSCAPE(vp) ){
		zap_image_list(vp);
		add_image(vp,dp,0,0);
	} else {
		/* If we are holding an image, release it */
		if( vp->vw_dp != NO_OBJ )
			release_image(QSP_ARG  vp->vw_dp);
		vp->vw_dp=dp;
		/* make sure this image doesn't get deleted out from under us */
		dp->dt_refcount++;
	}

	for(i=0;i<dp->dt_frames;i++){
		vp->vw_frameno=i;
		//refresh_image(QSP_ARG  vp);
		usleep(16000);	/* approx 16 msec */
	}
	select_viewer(QSP_ARG  vp);
}

void load_viewer( QSP_ARG_DECL  Viewer *vp, Data_Obj *dp )
{
	if( dp->dt_frames != 1 ){
sprintf(error_string,"load_viewer:  Object %s has %d frames, calling old_load_viewer().",
dp->dt_name,dp->dt_frames);
advise(error_string);
		old_load_viewer(QSP_ARG  vp,dp);
		return;
	}

	/* This caused a memory leak bug - add a zap_image_list()
	 * if we want this here...
	 */

	/* add_image(vp,dp,0,0); */

	embed_image(QSP_ARG  vp,dp,0,0);
}


#endif /* HAVE_X11 */

