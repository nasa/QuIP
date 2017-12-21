
#include "quip_config.h"

#ifdef HAVE_X11_EXT

#include "quip_prot.h"
#include "viewer.h"
#include "xsupp.h"

Viewer * _init_shm_viewer(QSP_ARG_DECL  const char *name, int width, int height, int depth)
{
	Viewer *vp;

	vp = viewer_init(name,width,height,0);
	if( vp == NULL ) return vp;

	make_grayscale(0,256);	// do we need this here???

	show_viewer(vp);
	shm_setup(vp);

	/* use the depth of the first viewer to determine the
	 * scanning increment...
	 */

	return vp;
}

void display_to_shm_viewer(Viewer *vp,Data_Obj *dp)
{
	update_shm_viewer(vp, ((char *)OBJ_DATA_PTR(dp)),
		4 /* increment */,
		1 /* this means advance src component - ??? */,
		vp->vw_width,vp->vw_height, 0,0);
}


#endif // HAVE_X11_EXT
