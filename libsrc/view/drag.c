#include "quip_config.h"

char VersionId_viewer_drag[] = QUIP_VERSION_STRING;

#ifdef HAVE_X11

/* draggable objects for xlib !? */

#include "data_obj.h"
#include "getbuf.h"
#include "savestr.h"
#include "node.h"
#include "viewer.h"
#include "items.h"

ITEM_INTERFACE_DECLARATIONS(Draggable,dragg)

void make_dragg(QSP_ARG_DECL  const char *name,Data_Obj *bm,Data_Obj *dp)
{
	Draggable *dgp;

	if( !dp_same_size(QSP_ARG  bm,dp,"make_dragg") ){
		WARN("image/bitmap size mismatch");
		return;
	}
	if( bm->dt_prec != PREC_BIT ){
		sprintf(error_string,"Object %s has precision %s, should be %s",
			bm->dt_name,name_for_prec(bm->dt_prec),name_for_prec(PREC_BIT));
		WARN(error_string);
		return;
	}
	if( dp->dt_prec != PREC_BY && dp->dt_prec != PREC_UBY ){
		sprintf(error_string,"Image %s (for draggable object) has %s precision, should be %s or %s",
			dp->dt_name,name_for_prec(dp->dt_prec),
			name_for_prec(PREC_BY),name_for_prec(PREC_UBY));
		WARN(error_string);
		return;
	}

	dgp = new_dragg(QSP_ARG  name);
	if( dgp == NO_DRAGG ) return;

	dgp->dg_width = (int) dp->dt_cols;
	dgp->dg_height = (int) dp->dt_rows;
	dgp->dg_bitmap = bm;
	dgp->dg_image = dp;
	dgp->dg_np = mk_node(dgp);
}

#endif /* HAVE_X11 */

