/* collect all the external submenu declarations that we need */

#ifdef __cplusplus
extern "C" {
#endif


#include "query.h"

extern COMMAND_FUNC( fiomenu );

extern COMMAND_FUNC( protomenu );

extern COMMAND_FUNC( gps_menu );

extern COMMAND_FUNC( x_menu );

extern COMMAND_FUNC( genwin_menu ); 

extern COMMAND_FUNC( lutmenu ); 

extern COMMAND_FUNC( ocv_menu );

#ifdef HAVE_ADJTIMEX
extern COMMAND_FUNC( timex_menu );
#endif /* HAVE_ADJTIMEX */

// not a submenu - appears in two places???
extern COMMAND_FUNC( do_xsync );
extern COMMAND_FUNC( call_event_funcs );

/* portmenu.c */
extern COMMAND_FUNC( portmenu );

/* pp_menu.c */
extern COMMAND_FUNC( parport_menu );

extern COMMAND_FUNC( pgr_menu );

extern COMMAND_FUNC( gl_menu );

extern COMMAND_FUNC( cuda_menu );

extern COMMAND_FUNC( datamenu );

extern COMMAND_FUNC( do_exprs );

/* vl_menu.c */
extern COMMAND_FUNC( vl_menu );

extern COMMAND_FUNC( salac_menu );

extern COMMAND_FUNC( soundmenu );

extern COMMAND_FUNC( do_requant );

extern COMMAND_FUNC( warmenu );

extern COMMAND_FUNC( rv_menu );

extern COMMAND_FUNC( moviemenu );

extern COMMAND_FUNC( mseq_menu );

extern COMMAND_FUNC( stepmenu );

extern COMMAND_FUNC( knoxmenu );
extern COMMAND_FUNC( viewmenu );
extern COMMAND_FUNC( nrmenu );
extern COMMAND_FUNC( v4l2_menu );
extern COMMAND_FUNC( meteor_menu );
extern COMMAND_FUNC( visca_menu );
extern COMMAND_FUNC( dv_menu );
extern COMMAND_FUNC(mouse_menu);
extern COMMAND_FUNC( pipemenu );
extern COMMAND_FUNC( thread_menu );
extern COMMAND_FUNC( macmenu );
extern COMMAND_FUNC( ittyp_menu );
extern COMMAND_FUNC(ser_menu);
extern COMMAND_FUNC( varmenu );
extern COMMAND_FUNC( sched_menu );
extern COMMAND_FUNC( seq_menu );
extern COMMAND_FUNC( svr_menu );
extern COMMAND_FUNC( stty_menu );
extern COMMAND_FUNC( togclobber );
/* features.c */
extern COMMAND_FUNC(do_list_features);

extern COMMAND_FUNC(picmenu);

extern COMMAND_FUNC( gsl_menu );

extern COMMAND_FUNC( atc_menu );

/* dastst.c */
extern COMMAND_FUNC( aio_menu );


#ifdef __cplusplus
}
#endif

