
/* globals used just in this module */
extern u_long xdebug;
extern int simulating_luts;

/* dpy.c */
extern List *displays_list(SINGLE_QSP_ARG_DECL);

/* event.c */
extern void i_loop(SINGLE_QSP_ARG_DECL);
extern void discard_events(SINGLE_QSP_ARG_DECL);

/* lut_xlib.c */
extern void set_curr_win(Window win);
extern u_long simulate_lut_mapping(Viewer *vp, u_long color);

