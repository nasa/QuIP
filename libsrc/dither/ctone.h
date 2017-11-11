#include "data_obj.h"

#define MAXCOLS	512

extern COMMAND_FUNC( getwhite );
extern void rgb2o(QSP_ARG_DECL  float *);
extern void o2rgb(QSP_ARG_DECL  float *);

/* globals */

//extern int thebest[3];
//extern float desired[3][MAXCOLS];
extern int know_white;

/* qinit.c */

extern void qinit(char *);

/* cdiff.c */
extern void ctoneit(QSP_ARG_DECL  Data_Obj *,Data_Obj *);

/* getbest.c */
extern int getbest(QSP_ARG_DECL  int);
#define showvec(p) _showvec(QSP_ARG  p)
extern void _showvec(QSP_ARG_DECL  float *);

/* ctone.c */
extern COMMAND_FUNC( do_ctone_menu );

/* rb2rgb.c */
extern COMMAND_FUNC( set_matrices );
extern COMMAND_FUNC( set_lumscal );


