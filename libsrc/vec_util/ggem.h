

/* prototypes for routines hacked from graphics gems */

#include "data_obj.h"

extern void ggem_fill(QSP_ARG_DECL  int x,int y,int width,int height,int nv,int tolerance);
extern void gen_fill(incr_t x,incr_t y,Data_Obj *dp,int (*inside_func)(long,long),
	void (*fill_func)(long,long) );

