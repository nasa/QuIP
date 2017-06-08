#include "quip_config.h"

#include <stdio.h>
#include "debug.h"
#include "data_obj.h"
#include "quip_menu.h"
#include "quip_prot.h"
#include "getbuf.h"
//#include "dataprot.h"
//#include "menuname.h"
//#include "rn.h"			/* set_seed() */

static COMMAND_FUNC( do_getmean );
static COMMAND_FUNC( do_equate );
static COMMAND_FUNC( do_cpybuf );
static COMMAND_FUNC( do_rnd );
static COMMAND_FUNC( do_uni );

static COMMAND_FUNC( do_getmean )
{
	Data_Obj *dp;

	dp=PICK_OBJ("");
	getmean(QSP_ARG  dp);
}

static COMMAND_FUNC( do_equate )
{
	Data_Obj *dp;
	double v;

	dp=PICK_OBJ("");
	v=HOW_MUCH("value");
	if( dp==NULL ) return;

	dp_equate(QSP_ARG  dp,v);
}

static COMMAND_FUNC( do_cpybuf )
{
	Data_Obj *dp_fr, *dp_to;

	dp_to=PICK_OBJ("destination data object");
	dp_fr=PICK_OBJ("source data object");
	if( dp_to==NULL || dp_fr==NULL ) return;

	dp_copy(QSP_ARG  dp_to,dp_fr);
}

static COMMAND_FUNC( do_rnd )
{
	Data_Obj *dp;
	int imax,imin;

	dp=PICK_OBJ("byte buffer" );
	imin=(int)HOW_MANY("minimum random value");
	imax=(int)HOW_MANY("maximum random value");
	if( dp==NULL ) return;
	i_rnd(QSP_ARG  dp,imin,imax);
}

#ifdef NOT_YET
static COMMAND_FUNC( do_seed )
{
	long n;

	n=HOW_MANY("seed for uniform random number generator");
	set_seed(QSP_ARG  n);	/* make sure proper flag is set, see rn.c */
}
#endif /* NOT_YET */

static COMMAND_FUNC( do_uni )
{
	Data_Obj *dp;

	/* need to seed this generator... */

	dp=PICK_OBJ("float data object");
	if( dp==NULL ) return;
	dp_uni(QSP_ARG  dp);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(memops_menu,s,f,h)

MENU_BEGIN(memops)

ADD_CMD( copy,		do_cpybuf,	copy buffer	)
ADD_CMD( equate,	do_equate,	set memory buffer to constant value	)
ADD_CMD( mean,		do_getmean,	compute mean of buffer	)
ADD_CMD( randomize,	do_rnd,		randomize byte buffer	)
ADD_CMD( uniform,	do_uni,		randomize float buffer	)
#ifdef NOT_YET
ADD_CMD( seed_uni,	do_seed,	set seed for uniform generator	)
#endif /* NOT_YET */

MENU_END(memops)


COMMAND_FUNC( buf_ops )
{
	PUSH_MENU(memops);
}

