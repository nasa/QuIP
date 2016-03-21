#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_MATH_H
#include <math.h>
#endif


extern float lum_weight, rg_weight, by_weight;
#include "quip_prot.h"
#include "ctone.h"
#include "qlevel.h"

/* globals - do we need them all? */
//static char *Progname;
//static int opicfd[3], picfd[3];

//int nlevels;

#ifdef NOT_USED
char qname[80];
COMMAND_FUNC( getqf )
{
	strcpy(qname,NAMEOF("quantization file"));
}
#endif /* NOT_USED */

static COMMAND_FUNC( set_weights )
{
	lum_weight=(float)HOW_MUCH("weight for luminance");
	rg_weight=(float)HOW_MUCH("weight for R-G");
	by_weight=(float)HOW_MUCH("weight for B-Y");
}

static COMMAND_FUNC( do_ctone )
{
	Data_Obj *dpto,*dpfr;

	dpto = PICK_OBJ("destination image");
	dpfr = PICK_OBJ("source image");
	if( dpto == NO_OBJ || dpfr == NO_OBJ ) return;
	ctoneit(QSP_ARG  dpto,dpfr);
}

static COMMAND_FUNC( set_levels )
{
	int i;

	nlevels=(int)HOW_MANY("number of quantization levels");
	for(i=0;i<nlevels;i++){
		sprintf(ERROR_STRING,"value for level %d",i+1);
		quant_level[i] = HOW_MANY(ERROR_STRING);
	}
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(options_menu,s,f,h)

MENU_BEGIN(options)
ADD_CMD( weights,	set_weights,	specify weighting factors )
ADD_CMD( levels,	set_levels,	set the number and values of quantization levels )
ADD_CMD( scan,	do_ctone,	process an image )
ADD_CMD( white,	getwhite,	specify white point )
ADD_CMD( lumscal,	set_lumscal,	specify luminance scaling factors )
ADD_CMD( matrices,	set_matrices,	specify objects to use for color transformation matrices )
MENU_END(options)

COMMAND_FUNC( do_ctone_menu )
{
	PUSH_MENU(options);
}



