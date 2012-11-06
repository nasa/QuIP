#include "quip_config.h"

char VersionId_dither_ctone[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_MATH_H
#include <math.h>
#endif


extern float lum_weight, rg_weight, by_weight;
#include "ctone.h"
#include "qlevel.h"

/* globals - do we need them all? */
char *Progname;
char qname[80];
int opicfd[3], picfd[3];

int nlevels;

COMMAND_FUNC( getqf )
{
	strcpy(qname,NAMEOF("quantization file"));
}

COMMAND_FUNC( set_weights )
{
	lum_weight=HOW_MUCH("weight for luminance");
	rg_weight=HOW_MUCH("weight for R-G");
	by_weight=HOW_MUCH("weight for B-Y");
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

	nlevels=HOW_MANY("number of quantization levels");
	for(i=0;i<nlevels;i++){
		sprintf(ERROR_STRING,"value for level %d",i+1);
		quant_level[i] = HOW_MANY(ERROR_STRING);
	}
}

static Command opt_ctbl[]={
{ "weights",	set_weights,	"specify weighting factors"	},
{ "levels",	set_levels,	"set the number and values of quantization levels"	},
{ "scan",	do_ctone,	"process an image"		},
{ "white",	getwhite,	"specify white point"		},
{ "lumscal",	set_lumscal,	"specify luminance scaling factors"		},
{ "matrices",	set_matrices,	"specify objects to use for color transformation matrices"		},
{ "quit",	popcmd,		"exit submenu"			},
{ NULL_COMMAND							}
};

COMMAND_FUNC( ctone_menu )
{
	PUSHCMD(opt_ctbl,"options");
}



