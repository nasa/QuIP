#include "quip_config.h"

#include <stdio.h>
#include <math.h>
#include "quip_prot.h"
#include "cmaps.h"
#include "lut_cmds.h"

#define MAX_BIT_PLANES	8

float white_point[3] = {(float)127.0,(float)127.0,(float)127.0};

static COMMAND_FUNC( do_setwhite )
{
	float white[3];

	white[0]=(float)how_much("white point red component");
	white[1]=(float)how_much("white point green component");
	white[2]=(float)how_much("white point blue component");
	setwhite(white);
}

static COMMAND_FUNC( do_lvlspercomp )
{
	int n;

	n=(int)how_many("number of levels per component");
	set_lvls_per_comp(n);
}

static COMMAND_FUNC( do_set_bitvecs )
{
	int nplanes;
	int i;
	float vectbl[MAX_BIT_PLANES][3];

	nplanes=get_ncomps();
	for(i=0;i<nplanes;i++){
		vectbl[i][0] = (float)how_much("red component");
		vectbl[i][1] = (float)how_much("green component");
		vectbl[i][2] = (float)how_much("blue component");
	}
	set_bit_vecs(vectbl);
}

static COMMAND_FUNC( do_bitplanes )
{
	int nplanes;
	int i;
	float amplist[MAX_BIT_PLANES];

	nplanes = (int)how_many("number of image components");
	for(i=0;i<nplanes;i++)
		amplist[i]=(float)how_much("vector amplitude");
	set_bitplanes(nplanes, amplist);
}

static COMMAND_FUNC( do_set_base )
{
	int i;

	i=(int)how_many("index of base color");
	if( i< 0 || i > (N_COLORS-1) ){
		warn("invalid base index");
		return;
	}
	set_base_index(i);
}

#ifdef FOOBAR
static COMMAND_FUNC( do_drift_plane )
{
	int k;
	struct drift_plane dp;
	float vectbl[MAX_BIT_PLANES][3];

	/* Sets base_index */
	do_set_base();

	/* Set white point r,g,b */
	do_setwhite();

	dp.ndirections	= how_many("number of directions");

	if (dp.ndirections > MAX_DIRECTIONS) {
		sprintf(w.w_str,"ndirections set to max value of %d",
			MAX_DIRECTIONS);
		warn(w.w_str);
		dp.ndirections = MAX_DIRECTIONS;
	}

	for (k=0; k<dp.ndirections; k++) {
		advise("");
		sprintf(w.w_str,"For direction #%d enter:",k);
		advise(w.w_str);
		dp.direction[k] = (float)how_much("direction");
		dp.segprefix[k] = savestr(nameof(
			"direction segment buffer name (sans extension)"));
	}
	advise("");

	dp.ngratings = how_many("number of gratings");

	dp.nplanes	= 2 * dp.ngratings;
	for(k=0;k<dp.nplanes;k++) {
		vectbl[k][0] = (float)(white_point[0] / sqrt(2.0));
		vectbl[k][1] = (float)(white_point[1] / sqrt(2.0));
		vectbl[k][2] = (float)(white_point[2] / sqrt(2.0));
	}

	set_bit_vecs(dp.nplanes,vectbl);

	dp.rduration  = how_many("ramp duration (time in frames)");
	dp.pduration  = how_many("peek duration (time in frames)");
	dp.nrefresh  = how_many("number of refreshes (vblanks, delays)");

	if (dp.nplanes > MAX_BIT_PLANES) {
		sprintf(w.w_str,
		"nplanes set to max value of %d", MAX_BIT_PLANES);
		warn(w.w_str);
		dp.nplanes = MAX_BIT_PLANES;
	}

	for (k=0; k<dp.ngratings; k++) {
		advise("");
		sprintf(w.w_str,"For grating #%d enter:",k);
		advise(w.w_str);
		dp.period[k] = how_many("period of grating in frames");
		dp.contrast[k] = (float)how_much("amplitude contrast");
		dp.start_phase[k] = (float)how_much("start phase in degs.");
	}

	make_drift_plane(&dp);
	for (k=0; k<dp.ndirections; k++)
		givbuf((char *)dp.segprefix[k]);
}
#endif

static COMMAND_FUNC( do_sine_mod )
{
	int nf,per;
	int nc;
	float phase[8];
	const char *s;
	int i;

	nf=(int)how_many("number of frames");
	per=(int)how_many("period in frames");
	s=nameof("lut buffer name stem");

	nc=get_ncomps();
	for(i=0;i<nc;i++)
		phase[i] = (float)(i*2*atan(1));

	sine_mod_amp(nf,phase,per,(float *)NULL,s);
}

static COMMAND_FUNC( do_set_ncomps )
{
	int nc;

	nc=(int)how_many("number of image components");
	set_ncomps(nc);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(bitplanes_menu,s,f,h)

MENU_BEGIN(bitplanes)
ADD_CMD( ncomps,	do_set_ncomps,	set number of image components )
ADD_CMD( amplitudes,	do_bitplanes,	set image component amplitudes )
ADD_CMD( base_index,	do_set_base,	set base index for image components )
ADD_CMD( nlevels,	do_lvlspercomp,	specify number of levels per image component )
#ifdef FOOBAR
ADD_CMD( drift_plane,	do_drift_plane,	plane drifting )
#endif
ADD_CMD( vectors,	do_set_bitvecs,	specify bitplane color vectors )
ADD_CMD( white,		do_setwhite,	set white point for modulation )
ADD_CMD( sine,		do_sine_mod,	sinusoidal modulation )
MENU_END(bitplanes)

COMMAND_FUNC( do_bit_menu )
{
	CHECK_AND_PUSH_MENU(bitplanes);
}

