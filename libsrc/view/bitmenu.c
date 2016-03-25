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

	white[0]=(float)HOW_MUCH("white point red component");
	white[1]=(float)HOW_MUCH("white point green component");
	white[2]=(float)HOW_MUCH("white point blue component");
	setwhite(white);
}

static COMMAND_FUNC( do_lvlspercomp )
{
	int n;

	n=(int)HOW_MANY("number of levels per component");
	set_lvls_per_comp(n);
}

static COMMAND_FUNC( do_set_bitvecs )
{
	int nplanes;
	int i;
	float vectbl[MAX_BIT_PLANES][3];

	nplanes=get_ncomps();
	for(i=0;i<nplanes;i++){
		vectbl[i][0] = (float)HOW_MUCH("red component");
		vectbl[i][1] = (float)HOW_MUCH("green component");
		vectbl[i][2] = (float)HOW_MUCH("blue component");
	}
	set_bit_vecs(vectbl);
}

static COMMAND_FUNC( do_bitplanes )
{
	int nplanes;
	int i;
	float amplist[MAX_BIT_PLANES];

	nplanes = (int)HOW_MANY("number of image components");
	for(i=0;i<nplanes;i++)
		amplist[i]=(float)HOW_MUCH("vector amplitude");
	set_bitplanes(QSP_ARG  nplanes, amplist);
}

static COMMAND_FUNC( do_set_base )
{
	int i;

	i=(int)HOW_MANY("index of base color");
	if( i< 0 || i > (N_COLORS-1) ){
		WARN("invalid base index");
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

	dp.ndirections	= HOW_MANY("number of directions");

	if (dp.ndirections > MAX_DIRECTIONS) {
		sprintf(w.w_str,"ndirections set to max value of %d",
			MAX_DIRECTIONS);
		WARN(w.w_str);
		dp.ndirections = MAX_DIRECTIONS;
	}

	for (k=0; k<dp.ndirections; k++) {
		advise("");
		sprintf(w.w_str,"For direction #%d enter:",k);
		advise(w.w_str);
		dp.direction[k] = (float)HOW_MUCH("direction");
		dp.segprefix[k] = savestr(NAMEOF(
			"direction segment buffer name (sans extension)"));
	}
	advise("");

	dp.ngratings = HOW_MANY("number of gratings");

	dp.nplanes	= 2 * dp.ngratings;
	for(k=0;k<dp.nplanes;k++) {
		vectbl[k][0] = (float)(white_point[0] / sqrt(2.0));
		vectbl[k][1] = (float)(white_point[1] / sqrt(2.0));
		vectbl[k][2] = (float)(white_point[2] / sqrt(2.0));
	}

	set_bit_vecs(dp.nplanes,vectbl);

	dp.rduration  = HOW_MANY("ramp duration (time in frames)");
	dp.pduration  = HOW_MANY("peek duration (time in frames)");
	dp.nrefresh  = HOW_MANY("number of refreshes (vblanks, delays)");

	if (dp.nplanes > MAX_BIT_PLANES) {
		sprintf(w.w_str,
		"nplanes set to max value of %d", MAX_BIT_PLANES);
		WARN(w.w_str);
		dp.nplanes = MAX_BIT_PLANES;
	}

	for (k=0; k<dp.ngratings; k++) {
		advise("");
		sprintf(w.w_str,"For grating #%d enter:",k);
		advise(w.w_str);
		dp.period[k] = HOW_MANY("period of grating in frames");
		dp.contrast[k] = (float)HOW_MUCH("amplitude contrast");
		dp.start_phase[k] = (float)HOW_MUCH("start phase in degs.");
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

	nf=(int)HOW_MANY("number of frames");
	per=(int)HOW_MANY("period in frames");
	s=NAMEOF("lut buffer name stem");

	nc=get_ncomps();
	for(i=0;i<nc;i++)
		phase[i] = (float)(i*2*atan(1));

	sine_mod_amp(QSP_ARG  nf,phase,per,(float *)NULL,s);
}

static COMMAND_FUNC( do_set_ncomps )
{
	int nc;

	nc=(int)HOW_MANY("number of image components");
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
	PUSH_MENU(bitplanes);
}

