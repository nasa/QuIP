#include "quip_config.h"

#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#include "data_obj.h"
#include "getbuf.h"
#include "debug.h"
#include "optimize.h"

/* these stay in fixed order */

#define MAX_FS_PARAMS	3

#define FREQ_NAME	"frequency"
#define PHASE_NAME	"phase"
#define AMP_NAME	"amplitude"

/* For stepit, the max and min are used, but for the simplex method they are not!
 * In the simplex method, the max's are used to set the initial simplex vertices...
 */

static Opt_Param default_freq,default_phase,default_amp;

#define FREQ_INDEX	0
#define PHASE_INDEX	1
#define AMP_INDEX	2

static int vary_flag[3]={1,1,1};

static Opt_Param *prm_pp[3];

static float *target;
static u_long nsamps;

static float the_freq,the_phase,the_amp;

static void init_sine_defaults(void);

COMMAND_FUNC( do_fix_prm )
{
	float value;
	Opt_Param *opp;

	opp = PICK_OPT_PARAM("");
	value=how_much("value");

	if( opp==NULL ) return;

	opp->ans = value;

	if( !strcmp(opp->op_name,FREQ_NAME) ){
		vary_flag[FREQ_INDEX] = 0;
	} else if( !strcmp(opp->op_name,PHASE_NAME) ){
		vary_flag[PHASE_INDEX] = 0;
	} else if( !strcmp(opp->op_name,AMP_NAME) ){
		vary_flag[AMP_INDEX] = 0;
	}
}
	
void sine_hints(float freq, float phase, float amp)
{
	prm_pp[FREQ_INDEX]->ans  = freq;
	prm_pp[PHASE_INDEX]->ans = phase;
	prm_pp[AMP_INDEX]->ans   = amp;

	the_freq  = freq;
	the_phase = phase;
	the_amp   = amp;
}

float fitsine_error()
{
	float	err;
	float arg,arginc;
	float resid;
	u_long i;

	the_freq  = prm_pp[FREQ_INDEX]->ans;
	the_amp   = prm_pp[AMP_INDEX]->ans;
	the_phase = prm_pp[PHASE_INDEX]->ans;

	arg = 8 * atan(1.0) * the_phase / 360;

	/* here we take the frequency to be in cycles per interval */
	arginc = 8*atan(1.0)*the_freq/nsamps;

	err = 0.0;
	for(i=0;i<nsamps;i++){
		resid = target[i] - the_amp * sin( arg );
		err  += resid*resid;
		arg  += arginc;
	}

	if( verbose )
		printf("fitsine_error:  freq %g   phase %g   amp %g      error %g\n",
			the_freq,the_phase,the_amp,err);

	return(err);
}

static void init_sine_defaults()
{
	/* these values used to be at a table, but that broke
	 * with the new Item struct going at the top...
	 */

	default_freq.op_name=FREQ_NAME;
	default_freq.ans = 1.0;
	default_freq.maxv = 20.0;
	default_freq.minv = 0.1;
	default_freq.delta = 0.1;
	default_freq.mindel = 0.001;

	default_phase.op_name=PHASE_NAME;
	default_phase.ans = 0.0;
	default_phase.maxv = 720.0;
	default_phase.minv = -720.0;
	default_phase.delta = 10.0;
	default_phase.mindel = 0.001;

	default_amp.op_name=AMP_NAME;
	default_amp.ans = 5.0;
	default_amp.maxv = 100.0;
	default_amp.minv = 0.1;
	default_amp.delta = 0.1;
	default_amp.mindel = 0.001;
}

COMMAND_FUNC( fitsine )
{
	Data_Obj *dp;
	char str[128];
	static int defaults_inited=0;

	if( !defaults_inited ){
		init_sine_defaults();
		defaults_inited++;
	}

	dp=PICK_OBJ( "signal vector" );
	if( dp==NULL ) return;

	sprintf(msg_str, "Attempting to fit %s by varying",OBJ_NAME(dp));
	prt_msg_frag(msg_str);

	delete_opt_params(SINGLE_QSP_ARG);

	if( vary_flag[FREQ_INDEX] )
		prm_pp[FREQ_INDEX]  = add_opt_param(&default_freq);
	else
		prm_pp[FREQ_INDEX] = &default_freq;

	if( vary_flag[PHASE_INDEX] )
		prm_pp[PHASE_INDEX]  = add_opt_param(&default_phase);
	else
		prm_pp[PHASE_INDEX]  = &default_phase;

	if( vary_flag[AMP_INDEX] )
		prm_pp[AMP_INDEX]  = add_opt_param(&default_amp);
	else
		prm_pp[AMP_INDEX]  = &default_amp;

	nsamps = OBJ_COLS(dp);
	target = (float *)(OBJ_DATA_PTR(dp));

	optimize(fitsine_error);

	the_freq = prm_pp[FREQ_INDEX]->ans;
	the_amp = prm_pp[AMP_INDEX]->ans;
	the_phase = prm_pp[PHASE_INDEX]->ans;

	if( verbose )
		printf("final values:  freq %g   phase %g   amp  %g\n",
			the_freq, the_phase, the_amp );

	while( the_phase > 360 ) the_phase -= 360;
	while( the_phase < 0 )   the_phase += 360;

	sprintf(str,"%g",the_freq);
	ASSIGN_VAR("sine_freq",str);
	sprintf(str,"%g",the_phase);
	ASSIGN_VAR("sine_phase",str);
	sprintf(str,"%g",the_amp);
	ASSIGN_VAR("sine_amp",str);
}



