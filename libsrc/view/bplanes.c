#include "quip_config.h"

/* bplanes.c	non-interactive subroutine library */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include "data_obj.h"
#include "quip_prot.h"
#ifndef FALSE
#define  FALSE 0
#endif /* FALSE */
#include "linear.h"
#include "cmaps.h"

#define MAX_COMPS		8
#define MAX_LEVELS		16

static float	white_point[3]={(float)127.0,(float)127.0,(float)127.0};
static float	vector_table[MAX_COMPS][3];
static int	base_index=0;
static int	lvls_per_comp=2;
static int	n_comps=(-1);
static float	mult[MAX_LEVELS];	/* goes from -1 to 1, lvls_per_comp-1 steps */
static float *	amplist;
/* the digits array is used to know - what? */
/* it indexes the mult array...
 */
static int 	digits[MAX_COMPS];

#ifdef SINE_TBL
extern double t_sin(double);
extern double t_cos(double);
#endif /* SINE_TBL */

void set_lvls_per_comp(int n)
{
	char str[256];

	if( n < 2 || n > MAX_LEVELS ){
		sprintf(str, "bad number of bits per component, using %d",
			MAX_LEVELS);
		NWARN(str);
		n = MAX_LEVELS;
	}  
	lvls_per_comp = n ;
}

/* Set a single color 
 * the digits array tells us the level of each component
 */

void set_c_amps(QSP_ARG_DECL  int index)
{
	float fctr, color[3];
	int i,r,g,b;

	color[0] = white_point[0];
	color[1] = white_point[1];
	color[2] = white_point[2];
	for(i=0;i<n_comps;i++){
		fctr = amplist[i] * mult[digits[i]];
		color[0] += fctr * vector_table[i][0];
		color[1] += fctr * vector_table[i][1];
		color[2] += fctr * vector_table[i][2];
	}
	r=(int)(color[0]+0.5);
	g=(int)(color[1]+0.5);
	b=(int)(color[2]+0.5);
	setcolor(index,r,g,b);
}

/* What does this do?
 *
 * digit is the place (bit), we call this once for each component
 */

void count(QSP_ARG_DECL  int digit,int offset)
{
	int i,osi;	/* offset increment */

	osi=1;
	for(i=0;i<digit;i++)
		osi *= lvls_per_comp;

	for(i=0;i<lvls_per_comp;i++){
		digits[digit] = i;
		if( digit == 0 ){	/* ones place, doit */
			set_c_amps(QSP_ARG  offset+i);
		} else {
			count(QSP_ARG  digit-1,offset+i*osi);
		}
	}
}

void set_ncomps(int n)
{
	char str[256];

	if( n > MAX_COMPS ) {
		sprintf(str,
			"too many image components specified, using %d",
			MAX_COMPS);
		NWARN(str);
		n = MAX_COMPS;
	} else if( n < 2 ){
		NWARN("must be at least two components, set to 2");
		n = 2;
	}
	n_comps = n;
}

int get_ncomps(void)
{
	return(n_comps);
}

void set_comp_amps(QSP_ARG_DECL  float *amps)
{
	int i;
	float minc;

	if( n_comps < 0 ){
		NWARN("must specify number of components");
		return;
	}

	/* set up multipliers */
	minc = (float)(2.0 / ((float)lvls_per_comp-1.0));
	mult[0] = (float)-1.0;
	for(i=1;i<lvls_per_comp;i++)
		mult[i] = mult[i-1] + minc;

	amplist = amps;

	/* count base n */

	push_cm_state();
	CLR_CM_STATE(IMMEDIATE);
	count(QSP_ARG  n_comps-1,base_index);
	pop_cm_state();

	update_if();
}

void sine_mod_amp(QSP_ARG_DECL  int nframes,float *phases,int period,float *envelope,const char *lutstem)
{
	float amps[MAX_COMPS];
	float arginc;
	float factor;
	char str[32];
	int i,j;

	arginc = (float)(8*atan(1)/period);
	for(i=0;i<nframes;i++){
		sprintf(str,"%s%d",lutstem,i);
		if( new_colormap(QSP_ARG  str) == NULL )
			NERROR1("error creating LUT buffer");
		if( envelope != ((float *)NULL) )
			factor=envelope[i];
		else
			factor=1;
		for(j=0;j<n_comps;j++){
#ifdef SINE_TBL
			amps[j]=(float)(factor*t_sin(phases[j]));
#else /* ! SINE_TBL */
			amps[j]=(float)(factor*sin(phases[j]));
#endif /* ! SINE_TBL */
			phases[j]+=arginc;
		}
		set_comp_amps(QSP_ARG  amps);
		index_alpha(i,0,255);
	}
}


void set_base_index(int i)
{
	if( i<0 || i>DACMAX ) {
		NWARN("base value out of range, using 0");
		base_index=0;
		return;
	}
	base_index=i;
}

int get_base_index(void)
{
	return(base_index);
}

void setwhite(float *white)
{
	white_point[0] = white[0];
	white_point[1] = white[1];
	white_point[2] = white[2];
	SET_CM_FLAG( SETWHITE );
}

void set_bit_vecs( float veclist[MAX_COMPS][3] )
{
	int i;

	if( n_comps < 0 ){
		NWARN("must set number of components before specifing vectors");
		return;
	}
	for(i=0;i<n_comps;i++){
		vector_table[i][0] = veclist[i][0];
		vector_table[i][1] = veclist[i][1];
		vector_table[i][2] = veclist[i][2];
	}  
	SET_CM_FLAG( SETBITVECS );
}

/* backwards compatibility */

void set_bitplanes(QSP_ARG_DECL  int nplanes,float *amps)
{
	/* BUG here is where we should be checking status for white & vectors */

	set_ncomps(nplanes);
	set_comp_amps(QSP_ARG  amps);
}

void set_bits_per_comp(int n)
{
	int nl=1;

	while(n--)
		nl *= 2;

	set_lvls_per_comp(nl);
}

