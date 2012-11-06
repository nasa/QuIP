#include "quip_config.h"

char VersionId_luts_linear[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "debug.h"
#include "cmaps.h"
#include "linear.h"
#include "savestr.h"
#include "items.h"

#define MAX_CHAR		80

/* global */
/*short lintbl[N_COMPS][MAX_LIN_LVLS]; */
Data_Obj *default_lt_dp=NO_OBJ;		/* BUG this object needs to be locked, or something... */
					/* a user could delete it from the data menu, creating a
					 * dangling pointer.
					 */

int phosmax=PHOSMAX;

static u_int n_lin_lvls=N_LIN_LVLS;

#define CURR_LIN_DATA(comp,index)		LT_DATA(current_dpyp->c_lt_dp,comp,index)

Data_Obj *new_lintbl( QSP_ARG_DECL  const char * name )
{
	Data_Obj *lt_dp;

	//CHECK_DPY("new_lintbl")

#ifdef HAVE_CUDA
	push_data_area(ram_area);
#endif
	lt_dp = mk_vec(QSP_ARG  name,n_lin_lvls,N_COMPS,PREC_UIN);
#ifdef HAVE_CUDA
	pop_data_area();
#endif
	if( lt_dp == NO_OBJ ) return(lt_dp);

	/*lin_setup(lt_dp,DEF_GAM,DEF_VZ); */		/* seems a bit wasteful? */
	return(lt_dp);
}

void set_n_linear(QSP_ARG_DECL  int n)
{
	if( n < 2 ){
		WARN("number of linearization levels must be > 1");
		return;
	}
	if( n >= MAX_LIN_LVLS ){
		sprintf(error_string,
	"Number of linearization levels resetricted to a max of %d",
			MAX_LIN_LVLS);
		WARN(error_string);
		n = MAX_LIN_LVLS;
	}
	sprintf(error_string,
		"Previous number of linearization levels was %d",n_lin_lvls);
	advise(error_string);
	n_lin_lvls = n;
	phosmax = n-1;
}

void lin_setup(QSP_ARG_DECL  Data_Obj *lt_dp,double gam,double vz)
{
	double b,e,k,x;
	u_int i,j;
	long stat=LININIT;
	u_short *sp,d;

	if( verbose ){
		sprintf(error_string,
			"Setting up linearization table with %d entries",
			n_lin_lvls);
		advise(error_string);
		sprintf(error_string,"gamma = %g,   vzero = %g",gam,vz);
		advise(error_string);
	}

	/* insure that an input of phosmax results in something slightly
	 * less than DACMAX */

	k = (double)(phosmax+1) ;
	k /= pow((double)((double)DACMAX-vz),(double)gam);

	e = 1.0 / (double)gam ;

	sp = (u_short *) lt_dp->dt_data;

	for(j=0;j<N_COMPS;j++)
		LT_DATA(lt_dp,j,0)=0;
	for(i=1;i<n_lin_lvls;i++){
		b=((double)i)/k;
		x=pow( b, e );
		d = (x + (double)vz);
		/* used to check for positive, but now d is unsigned */
		if( d > DACMAX ){
			sprintf(error_string,"lin_setup value out of range:  i=%d  tbl=%d",i,d);
			WARN(error_string);
			stat=0;
		}
		for(j=0;j<N_COMPS;j++)
			LT_DATA(lt_dp,j,i) = d;
	}
	SET_CM_FLAG( stat );
}

void lininit(SINGLE_QSP_ARG_DECL)
{
	CHECK_DPY("lininit")

#ifdef HAVE_X11
	lin_setup(QSP_ARG  current_dpyp->c_lt_dp,DEF_GAM,DEF_VZ);
#endif /* HAVE_X11 */
}

#ifdef HAVE_X11
void install_default_lintbl(QSP_ARG_DECL  Dpyable *dpyp)
{
	if(default_lt_dp==NO_OBJ){
		default_lt_dp = new_lintbl(QSP_ARG  "default_lintbl");
		lin_setup(QSP_ARG  default_lt_dp,DEF_GAM,DEF_VZ);		/* seems a bit wasteful? */
	}
	dpyp->c_lt_dp = default_lt_dp;
	default_lt_dp->dt_refcount ++;
}
#endif /* HAVE_X11 */

#ifdef FOOBAR
void linrd(name)
char *name;
{
	u_int i,j;
	FILE	*fp;
	long stat=(LINRD|LININIT);
	int factor;
	u_int n;
	int value;

	/* this should be N_COMPS, but we have no lin data for alpha */

	for(j=0;j< 3 /* N_COMPS */ ;j++) {
		char filename[256];

		sprintf(filename, "%s.%c", name, '0' + j);

		if(!(fp = tryopen(filename, "r"))) {
			stat=0;
			continue;
		}

		sprintf(error_string,"Reading linearization data from %s",
			filename);
		advise(error_string);

		n=0;
		while( fscanf(fp,"%d",&value) == 1 ){
			if( value < 0 || value > DACMAX ) {
				sprintf(error_string,
			"linearization value out of range:  i=%d  tbl=%d",
					n,value);
				warn (error_string);
				stat=0;
			}
			if( n < MAX_LIN_LVLS )
				CURR_LIN_DATA(j,n) = value;
			n++;
		}
		if( j== 0 )
			n_lin_lvls = n;
		else if( n != n_lin_lvls )
	NWARN("component linearization files have different numbers of values");

		fclose(fp);
	}

	/* set alpha to a ramp */

	factor = n_lin_lvls/DACMAX;
	for(i=0;i<n_lin_lvls;i++)
		CURR_LIN_DATA(3,i)=i/factor;

	SET_CM_FLAG( stat );
}

void linwt(name)
char *name;
{
	u_int i,j;
	FILE	*fp;

	for(j=0;j<N_COMPS;j++) {
		char filename[256];

		sprintf(filename, "%s.%c", name, '0' + j);

		if(!(fp = TRYNICE(filename, "w"))) {
			continue;
		}

		sprintf(error_string,
			"Writing linearization data to %s",filename);
		advise(error_string);

		for(i=0;i<n_lin_lvls;i++) 
			fprintf(fp,"%d\n",CURR_LIN_DATA(j,i));
		fclose(fp);
	}
}

void print_lin(start, n)
u_int start, n;
{
	u_int i;

	CHECK_DPY("print_lin")

	if( CM_FLAG_IS_CLR( LININIT ) )
		lin_setup(current_dpyp->c_lt_dp,DEF_GAM,DEF_VZ);

	if( start < 0 || start >= n_lin_lvls ){
		sprintf(error_string,"Start value %d must be in the range 0-%d",
			start,n_lin_lvls-1);
		NWARN(error_string);
		return;
	}
	if( n <= 0 ){
		NWARN("number of entries to print must be positive");
		return;
	}
	if( start+n > n_lin_lvls ){
		NWARN("requested linearization entries out of range");
		return;
	}

	for(i=start;i<n_lin_lvls && i<=start+n;i++){

		sprintf(msg_str,"%d:\t\t%d\t%d\t%d",
			i,CURR_LIN_DATA(0,i),CURR_LIN_DATA(1,i),CURR_LIN_DATA(2,i));
		prt_msg(msg_str);
	}
}
#endif /* FOOBAR */


#ifdef FOOBAR
/*
 * Install the default linearization if none has been read in or set
 */

void insure_linearization()
{
	if( CM_FLAG_IS_CLR( LININIT ) ){
		if( verbose )
			advise("LINCHK:  installing default linearization!");
		lin_setup(current_dpyp->c_lt_dp,DEF_GAM,DEF_VZ);
	}
}
#endif /* FOOBAR */


void set_lintbl(QSP_ARG_DECL  Data_Obj *lt_dp)
{
	CHECK_DPY("set_lintbl")

	/* BUG check that this object is a valid lintbl */
#ifdef HAVE_X11
	current_dpyp->c_lt_dp = lt_dp;
#endif /* HAVE_X11 */
}

