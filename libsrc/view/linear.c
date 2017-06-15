#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "cmaps.h"
#include "check_dpy.h"
#include "linear.h"

#define MAX_CHAR		80

/* global */
/*short lintbl[N_COMPS][MAX_LIN_LVLS]; */
Data_Obj *default_lt_dp=NULL;		/* BUG this object needs to be locked... */
					/* a user could delete it from the data menu,
					 * creating a dangling pointer.
					 */

int phosmax=PHOSMAX;

static u_int n_lin_lvls=N_LIN_LVLS;

#define CURR_LIN_DATA(comp,index)	LT_DATA( DPA_LINTBL_OBJ(current_dpyp),comp,index)

Data_Obj *new_lintbl( QSP_ARG_DECL  const char * name )
{
	Data_Obj *lt_dp;

	//CHECK_DPY("new_lintbl")

#ifdef HAVE_CUDA
	push_data_area(ram_area_p);
#endif
	lt_dp = mk_vec(QSP_ARG  name,n_lin_lvls,N_COMPS,PREC_FOR_CODE(PREC_UIN));
#ifdef HAVE_CUDA
	pop_data_area();
#endif
	if( lt_dp == NULL ) return(lt_dp);

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
		sprintf(ERROR_STRING,
	"Number of linearization levels resetricted to a max of %d",
			MAX_LIN_LVLS);
		WARN(ERROR_STRING);
		n = MAX_LIN_LVLS;
	}
	sprintf(ERROR_STRING,
		"Previous number of linearization levels was %d",n_lin_lvls);
	advise(ERROR_STRING);
	n_lin_lvls = n;
	phosmax = n-1;
}

void lin_setup(QSP_ARG_DECL  Data_Obj *lt_dp,double gam,double vz)
{
	double b,e,k,x;
	u_int i,j;
	long stat=LININIT;
	//u_short *sp;
    u_short d;

	if( verbose ){
		sprintf(ERROR_STRING,
			"Setting up linearization table with %d entries",
			n_lin_lvls);
		advise(ERROR_STRING);
		sprintf(ERROR_STRING,"gamma = %g,   vzero = %g",gam,vz);
		advise(ERROR_STRING);
	}

	/* insure that an input of phosmax results in something slightly
	 * less than DACMAX */

	k = (double)(phosmax+1) ;
	k /= pow((double)((double)DACMAX-vz),(double)gam);

	e = 1.0 / (double)gam ;

	//sp = (u_short *) OBJ_DATA_PTR(lt_dp);

	for(j=0;j<N_COMPS;j++)
		LT_DATA(lt_dp,j,0)=0;
	for(i=1;i<n_lin_lvls;i++){
		b=((double)i)/k;
		x=pow( b, e );
		d = (u_short)(x + (double)vz);
		/* used to check for positive, but now d is unsigned */
		if( d > DACMAX ){
			sprintf(ERROR_STRING,"lin_setup value out of range:  i=%d  tbl=%d",i,d);
			WARN(ERROR_STRING);
			stat=0;
		}
		for(j=0;j<N_COMPS;j++)
			LT_DATA(lt_dp,j,i) = d;
	}
	SET_CM_FLAG( stat );
}

#ifdef NOT_USED
void lininit(SINGLE_QSP_ARG_DECL)
{
	CHECK_DPYP("lininit")

#ifdef HAVE_X11
	lin_setup(QSP_ARG   DPA_LINTBL_OBJ(current_dpyp),DEF_GAM,DEF_VZ);
#endif /* HAVE_X11 */
}
#endif /* NOT_USED */

#ifdef HAVE_X11
void install_default_lintbl(QSP_ARG_DECL  Dpyable *dpyp)
{
	if(default_lt_dp==NULL){
		default_lt_dp = new_lintbl(QSP_ARG  "default_lintbl");
		lin_setup(QSP_ARG  default_lt_dp,DEF_GAM,DEF_VZ);		/* seems a bit wasteful? */
	}
	 DPA_LINTBL_OBJ(dpyp) = default_lt_dp;
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

		sprintf(ERROR_STRING,"Reading linearization data from %s",
			filename);
		advise(ERROR_STRING);

		n=0;
		while( fscanf(fp,"%d",&value) == 1 ){
			if( value < 0 || value > DACMAX ) {
				sprintf(ERROR_STRING,
			"linearization value out of range:  i=%d  tbl=%d",
					n,value);
				warn (ERROR_STRING);
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

		sprintf(ERROR_STRING,
			"Writing linearization data to %s",filename);
		advise(ERROR_STRING);

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
		lin_setup( DPA_LINTBL_OBJ(current_dpyp),DEF_GAM,DEF_VZ);

	if( start < 0 || start >= n_lin_lvls ){
		sprintf(ERROR_STRING,"Start value %d must be in the range 0-%d",
			start,n_lin_lvls-1);
		NWARN(ERROR_STRING);
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
		lin_setup( DPA_LINTBL_OBJ(current_dpyp),DEF_GAM,DEF_VZ);
	}
}
#endif /* FOOBAR */


void set_lintbl(QSP_ARG_DECL  Data_Obj *lt_dp)
{
	CHECK_DPYP("set_lintbl")

	/* BUG check that this object is a valid lintbl */
#ifdef HAVE_X11
	 DPA_LINTBL_OBJ(current_dpyp) = lt_dp;
#endif /* HAVE_X11 */
}

