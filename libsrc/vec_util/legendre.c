#include "quip_config.h"

char VersionId_vec_util_legendre[] = QUIP_VERSION_STRING;

/*
 * Legendre polynomials for spherical Fourier transforms...
 *
 * based on Swartztrauber, SIAM J Numer. Anal., 16 934-949, 1979.
 *
 * theta is latitude, 0 - pi
 * phi is longitude 0 - 2 pi
 *
 * N number of theta samples
 * M number of phi samples
 * equal angular spacing -> M ~ 2N
 */

#include <stdio.h>
#include <math.h>

#include "data_obj.h"
#include "items.h"
#include "getbuf.h"
#include "debug.h"

extern u_long polydebug;

#include "polynm.h"

/* local prototypes */

static void add_coeffs(Coefficient *,Coefficient *,Coefficient *);
static void mul_coeffs(Coefficient *,Coefficient *,Coefficient *);
static void scale_coeff(Coefficient *,Coefficient *,double);
static void norm_coeff(Coefficient *);
static void clear_exponent(Coefficient *);
static void show_coeff(Coefficient *cfp);
static void coeff_power(Coefficient *dst,Coefficient *src,int n);
static Polynomial * scale_poly_coeffs(QSP_ARG_DECL  Polynomial *pp,Coefficient *cfp);
static void invert_coeff(Coefficient *cfp);
static Polynomial *rename_poly(QSP_ARG_DECL  Polynomial *pp,char *newname);
static void sqrt_coeff(Coefficient *cfp);
static Polynomial *norm_legendre_poly(QSP_ARG_DECL  int m,int n);

ITEM_INTERFACE_DECLARATIONS(Polynomial,polynm)

void factorial_series(Coefficient *cfp,int start,int n)
{
	double ds;

	cfp->value=1.0;
	cfp->exponent=0;

	ds=start;

	if( start < n ){
		NWARN("factorial_series:  start less than n");
		return;
	}

	while(n--){
		scale_coeff(cfp,cfp,ds);
		ds -= 1.0;
	}
}

void factorial(Coefficient *cfp,int n)
{
	double dn;

	dn = n;

	cfp->value=1.0;
	cfp->exponent=0;

	while( dn > 1.0 ){
		scale_coeff(cfp,cfp,dn);
		dn -= 1.0;
	}
}

void set_coeff(Coefficient *cfp,double v)
{
	cfp->value = v;
	cfp->exponent = 0;
	norm_coeff(cfp);
}

static void coeff_power(Coefficient *dst,Coefficient *src,int n)
{
	dst->value = pow(src->value,(double)n);
	dst->exponent = src->exponent * n;
	norm_coeff(dst);
}

static void sqrt_coeff(Coefficient *cfp)
{
	/* first make sure the exponent is even */

	if( cfp->exponent & 1 ){
		cfp->value *= 10.0;
		cfp->exponent --;
	}

	cfp->value = sqrt(cfp->value);
	cfp->exponent /= 2;
}

static void invert_coeff(Coefficient *cfp)
{
	if( cfp->value == 0.0 ){
		NWARN("invert_coeff:  divide by zero!?");
		return;
	}

	cfp->value = 1/cfp->value;
	cfp->exponent *= -1;

	norm_coeff(cfp);
}

static void add_coeffs(Coefficient *cfp,Coefficient *cfp1,Coefficient *cfp2)
{
	Coefficient cf1,cf2;

	cf1 = *cfp1;
	cf2 = *cfp2;

	/* make the exponents the same.
	 * the values should initially have a value between 1 and 10.
	 *
	 * if the exponents are both positive, move up to the larger.
	 */

	while ( cf1.exponent > cf2.exponent ){
		cf2.value /= 10.0;
		cf2.exponent ++;
	}
	while ( cf2.exponent > cf1.exponent ){
		cf1.value /= 10.0;
		cf1.exponent ++;
	}

	/* now the exponents are equal */

	cfp->value = cf1.value+cf2.value;
	cfp->exponent = cf1.exponent;
}

static void mul_coeffs(Coefficient *cfp,Coefficient *cfp1,Coefficient *cfp2)
{
	cfp->value = cfp1->value * cfp2->value;
	cfp->exponent = cfp1->exponent + cfp2->exponent;
	norm_coeff(cfp);
}

static void norm_coeff(Coefficient *cfp)
{
	if( cfp->value == 0.0 ) {
		cfp->exponent=0;
		return;
	}

	while( fabs(cfp->value) >= 10.0 ){
		if( cfp->value == 0.0 ) {
NWARN("truncating");
			cfp->exponent=0;
			return;
		}
		cfp->value /= 10.0;
		cfp->exponent ++;
	}
	while( fabs(cfp->value) <= 0.1 ){
		if( cfp->value == 0.0 ) {
NWARN("truncating");
			cfp->exponent=0;
			return;
		}
		cfp->value *= 10.0;
		cfp->exponent --;
	}
}

static void clear_exponent(Coefficient *cfp)
{
	while( cfp->exponent > 0 ){
		cfp->value *= 10.0;
		cfp->exponent --;
	}
	while( cfp->exponent < 0 ){
		cfp->value /= 10.0;
		cfp->exponent ++;
	}
}

static void scale_coeff(Coefficient *cfp,Coefficient *srcp,double factor)
{
	*cfp = *srcp;
	cfp->value *= factor;
	norm_coeff(cfp);
}

void zap_poly(QSP_ARG_DECL  Polynomial *pp)
{
	givbuf(pp->coeff);
	del_polynm(QSP_ARG  pp->poly_name);
}

Polynomial *new_poly(QSP_ARG_DECL  int n)
{
	Polynomial *pp;
	static int tmp_index=1;
	char s[32];

	sprintf(s,"tmp_poly%d",tmp_index++);
	pp = new_polynm(QSP_ARG  s);
	pp->order = n;
	pp->coeff = (Coefficient *)getbuf( (n+1) * sizeof(Coefficient) );
	return(pp);
}

static Polynomial *rename_poly(QSP_ARG_DECL  Polynomial *pp,char *newname)
{
	Polynomial *npp;

	npp = new_polynm(QSP_ARG  newname);
	if( npp == NO_POLY ) return(npp);

	npp->order = pp->order;
	npp->coeff = pp->coeff;
	del_polynm(QSP_ARG  pp->poly_name);
	return(npp);
}

Polynomial *add_polys(QSP_ARG_DECL  Polynomial *pp1,Polynomial *pp2)
{
	int i,l,n;
	Polynomial *pp, *hpp, *lpp;

	if( pp1->order > pp2->order ){
		hpp = pp1;
		lpp = pp2;
	} else {
		hpp = pp2;
		lpp = pp1;
	}

	n = hpp->order;
	l = lpp->order;

	pp = new_poly(QSP_ARG  n);

	for(i=0;i<=l;i++)
		add_coeffs(&pp->coeff[i],&hpp->coeff[i],&lpp->coeff[i]);
	for(;i<=n;i++)
		pp->coeff[i] = hpp->coeff[i];

	return(pp);
}

Polynomial *mul_polys(QSP_ARG_DECL  Polynomial *pp1,Polynomial *pp2)
{
	Polynomial *pp;
	int i,j,k;

	pp = new_poly(QSP_ARG   pp1->order + pp2->order );

	for(i=0;i<=pp->order;i++){
		pp->coeff[i].value = 0.0;
		pp->coeff[i].exponent = 0;
	}

	for(i=0;i<=pp1->order;i++){
		for(j=0;j<=pp2->order;j++){
			Coefficient tmpc;

			k = i+j;
			mul_coeffs(&tmpc,&pp1->coeff[i],&pp2->coeff[j]);
			add_coeffs(&pp->coeff[k],&pp->coeff[k],&tmpc);
		}
	}
	return(pp);
}

static Polynomial * scale_poly_coeffs(QSP_ARG_DECL  Polynomial *pp,Coefficient *cfp)
{
	Polynomial *new_pp;
	int i;

	new_pp=new_poly(QSP_ARG  pp->order);
	for(i=0;i<=pp->order;i++)
		mul_coeffs(&new_pp->coeff[i], &pp->coeff[i], cfp );
	return(new_pp);
}

Polynomial *scale_poly(QSP_ARG_DECL  Polynomial *pp,double factor)
{
	Polynomial *new_pp;
	int i;

	new_pp=new_poly(QSP_ARG  pp->order);
	for(i=0;i<=pp->order;i++)
		scale_coeff(&new_pp->coeff[i], &pp->coeff[i], factor );
	return(new_pp);
}

Polynomial *diff_poly(QSP_ARG_DECL  Polynomial *pp)
{
	Polynomial *new_pp;
	int i;

	if( pp->order == 0 )
		return(NO_POLY);

	new_pp = new_poly(QSP_ARG  pp->order - 1 );
	for(i=1;i<=pp->order;i++){
		scale_coeff(&new_pp->coeff[i-1], &pp->coeff[i], (double) i );
	}
	return(new_pp);
}

Polynomial *exp_poly(QSP_ARG_DECL  Polynomial *pp,int n)
{
	Polynomial *tmp_pp, *new_pp;
	int i;

	if( n==0 ){
		new_pp=new_poly(QSP_ARG  0);
		new_pp->coeff[0].value=1.0;
		new_pp->coeff[0].exponent=0;
		return(new_pp);
	}

	/* duplicate the input polynomial */
	tmp_pp=new_poly(QSP_ARG  pp->order);
	for(i=0;i<=pp->order;i++)
		tmp_pp->coeff[i] = pp->coeff[i];


	/* now do the general case */

	while( n >= 2 ){
		new_pp = mul_polys(QSP_ARG  pp,tmp_pp);
		zap_poly(QSP_ARG  tmp_pp);
		tmp_pp = new_pp;
		n--;
	}
	return(tmp_pp);
}

static void show_coeff(Coefficient *cfp)
{
	sprintf(msg_str,"\t%g %ld",cfp->value,cfp->exponent);
	prt_msg(msg_str);
}

double eval_poly(Polynomial *pp,double x)
{
	int i;
	Coefficient answer;

	/* high order coefficients come at the end */

	i=pp->order;

	answer.value = 0.0;
	answer.exponent = 0;
	while( i > 0 ){
		add_coeffs(&answer,&answer, &pp->coeff[i]);
if( debug & polydebug ) show_coeff(&answer);
		scale_coeff(&answer,&answer, x );
if( debug & polydebug ) show_coeff(&answer);
		i--;
	}
	add_coeffs(&answer,&answer, &pp->coeff[0]);
if( debug & polydebug ) show_coeff(&answer);
	clear_exponent(&answer);
if( debug & polydebug ) show_coeff(&answer);
	
	return(answer.value);
}


void show_poly(Polynomial *pp)
{
	int i;

	sprintf(msg_str,"%s, order %d,",pp->poly_name,pp->order);
	prt_msg(msg_str);
	i=pp->order;
	msg_str[0]=0;
	while( i >= 0 ){
		show_coeff(&pp->coeff[i]);
		i--;
	}
	prt_msg("");
}

/* This is not the "Legendre polynomial" (which is
 * the associated Legendre function with m=0),
 * rather it is the polynomial used in rodrigues' formula,
 * see Talman p 164, Swarztrauber p 936.
 */

Polynomial *legendre_poly(QSP_ARG_DECL  int m,int n)
{
	char str[128];
	char name[128];
	Polynomial *pp1, *pp2;
	int i;
	Coefficient cf,cf2;

	sprintf(name,"Leg.%d.%d",m,n);
	pp1 = polynm_of(QSP_ARG  name);
	if( pp1 != NO_POLY ) return(pp1);

	sprintf(str,"tmp.%d.%d",m,n);

	pp1 = new_polynm(QSP_ARG  str);
	if( pp1 == NO_POLY ) return(pp1);

	pp1->order = 2;
	pp1->coeff = (Coefficient *)getbuf(3*sizeof(Coefficient));

	/* x^2 - 1 */
	pp1->coeff[0].value = -1.0;
	pp1->coeff[1].value =  0.0;
	pp1->coeff[2].value =  1.0;
	pp1->coeff[0].exponent = 0;
	pp1->coeff[1].exponent = 0;
	pp1->coeff[2].exponent = 0;

	pp2 = exp_poly(QSP_ARG  pp1,n);
	zap_poly(QSP_ARG  pp1);

	for(i=0;i<(m+n);i++){
		pp1 = diff_poly(QSP_ARG  pp2);
		zap_poly(QSP_ARG  pp2);
		pp2 = pp1;
	}

	cf.value=2.0;
	cf.exponent=0;
	coeff_power(&cf,&cf,n);
	factorial(&cf2,n);
	mul_coeffs(&cf,&cf,&cf2);
	invert_coeff(&cf);

	pp1 = scale_poly_coeffs(QSP_ARG  pp2,&cf);
	zap_poly(QSP_ARG  pp2);

	pp2=rename_poly(QSP_ARG  pp1,name);
	return(pp2);
}

void tabulate_polynomial(Data_Obj *ydp,Data_Obj *xdp,Polynomial *pp)
{
	float *dst,*src;
	dimension_t i,j;

	/* BUG should check types, sizes, etc */

	for(i=0;i<ydp->dt_rows;i++){

		dst = (float *)ydp->dt_data;
		dst += i * ydp->dt_rinc;

		src = (float *)xdp->dt_data;
		src += i * xdp->dt_rinc;

		for(j=0;j<ydp->dt_cols;j++){
			*dst = eval_poly(pp,*src);

			dst += ydp->dt_pinc;
			src += xdp->dt_pinc;
		}
	}
}

void tabulate_legendre(QSP_ARG_DECL  Data_Obj *ydp,Data_Obj *xdp,int m,int n)
{
	float *dst,*src;
	dimension_t i,j;
	Polynomial *pp;

	pp = legendre_poly(QSP_ARG  m,n);

	for(i=0;i<ydp->dt_rows;i++){

		dst = (float *)ydp->dt_data;
		dst += i * ydp->dt_rinc;

		src = (float *)xdp->dt_data;
		src += i * xdp->dt_rinc;

		for(j=0;j<ydp->dt_cols;j++){
			float theta,ct,st;
			float t;

			theta = *src;

			ct = cos(theta);
			st = sin(theta);
			t = eval_poly(pp,ct);
			t *= pow(st,(double)m);

			/* this is now incorporated into the polynomial */
			/* / ( pow(2.0,(double)n) * factorial(n) ); */

			*dst = t;

			dst += ydp->dt_pinc;
			src += xdp->dt_pinc;
		}
	}
	zap_poly(QSP_ARG  pp);
}


static Polynomial *norm_legendre_poly(QSP_ARG_DECL  int m,int n)
{
	char name[64];
	Coefficient sc;
	Polynomial *pp,*pp2;

	sprintf(name,"Norm.%d.%d",m,n);
	pp = polynm_of(QSP_ARG  name);
	if( pp != NO_POLY ) return(pp);

	pp = legendre_poly(QSP_ARG  m,n);

	/* normalize */
	sc.value=1.0;
	sc.exponent=0;
	factorial_series(&sc,n+m,2*m);	/* from literature */
	/* factorial_series(&sc,2*n,n+m); */		/* from intuition */

prt_msg("factorial series");
show_coeff(&sc);
	invert_coeff(&sc);

prt_msg("inverted factorial series");
show_coeff(&sc);
	scale_coeff(&sc,&sc,(double)((2*n+1)*0.5) );

prt_msg("scaled");
show_coeff(&sc);
	sqrt_coeff(&sc);

prt_msg("sqrt'd");
show_coeff(&sc);
	pp2 = scale_poly_coeffs(QSP_ARG  pp,&sc);
	zap_poly(QSP_ARG  pp);

	pp=rename_poly(QSP_ARG  pp2,name);

	return(pp);
}



/*
 * Tabulate the normalized legendre function
 */

void tabulate_Pbar(QSP_ARG_DECL  Data_Obj *ydp,Data_Obj *xdp,int m,int n)
{
	float *dst,*src;
	dimension_t i,j;
	Polynomial *pp;

	pp = norm_legendre_poly(QSP_ARG  m,n);

	for(i=0;i<ydp->dt_rows;i++){

		dst = (float *)ydp->dt_data;
		dst += i * ydp->dt_rinc;

		src = (float *)xdp->dt_data;
		src += i * xdp->dt_rinc;

		for(j=0;j<ydp->dt_cols;j++){
			float theta,ct,st;
			float t;

			theta = *src;

			ct = cos(theta);
			st = sin(theta);
			t = eval_poly(pp,ct);
			t *= pow(st,(double)m) ;

			/* now incorporated into the polynomial */
			/* / ( pow(2.0,(double)n) * factorial(n) ); */

			*dst = t;

			dst += ydp->dt_pinc;
			src += xdp->dt_pinc;
		}
	}
	/* zap_poly(QSP_ARG  pp); */
}

