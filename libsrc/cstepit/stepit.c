
#include "quip_config.h"

/*
 * Implementation of the stepit subroutine in C; translated from the Fortran
 * routine composed by J. P. Chandler.
 *
 * Sclar	2/5/87
 * Movshon	1/21/91
 * Mulligan	4/23/98
 *
 */

#include <math.h>
#include <signal.h>
#include <stdio.h>
#include "cstepit.h"
#include "quip_prot.h"
#include "query_stack.h"
#include "sigpush.h"		/* sigpush(), sigpop() */

#define	amax1(a,b) ((a>b)?a:b)
#define	amin1(a,b) ((a<b)?a:b)

/* things from quick.c that need to be here... */

static int ccflag;
static int nf;
static double fit_err[VARS][VARS];	// used to be just err, but symbol collision on Mac OS - jbm
static double fobj;
static int ntrace=(-1);
static int n_params,mask[VARS];
static double deltax[VARS],delmin[VARS];
static double x[VARS];
static double xmin[VARS],xmax[VARS];

/* Everything below is being declared external */

static double  vec[VARS], trial[VARS], xsave[VARS], fstor[VARS], dx[VARS];
static double  oldvec[VARS], salvo[VARS], xosc[VARS][VVARS], chiosc[VVARS];
static double  fbest, fprev, uhuge;
static double  ratio = 10.;
static double  colin = .99;
static double  ack = 2.;
static double  signif = 1.e6;	/* original value was 2.e8 */

static int     nflat[VARS];
static int     mosque = VARS;
static int     ncomp = VARS;

/* local prototypes */


static void ccseen(void);
static void done(void);
#ifdef FOOBAR
static void intrupt(void);
#endif /* FOOBAR */

/*
 * Here if no free parameters given
 */

#define noparams(cstepit_fn) _noparams(QSP_ARG  cstepit_fn)

static void _noparams(QSP_ARG_DECL  void (*cstepit_fn)(void))
{
	advise("calling function with initial parameters...");
	(*cstepit_fn)();
	done();
}


void halt_cstepit(void)
{
	ccflag=1;	/* simulate the user typing ^C */
}

/*
 * stepit() returns 0 if it completed normally, 1 if interrupted
 */

/*int stepit(cstepit_fn)*/
int _stepit (QSP_ARG_DECL  void (*cstepit_fn)(void))
{
	int     nactiv, nack, ncirc, nosc, nah, nzip, jock, i, j, k;
	int     ngiant, ntry, nretry, kl, nt, na, nz, ngate, nflag;
	double  a, vacd, compar, sub, p, save, del, cinder, denom;
	double  sumo, sumv, cosine, coxcom, chibak, traal, trail;
	/* double  pack */
	double cander, xij, steps, atempo;
#ifdef FOOBAR
	/* Clean exit on ^C */
	/* signal(SIGINT, intrupt); */
	sigpush(SIGINT,intrupt);
#endif /* FOOBAR */

	del=coxcom=0.0;		// quiet compiler
	nah=0;			// quiet compiler

//#ifdef SUN
//	uhuge = max_normal();
//#else	/* ! SUN */
//	uhuge = HUGE_VAL;		/* infinity, on SGI */
//#endif /* ! SUN */

#ifdef HUGE_VAL
	uhuge = HUGE_VAL;
#else
#error HUGE_VAL is not defined!?
#endif

	if (n_params <= 0) {
		noparams(cstepit_fn);
#ifdef FOOBAR
		sigpop(SIGINT);
#endif /* FOOBAR */
		return(0);
	}
	for (;;) {
		nactiv = 0;

		for (i = 0; i < n_params; i++) {
			if (mask[i])
				continue;
			atempo = signif*fabs(deltax[i]) - fabs(x[i]);
			if (atempo <= 0.0f) {
				if (x[i] != 0 )
					deltax[i] = .01*x[i];
				else
					deltax[i] = .01;
			}
			if (delmin[i] == 0.)
				delmin[i] = deltax[i] / signif;

			if (xmax[i] <= xmin[i]) {
				xmax[i] = uhuge;
				xmin[i] = -1.*uhuge;
			}
			nactiv++;
			vacd = amin1(xmax[i], x[i]);
			x[i] = amax1(xmin[i], vacd);
		}

		compar = 0.;

		if ((nactiv - 1) < 0)
			for (j = 0; j < n_params; j++)
				mask[j] = 0;
		else
			break;	/* this is how we leave loop */
	}

	if ((nactiv - 1) > 0) {
		a = (float) nactiv;
		sub = 2. / (a - 1.);
		p = 2. * (1. / sqrt(a) / (1. - pow(.5, sub)) - 1.);
		compar = amin1(9.99e-1, fabs((1. - pow((1. - colin), sub)) * (1. + p * (1. - colin))));
	}
 /* here we are equivalent to line 190 of stepit.f */

	/* if (ntrace >= 0) { 			   } */
	if( verbose ){

		printf("\nMask  =");
		for (j = 0; j < n_params; j++)
			printf("%10d", mask[j]);
		printf("\nX     =");
		for (j = 0; j < n_params; j++)
			printf("%10.4g", x[j]);
		printf("\nXmax  =");
		for (j = 0; j < n_params; j++)
			printf("%10.4g", xmax[j]);
		printf("\nXmin  =");
		for (j = 0; j < n_params; j++)
			printf("%10.4g", xmin[j]);
		printf("\nDeltax=");
		for (j = 0; j < n_params; j++)
			printf("%10.4g", deltax[j]);
		printf("\nDelmin=");
		for (j = 0; j < n_params; j++)
			printf("%10.4g", delmin[j]);
		printf("\n");
		fflush(stdout);
		/* BUG use advise() or prt_msg() */
	}

	(*cstepit_fn)();

	if (ccflag) {
		ccseen();
#ifdef FOOBAR
	sigpop(SIGINT);
#endif /* FOOBAR */
		return(1);
	}
	nf = jock = 1;

/* equivalent of line 290 of stepit.f */

	if (ntrace >= 0){
		printf("\n\n%d variables, %d active, initial fobj = %g", n_params, nactiv, fobj);
		fflush(stdout);
	}
	if (ntrace > 0) {
		printf("\nncomp = %d   ratio = %5.1f   ack = %5.1f  ", ncomp, ratio, ack);
		printf("colin = %6.3f   ", colin);
		printf("compar = %6.3f\n", compar);
		fflush(stdout);
	}
	if (n_params <= 0) {
		noparams(cstepit_fn);
#ifdef FOOBAR
	sigpop(SIGINT);
#endif /* FOOBAR */
		return(0);
	}
	if (ntrace > 0){
		fprintf(stderr,"\nTrace map of the minimalization process\n\n");
		fflush(stderr);
	}

	for (i = 0; i < n_params; i++) {
		dx[i] = deltax[i];
		vec[i] = 0.;
		for (j = 0; j < mosque; j++)
			fit_err[i][j] = 0.;
	}

	fbest = fobj;
	nosc = 0;
L380:	ncirc = 0;
	nzip = 0;

/*
 * Main loop for cycling through the variables; First trial step with
 * each variable is separate
 */

L390:	nack = 0;

	for (i = 0; i < n_params; i++) {

		oldvec[i] = vec[i];
		vec[i] = 0.;
		trial[i] = 0.;

		if (mask[i]) {
			nflat[i] = 1;
			continue;
		}
		nack++;
		save = x[i];

		if (signif * fabs(dx[i]) <= fabs(x[i]))
			goto L580;

		x[i] = save + dx[i];
		if (jock > 0)
			jock = 0;
		nflag = 1;

		if (x[i] <= xmin[i] || x[i] >= xmax[i]) {
			nflag += 3;
			goto L490;
		}
		(*cstepit_fn)();

		if (ccflag) {
			ccseen();
#ifdef FOOBAR
	sigpop(SIGINT);
#endif /* FOOBAR */
			return(1);
		}
		nf++;
		fprev = fobj;

		if ((fobj - fbest) < 0.)
			goto L620;
		if ((fobj - fbest) == 0.)
			nflag++;
L490:		x[i] = save - dx[i];
		if (x[i] <= xmin[i] || x[i] >= xmax[i])
			goto L590;

		(*cstepit_fn)();

		if (ccflag) {
			ccseen();
#ifdef FOOBAR
	sigpop(SIGINT);
#endif /* FOOBAR */
			return(1);
		}
		nf++;

		if ((fobj - fbest) < 0.)
			goto L610;
		if ((fobj - fbest) == 0.)
			nflag++;

		if ((nflag - 3) == 0)
			goto L580;
		if ((nflag - 3) > 0)
			goto L590;

		if ((fobj - fprev) * (fprev - 2. * fbest + fobj) == 0)
			goto L590;

		trial[i] = .5 * dx[i] * (fobj - fprev) / (fprev - 2. * fbest + fobj);
		vec[i] += del / fabs(dx[i]);
		nflat[i] = 0;
		x[i] = save + trial[i];

		(*cstepit_fn)();

		if (ccflag) {
			ccseen();
#ifdef FOOBAR
	sigpop(SIGINT);
#endif /* FOOBAR */
			return(1);
		}
		nf++;

		if (fobj < fbest) {
			fbest = fobj;
			jock = 1;
			goto L600;
		}
		trial[i] = vec[i] = 0.;
		goto L590;

L580:		vec[i] = 0.;
		nflat[i] = 1;
L590:		x[i] = save;
L600:		ncirc++;

		if ((ncirc - nactiv) < 0)
			goto L690;
		goto L1430;

L610:		dx[i] = -1.0 * dx[i];

/*
 * A lower value has been found, hence, this variable will
 * change.
 */

L620:		ncirc = 0;
		del = dx[i];
L630:		fprev = fbest;
		fbest = fobj;
		vec[i] += del / fabs(dx[i]);
		nflat[i] = 0;
		trial[i] += del;
		del *= ack;
		save = x[i];
		x[i] = save + del;
		if (x[i] <= xmin[i] || x[i] >= xmax[i])
			goto L680;

		(*cstepit_fn)();

		if (ccflag) {
			ccseen();
#ifdef FOOBAR
	sigpop(SIGINT);
#endif /* FOOBAR */
			return(1);
		}
		nf++;

		if (fobj < fbest)
			goto L630;
		cinder = 0.;
		denom = ack * fprev - (ack + 1.) * fbest + fobj;
		if ( denom != 0 )
			cinder = (.5 / ack) * (pow(ack, 2.) * fprev - (pow(ack, 2.) - 1.)
			    * fbest - fobj) / denom;
		x[i] = save + cinder * del;

		(*cstepit_fn)();

		if (ccflag) {
			ccseen();
#ifdef FOOBAR
	sigpop(SIGINT);
#endif /* FOOBAR */
			return(1);
		}
		nf++;

		if (fobj >= fbest)
			goto L680;
		fbest = fobj;
		trial[i] += cinder * del;
		vec[i] += cinder * del / fabs(dx[i]);
		goto L690;
L680:		x[i] = save;
L690:		if (nzip < 1)
			goto L1340;

		if (fabs(vec[i]) >= ack) {
			dx[i] = ack * fabs(dx[i]);
			vec[i] /= ack;
			oldvec[i] /= ack;
			for (j = 0; j < mosque; j++)
				fit_err[i][j] /= ack;
			if (ntrace > 0){
				fprintf(stderr,"\nStep size %d increased to %11.4g", i, dx[i]);
				fflush(stderr);
			}
		}
		for (j = 0, sumo = sumv = 0.; j < n_params; j++) {
			sumo += pow(oldvec[j], 2.);
			sumv += pow(vec[j], 2.);
		}
		if ((sumo * sumv) <= 0.)
			goto L1340;

		sumo = sqrt(sumo);
		sumv = sqrt(sumv);
		for (j = 0, cosine = 0.; j < n_params; j++)
			cosine += (oldvec[j] / sumo) * (vec[j] / sumv);
		na = nack - nactiv;
		nz = nzip - 1;

		if (nz < 0)
			goto L1340;
		/* braces added here to avoid ambiguous "else"...
		 * jbm, 1/10/00
		 * Intent seemed clear enough from indentation,
		 * but who indented???
		 */
		if (nz == 0){
			if (na < 0)
				goto L1340;
			else if (cosine < compar)
				goto L1340;
		}
		if (na < 0)
			if (cosine < compar)
				goto L1340;
		if (nzip >= ncomp)
			goto L830;
		if (cosine < compar)
			goto L1340;

/* equivalent at this point to line 830 of stepit.f */

L830:		if (ntrace > 0) {
			fprintf(stderr,"\nfobj = %15.8g ", fbest);
			for (j = 0; j < i + 1; j++)
				fprintf(stderr,"\nNumber of steps = %9.2g ", vec[j]);
			fflush(stderr);
		}
		ngiant = ntry = nretry = 0;
		kl = 0;
		nosc++;

		if (nosc > mosque) {
			nosc = mosque;
			for (k = 1; k < mosque; k++) {
				chiosc[k - 1] = chiosc[k];
				for (j = 0; j < n_params; j++) {
					xosc[j][k - 1] = xosc[j][k];
					fit_err[j][k - 1] = fit_err[j][k];
				}
			}
		}
		for (j = 0; j < n_params; j++) {
			xosc[j][nosc] = x[j];
			fit_err[j][nosc] = vec[j] / sumv;
		}

		chiosc[nosc] = fbest;
		if (nosc < 3)
			goto L960;

/*
 * Search for a previous successful giant step in a direction
 * more nearly parallel to the direction of the proposed ste
 * than was the immediately previous one
 */

		for (coxcom = 0., j = 0; j < n_params; j++)
			coxcom += fit_err[j][nosc] * fit_err[j][nosc - 1];
		nah = nosc - 2;
L930:		ntry = 0;
		for (k = kl; k < nah + 1; k++) {
			nretry = nah - k;
			for (j = 0, cosine = 0.; j < n_params; j++)
				cosine += fit_err[j][nosc] * fit_err[j][k];
			if (cosine > coxcom)
				goto L970;
		}
L960:		chibak = fstor[i];
		goto L1020;

L970:		ntry = 1;
		kl = k + 1;
		if (ntrace > 0) {
			nt = nosc - k;
			fprintf(stderr,"\n****Possible oscillation with period  %d", nt);
			fflush(stderr);
		}
		for (j = 0; j < n_params; j++) {
			salvo[j] = trial[j];
			trial[j] = (x[j] - xosc[j][k]) / ack;
		}

		chibak = fbest + (chiosc[k] - fbest) / ack;

L1020:		for (j = 0; j < n_params; j++) {
			xsave[j] = x[j];
			trial[j] *= ack;
			if (mask[j])
				continue;
			traal = x[j] + trial[j];
			trail = amin1(traal, xmax[j]);
			x[j] = amax1(trail, xmin[j]);
		}

		jock = 0;

		(*cstepit_fn)();

		if (ccflag) {
			ccseen();
#ifdef FOOBAR
	sigpop(SIGINT);
#endif /* FOOBAR */
			return(1);
		}
		nf++;

		if (fobj >= fbest)
			goto L1080;
		chibak = fbest;
		fbest = fobj;
		ngiant++;
		if (ntrace > 0) {
			fprintf(stderr,"\nfobj = %15.8g ", fbest);
			for (j = 0; j < n_params; j++)
				fprintf(stderr,"\nx[%d] = %11.4g", j, x[j]);
			fflush(stderr);
		}
		goto L1020;

L1080:		if (nretry <= 0)
			goto L1100;
		if (ngiant <= 0)
			goto L1150;
L1100:		cinder = (.5 / ack) * (pow(ack, 2.) * chibak - (pow(ack, 2.) - 1.) *
		    fbest - fobj) / (ack * chibak - (ack + 1.) * fbest + fobj);

		for (j = 0; j < n_params; j++) {
			if (mask[j])
				continue;
			cander = xsave[j] + cinder * trial[j];
			xij = amin1(cander, xmax[j]);
			x[j] = amax1(xij, xmin[j]);
		}

		jock = 0;

		(*cstepit_fn)();

		if (ccflag) {
			ccseen();
#ifdef FOOBAR
	sigpop(SIGINT);
#endif /* FOOBAR */
			return(1);
		}
		nf++;

		if (fobj >= fbest) {
			if ((ngiant) || (ntry == 0))
				for (j = 0; j < n_params; j++) {
					trial[j] /= ack;
					x[j] = xsave[j];
				}

			else
				for (j = 0; j < n_params; j++) {
					trial[j] = salvo[j];
					x[j] = xsave[j];
				}

			goto L1151;

	L1150:		for (j = 0; j < n_params; j++) {
				trial[j] /= ack;
				x[j] = xsave[j];
			}

	L1151:		if (ntrace > 0) {
				fprintf(stderr,"\nChisq = %15.8g after %d giant steps", fbest, ngiant);
				for (j = 0; j < n_params; j++)
					fprintf(stderr,"\nx[%d] = %11.4g", j, x[j]);
				fflush(stderr);
			}
			if (ngiant > 0)
				goto L1310;
			if (nretry > 0)
				goto L930;
			if (ntry == 0)
				goto L1330;

			ntry = 0;
			goto L960;

		};		/* replaces goto 1280 */

		fbest = fobj;
		jock = 1;

		if (ntrace > 0) {
			steps = (double) ngiant + cinder;
			fprintf(stderr,"\n Chisq = %15.8g after %6.1f giant steps", fbest, steps);
			fflush(stderr);
		}
L1310:		if (ntry)
			nosc = 0;

		goto L380;

L1330:		nosc--;

		if (0 > nosc)
			nosc = 0;
L1340:		fstor[i] = fbest;
	}
/* equivalent to line 1350 in steptit.f- end of the big loop */

/*
 * Another cycle through the variables has been completed. Print
 * another line of traces.
 */

	if (ntrace > 0) {
		fprintf(stderr,"\nCHISQ = %15.8g ", fbest);
		for (j = 0; j < i + 1; j++)
			fprintf(stderr,"\nNumber of steps = %9.2g ", vec[j]);
		fflush(stderr);
	}
	if ((nzip == 0) && ntrace > 0){
		for (j = 0; j < n_params; j++)
			fprintf(stderr,"\nx[%d] = %11.4g", j, x[j]);
		fflush(stderr);
	}
	nzip++;
	goto L390;

/* A minimum has been found- print the remaining traces	*/

L1430:	if (ntrace > 0) {
		fprintf(stderr,"\nCHISQ = %15.8g ", fbest);
		for (j = 0; j < i + 1; j++)
			fprintf(stderr,"\nNumber of steps = %9.2g ", vec[j]);
		fflush(stderr);
	}
	nosc = 0;
	ngate = 1;
	for (j = 0; j < n_params; j++) {
		if ((mask[j]) || nflat[j] > 0 || (fabs(dx[j]) <=
			fabs(delmin[j])))
			goto L1520;
		ngate = 0;
L1520:		dx[j] /= ratio;
	}

	if (ngate > 0)
		goto L1600;

	if (ntrace > 0) {
		fprintf(stderr,"\nSteps reduced to:");
		for (j = 0; j < i + 1; j++)
			fprintf(stderr,"\n%11.4g", dx[j]);
		fflush(stderr);
	}
	goto L380;
L1600:	fobj = fbest;
	done();
#ifdef FOOBAR
	sigpop(SIGINT);
#endif /* FOOBAR */
	return(0);
}

/*
 * Here on SIGINT
 */

#ifdef FOOBAR
static void intrupt()
{
	ccflag++;
	sigpop(SIGINT);
}
#endif /* FOOBAR */

/*
 * Here if control-C noticed
 */

static void ccseen()
{

	ccflag = 0;
	if( verbose )
		NADVISE("Subroutine stepit terminated by operator");

	fobj = fbest;
	done();
}

/*
 * Report and exit
 */

static void done()
{
	int     i;

	if (ntrace < 0)
		return;
	printf("\n\nFit after %d iterations", nf);
	printf("\nFit values:");
	for (i = 0; i < n_params; i++)
		printf(" %10.6g", x[i]);
	printf("\nResidual error: %11.4g\n", fobj);
	fflush(stdout);
}

void setfobj(double value)
{
	fobj = value;
}

void getvals(double *arr, int n)
{
	int i;

	for(i=0;i<n;i++)
		arr[i] = x[i];
}

int _reset_n_opt_params(QSP_ARG_DECL  int n)
{
	if( n<=0 ){
		sprintf(ERROR_STRING,
	"requested n_params %d must be positive (and <= %d)",
			n,VARS);
		warn(ERROR_STRING);
		n=1;
	} else if( n>VARS ){
		sprintf(ERROR_STRING,
	"requested n_params %d is too large, must be <= %d",
			n,VARS);
		warn(ERROR_STRING);
		n=VARS;
	}
	return( n_params=n );
}

void _set_opt_param_vals(QSP_ARG_DECL  double *arr, int n)
{
	int i;

	if( n_params == 0 )
		n_params=n;
	else if( n!=n_params ){
		sprintf(ERROR_STRING,"setvals:  n_params = %d, n = %d",n_params,n);
		advise(ERROR_STRING);
		warn("parameter count mismatch");
	}

	for(i=0;i<n;i++)
		x[i] = arr[i];
}

void _set_opt_param_minmax(QSP_ARG_DECL  double *minarr, double *maxarr, int n)
{
	int i;

	if( n_params==0 )
		n_params=n;
	else if( n!=n_params ){
		sprintf(ERROR_STRING,"setminmax:  n_params = %d, n = %d",n_params,n);
		advise(ERROR_STRING);
		warn("parameter count mismatch");
	}

	for(i=0;i<n;i++){
		xmin[i] = minarr[i];
		xmax[i] = maxarr[i];
	}
}

void _set_opt_param_delta(QSP_ARG_DECL  double *delarr, double *dmnarr, int n)
{
	int i;

	if( n_params==0 )
		n_params=n;
	else if( n!=n_params ){
		sprintf(ERROR_STRING,"setminmax:  n_params = %d, n = %d",n_params,n);
		advise(ERROR_STRING);
		warn("parameter count mismatch");
	}

	for(i=0;i<n;i++){
		deltax[i] = delarr[i];
		delmin[i] = dmnarr[i];
	}
}

void settrace(int nt)
{
	ntrace = nt;
}

void setmaxcalls(int n)
{
}

