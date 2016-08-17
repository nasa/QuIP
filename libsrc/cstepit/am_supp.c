#include "quip_config.h"

#ifdef HAVE_NUMREC
#ifdef USE_NUMREC

/*
 * linkage to numerical recipes amoeba routine
 */

#include <math.h>

#include "quip_prot.h"
//#include "fitsine.h"
#include "optimize.h"

static Query_Stack *am_qsp=NULL;

/* local prototypes */

static float amoeba_scr_funk(float *p);
static void run_amoeba(float (*func)(float *));
static void show_simplex_verts(void);
static void init_simplex(void);
static float (*user_c_func)(SINGLE_QSP_ARG_DECL);


static float simplex_vertices[MAX_OPT_PARAMS+1][MAX_OPT_PARAMS];

void halt_amoeba(void)
{
	NWARN("Sorry, don't know how to halt amoeba!?");
}

static void show_simplex_verts()
{
	int i,j,n;

	n= eltcount( opt_param_list(SGL_DEFAULT_QSP_ARG) );

	for(i=0;i<=n;i++){
		printf("simplex vertex %d:\n",i);
		for(j=0;j<n;j++)
			printf("\t%g",simplex_vertices[i][j]);
		printf("\n");
	}
}

static void init_simplex()
{
	List *lp;
	Node *np;
	int n;
	int i,j;

	lp=opt_param_list(SGL_DEFAULT_QSP_ARG);
	if( lp == NO_LIST ){
		NWARN("init_simplex:  no params!?");
		return;
	}
	n = eltcount(lp);
	np=lp->l_head;

	i=0;
	while(np!=NO_NODE){
		Opt_Param *opp;

		opp=(Opt_Param*)np->n_data;
		for(j=0;j<n;j++){
			if( i!=j )
				simplex_vertices[j][i] = opp->ans;
			else
				simplex_vertices[j][i] = opp->ans+opp->delta;
		}
		simplex_vertices[n][i] = opp->ans;
		np=np->n_next;
		i++;
	}
	if( verbose )
		show_simplex_verts();
}

static float amoeba_scr_funk(float *p)
{
	char str[128];
	float	err;
	Variable *vp;
	int i;
	List *lp;
	Node *np;
#ifdef THREAD_SAFE_QUERY
	Query_Stack *qsp;
    
	qsp = am_qsp;
#endif // THREAD_SAFE_QUERY

	lp=opt_param_list(SGL_DEFAULT_QSP_ARG);
	if( lp==NO_LIST ) {
		NWARN("no parameters!?");
		return(0.0);
	}

	np=lp->l_head;

	i=0;
	while(np!=NO_NODE){
		Opt_Param *opp;

		opp=(Opt_Param *)np->n_data;
		sprintf(str,"%g",p[i+1]);
		ASSIGN_VAR(opp->op_name,str);
		i++;
		np=np->n_next;
	}

	digest(DEFAULT_QSP_ARG  opt_func_string, OPTIMIZER_FILENAME);
	
	vp=var__of(QSP_ARG  "error");
	if( vp == NO_VARIABLE ) {
		NWARN("variable \"error\" not set!!!");
		err=0.0;
	} else sscanf(VAR_VALUE(vp),"%g",&err);

	return(err);
}

static float amoeba_c_funk(float *p)
{
	float	err;
	int i;
	List *lp;
	Opt_Param *opp;
	Node *np;

	lp=opt_param_list(SGL_DEFAULT_QSP_ARG);
	np=lp->l_head;

	i=0;
	while(np!=NO_NODE){
		opp=(Opt_Param *)np->n_data;
		opp->ans=p[i+1];
		np=np->n_next;
		i++;
	}

	err = (*user_c_func)(SGL_DEFAULT_QSP_ARG);

	return(err);
}

static float *p_rowlist[MAX_OPT_PARAMS+1];
static float y[MAX_OPT_PARAMS];

void run_amoeba_scr(SINGLE_QSP_ARG_DECL)
{
	am_qsp = THIS_QSP;
	run_amoeba( amoeba_scr_funk );
}

void run_amoeba_c( QSP_ARG_DECL  float (*func)(SINGLE_QSP_ARG_DECL) )
{
	user_c_func = func;
	run_amoeba( amoeba_c_funk );
}

static void run_amoeba( float (*func)(float *) )
{
	float ftol;
	int nfunk;
	int i;
	int n;

	init_simplex();

	/* make a matrix p[][] of simplex vertices */

	/* initialize row list */

	n = eltcount( opt_param_list(SGL_DEFAULT_QSP_ARG) );

	for(i=0;i<n+1;i++)
		p_rowlist[i] = (&simplex_vertices[i][0])  - 1;

	/* evaluate the simplex points, in y */

	for(i=0;i<n+1;i++)
		y[i] = (*func)(p_rowlist[i]);

	/* call amoeba */

	/* not sure how to properly set this!? */
	ftol=0.001f;

/*
if( verbose ){
printf("calling amoeba:\n");
printf("p_rowlist = 0x%x, y = 0x%x\n",p_rowlist-1,y-1);
printf("ndim = %d\n",n);
printf("ftol = %f\n",ftol);
printf("funk = 0x%x\n",&func);
printf("&nfunk = 0x%x\n",&nfunk);
}
*/

	/* We subtract 1 here because of Fortran indexing?
	 * The compiler thinks we've made a mistake...
	 */

	amoeba(p_rowlist-1,y-1,n,ftol,func, &nfunk);
}

#endif /* USE_NUMREC */
#endif /* HAVE_NUMREC */

