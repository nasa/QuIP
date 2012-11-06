

#include "quip_config.h"

char VersionId_newvec_dispatch[] = QUIP_VERSION_STRING;

#include <string.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>	/* getpid() */
#endif

#if N_PROCESSORS>1
/* BUG - should have ifdef for pthreads here too... */
#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif
#endif

#include "nvf.h"
#include "my_cpuid.h"
#include "os_check.h"
#include "debug.h"

/* int n_processors=N_PROCESSORS; */
int n_processors=1;
int for_real=1;	/* if zero then just parse commands */


/* globals */
int use_sse_extensions=0;
Vec_Func *this_vfp;

#if N_PROCESSORS>1

static int parallelized;	/* set to 1 after parallelization */
				/* or if impossible! */

/* These are the data to be passed to each data processing thread */


/* we give a thread work EITHER with a function (pi_func) and args,
 * OR with a vector function (pi_vfp), and a level with which to
 * call vectorize()
 */

typedef struct proc_info {
	Vec_Obj_Args *	pi_oap;
	void		(* pi_func)(Vec_Obj_Args *);
	Vec_Func *	pi_vfp;
	int		pi_level;
	int		pi_index;
} Proc_Info;

static Proc_Info pi[N_PROCESSORS];
static pthread_t dp_thr[N_PROCESSORS];
static int n_threads_started=0;

#endif /* N_PROCESSORS>1 */


#if N_PROCESSORS > 1
#ifdef DEBUG

/* Different threads each need to use their own message
 * buffer to avoid conflicts.
 */

static void mpnadvise(int index, const char *msg)
{
	char str[LLEN];

	sprintf(str,"thread %d:  %s",index,msg);
	NADVISE(str);
}
	
#endif /* DEBUG */

/* sched_yield doesn't really do what we want... */

/* These are commented out, presumably because it slows things
 * down too much.  Not yielding the processor could work when
 * the number of threads is less than or equal to the number
 * of actual processors...
 */


//#define YIELD_PROC()		usleep(10)
//#define YIELD_PROC		usleep(1);
#define YIELD_PROC


#endif /* N_PROCESSORS > 1 */

COMMAND_FUNC( set_use_sse )
{
	int yn;

	yn = ASKIF("Use SSE extensions");

#ifdef USE_SSE
	if( yn ){
		/* make sure that processor and OS are ok */
		if( ! cpu_supports_mmx() ){
			WARN("This CPU does not support the SSE extensions");
			return;
		}
#ifdef FOOBAR
		/* We are on the 2.6 kernel now, there's no going back! */
		if( ! os_supports_mmx() ){
			WARN("OS version must be at least 2.4 for SSE extensions");
			return;
		}
#endif /* FOOBAR */
		use_sse_extensions = 1;
	} else	use_sse_extensions = 0;

	sprintf(ERROR_STRING,"use_sse_extensions = %d",use_sse_extensions);
	NADVISE(ERROR_STRING);
#else /* ! USE_SSE */
	WARN("No support for SSE extensions; recompile with USE_SSE defined.");
#endif /* ! USE_SSE */
}

COMMAND_FUNC( set_n_processors )
{
	int n;

	n=HOW_MANY("number of processors to use");
	if( n >=1 && n <= N_PROCESSORS ){
		n_processors = n;
	} else {
		sprintf(ERROR_STRING,"%d processors requested, %d max on this machine",
			n,N_PROCESSORS);
		WARN(ERROR_STRING);
	}
}

#if N_PROCESSORS > 1

static void *data_processor(void *argp)
{
	Proc_Info *pip;
#ifdef DEBUG
	char mystring[LLEN];
	int myindex;
#endif /* DEBUG */
//int advised;
//int n_waits=0;

	pip = (Proc_Info*) argp;
#ifdef DEBUG
	myindex=pip->pi_index;
#endif /* DEBUG */
//mpnadvise(myindex,"BEGIN");

	while(1){
		/* wait for something to do */
//advised=0;
		while( pip->pi_oap == NO_VEC_OBJ_ARGS ){
//if( ! advised ){
//advised=1;
//mpnadvise(myindex,"waiting for work");
//}

//n_waits++;
//if( n_waits >= 10000 ){
//mpnadvise(myindex,"still waiting for work");
//n_waits=0;
//}
			/* We seem to get stuck here (with no YIELD_PROC),
			 * but it's fixed when we put in periodic
			 * printf's...  try w/ YIELD_PROC
			 */
			YIELD_PROC
		}
//sprintf(mystring,"dataproc READY  pip = 0x%lx, pip->pi_oap = 0x%lx",
//(int_for_addr)pip,(int_for_addr)pip->pi_oap);
//mpnadvise(myindex,mystring);

		if( pip->pi_func == NULL )
			/*
			subvectorize(pip->pi_vfp,pip->pi_oap,pip->pi_level);
			*/
			NERROR1("data_processor:  Sorry, don't know how to subvectorize!?");
		else {
#ifdef DEBUG
if( debug & veclib_debug ){
sprintf(mystring,"thread[%d]:  func = 0x%lx",pip->pi_index,(int_for_addr)pip->pi_func);
mpnadvise(myindex,mystring);
/* show_obj_args is not thread safe */
private_show_obj_args(mystring,pip->pi_oap,advise);
}
#endif /* DEBUG */
			(*(pip->pi_func))(pip->pi_oap);
//mpnadvise(myindex,"BACK from function call");
		}

//sprintf(mystring,"dataproc FINISHED working  pip = 0x%lx, pip->pi_oap = 0x%lx",
//(int_for_addr)pip,(int_for_addr)pip->pi_oap);
//mpnadvise(myindex,mystring);
		/* indicate work finished */
		pip->pi_oap = NO_VEC_OBJ_ARGS;

	}
	return(NULL);	/* NOTREACHED */
} /* end data_processor */

static void start_dataproc_threads(void)
{
	int i;
	pthread_attr_t attr1;

	pthread_attr_init(&attr1);	/* initialize to default values */
	pthread_attr_setinheritsched(&attr1,PTHREAD_INHERIT_SCHED);

	for(i=n_threads_started;i<(n_processors-1);i++){
sprintf(DEFAULT_ERROR_STRING,"creating new data processing thread %d",i);
NADVISE(DEFAULT_ERROR_STRING);
		pi[i].pi_oap = NO_VEC_OBJ_ARGS;
		pi[i].pi_func = NULL;
		pi[i].pi_vfp = NO_NEW_VEC_FUNC;
		pi[i].pi_level = (-1);
		pi[i].pi_index = i;
		pthread_create(&dp_thr[i],&attr1,data_processor,&pi[i]);
	}
	n_threads_started=i;
}

static void wait_for_threads(void)
{
	int i,busy;

//void *v_oap[N_PROCESSORS];
//for(i=0;i<(n_processors-1);i++)
//v_oap[i]=NULL;
	busy=1;
	while(busy){
		YIELD_PROC
		/* YIELD_PROC(); */
		/* give the data processors a chance to run */
		busy=0;
		for(i=0;i<(n_processors-1);i++)
/*
advise("waiting for data processing thread");
*/
			if( pi[i].pi_oap != NO_VEC_OBJ_ARGS ){
//if( v_oap[i] != pi[i].pi_oap ){
//sprintf(DEFAULT_ERROR_STRING,"waiting for thread %d, oap = 0x%lx",i,(int_for_addr)pi[i].pi_oap);
//advise(DEFAULT_ERROR_STRING);
//v_oap[i] = pi[i].pi_oap;
//}
				busy=1;
			}
//else {
//sprintf(DEFAULT_ERROR_STRING,"wait_for_threads: %d, oap = 0x%lx",i,(int_for_addr)pi[i].pi_oap);
//advise(DEFAULT_ERROR_STRING);
//}
	}
//advise("wait_for_threads:  DONE waiting for all threads");
}

#endif /* N_PROCESORS > 1 */

#if N_PROCESSORS > 1


#define D_TOP		0
#define D_WAITING	1
#define D_WORKING	2
#define D_DONE		3
#define L_WAITING	4
#define L_CHECKING	5
#define L_BUSY		6
#define L_DONE		7

/* We used to auto-initialize here, but the compiler doesn't warn us
 * any more if we don't have the full number of initializers
 */

static int tmp_obj_names_inited=0;

static const char *tmp_obj_name[1+MAX_SRC_OBJECTS][N_PROCESSORS];
static Data_Obj *tmp_obj[1+MAX_SRC_OBJECTS][N_PROCESSORS];

static void init_tmp_obj_names(void)
{
	int i_proc;
	int i_src;
	char name[32];	// 32 is overkill

	for(i_proc=0;i_proc<N_PROCESSORS;i_proc++){
		sprintf(name,"_tmp_pdst_%d",i_proc);
		tmp_obj_name[0][i_proc] = savestr(name);
		tmp_obj[0][i_proc]=NO_OBJ;
		for(i_src=0;i_src<MAX_SRC_OBJECTS;i_src++){
			sprintf(name,"_tmp_psrc%d_%d",i_src+1,i_proc);
			tmp_obj_name[i_src+1][i_proc] = savestr(name);
			tmp_obj[i_src+1][i_proc]=NO_OBJ;
		}
	}
	tmp_obj_names_inited=1;
}

void launch_threads(QSP_ARG_DECL  void (*func)(Vec_Obj_Args *), Vec_Obj_Args oa[])
{
	int i;

	if( n_threads_started < (n_processors-1) )
		start_dataproc_threads();

	for(i=0;i<(n_processors-1);i++){
		pi[i].pi_func = func;
		/* set this last */
		pi[i].pi_oap = &oa[i];
#ifdef DEBUG
if( debug & veclib_debug ){
sprintf(DEFAULT_ERROR_STRING,"launch_threads: enabling data processing thread %d, pi_oap = 0x%lx",
i,(int_for_addr)pi[i].pi_oap);
NADVISE(DEFAULT_ERROR_STRING);
//show_obj_args(&oa[i]);
}
#endif /* DEBUG */
	}

	/* this thread does some of the work too! */
	(*func)(&oa[n_processors-1]);
	
	wait_for_threads();

	/* now we might consider deleting the temp objects... */
	for(i=0;i<n_processors;i++){
		int j;
		for(j=0;j<(MAX_SRC_OBJECTS+1);j++){
			if( tmp_obj[j][i] != NO_OBJ ){
				delvec(QSP_ARG  tmp_obj[j][i]);
				tmp_obj[j][i] = NO_OBJ;
			}
		}
	}
}


static Scalar_Value private_scalar[N_PROCESSORS];

#endif /* N_PROCESSORS > 1 */

/* FIXIT
 * Takes an argument, and splits it into N arguments, one per processor.
 * uses variables i_dim, n_processors
 *
 */

#define FIXIT( arg_dp, arg_index )						\
{										\
	index_t offsets[N_DIMENSIONS]={0,0,0,0,0};				\
										\
	n_per_thread = oap->arg_dp->dt_type_dim[i_dim] / n_processors;	\
/*sprintf(DEFAULT_ERROR_STRING,"FIXIT %s (index = %d):  i_dim = %d   n_per_thread = %ld",		\
oap->arg_dp->dt_name, arg_index, i_dim,	\
							n_per_thread);		\
NADVISE(DEFAULT_ERROR_STRING);*/								\
	dimset = oap->arg_dp->dt_type_dimset;					\
	/* special case the last chunk (later) in case of a remainder... */	\
	dimset.ds_dimension[i_dim] = n_per_thread;				\
										\
	for(i=0;i<n_processors;i++){						\
		if( i==(n_processors-1) ){					\
			/* take care of remainder... */				\
			dimset.ds_dimension[i_dim] =				\
	oap->arg_dp->dt_type_dim[i_dim] - n_per_thread * (n_processors-1);	\
		}								\
/*sprintf(DEFAULT_ERROR_STRING,"thread %d:  offset = %ld, n = %ld",i,			\
 * offsets[i_dim],dimset.ds_dimension[i_dim]);					\
NADVISE(DEFAULT_ERROR_STRING);*/								\
		oa_tbl[i].arg_dp = tmp_obj[arg_index][i] =			\
			mk_subseq(QSP_ARG  tmp_obj_name[arg_index][i],		\
					oap->arg_dp,offsets,&dimset);		\
/*sprintf(DEFAULT_ERROR_STRING,"tmp_obj[%d][%d] %s created",				\
arg_index,i,tmp_obj[arg_index][i]->dt_name);					\
NADVISE(DEFAULT_ERROR_STRING);*/								\
		offsets[i_dim] += n_per_thread;					\
	}									\
}


/* vec_dispatch  -  call the library routine */

void vec_dispatch(QSP_ARG_DECL  Vec_Func *vfp,Vec_Obj_Args *oap)
{
	/* This is a kludge; we remember the function in a global so we can print
	 * the function name from nullf if there is a problem...
	 * BUG this is not thread-safe!?  Add a variable to Query_Stream
	 * and it will be ok...
	 */

	this_vfp = vfp;


	/* BUG the chain block should be set up earlier, when
	 * the global arg vars are set!?
	 *
	 * We did some stuff on is_chaining here (see vec_dispatch()),
	 * but let's get this working...
	 */

#ifdef DEBUG
	if( debug & veclib_debug ) {
		sprintf(DEFAULT_ERROR_STRING,"\nvec_dispatch:  Function %s",
			vfp->vf_name);
		NADVISE(DEFAULT_ERROR_STRING);

		show_obj_args(oap);
	}
#endif /* DEBUG */

#if N_PROCESSORS > 1
	/* Don't try to parallelize anything with a scalar destination...
	 * Things like vsum are possible but hard and we're punting.
	 * Scalar assignments have no place here, and this will catch them.
	 */

	if( (!parallelized) && n_processors > 1 && ! IS_SCALAR(oap->oa_dest) ){
		int i, i_dim;

		Vec_Obj_Args oa_tbl[N_PROCESSORS];
		/* u_long n_per_proc,n_extra; */

		if( vfp->vf_flags & NEIGHBOR_COMPARISONS )	/* vsum, vmaxv, vmaxg etc */
			goto use_single_processor;


		/* The bitmap arg is used with vv_select, vs_select, ss_select
		 *
		 * The operations with bitmap result have the bitmap as one
		 * of the official vector args, which is handled correctly
		 * without any special treatent.  --Unless the number per
		 * processor is lt 1!?
		 */
		if( vfp->vf_flags & BITMAP_SRC )
			goto use_single_processor;
		else if( vfp->vf_flags & BITMAP_DST )
			goto use_single_processor;

		/* The strategy here is to subdivide the highest non-1 dimension.
		 * We scan the destination object - this is great for outer ops,
		 * we subdivide the source objects only if they have a non-1 size
		 * for the given dimension.  But projection ops won't work - ?
		 *
		 * There is also a problem for operations on scalar operations,
		 * e.g.  VSet img[0][0] 5
		 */

		/* copy the args by default */
		for(i=0;i<n_processors;i++){
			oa_tbl[i] = *oap;
		}

		if( !tmp_obj_names_inited )
			init_tmp_obj_names();

		for(i_dim=N_DIMENSIONS-1;i_dim>=0;i_dim--){
			u_long n_per_thread;
			Dimension_Set dimset;

			// BUG? this dimension being not equal to 1
			// is not really the best test, we are counting
			// down in the hopes that each thread will get
			// a contiguous block, but what if this dimension is 2?
			// and the number of processors is much larger?
			if( oap->oa_dest->dt_type_dim[i_dim] != 1 ){
				FIXIT(oa_dest,0)
				if( oap->oa_1 != NO_OBJ &&
						oap->oa_1->dt_type_dim[i_dim] > 1 )
					FIXIT(oa_1,1)
				if( oap->oa_2 != NO_OBJ &&
						oap->oa_2->dt_type_dim[i_dim] > 1 )
					FIXIT(oa_2,2)
				/* BUG more than 2 source args? */
				/* make sure we exit loop now */
				i_dim = (-2);	// show that we found something...
			}
		}
#ifdef CAUTIOUS
		if( i_dim == -1 ){
NWARN("CAUTIOUS:  could't find a dimension to subdivide for multi-proc!?");
show_obj_args(oap);
		}
#endif /* CAUTIOUS */

		if( vfp->vf_code == FVRAMP1D ){
			/* need to fix starting value scalars for vramp... */

			/* BUG we'd like to do this in a precision independent way,
			 * but we need some additional functions to do scalar math.
			 */

			float inc;
NWARN("Arghh - probably botching vramp scalar args for multiple processors...");
			if( oa_tbl[0].oa_dest->dt_prec != PREC_SP ){
				NWARN("sorry, can only fix vramp multiprocessor args for float precision");
				return;
			}
			extract_scalar_value( &private_scalar[0], oap->oa_s1 );
			extract_scalar_value( (Scalar_Value *)(&inc), oap->oa_s2 );
			for(i=1;i<n_processors;i++){
				/*
				private_scalar[i].u_f = private_scalar[i-1].u_f
					+ va[i-1].arg_n1 * inc;
				va[i].arg_scalar1 = &private_scalar[i];
				*/
				/* BUG need to fix ramp args */
				oa_tbl[i].oa_s1 = /* Need more objects */ NO_OBJ;
			}
		} else if( vfp->vf_code == FVRAMP2D ){
			NWARN("NEED TO FIX CODE FOR MULTIPROC RAMP2D");
			goto use_single_processor;
		}

		/* BUG need to fix bitmaps */
		/* what about fixing bitmaps??? */
		/* BUG functions that retunr scalars need special treatment */


		/* Now we're ready to turn the threads loose
		 * on this computation.
		 */
#ifdef DEBUG
		if( debug & veclib_debug ) {
			sprintf(DEFAULT_ERROR_STRING,"\n\nvec_dispatch %s, using %d processors\n",
				vfp->vf_name,n_processors);
			NADVISE(DEFAULT_ERROR_STRING);
	}
#endif /* DEBUG */

		/* launch n-1 threads.
		 * The last thread is the control one?
		 * i.e. this one?
		 */
//sprintf(DEFAULT_ERROR_STRING,"calling launch_threads, vf_code = %d, functype = %d, func = 0x%lx",
//vfp->vf_code,oap->oa_functype,(int_for_addr)vfa_tbl[vfp->vf_code].vfa_func[oap->oa_functype]);
//NADVISE(DEFAULT_ERROR_STRING);
		launch_threads(QSP_ARG  vfa_tbl[vfp->vf_code].vfa_func[oap->oa_functype],oa_tbl);
		return;
	}
use_single_processor:

#ifdef DEBUG
	if( debug & veclib_debug ) {
		sprintf(DEFAULT_ERROR_STRING,"vec_dispatch %s, using single processor\n",
			vfp->vf_name);
		NADVISE(DEFAULT_ERROR_STRING);
		/* show_obj_args(oap); */
	}
#endif /* DEBUG */

#endif	/* N_PROCESSORS > 1 */

#ifdef CAUTIOUS
	if( oap->oa_argsprec < 0 || oap->oa_argsprec >= N_ARGSET_PRECISIONS ){
		sprintf(ERROR_STRING,"CAUTIOUS:  vec_dispatch:  bad argset precision %d",oap->oa_argsprec);
		ERROR1(ERROR_STRING);
	}
	if( vfp->vf_code != vfa_tbl[vfp->vf_code].vfa_code ){
		sprintf(ERROR_STRING,"CAUTIOUS: Code mismatch, table entry %d has code %d - expected %s",
			vfp->vf_code,
			vfa_tbl[vfp->vf_code].vfa_code,vfp->vf_name);
		ERROR1(ERROR_STRING);
	}
#endif /* CAUTIOUS */

	/* not really a BUG, but this may be redundant... */
	/* Where should this be done??? */
	oap->oa_functype = FUNCTYPE_FOR(oap->oa_argsprec,oap->oa_argstype);

#ifdef DEBUG
if( debug & veclib_debug ){
sprintf(ERROR_STRING,"vec_dispatch:  calling tabled function, code = %d, functype = %d",
vfp->vf_code,oap->oa_functype);
NADVISE(ERROR_STRING);
show_obj_args(oap);
}
#endif /* DEBUG */

	(*vfa_tbl[vfp->vf_code].vfa_func[oap->oa_functype])(oap);

} /* end vec_dispatch */




