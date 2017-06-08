#include "quip_config.h"

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
//#include "debug.h"
//#include "warn.h"
#include "quip_prot.h"
#include "platform.h"
#include "debug.h"

/* int n_processors=N_PROCESSORS; */
int n_processors=1;
int for_real=1;	/* if zero then just parse commands */


/* globals */
int use_sse_extensions=0;
const Vector_Function *this_vfp;

#if N_PROCESSORS>1

static int parallelized;	/* set to 1 after parallelization */
				/* or if impossible! */

/* These are the data to be passed to each data processing thread */


/* we give a thread work EITHER with a function (pi_func) and args,
 * OR with a vector function (pi_vfp), and a level with which to
 * call vectorize()
 */

typedef struct proc_info {
	Vec_Obj_Args *		pi_oap;
	void		(* 	pi_func)(HOST_CALL_ARG_DECLS);
	int			pi_vf_code;
	int			pi_level;
	int			pi_index;
	Query_Stack *		pi_qsp;
} Proc_Info;

static Proc_Info pi[N_PROCESSORS];
static pthread_t dp_thr[N_PROCESSORS];
static int n_threads_started=0;

#endif /* N_PROCESSORS>1 */


#if N_PROCESSORS > 1
#ifdef QUIP_DEBUG

/* Different threads each need to use their own message
 * buffer to avoid conflicts.
 */

#define MAX_MSG_LEN	256
#define MAX_DIGITS	8	// overkill

static void mpnadvise(int index, const char *msg)
{
	char str[MAX_MSG_LEN+1];
	if( strlen("thread :  ")+MAX_DIGITS+strlen(msg) >= MAX_MSG_LEN )
		NERROR1("mpnadvise:  need to increase MAX_MSG_LEN!?");

	sprintf(str,"thread %d:  %s",index,msg);
	NADVISE(str);
}
	
#endif /* QUIP_DEBUG */

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
#ifdef USE_SSE
	int yn;
    
	yn = ASKIF("Use SSE extensions");
    
	if( yn ){
		/* make sure that processor and OS are ok */
		if( ! cpu_supports_sse() ){
			// Do we need SSE or SSE2??? BUG?
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
	ASKIF("Use SSE extensions (ineffective)");
	WARN("No support for SSE extensions; recompile with USE_SSE defined.");
#endif /* ! USE_SSE */
}

COMMAND_FUNC( set_n_processors )
{
	int n;

	n=(int)HOW_MANY("number of processors to use");
	if( n >=1 && n <= N_PROCESSORS ){
		n_processors = n;
	} else {
		sprintf(ERROR_STRING,"%d processors requested, %d max on this machine",
			n,N_PROCESSORS);
		WARN(ERROR_STRING);
	}
}

#ifdef MAX_DEBUG
static void dump_tbl_row(Vec_Func_Array *row)
{
	int i;

	fprintf(stderr,"code = %d\n",row->vfa_code);
	for(i=0;i<N_FUNCTION_TYPES;i++){
		fprintf(stderr,"%d:\t0x%lx\n",i,(u_long)row->vfa_func[i]);
	}
}
#endif // MAX_DEBUG


debug_flag_t veclib_debug=0;
#if N_PROCESSORS > 1

static void *data_processor(void *argp)
{
	Proc_Info *pip;
#ifdef QUIP_DEBUG
#define BUF_LEN	256
	char mystring[BUF_LEN];	// BUG need to insure no overrun
	int myindex;
#endif /* QUIP_DEBUG */
	Query_Stack *qsp;
//int advised;
//int n_waits=0;

	pip = (Proc_Info*) argp;
	qsp = pip->pi_qsp;
#ifdef QUIP_DEBUG
	myindex=pip->pi_index;
#endif /* QUIP_DEBUG */
//mpnadvise(myindex,"BEGIN");

	while(1){
		/* wait for something to do */
//advised=0;
		while( pip->pi_oap == NULL ){
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
#ifdef QUIP_DEBUG
if( debug & veclib_debug ){
sprintf(mystring,"thread[%d]:  func = 0x%lx",pip->pi_index,(int_for_addr)pip->pi_func);
mpnadvise(myindex,mystring);
/* show_obj_args is not thread safe */
private_show_obj_args(QSP_ARG  mystring,pip->pi_oap,_advise);
}
#endif /* QUIP_DEBUG */
			(*(pip->pi_func))(pip->pi_vf_code,pip->pi_oap);
//mpnadvise(myindex,"BACK from function call");
		}

//sprintf(mystring,"dataproc FINISHED working  pip = 0x%lx, pip->pi_oap = 0x%lx",
//(int_for_addr)pip,(int_for_addr)pip->pi_oap);
//mpnadvise(myindex,mystring);
		/* indicate work finished */
		pip->pi_oap = NULL;

	}
	return(NULL);	/* NOTREACHED */
} /* end data_processor */

static void start_dataproc_threads(SINGLE_QSP_ARG_DECL)
{
	int i;
	pthread_attr_t attr1;
	char thread_name[32];

	pthread_attr_init(&attr1);	/* initialize to default values */
	pthread_attr_setinheritsched(&attr1,PTHREAD_INHERIT_SCHED);

	for(i=n_threads_started;i<(n_processors-1);i++){
sprintf(DEFAULT_ERROR_STRING,"creating new data processing thread %d",i);
NADVISE(DEFAULT_ERROR_STRING);
		pi[i].pi_oap = NULL;
		pi[i].pi_func = NULL;
		pi[i].pi_vf_code = (-1);	// BUG need to set this?
		pi[i].pi_level = (-1);
		pi[i].pi_index = i;
		// should each thread have its own qsp?
		sprintf(thread_name,"compute_thread_%d",i);
		pi[i].pi_qsp = new_query_stack(QSP_ARG  thread_name);
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
			if( pi[i].pi_oap != NULL ){
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
		tmp_obj[0][i_proc]=NULL;
		for(i_src=0;i_src<MAX_SRC_OBJECTS;i_src++){
			sprintf(name,"_tmp_psrc%d_%d",i_src+1,i_proc);
			tmp_obj_name[i_src+1][i_proc] = savestr(name);
			tmp_obj[i_src+1][i_proc]=NULL;
		}
	}
	tmp_obj_names_inited=1;
}

void launch_threads(QSP_ARG_DECL
	void (*func)(HOST_CALL_ARG_DECLS),
	int vf_code, Vec_Obj_Args oa[])
{
	int i;

	if( n_threads_started < (n_processors-1) )
		start_dataproc_threads(SINGLE_QSP_ARG);

	for(i=0;i<(n_processors-1);i++){
		pi[i].pi_func = func;
		/* set this last */
		pi[i].pi_oap = &oa[i];
		pi[i].pi_vf_code = vf_code;
#ifdef QUIP_DEBUG
if( debug & veclib_debug ){
sprintf(DEFAULT_ERROR_STRING,"launch_threads: enabling data processing thread %d, pi_oap = 0x%lx",
i,(int_for_addr)pi[i].pi_oap);
NADVISE(DEFAULT_ERROR_STRING);
//show_obj_args(QSP_ARG  &oa[i]);
}
#endif /* QUIP_DEBUG */
	}

	/* this thread does some of the work too! */
	(*func)(vf_code,&oa[n_processors-1]);
	
	wait_for_threads();

	/* now we might consider deleting the temp objects... */
	for(i=0;i<n_processors;i++){
		int j;
		for(j=0;j<(MAX_SRC_OBJECTS+1);j++){
			if( tmp_obj[j][i] != NULL ){
				delvec(QSP_ARG  tmp_obj[j][i]);
				tmp_obj[j][i] = NULL;
			}
		}
	}
}


static Scalar_Value private_scalar[N_PROCESSORS];


/* FIXIT
 * Takes an argument, and splits it into N arguments, one per processor.
 * uses variables i_dim, n_processors
 *
 */

#define FIXIT( dest_dp, dp, arg_index )						\
{										\
	index_t offsets[N_DIMENSIONS]={0,0,0,0,0};				\
										\
	n_per_thread = OBJ_TYPE_DIM(dp,i_dim) / n_processors;	\
/*sprintf(DEFAULT_ERROR_STRING,"FIXIT %s (index = %d):  i_dim = %d   n_per_thread = %ld",		\
OBJ_NAME(dp), arg_index, i_dim,	\
							n_per_thread);		\
NADVISE(DEFAULT_ERROR_STRING);*/								\
	COPY_DIMS(dsp,OBJ_TYPE_DIMS(dp));					\
	/* special case the last chunk (later) in case of a remainder... */	\
	SET_DIMENSION(dsp,i_dim, n_per_thread);				\
										\
	for(i=0;i<n_processors;i++){						\
		if( i==(n_processors-1) ){					\
			/* take care of remainder... */				\
			SET_DIMENSION(dsp,i_dim,				\
	OBJ_TYPE_DIM(dp,i_dim) - n_per_thread * (n_processors-1) );		\
		}								\
/*sprintf(DEFAULT_ERROR_STRING,"thread %d:  offset = %ld, n = %ld",i,			\
 * offsets[i_dim],DIMENSION(dsp,i_dim));					\
NADVISE(DEFAULT_ERROR_STRING);*/								\
		dest_dp = tmp_obj[arg_index][i] =			\
			mk_subseq(QSP_ARG  tmp_obj_name[arg_index][i],		\
					dp,offsets,dsp);		\
/*sprintf(DEFAULT_ERROR_STRING,"tmp_obj[%d][%d] %s created",				\
arg_index,i,OBJ_NAME(tmp_obj[arg_index][i]));					\
NADVISE(DEFAULT_ERROR_STRING);*/								\
		offsets[i_dim] += n_per_thread;					\
	}									\
}


// We want to do this, only for the cpu platform!

static int multiprocessor_dispatch(QSP_ARG_DECL  const Vector_Function *vfp,
		Vec_Obj_Args *oap )
{
	int i, i_dim;
	Vec_Obj_Args oa_tbl[N_PROCESSORS];


	// Now this is general code for all platforms?

	/* Don't try to parallelize anything with a scalar destination...
	 * Things like vsum are possible but hard and we're punting.
	 * Scalar assignments have no place here, and this will catch them.
	 */

	if( parallelized )
		return -1;
	if( ! IS_SCALAR(oap->oa_dest) )
		return -1;

	/* u_long n_per_proc,n_extra; */

	if( vfp->vf_flags & NEIGHBOR_COMPARISONS )	/* vsum, vmaxv, vmaxg etc */
		return -1;


	/* The bitmap arg is used with vv_select, vs_select, ss_select
	 *
	 * The operations with bitmap result have the bitmap as one
	 * of the official vector args, which is handled correctly
	 * without any special treatent.  --Unless the number per
	 * processor is lt 1!?
	 */
	if( vfp->vf_flags & BITMAP_SRC )
		return -1;
	else if( vfp->vf_flags & BITMAP_DST )
		return -1;

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
		Dimension_Set *dsp;

		INIT_DIMSET_PTR(dsp)

		// BUG? this dimension being not equal to 1
		// is not really the best test, we are counting
		// down in the hopes that each thread will get
		// a contiguous block, but what if this dimension is 2?
		// and the number of processors is much larger?
		if( OBJ_TYPE_DIM(oap->oa_dest,i_dim) != 1 ){
			FIXIT(OA_DEST(&oa_tbl[i]),OA_DEST(oap),0)
			if( OA_SRC1(oap) != NULL &&
					OBJ_TYPE_DIM(OA_SRC1(oap),i_dim) > 1 )
				FIXIT(OA_SRC1(&oa_tbl[i]),OA_SRC1(oap),1)
			if( OA_SRC2(oap) != NULL &&
					OBJ_TYPE_DIM(OA_SRC2(oap),i_dim) > 1 )
				FIXIT(OA_SRC2(&oa_tbl[i]),OA_SRC2(oap),2)
			/* BUG more than 2 source args? */
			/* make sure we exit loop now */
			i_dim = (-2);	// show that we found something...
		}
	}
//#ifdef CAUTIOUS
//	if( i_dim == -1 ){
//NWARN("CAUTIOUS:  could't find a dimension to subdivide for multi-proc!?");
//show_obj_args(QSP_ARG  oap);
//	}
//#endif /* CAUTIOUS */

	assert( i_dim != (-1) );

	if( VF_CODE(vfp) == FVRAMP1D ){
		/* need to fix starting value scalars for vramp... */

		/* BUG we'd like to do this in a precision independent way,
		 * but we need some additional functions to do scalar math.
		 */

		float inc;
NWARN("Arghh - probably botching vramp scalar args for multiple processors...");
		if( OBJ_PREC(OA_DEST(&oa_tbl[0])) != PREC_SP ){
			NWARN("sorry, can only fix vramp multiprocessor args for float precision");
			return -1;
		}
		//extract_scalar_value( &private_scalar[0], OA_SVAL1(oap) );
		//extract_scalar_value( (Scalar_Value *)(&inc), OA_SVAL2(oap) );
		extract_scalar_value(QSP_ARG  &private_scalar[0], OA_SCLR1(oap) );
		extract_scalar_value(QSP_ARG  (Scalar_Value *)(&inc), OA_SCLR2(oap) );
		for(i=1;i<n_processors;i++){
			/*
			private_scalar[i].u_f = private_scalar[i-1].u_f
				+ va[i-1].arg_n1 * inc;
			va[i].arg_scalar1 = &private_scalar[i];
			*/
			/* BUG need to fix ramp args */
			oa_tbl[i].oa_sdp[0] = /* Need more objects */ NULL;
		}
	} else if( VF_CODE(vfp) == FVRAMP2D ){
		WARN("NEED TO FIX CODE FOR MULTIPROC RAMP2D");
		return -1;
	}

	/* BUG need to fix bitmaps */
	/* what about fixing bitmaps??? */
	/* BUG functions that retunr scalars need special treatment */


	/* Now we're ready to turn the threads loose
	 * on this computation.
	 */
#ifdef QUIP_DEBUG
	if( debug & veclib_debug ) {
		sprintf(ERROR_STRING,"\n\nvec_dispatch %s, using %d processors\n",
			VF_NAME(vfp),n_processors);
		ADVISE(DEFAULT_ERROR_STRING);
	}
#endif /* QUIP_DEBUG */

// BUG - add support for different platforms
// this used to just be vfa_tbl

	/* launch n-1 threads.
	 * The last thread is the control one?
	 * i.e. this one?
	 */
//sprintf(DEFAULT_ERROR_STRING,"calling launch_threads, vf_code = %d, functype = %d, func = 0x%lx",
//VF_CODE(vfp),OA_FUNCTYPE(oap),(int_for_addr)vl2_vfa_tbl[VF_CODE(vfp)].vfa_func[OA_FUNCTYPE(oap)]);
//NADVISE(DEFAULT_ERROR_STRING);
	// We can use vl2_vfa_tbl here because we know we are on
	// the cpu platform...
	launch_threads(QSP_ARG  vl2_vfa_tbl[VF_CODE(vfp)].vfa_func[OA_FUNCTYPE(oap)],VF_CODE(vfp),oa_tbl);
	return 0;
}
#endif /* N_PROCESSORS > 1 */

int platform_dispatch( QSP_ARG_DECL  const Compute_Platform *cpp,
				const Vector_Function *vfp,
				Vec_Obj_Args *oap )
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

#ifdef QUIP_DEBUG
	if( debug & veclib_debug ) {
		sprintf(DEFAULT_ERROR_STRING,"\nvec_dispatch:  Function %s",
			VF_NAME(vfp));
		NADVISE(DEFAULT_ERROR_STRING);

		show_obj_args(QSP_ARG  oap);
	}
#endif /* QUIP_DEBUG */

#if N_PROCESSORS > 1
	// BUG need to perform this test:
	// If platform is CPU, and n_processors>1,
	// then call multiprocessor_dispatch

	if( IS_CPU_DEVICE( OA_PFDEV(oap) ) && n_processors > 1 ){
		if( multiprocessor_dispatch(QSP_ARG  vfp, oap ) == 0 )
			return 0;
	}
#endif	/* N_PROCESSORS > 1 */

//#ifdef CAUTIOUS
//	if( /* OA_ARGSPREC(oap) < 0 || */ OA_ARGSPREC(oap) >= N_ARGSET_PRECISIONS ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  vec_dispatch:  bad argset precision %d",OA_ARGSPREC(oap));
//		ERROR1(ERROR_STRING);
//	}

	assert( OA_ARGSPREC(oap) < N_ARGSET_PRECISIONS );

#ifdef CAUTIOUS
	if( PF_FUNC_TBL(cpp) == NULL ){
		sprintf(ERROR_STRING,
"CAUTIOUS:  platform_dispatch:  vfa_tbl has not been set for platform %s!?",
			PLATFORM_NAME(cpp));
		WARN(ERROR_STRING);
	}
#endif // CAUTIOUS

	assert( PF_FUNC_TBL(cpp) != NULL );

#ifdef CAUTIOUS
	if( VF_CODE(vfp) != PF_FUNC_TBL(cpp)[VF_CODE(vfp)].vfa_code ){
		sprintf(ERROR_STRING,
"CAUTIOUS:  platform_dispatch:  table entry %d has code %d - expected %s",
			VF_CODE(vfp),
			PF_FUNC_TBL(cpp)[VF_CODE(vfp)].vfa_code,VF_NAME(vfp));
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,
"platform_dispatch:  vfa table for platform %s may not be sorted?",
			PLATFORM_NAME(cpp));
		WARN(ERROR_STRING);
	}
#endif /* CAUTIOUS */

	assert( VF_CODE(vfp) == PF_FUNC_TBL(cpp)[VF_CODE(vfp)].vfa_code );

	/* not really a BUG, but this may be redundant... */
	/* Where should this be done??? */
	OA_FUNCTYPE(oap) = FUNCTYPE_FOR(OA_ARGSPREC(oap),OA_ARGSTYPE(oap));

#ifdef QUIP_DEBUG
if( debug & veclib_debug ){
sprintf(ERROR_STRING,"vec_dispatch:  calling tabled function, code = %d, functype = %d",
VF_CODE(vfp),OA_FUNCTYPE(oap));
NADVISE(ERROR_STRING);
show_obj_args(QSP_ARG  oap);
}
#endif /* QUIP_DEBUG */

#ifdef MAX_DEBUG
dump_tbl_row(&(PF_FUNC_TBL(cpp)[VF_CODE(vfp)]));
fprintf(stderr,"Calling function at 0x%lx\n",
(u_long)PF_FUNC_TBL(cpp)[VF_CODE(vfp)].vfa_func[OA_FUNCTYPE(oap)]);
#endif // MAX_DEBUG

	// Do it!
//fprintf(stderr,"platform_dispatch calling function %s from vfa_tbl...\n",VF_NAME(vfp));
	(*(PF_FUNC_TBL(cpp)[VF_CODE(vfp)].vfa_func[OA_FUNCTYPE(oap)]))(VF_CODE(vfp),oap);
	return 0;

} /* end platform_dispatch */

// Make sure that all of the non-null objects
// have the same device, and return the platform...

#define CHECK_OBJ_PLATFORM(dp)					\
								\
	if( dp != NULL ){					\
		if( pdp == NULL ){				\
			pdp = OBJ_PFDEV(dp);			\
		} else {					\
			if( OBJ_PFDEV(dp) != pdp ){		\
	sprintf(ERROR_STRING,"Object %s has wrong device (%s) - expected %s",	\
	OBJ_NAME(dp),PFDEV_NAME(OBJ_PFDEV(dp)),PFDEV_NAME(pdp));	\
				WARN(ERROR_STRING);		\
				return NULL;			\
			}					\
		}						\
	}
				
static Compute_Platform * set_oargs_platform(QSP_ARG_DECL  Vec_Obj_Args *oap)
{
	Platform_Device *pdp=NULL;
	int i;

	SET_OA_PFDEV(oap,NULL);
	CHECK_OBJ_PLATFORM(OA_DEST(oap))
	for(i=0;i<MAX_N_ARGS;i++){
		CHECK_OBJ_PLATFORM(OA_SRC_OBJ(oap,i));
	}
	for(i=0;i<MAX_RETSCAL_ARGS;i++){
		CHECK_OBJ_PLATFORM(OA_SCLR_OBJ(oap,i));
	}
	SET_OA_PFDEV(oap,pdp);
	return PFDEV_PLATFORM(pdp);
}

int platform_dispatch_by_code( QSP_ARG_DECL   int code, Vec_Obj_Args *oap )
{
	Compute_Platform *cpp;
	Vector_Function *vfp;

	assert( code >= 0 && code < N_VEC_FUNCS );

	vfp = &(vec_func_tbl[code]);
	cpp = set_oargs_platform(QSP_ARG  oap);
	return platform_dispatch( QSP_ARG  cpp, vfp, oap );
}

void dp_convert(QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp )
{
	int code;
	Vec_Obj_Args oa1,*oap=&oa1;

	assert( dst_dp != NULL );
	assert( src_dp != NULL );

	switch( OBJ_MACH_PREC(dst_dp) ){
		case PREC_BY: code=FVCONV2BY; break;
		case PREC_IN: code=FVCONV2IN; break;
		case PREC_DI: code=FVCONV2DI; break;
		case PREC_LI: code=FVCONV2LI; break;
		case PREC_UBY: code=FVCONV2UBY; break;
		case PREC_UIN: code=FVCONV2UIN; break;
		case PREC_UDI: code=FVCONV2UDI; break;
		case PREC_ULI: code=FVCONV2ULI; break;
		case PREC_SP: code=FVCONV2SP; break;
		case PREC_DP: code=FVCONV2DP; break;
		default:
			sprintf(ERROR_STRING,
"dp_convert:  destination object %s has unexpected precision (%s)!?",
				OBJ_NAME(dst_dp),
				NAME_FOR_PREC_CODE(OBJ_MACH_PREC(dst_dp)) );
			WARN(ERROR_STRING);
			return;
	}

	clear_obj_args(oap);
	setvarg2(oap,dst_dp,src_dp);
	// Need to set argset precision to match source, not destination...
	SET_OA_ARGSPREC(oap, ARGSET_PREC( OBJ_PREC( src_dp ) ) );
//fprintf(stderr,"dp_convert, dispatching conversion of %s to %s, func %s (code = %d)\n",
//OBJ_NAME(dst_dp),OBJ_NAME(src_dp),
//VF_NAME( &(vec_func_tbl[code]) ),
//code);
	platform_dispatch_by_code(QSP_ARG  code, oap );
}

