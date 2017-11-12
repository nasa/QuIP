#include "quip_config.h"
#include "quip_prot.h"
#include "my_fb.h"

#ifdef HAVE_GENLOCK

/* soft genlock of dual head matrox 450
 *
 * We begin by creating a thread for each head, and noting the times
 * of the vertical blanking on each...
 */


#include <stdio.h>

#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>	/* getpid() */
#endif

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>	/* gettimeofday() */
#endif

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif


/* from /usr/src/linux/include/linux/matroxfb.h */
#define FBIO_WAITFORVSYNC	_IOW('F', 0x20, u_int32_t)

#ifdef HAVE_LINUX_FB_H
#include <linux/fb.h>
#endif




static pthread_t fb_thr[MAX_HEADS];
static pthread_attr_t attr1;

typedef struct per_process_info {
	int		ppi_index;
	int		ppi_flags;
	pid_t		ppi_pid;
	int		ppi_fd;
#ifdef THREAD_SAFE_QUERY
	Query_Stack *	ppi_qsp;
#endif // THREAD_SAFE_QUERY
} Proc_Info;

typedef struct vboard_info {
	FB_Info *	vbi_fbip;
	u_long *	vbi_buf;	/* for time delta's */
	u_long		vbi_count;
	u_long		vbi_inc;
#ifdef THREAD_SAFE_QUERY
	Query_Stack *	vbi_qsp;
#endif // THREAD_SAFE_QUERY
} VBoard_Info;

typedef struct pport_info {
	ParPort *	ppti_ppp;
	u_long *	ppti_buf;
	u_long		ppti_count;
	u_long		ppti_inc;
} PPort_Info;

#define DRIFT_MEMORY	0.75	/* using a fading average gives us
				 * a little immunity to sampling jitter */

#define GL_REFRACTORY	20	/* wait this many updates between adjustments */

/* If DRIFT_MEMORY were 0, then we wouldn't have to wait...
 * The ideal value for GL_REFRACTORY is a function of DRIFT_MEMORY
 * and should be computed in a more principled way.  What we want
 * to have happen is for the frame time to oscillate between
 * two values (with unequal duty cycle), keeping the latency
 * within <drift> of zero.  We ought to be able to calculate
 * the expected change in drift for a change in the number of lines...
 * For example, at a nominal 60 Hz (16.67 msec/frame), and a
 * resolution of 1280x1024, adding one line should change the
 * frame time by approximately 16 usecs.  A latency value less than
 * this is not worth correcting...
 */
#define LATENCY_THRESHOLD	33	/* 33 usecs */

int genlock_active=0;
Genlock_Info gli1;

static Proc_Info ppi[MAX_HEADS];
static u_long n_stamps[MAX_HEADS];
static u_long count[MAX_HEADS];

static struct timeval tv_now[MAX_HEADS];
static struct timeval tv_prev[MAX_HEADS];

#define MAX_STAMPS	512
static long delta[MAX_HEADS][MAX_STAMPS];
static double delta_accum[MAX_HEADS];
static double delta_avg[MAX_HEADS][MAX_STAMPS];
static struct timezone tz;

/* WHERE IS THIS DEFINED??? */
//int wait_til_vblank_transition(QSP_ARG  int fd);


static pthread_mutex_t genlock_mutex;
static ParPort *the_ppp=NULL;
static pthread_t vboard_thr, pport_thr;
static pthread_t gl_thr;
static struct timeval *tvp=NULL,tv0,tv1;


static long compute_latency(struct timeval *later_tvp, struct timeval *earlier_tvp)
{
	long d_secs, d_usecs;

	d_secs = later_tvp->tv_sec - earlier_tvp->tv_sec;
	d_usecs = later_tvp->tv_usec - earlier_tvp->tv_usec;

	return( 1000000*d_secs + d_usecs );
}

static u_long delta_usecs()
{
	if( tvp == NULL ){
		gettimeofday(&tv0,&tz);
		tvp = &tv0;
		return(0);
	}
	gettimeofday(&tv1,&tz);
	return( compute_latency(&tv1,&tv0) );
}

static int read_vbl_state(QSP_ARG_DECL  int fd)
{
	struct fb_vblank vbl_info;

	if(ioctl(fd, FBIOGET_VBLANK, &vbl_info)<0) {
		perror("ioctl");
		WARN("ioctl FBIOGET_VBLANK failed!\n");
		return -1;
	}
	return( vbl_info.flags & FB_VBLANK_VBLANKING );
}

static int wait_til_vblank_transition(QSP_ARG_DECL  int fd)
{
	int original_state, current_state;
	original_state = read_vbl_state(QSP_ARG  fd);

	do {
		current_state = read_vbl_state(QSP_ARG  fd);
	} while( current_state == original_state );

	return current_state;
}

static void *vboard_daemon(void *argp)
{
	VBoard_Info *vbip;
	u_long n;
	u_long *lp;
#ifdef THREAD_SAFE_QUERY
	Query_Stack *qsp;
#endif // THREAD_SAFE_QUERY

	vbip = (VBoard_Info *) argp;
	n = vbip->vbi_count;
	lp = vbip->vbi_buf;
#ifdef THREAD_SAFE_QUERY
	qsp = vbip->vbi_qsp;
#endif // THREAD_SAFE_QUERY

	while(n--){
		wait_til_vblank_transition(QSP_ARG  vbip->vbi_fbip->fbi_fd);
		*lp = delta_usecs();
		lp += vbip->vbi_inc;
	}
	return(NULL);
}

static void *pport_logger(void *argp)
{
	PPort_Info *ppip;
	u_long n;
	u_long *lp;
	int s;

	ppip = (PPort_Info *) argp;
	n = ppip->ppti_count;
	lp = ppip->ppti_buf;

	while(n--){
		s=read_til_transition(ppip->ppti_ppp,8);
		*lp = delta_usecs();
		lp += ppip->ppti_inc;
	}
	return(NULL);
}

static void decrease_frame_time(QSP_ARG_DECL  FB_Info *fbip)
{
	/* get_genlock_mutex(); */
	if (ioctl(fbip->fbi_fd,FBIOGET_VSCREENINFO, &fbip->fbi_var_info)<0) {
		perror("ioctl error getting variable screeninfo");
	} else {	
		if( fbip->fbi_var_info.lower_margin == 0 ){
			/* we could decrease upper */
			WARN("decrease_frame_time:  lower_margin is already 0!?");
		} else if(fbip->fbi_var_info.lower_margin < 0 ){
			WARN("decrease_frame_time:  lower_margin is negative!?!?!?");
		} else {
			fbip->fbi_var_info.lower_margin --;
/*
printf("decrease_frame_time:  lower_margin reduced to %d\n",fbip->fbi_var_info.lower_margin);
*/

			fbip->fbi_var_info.activate |= FB_ACTIVATE_NOW;

			if(ioctl(fbip->fbi_fd, FBIOPUT_VSCREENINFO, &fbip->fbi_var_info)<0) {
				perror("ioctl FBIOPUT_VSCREENINFO");
			}
		}
	}
	/* rls_genlock_mutex(); */
}

static void increase_frame_time(QSP_ARG_DECL  FB_Info *fbip)
{
	/* get_genlock_mutex(); */
	if (ioctl(fbip->fbi_fd,FBIOGET_VSCREENINFO, &fbip->fbi_var_info)<0) {
		perror("ioctl error getting variable screeninfo");
	} else {	
		if( fbip->fbi_var_info.lower_margin >= 20 ){
			/* we could decrease upper */
			WARN("increase_frame_time:  lower_margin is already >= 20!?");
		} else {
			fbip->fbi_var_info.lower_margin ++;
/*
printf("increase_frame_time:  lower_margin increased to %d\n",fbip->fbi_var_info.lower_margin);
*/

			fbip->fbi_var_info.activate |= FB_ACTIVATE_NOW;

			if(ioctl(fbip->fbi_fd, FBIOPUT_VSCREENINFO, &fbip->fbi_var_info)<0) {
				perror("ioctl FBIOPUT_VSCREENINFO");
			}
		}
	}
	/* rls_genlock_mutex(); */
}


static void adjust_genlock(QSP_ARG_DECL  Genlock_Info *glip,int index)
{
	/* We want to keep the fb latency close to zero,
	 * but because we measure drift with a fading average,
	 * we need to wait a little bit between adjustments.
	 */
	long latency,drift;
	latency = glip->gli_fb_latency[index];
	drift = glip->gli_drift[index];
			
	if( latency > 8333 ) latency -= 16666;

/*
printf("adjust_genlock:  latency = %ld, drift = %ld\n",latency,drift);
*/
			
	if( latency > LATENCY_THRESHOLD ){
		/* need to advance vbl */
		if( drift > 0 ){
			/* need to decrease frame time */

			/* wait before adjusting */
			glip->gli_refractory[index]=GL_REFRACTORY;

			decrease_frame_time(QSP_ARG  glip->gli_fbip[index]);
		} else {
			/* negative drift is improving lock */

			/* check again at next update */
			glip->gli_refractory[index]=1;
		}
	} else if( latency < -LATENCY_THRESHOLD ){
		/* need to retard vbl */
		if( drift < 0 ){
			/* need to increase frame time */
			increase_frame_time(QSP_ARG  glip->gli_fbip[index]);

			/* wait before adjusting */
			glip->gli_refractory[index]=GL_REFRACTORY;
		} else {
			/* positive drift is improving lock */
			glip->gli_refractory[index]=1;	/* check again at next update */
		}
	} else {
		glip->gli_refractory[index]=1;	/* check again at next update */
	}
}


/* The genlock daemon simply records the most recent transition times
 * from ALL devices.  Because this daemon ties up a processor, we
 * can't have one per device...
 */

#define PARPORT_SYNC_MASK	8	/* wired to bit 3 of status word, pin 15 */

static void *genlock_daemon(void *argp)
{
	Genlock_Info *glip;
	int original_pp_value, current_pp_value;
	int original_vbl_state[MAX_HEADS], current_vbl_state[MAX_HEADS];
	int i;
	struct timeval tv_now;
	struct timezone tz;
	long prev_latency;
	Query_Stack *qsp;

	glip = (Genlock_Info *) argp;
	qsp = glip->gli_qsp;

	original_pp_value = read_parport_status(glip->gli_ppp) & PARPORT_SYNC_MASK;
	for(i=0;i<glip->gli_n_heads;i++)
		original_vbl_state[i] = read_vbl_state(QSP_ARG  glip->gli_fbip[i]->fbi_fd);

	while(genlock_active){
		current_pp_value = read_parport_status(glip->gli_ppp) & PARPORT_SYNC_MASK;
		if( current_pp_value != original_pp_value ){
			gettimeofday(&tv_now,&tz);
			if( current_pp_value == 0 ){
				/* beginning of negative pulse */
				glip->gli_tv_pp[0] = tv_now;
				for(i=0;i<glip->gli_n_heads;i++){
					glip->gli_pp_latency[i] = compute_latency(&tv_now,&glip->gli_tv_fb[i][0]);
				}
			}
			else
				glip->gli_tv_pp[1] = tv_now;
			original_pp_value = current_pp_value;
		}
		for(i=0;i<glip->gli_n_heads;i++){
			current_vbl_state[i] = read_vbl_state(QSP_ARG  glip->gli_fbip[i]->fbi_fd);
			if( current_vbl_state[i] != original_vbl_state[i] ){
				gettimeofday(&tv_now,&tz);
				if( current_vbl_state[i] == 0 )
					/* end of vblank pulse */
					glip->gli_tv_fb[i][1] = tv_now;
				else {
					long drift;

					/* beginning of vblank pulse */
					glip->gli_tv_fb[i][0] = tv_now;
					prev_latency = glip->gli_fb_latency[i];
					glip->gli_fb_latency[i] = compute_latency(&tv_now,&glip->gli_tv_pp[0]);
					if( prev_latency > 0 ){
						drift = glip->gli_fb_latency[i] - prev_latency;
						if( drift > 10000 )
							drift -= 16666;
						if( drift < -10000 )
							drift += 16666;
/*
printf("drift = %ld\n",drift);
fflush(stdout);
*/
						/* fading average */
						glip->gli_drift[i] =
							DRIFT_MEMORY * glip->gli_drift[i] +
							(1-DRIFT_MEMORY) * drift;
						glip->gli_refractory[i] --;
						if( glip->gli_refractory[i] <= 0 )
							adjust_genlock(QSP_ARG  glip,i);
					}
				}
				original_vbl_state[i] = current_vbl_state[i];
			}
		}
	}
	return(NULL);
}

/* genlock_vblank polls vblank bits and the external sync bit until all heads are in blanking...
 * lower_margins are adjusted as needed.  This attempts to do the same thing as the daemon
 * above, but doesn't need to run in its own thread.
 */

void genlock_vblank(Genlock_Info *glip)
{
	int original_pp_value, current_pp_value;
	int original_vbl_state[MAX_HEADS], current_vbl_state[MAX_HEADS];
	int in_blanking[MAX_HEADS];
	int i;
	struct timeval tv_now;
	struct timezone tz;
	long prev_latency;
	int all_in_blanking;
#ifdef THREAD_SAFE_QUERY
	Query_Stack *qsp=glip->gli_qsp;
#endif // THREAD_SAFE_QUERY

	if( ! genlock_active ){
		WARN("genlock_vblank:  genlock is not initialized");
		return;
	}

	original_pp_value = read_parport_status(glip->gli_ppp) & PARPORT_SYNC_MASK;
	all_in_blanking=0;
	for(i=0;i<glip->gli_n_heads;i++){
		original_vbl_state[i] = read_vbl_state(QSP_ARG  glip->gli_fbip[i]->fbi_fd);
		in_blanking[i]=0;
	}

	while( ! all_in_blanking ){
		current_pp_value = read_parport_status(glip->gli_ppp) & PARPORT_SYNC_MASK;
		if( current_pp_value != original_pp_value ){
			gettimeofday(&tv_now,&tz);
			if( current_pp_value == 0 ){
				/* beginning of negative pulse */
				glip->gli_tv_pp[0] = tv_now;
				for(i=0;i<glip->gli_n_heads;i++){
					glip->gli_pp_latency[i] = compute_latency(&tv_now,&glip->gli_tv_fb[i][0]);
				}
			}
			else
				glip->gli_tv_pp[1] = tv_now;
			original_pp_value = current_pp_value;
		}
		for(i=0;i<glip->gli_n_heads;i++){
			current_vbl_state[i] = read_vbl_state(QSP_ARG  glip->gli_fbip[i]->fbi_fd);
			if( current_vbl_state[i] != original_vbl_state[i] ){
				gettimeofday(&tv_now,&tz);
				if( current_vbl_state[i] == 0 )
					/* end of vblank pulse */
					glip->gli_tv_fb[i][1] = tv_now;
				else {
					long drift;

					/* beginning of vblank pulse */
					glip->gli_tv_fb[i][0] = tv_now;
					prev_latency = glip->gli_fb_latency[i];
					glip->gli_fb_latency[i] = compute_latency(&tv_now,&glip->gli_tv_pp[0]);
					if( prev_latency > 0 ){
						drift = glip->gli_fb_latency[i] - prev_latency;
						if( drift > 10000 )
							drift -= 16666;
						if( drift < -10000 )
							drift += 16666;
						/* fading average */
						glip->gli_drift[i] =
							DRIFT_MEMORY * glip->gli_drift[i] +
							(1-DRIFT_MEMORY) * drift;
						glip->gli_refractory[i] --;
						if( glip->gli_refractory[i] <= 0 )
							adjust_genlock(QSP_ARG  glip,i);
					}
					in_blanking[i]=1;
				}
				original_vbl_state[i] = current_vbl_state[i];
			}
		}

		all_in_blanking = 1;
		for(i=0;i<glip->gli_n_heads;i++)
			all_in_blanking &= in_blanking[i];
	}
}

static void test_parport(void)
{
	int n,s;
	Data_Obj *dp;
	u_long *lp;
	FB_Info *fbip;
	pthread_attr_t attr1;
	VBoard_Info vbi1;
	PPort_Info ppti1;
	u_long l;

	pthread_attr_init(&attr1);	/* initialize to default values */
	pthread_attr_setinheritsched(&attr1,PTHREAD_INHERIT_SCHED);


	fbip = pick_fbi("frame buffer for VSYNC");
	dp = pick_obj("data vector for latencies");

	if( fbip == NULL || dp == NULL ) return;

	INSIST_RAM_OBJ(dp,"test_parport")

	if( OBJ_PREC(dp) != PREC_UDI ){
		sprintf(ERROR_STRING,"latency vector %s (%s) should have precision %s",
			OBJ_NAME(dp),PREC_NAME(OBJ_PREC_PTR(dp)),
				NAME_FOR_PREC_CODE(PREC_UDI));
		WARN(ERROR_STRING);
		return;
	}

	if( OBJ_COMPS(dp) != 2 ){
		sprintf(ERROR_STRING,"latency vector %s (%d) should have 2 components",
			OBJ_NAME(dp),OBJ_COMPS(dp));
		WARN(ERROR_STRING);
		return;
	}

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"latency vector %s should be contiguous",OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}

	n = OBJ_N_MACH_ELTS(dp)/2;
	lp = (u_long *) OBJ_DATA_PTR(dp);

	vbi1.vbi_fbip = fbip;
	vbi1.vbi_buf = lp;
	vbi1.vbi_count = n;
	vbi1.vbi_inc = 2;
#ifdef THREAD_SAFE_QUERY
	vbi1.vbi_qsp = THIS_QSP;
#endif // THREAD_SAFE_QUERY

	lp ++;

	if( the_ppp == NULL ){
		the_ppp=open_parport(NULL);	/* use default */
		if( the_ppp == NULL ) return;
	}

	ppti1.ppti_ppp = the_ppp;
	ppti1.ppti_buf = lp;
	ppti1.ppti_count = n;
	ppti1.ppti_inc = 2;

	tvp = NULL;				/* force re-zero */

	s=read_til_transition(the_ppp,8);
	if( s==0 )
		s=read_til_transition(the_ppp,8);
	/* should now be in the one state */
	/* this is the end of the (negative) pulse */

	l = delta_usecs();	/* zero the clock */

	pthread_create(&vboard_thr,&attr1,vboard_daemon,&vbi1);
	pthread_create(&pport_thr,&attr1,pport_logger,&ppti1);

	/* should wait for threads here... */
	if( pthread_join(vboard_thr,NULL) != 0 ){
		perror("pthread_join");
		WARN("error joining video board thread");
	}
	if( pthread_join(pport_thr,NULL) != 0 ){
		perror("pthread_join");
		WARN("error joining parallel port thread");
	}
}

int init_genlock(SINGLE_QSP_ARG_DECL)
{
	int n,i;

	if( genlock_active ){
		WARN("init_genlock:  genlock is already active!?");
		return -1;
	}

	n=HOW_MANY("number of frame buffers to genlock");
	if( n < 1 || n > MAX_HEADS ){
		sprintf(ERROR_STRING,"do_genlock:  number of frame buffers (%d) must be between 1 and %d",n,MAX_HEADS);
		WARN(ERROR_STRING);
		return -1;
	}
	gli1.gli_n_heads = n;
	for(i=0;i<n;i++){
		gli1.gli_fbip[i] = pick_fbi("frame buffer for VSYNC");
		gli1.gli_fb_latency[i] = (-1);
		gli1.gli_pp_latency[i] = (-1);
		gli1.gli_refractory[i]=GL_REFRACTORY;	/* wait before adjusting */
	}

	for(i=0;i<n;i++)
		if( gli1.gli_fbip[i] == NULL ) return -1;

	if( the_ppp == NULL ){
		the_ppp=open_parport(NULL);	/* use default */
		if( the_ppp == NULL ) return -1;
	}

	gli1.gli_ppp = the_ppp;
	gli1.gli_qsp = THIS_QSP;

	genlock_active=1;

	return 0;
}

static void start_genlock_daemon(void)
{
	if( init_genlock(SINGLE_QSP_ARG) < 0 ) return;

	pthread_attr_init(&attr1);	/* initialize to default values */
	pthread_mutex_init(&genlock_mutex,NULL);
	pthread_attr_setinheritsched(&attr1,PTHREAD_INHERIT_SCHED);


	pthread_create(&gl_thr,&attr1,genlock_daemon,&gli1);
}

void get_genlock_mutex(SINGLE_QSP_ARG_DECL)
{
	if( !genlock_active ){
		WARN("get_genlock_mutex:  genlock is not active");
		return;
	}
	pthread_mutex_lock(&genlock_mutex);
}

void rls_genlock_mutex(SINGLE_QSP_ARG_DECL)
{
	if( !genlock_active ){
		WARN("rls_genlock_mutex:  genlock is not active");
		return;
	}
	pthread_mutex_unlock(&genlock_mutex);
}

static void report_genlock_status( FB_Info *fbip )
{
	int i,index;
	float l2,l3;

	if( ! genlock_active ){
		WARN("report_genlock_status:  genlock is not active!?");
		return;
	}
	index = -1;
	for(i=0;i<gli1.gli_n_heads;i++)
		if( gli1.gli_fbip[i] == fbip ) index=i;

	if( index < 0 ){
		sprintf(ERROR_STRING,"report_genlock_status:  genlock is not active for frame buffer %s!?",fbip->fbi_name);
		WARN(ERROR_STRING);
		return;
	}

	l2 = gli1.gli_fb_latency[index] /1000.0;
	l3 = gli1.gli_drift[index] / 1000.0;
	sprintf(msg_str,"%s:\t\tvbl %g msec after ext pulse\t\tdrift = %g",fbip->fbi_name,l2,l3);
	prt_msg(msg_str);
}

static void *fb_pair_daemon(void *argp)
{
	Proc_Info *pip;
	int i,fd;
#ifdef THREAD_SAFE_QUERY
	Query_Stack *qsp;
#endif // THREAD_SAFE_QUERY

	pip = (Proc_Info*) argp;
	pip->ppi_pid = getpid();
	i=pip->ppi_index;
	fd=pip->ppi_fd;
#ifdef THREAD_SAFE_QUERY
	qsp = pip->ppi_qsp;
#endif // THREAD_SAFE_QUERY

	n_stamps[i]=0;
	count[i]=0;

	while(1) {
		long delta_sec, delta_usec;
		int s;

		usleep(1000);	/* yield processor */

		s=wait_til_vblank_transition(QSP_ARG  fd);		/* wait for current interval to end */
		s=wait_til_vblank_transition(QSP_ARG  fd);		/* wait for next interval to start */

		tv_prev[i] = tv_now[i];
		gettimeofday(&tv_now[i],&tz);
		n_stamps[i]++;

		if( n_stamps[i] >= 2 ){

			delta_sec = tv_now[i].tv_sec - tv_prev[i].tv_sec;
			delta_usec = tv_now[i].tv_usec - tv_prev[i].tv_usec;
			delta_usec += delta_sec * 1000000;

			if( n_stamps[i] < MAX_STAMPS ){
				delta[i][count[i]] = delta_usec;
				delta_accum[i] += delta_usec;
				delta_avg[i][count[i]] = delta_accum[i] / (count[i]+1);
				count[i] ++;
			}
/* fprintf(stderr,"%d\t%d\t%g\n",i,count[i],delta_avg[i][count[i]-1]);
fflush(stderr);
*/
		}
		if( n_stamps[i] >= MAX_STAMPS ){
			n_stamps[i] = count[i] = 0;
			delta_accum[i] = 0.0;
		}
		/* We use the delta's to estimate the true frequency... */

		/* BUG - we need to be locked here??? */

		if( count[0] > 0 && count[1] > 0 ){
			delta_sec = tv_now[0].tv_sec - tv_now[1].tv_sec;
			delta_usec = tv_now[0].tv_usec - tv_now[1].tv_usec;
			delta_usec += delta_sec * 1000000;
		}
			
	}

	return(NULL);

} /* end fb_pair_daemon() */

static void start_fb_threads(QSP_ARG_DECL  int n_frame_buffers,int* fd_arr)
{
	int i;

	pthread_attr_init(&attr1);	/* initialize to default values */
	pthread_attr_setinheritsched(&attr1,PTHREAD_INHERIT_SCHED);

	for(i=0;i<n_frame_buffers;i++){
		ppi[i].ppi_index=i;
		ppi[i].ppi_flags = 0;
		ppi[i].ppi_fd = fd_arr[i];
sprintf(ERROR_STRING,"calling pthread_create for thread %d",i);
advise(ERROR_STRING);
		pthread_create(&fb_thr[i],&attr1,fb_pair_daemon,&ppi[i]);
	}
}

static void fbpair_monitor( FB_Info *fbip1, FB_Info *fbip2 )
{
	int fd_arr[2];

	fd_arr[0] = fbip1->fbi_fd;
	fd_arr[1] = fbip2->fbi_fd;

	start_fb_threads(QSP_ARG  2,fd_arr);
}


#endif /* HAVE_GENLOCK */

COMMAND_FUNC( halt_genlock )
{
#ifdef HAVE_GENLOCK
	if( ! genlock_active ){
		WARN("genlock_halt:  genlock is not active!?");
		return;
	}

	genlock_active = 0;

	if( pthread_join(gl_thr,NULL) != 0 ){
		perror("pthread_join");
		WARN("error joining genlock thread");
	}
#else /* ! HAVE_GENLOCK */
	WARN("No genlock capability for this installation");
#endif /* ! HAVE_GENLOCK */
}

COMMAND_FUNC( do_report_genlock_status )
{
	FB_Info *fbip;

	fbip = pick_fbi("frame buffer");
	if( fbip == NULL ) return;

#ifdef HAVE_GENLOCK
	report_genlock_status(fbip);
#endif // HAVE_GENLOCK
}

COMMAND_FUNC( do_genlock )
{
#ifdef HAVE_GENLOCK
	start_genlock_daemon();
#endif // HAVE_GENLOCK
}


COMMAND_FUNC( do_fbpair_monitor )
{
	FB_Info *fbip1,*fbip2;

	fbip1 = pick_fbi("first frame buffer");
	fbip2 = pick_fbi("second frame buffer");
	if( fbip1 == NULL || fbip2 == NULL ) return;

#ifdef HAVE_GENLOCK
	fbpair_monitor(fbip1,fbip2);
#endif // HAVE_GENLOCK
}

COMMAND_FUNC( do_test_parport )
{
#ifdef HAVE_GENLOCK
	test_parport();
#endif // HAVE_GENLOCK
}

