

#ifdef ALLOW_RT_SCHED
extern int rt_is_on;
#define YIELD_PROC(time)	{ if( rt_is_on ) sched_yield(); else usleep(time); }
#else /* ! ALLOW_RT_SCHED */
#define YIELD_PROC(time)	usleep(time);
#endif /* ALLOW_RT_SCHED */


