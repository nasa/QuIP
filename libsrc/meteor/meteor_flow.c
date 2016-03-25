
/* This file used to be called flow.c, but it seemed to have
 * disappeared after the project had been moved to git - does
 * git have a problem with duplicate filenames (there is another
 * flow.c in v4l2), or did it just get lost in the shuffle?
 */

#include "quip_config.h"

#ifdef HAVE_METEOR

/* Like continuous capture to memory, but we use the handshaking (like in stream record)
 * to make sure our application is synchronized.
 *
 */

#ifdef HAVE_UNISTD_H
#include <unistd.h>			/* usleep */
#endif /* HAVE_UNISTD_H */

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif /* HAVE_SYS_TYPES_H */

#include "quip_prot.h"
#include "mmenu.h"			/* capture_code */
#include "ioctl_meteor.h"		/* meteor_mem struct defn */

/* globals */

static int oldest, newest;	/* indices into ringbuf
				 *
				 * In the absence of wrap-around,
				 * (1+newest-oldest) = n_active.
				 */

static COMMAND_FUNC( do_start_flow )
{
	/* Now start the meteor capturing */

	/* set synchronous mode */
	capture_code = METEORCAPFRM;
	_hiwat=num_meteor_frames-2;	/* stop when this many bufs are full (active?) */
	_lowat=_hiwat-1;		/* restart asap */

	/* zero the error counts */
	meteor_clear_counts();

	oldest = newest = 0;

	if( meteor_capture(SINGLE_QSP_ARG) < 0 )
		ERROR1("meteor_capture() failed");

	/* wait for the first frame */

	/* Apparently, a buffer is 'active' while it is being filled...
	 * therefore, we wait for this to be 2, so we know the first
	 * frame has been completely captured before we start writing.
	 * But we can do a little housekeeping while the first frame
	 * is filling...
	 */

	while( _mm->num_active_bufs < 1 ){
/*
sprintf(ERROR_STRING,"num_activ_bufs = %d",_mm->num_active_bufs);
advise(ERROR_STRING);
*/
		usleep(1000);
	}
}

static COMMAND_FUNC( do_stop_flow )
{
	meteor_stop_capture(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( update_vars )
{
	char s[32];

	/* we subtract 2 from cur_frame:  one because it is a fortran-style index (begins at 1),
	 * and another one because the current frame is the one which is being aquired and is
	 * not complete.  We want newest to index the newest complete frame.
	 *
	 * This makes sense, BUT it seems to be wrong!?
	 * We are getting a stale newest frame, so we try using 1...
	 * The answer(?):  there are two cur_frame variables in the driver, one in
	 * struct meteor, which is used by most of the driver code and which points
	 * to the currently grabbing frame, and the shared mem version which gets copied
	 * at the end of a frame interrupt - SO what we see in shared mem is NOT the
	 * currently active buffer!
	 */
/*
sprintf(ERROR_STRING,"cur_frame:\t%d\t\tnum_activ:\t%d\tmask = 0x%x",_mm->cur_frame,_mm->num_active_bufs,_mm->active);
advise(ERROR_STRING);
*/
	/*newest = (_mm->cur_frame + num_meteor_frames - 2 ) % num_meteor_frames; */
	newest = (_mm->cur_frame + num_meteor_frames - 1 ) % num_meteor_frames;
	/* when we tried this, we got frames with tearing, showing that we were reading */
	/* the active (grabbing) frame */
	/*newest = (_mm->cur_frame + num_meteor_frames ) % num_meteor_frames; */

	/* num_active_bufs counts the frame which is currently acquiring, so we subtract two... */
	/* NO num_active_bufs counts the frames which have been acquired so we subtract one... */
	oldest = (newest + num_meteor_frames - (_mm->num_active_bufs-1)) % num_meteor_frames;

	sprintf(s,"%d",oldest);
	ASSIGN_VAR("oldest",s);

	sprintf(s,"%d",newest);
	ASSIGN_VAR("newest",s);

/*
sprintf(ERROR_STRING,"update_vars:\toldest = %d\t\tnewest = %d",oldest,newest);
advise(ERROR_STRING);
*/
	sprintf(s,"%d",_mm->num_active_bufs);
	ASSIGN_VAR("n_active",s);

	sprintf(s,"0x%lx",(long)_mm->active);
	ASSIGN_VAR("active_mask",s);
}

static COMMAND_FUNC( do_wait_next )	/* wait til we have another frame */
{
	int start_n;

	start_n=_mm->num_active_bufs;
	if( start_n == _hiwat ){
		sprintf(ERROR_STRING,"do_wait_next:  ring buffer already containts %d active frames, will not advance until one is released.",start_n);
		WARN(ERROR_STRING);
		update_vars(SINGLE_QSP_ARG);
		return;
	}
/*
sprintf(ERROR_STRING,"do_wait_next BEGIN:  num_active_bufs = %d",start_n);
advise(ERROR_STRING);
*/
	while( _mm->num_active_bufs == start_n )
		;
	/* if we sleep here, out time seems to be quantized in units of 10 msec??? */
	/* maybe because task switching occurs at 100 Hz system timer? */
		/* usleep(1000); */	/* 1 msec */

	update_vars(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_wait_drip )	/* wait til we have at least one frame */
{
	while( _mm->num_active_bufs <= 1 ){
		usleep(10000);
	}
	update_vars(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_release_buffer )
{
	/* release this buffer,
	 * and increment oldest.
	 */
	if( _mm->num_active_bufs > 1 ){	/* don't release if only 1 active - it's being captured now! */
		_mm->num_active_bufs--;		/* shared w/ driver */
		/* let's assume that oldest is valid! */
		_mm->active &= ~(1<<oldest);
	}

	update_vars(SINGLE_QSP_ARG);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(flow_menu,s,f,h)

MENU_BEGIN(flow)
ADD_CMD( start,		do_start_flow,	start capture )
ADD_CMD( stop,		do_stop_flow,	stop capture )
ADD_CMD( wait,		do_wait_drip,	wait for at least one frame in memory )
ADD_CMD( next,		do_wait_next,	wait for the next frame )
ADD_CMD( release,	do_release_buffer,	release oldest buffer )
ADD_CMD( update,	update_vars,	refresh values of $oldest and $newest )
MENU_END(flow)

COMMAND_FUNC( meteor_flow_menu )
{
	PUSH_MENU(flow);
}

#endif /* HAVE_METEOR */

