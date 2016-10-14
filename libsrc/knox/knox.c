#include "quip_config.h"

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "query_stack.h"
#include "data_obj.h"

// BUG static vars not thread safe
static int range_ok;
static int doing_command_set = 0;	/* flag for sending set of commands */


#define MIN_SIGNAL_NUMBER	1
#define MAX_SIGNAL_NUMBER	8

#define CHECK_RANGE( name, number, min, max )			\
{								\
	range_ok=1;						\
	if( number < min || number > max ) {			\
		sprintf(ERROR_STRING,				\
"%s (%d) must be between %d and %d", name, number, min, max);	\
		WARN(ERROR_STRING);				\
		range_ok=0;					\
	}							\
}


#define GET_WITH_LIMITS( var, string, minval, maxval )			\
									\
	var = HOW_MANY(string);						\
	CHECK_RANGE(string,var, minval, maxval)				\
	if( !range_ok ) ret_stat=(-1);



#define GET_PATTERN( var, string )					\
									\
	GET_WITH_LIMITS( var, string, MIN_CROSSPOINT_PATTERN, MAX_CROSSPOINT_PATTERN)


#define GET_SIGNAL( var, string )					\
									\
	GET_WITH_LIMITS( var, string, MIN_SIGNAL_NUMBER, MAX_SIGNAL_NUMBER )


#define GET_KNOX_ARGS(arg_buf)		get_knox_args(QSP_ARG  arg_buf)


#ifndef HAVE_KNOX

#define NO_KNOX_MSG WARN("Sorry, no knox support in this build.");
#define DO_KNOX_CMD( code, args, error_msg )				\
	NO_KNOX_MSG

#define USES_NEWER_FIRMWARE(kdp)		0
#define USES_OLDER_FIRMWARE(kdp)		0

#else  // HAVE_KNOX

#define DO_KNOX_CMD( code, args, error_msg )				\
									\
	if( do_knox_cmd(QSP_ARG  code,args) < 0 ) {			\
		WARN(error_msg);					\
		return;							\
	}

#include "data_obj.h"
#include "knox.h"
#include "serbuf.h"		/* expected_response() */
#include "serial.h"
#include "ttyctl.h"		/* keyhit() */
//#include "version.h"

#define TIMER_STOPPED	1
#define TIMER_RUNNING	2

static int timer_state=TIMER_STOPPED;

static Knox_State knox_state;	/* not initialized! */

static COMMAND_FUNC( do_knox_timer_stop );


typedef struct knox_response_strings {
	const char *krs_pre_map_response;
	const char *krs_post_map_response;
	const char *krs_route_done_response;
	const char *krs_lamp_response;
	const char *krs_timer_response;
	const char *krs_stop_response;
	const char *krs_recall_response;
} Knox_Firmware;

/* After the software was fixed to read the echo's one at a time (3/26/07),
 * some of these response strings had to be changed (remove leading CR).
 * But not all have been tested...
 */

Knox_Firmware
	older_fw=
/* main rack switcher */
/* mini-rack (dirac) switcher */
{
	"",				/* pre-map response */
	" \nDONE\r\n",			/* post-map response */
	"DONE\r\n",			/* route done response */
	"",				/* lamp response */
	"DONE\r\n",			/* timer response */
	"DONE\r\n",			/* stop response */
	"DONE\r\n"			/* recall response */
},

newer_fw=
/* new switcher */
{
	"\n\r \r\n",			/* pre-map response */
	" ",				/* post-map response */
	"\n\r DONE\n\r",		/* route done response */
	"\n\r  TESTING\r\n DONE\r\n",	/* lamp response */
	"\n\r TIMER MODE ON\n\r",	/* timer response */
	"\n\r TIMER MODE OFF\n\r",	/* stop response */
	"\n\r DONE\n\r"			/* recall response */
}
;

typedef struct knox_device {
	char *		kd_name;
	int		kd_fd;
	Serial_Buffer *	kd_sbp;
	Knox_Firmware *	kd_fwp;
} Knox_Device;

#define USES_NEWER_FIRMWARE(kdp)		((kdp)->kd_fwp == &newer_fw)
#define USES_OLDER_FIRMWARE(kdp)		((kdp)->kd_fwp == &older_fw)

Knox_Device *curr_kdp=NULL;

// should be static?
//ITEM_INTERFACE_PROTOTYPES_STATIC(Knox_Device,knox_dev)
//ITEM_INTERFACE_DECLARATIONS_STATIC(Knox_Device,knox_dev)
static Item_Type * knox_dev_itp=NO_ITEM_TYPE;
static ITEM_INIT_FUNC(Knox_Device,knox_dev)
static ITEM_NEW_FUNC(Knox_Device,knox_dev)
static ITEM_CHECK_FUNC(Knox_Device,knox_dev)

// FOOBAR?
//#define PICK_KNOX_DEV(pmpt)		pick_knox_dev(QSP_ARG  pmpt)

#define KNOX_DONE_MSG		"DONE"
#define KNOX_ERROR_MSG		"ERROR"

static void show_routing_map(SINGLE_QSP_ARG_DECL)
{
	int i;

	prt_msg("");
	prt_msg("\toutput\tvideo\taudio");
	for(i=0;i<8;i++){
		sprintf(msg_str,"\t%d\t%d\t%d",i+1,
			knox_state.ks_video[i],knox_state.ks_audio[i]);
		prt_msg(msg_str);
	}
	prt_msg("");
}

static void get_map_response(QSP_ARG_DECL  int i_output)
{
	char str[32];
	int n;

	sprintf(str,"  OUTPUT  %d     VIDEO  ",i_output);
	expected_response(QSP_ARG  curr_kdp->kd_sbp,str);
	n=get_number(QSP_ARG  curr_kdp->kd_sbp);
	knox_state.ks_video[i_output-1] = n;
	expected_response(QSP_ARG  curr_kdp->kd_sbp,"     AUDIO  ");
	n=get_number(QSP_ARG  curr_kdp->kd_sbp);
	knox_state.ks_audio[i_output-1] = n;
	expected_response(QSP_ARG  curr_kdp->kd_sbp," \r\n");
}

static void get_condensed_map_response(QSP_ARG_DECL  int i_output)
{
	char str[32];
	int n;

	expected_response(QSP_ARG  curr_kdp->kd_sbp,str);

	sprintf(str,"%dV",i_output);
	expected_response(QSP_ARG  curr_kdp->kd_sbp,str);
	n=get_number(QSP_ARG  curr_kdp->kd_sbp);
	knox_state.ks_video[i_output-1] = n;
	expected_response(QSP_ARG  curr_kdp->kd_sbp,"A");
	n=get_number(QSP_ARG  curr_kdp->kd_sbp);
	knox_state.ks_audio[i_output-1] = n;
	expected_response(QSP_ARG  curr_kdp->kd_sbp,"\r\n");
}

/* We used to squirt the whole string out at once, but in that case we sometimes
 * didn't get the whole command echo'd back...  So now we send the characters
 * one-at-a-time, and listen for the echo before proceeding.
 */

static void send_knox_cmd(QSP_ARG_DECL  const char* buf)
{
	int i,n;
	char rstr[2];

/*
sprintf(ERROR_STRING,"send_knox_cmd:  sending \"%s\"",buf);
advise(ERROR_STRING);
*/
	n=strlen(buf);
	rstr[1]=0;
	for(i=0;i<n;i++){
		rstr[0]=buf[i];
		send_serial(QSP_ARG  curr_kdp->kd_fd, (u_char *)rstr, 1);
		expect_string(QSP_ARG  curr_kdp->kd_sbp,rstr);
	}

	/* append carriage return */

	rstr[0]='\r';
	send_serial(QSP_ARG  curr_kdp->kd_fd, (u_char *)rstr, 1);
	expect_string(QSP_ARG  curr_kdp->kd_sbp,rstr);
}

static int identify_firmware(SINGLE_QSP_ARG_DECL)
{
	int c;

	/* send a simple command and see what we get back */

	/* It turns out that this command turns on the timer, which puts
	 * the switcher into a mode where it seems to ignore the buttons
	 * and displays all 1's...
	 *
	 * Our solution is to issue the stop command automatically
	 * immediately after.
	 */

	send_knox_cmd(QSP_ARG  "T11");

	/* Now the echo is read by send_knox_cmd */
	/* expected_response(QSP_ARG  curr_kdp->kd_sbp,"T11\r"); */

	/* now check the next character;
	 * we expect EITHER " DONE\n\r" OR "DONE\r\n"
	 */

	c = buffered_char(QSP_ARG  curr_kdp->kd_sbp);
	if( c == '\n' ){
advise("Knox 8x8 video switcher firmware 1 (newer) detected");
		expected_response(QSP_ARG  curr_kdp->kd_sbp,"\n\r ");
		curr_kdp->kd_fwp = &newer_fw;
		/*sleep(2); */
		usleep(500000);
		/* Now see if there's a character */
		if( keyhit(  curr_kdp->kd_sbp->sb_fd ) ){
			/* If the timer was off, then it prints this, otherwise not!? */
			expected_response(QSP_ARG  curr_kdp->kd_sbp, "TIMER MODE ON\n\r");
	/*"\r\n\r TIMER MODE ON\n\r", */
		}
				/*curr_kdp->kd_fwp->krs_timer_response */
	} else if( c == 'D' ){
advise("Knox 8x8 video switcher firmware 2 (older) detected");
		expected_response(QSP_ARG  curr_kdp->kd_sbp,"DONE\r\n");
		curr_kdp->kd_fwp = &older_fw;
	} else {
		show_buffer(curr_kdp->kd_sbp);
		ERROR1("Unrecognized knox firmware!?");
		return(-1);
	}

	do_knox_timer_stop(SINGLE_QSP_ARG);		/* BUG elim global */

	return(0);
}

static int process_knox_reply(QSP_ARG_DECL  Knox_Cmd_Code code)
{
	int i;
	const char *s;

	switch(code){
		case KNOX_MAP_REPORT:
			s=curr_kdp->kd_fwp->krs_pre_map_response;
			if( s != NULL && *s != 0 )
				expected_response(QSP_ARG  curr_kdp->kd_sbp,
					curr_kdp->kd_fwp->krs_pre_map_response);
			/* the pre-map response is now an empty string? */
			for(i=1;i<=8;i++)
				get_map_response(QSP_ARG  i);
			expected_response(QSP_ARG  curr_kdp->kd_sbp,curr_kdp->kd_fwp->krs_post_map_response);
			break;
		case KNOX_CONDENSE_REPORT:
			sleep(2);
			expected_response(QSP_ARG  curr_kdp->kd_sbp,curr_kdp->kd_fwp->krs_pre_map_response);
			for(i=1;i<=8;i++)
				get_condensed_map_response(QSP_ARG  i);
			expected_response(QSP_ARG  curr_kdp->kd_sbp,curr_kdp->kd_fwp->krs_post_map_response);
			break;

		case KNOX_LAMP_TEST:
			expected_response(QSP_ARG  curr_kdp->kd_sbp,curr_kdp->kd_fwp->krs_lamp_response);
			break;

		case KNOX_STORE_CROSSPOINT:
		case KNOX_RECALL_CROSSPOINT:
			expected_response(QSP_ARG  curr_kdp->kd_sbp,curr_kdp->kd_fwp->krs_recall_response);
			break;

		case KNOX_STOP_TIMER:
			expected_response(QSP_ARG  curr_kdp->kd_sbp,curr_kdp->kd_fwp->krs_stop_response);
			timer_state = TIMER_STOPPED;
			break;

		case KNOX_SET_TIMER:
			if( curr_kdp->kd_fwp == &older_fw ){
				expected_response(QSP_ARG  curr_kdp->kd_sbp,curr_kdp->kd_fwp->krs_timer_response);
			} else {
				if( timer_state == TIMER_STOPPED ){
					expected_response(QSP_ARG  curr_kdp->kd_sbp,
							curr_kdp->kd_fwp->krs_timer_response);
				} else
					expected_response(QSP_ARG  curr_kdp->kd_sbp,"\r\n\r ");
			}
			timer_state = TIMER_RUNNING;
			break;

		case KNOX_SET_VIDEO:
		case KNOX_SET_AUDIO:
		case KNOX_SET_DIFF:
		case KNOX_SET_BOTH:
			/* this works for the switcher in the main rack */
			/* expected_response(QSP_ARG  "\r DONE\n\r"); */
			/* expected_response(QSP_ARG  "\rDONE\r\n"); */
			expected_response(QSP_ARG  curr_kdp->kd_sbp,curr_kdp->kd_fwp->krs_route_done_response);
			break;

		default:
			sprintf(ERROR_STRING,"process_knox_reply:  unhandled code %d (%s)",
				code,knox_tbl[code].kc_desc);
			WARN(ERROR_STRING);
			sleep(2);
			replenish_buffer(QSP_ARG  curr_kdp->kd_sbp,256);
			show_buffer(curr_kdp->kd_sbp);
			return(-1);
	}
	/* now check for DONE or ERROR ?? */
	return(0);
}

#endif // HAVE_KNOX

static int get_knox_args(QSP_ARG_DECL   char* arg_buf)
{
	int input, first_output;
	int ret_stat=0;

	GET_SIGNAL(input,"input number");

	/*
	if( current_mode == KNOX_SALVO_MODE ) {
		GET_SIGNAL(first_output,"first output number");
		GET_SIGNAL(last_output,"last output number");

		sprintf(arg_buf, "%d%d%d", first_output, last_output, input);
	} else {
	*/
		GET_SIGNAL(first_output,"output number");

		sprintf(arg_buf, "%d%d", first_output, input);
	/*
	}
	*/

	return(ret_stat);		
}

#ifdef HAVE_KNOX
static int do_knox_cmd(QSP_ARG_DECL  Knox_Cmd_Code code, char* args)
{
	char buf[LLEN];
	int stat;
	
	sprintf(buf, "%s", knox_tbl[code].kc_str);
	if( args ) strcat(buf, args);
	reset_buffer(curr_kdp->kd_sbp);
	send_knox_cmd(QSP_ARG  buf);

	/* Now we read the echo char-by-char in send_knox_cmd */
	/* expected_response(QSP_ARG  curr_kdp->kd_sbp,buf); */

	/* This works for the switcher in the main video rack */
	/* expected_response(QSP_ARG  "\r\n"); */
	/* expected_response(QSP_ARG  "\r"); */

	stat=process_knox_reply(QSP_ARG  code);
	return(stat);
}
#endif // HAVE_KNOX

static COMMAND_FUNC( do_route_both )
{
	char knox_args[LLEN];

	if( GET_KNOX_ARGS(knox_args) < 0 ) return;

	DO_KNOX_CMD(KNOX_SET_BOTH,knox_args,"Unable to route audio and video!")
}

static COMMAND_FUNC( do_route_diff )
{
	char knox_args[LLEN];
	int output, video_input, audio_input;
	int ret_stat=0;

	GET_SIGNAL(video_input,"video input number");
	GET_SIGNAL(audio_input,"audio input number");
	GET_SIGNAL(output,"output number");

	if( ret_stat < 0 ) return;

	sprintf(knox_args, "%d%d%d", output, video_input, audio_input);

	DO_KNOX_CMD( KNOX_SET_DIFF,knox_args,"Unable to route audio and video from different inputs!")
}

static COMMAND_FUNC( do_route_video )
{
	char knox_args[LLEN];

	if( GET_KNOX_ARGS(knox_args) < 0 ) return;

	DO_KNOX_CMD(KNOX_SET_VIDEO,knox_args,"Unable to route video alone!")
}

static COMMAND_FUNC( do_route_audio )
{
	char knox_args[LLEN];

	if( GET_KNOX_ARGS(knox_args) < 0 ) return;

	DO_KNOX_CMD( KNOX_SET_AUDIO, knox_args,"Unable to route audio!?")
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(route_menu,s,f,h)

MENU_BEGIN(route)
ADD_CMD( both,	do_route_both,	route both audio and video (same input) )
ADD_CMD( diff,	do_route_diff,	route both audio and video (different inputs) )
ADD_CMD( video,	do_route_video,	route video alone )
ADD_CMD( audio,	do_route_audio,	route audio alone )
MENU_END(route)


static COMMAND_FUNC( do_knox_route_cmds )
{
	PUSH_MENU(route);	
}	

#ifdef NOT_YET
/*
 * We used to use the same command table for salvo command etc, setting global flag (commented out???)
 * to communicate the mode.  However, now that the prompt is part of the menu structure, this makes less
 * sense.  Better to have a separate command that sets the routing mode.
 */

static COMMAND_FUNC( do_knox_salvo_cmds )
{
	if( doing_command_set ) {
		WARN("command set begun, end command set first for salvo commands");
		return;
	}
	/* current_mode = KNOX_SALVO_MODE; */
	PUSH_MENU(knox_route_menu);
}

	
static COMMAND_FUNC( do_knox_conf_cmds )
{
	if( doing_command_set ) {
		WARN("begin command set issued, end command set first for conference commands");
		return;
	}
	/* current_mode = KNOX_CONFERENCE_MODE; */
	PUSH_MENU(knox_route_menu, "conference");
}
#endif /* NOT_YET */

static COMMAND_FUNC( do_take_set_cmds )
{
	if( !doing_command_set ) {
		advise("CAUTIOUS: command set not begun to execute commands!?");
		return;
	}

	DO_KNOX_CMD(KNOX_TAKE_COMMAND, NULL, "Unable to take current commands sent!")
}

static COMMAND_FUNC( do_begin_set_cmds )
{ 
	doing_command_set = 1;
}

static COMMAND_FUNC( do_end_set_cmds )
{
	do_take_set_cmds(SINGLE_QSP_ARG);
	doing_command_set = 0;
}

/* The manual says that we can issue a string of commands to
 * be executed one after another. However, when we are
 * doing a string of commands, we can only do regular
 * route commands, no salvo or conference commands.
 */

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(knox_cmd_menu,s,f,h)

MENU_BEGIN(knox_cmd)
ADD_CMD( route,		do_knox_route_cmds,	regular route audio and video )
//ADD_CMD( salvo,		do_knox_salvo_cmds,	salvo route audio and video )
//ADD_CMD( conference,	do_knox_conf_cmds,	conference route audio and video )
ADD_CMD( begin,		do_begin_set_cmds,	begin a set of routing commands )
ADD_CMD( end_set,	do_end_set_cmds,	end set of commands and send )
ADD_CMD( take,		do_take_set_cmds,	execute current set of commands sent )
MENU_END(knox_cmd)

static COMMAND_FUNC( do_knox_main_cmds )
{
	PUSH_MENU(knox_cmd);
}

#define MIN_CROSSPOINT_PATTERN	1
#define MAX_CROSSPOINT_PATTERN	8

static COMMAND_FUNC( do_knox_recall )
{
	char args[LLEN];
	int pattern;
	int ret_stat=0;

	GET_PATTERN(pattern,"switcher pattern index");
	if( ret_stat < 0 ){
		return;
	}

	sprintf(args, "%d", pattern);
	DO_KNOX_CMD(KNOX_RECALL_CROSSPOINT, args, "Unable to load crosspoint pattern!")
}

#define MIN_STORE_CROSSPOINT	1
#define MAX_STORE_CROSSPOINT	6

static COMMAND_FUNC( do_knox_store )
{
	char args[LLEN];
	int pattern_index;
	int ret_stat=0;

	GET_PATTERN(pattern_index,"switcher pattern index");
	if( ret_stat < 0 ) return;

	sprintf(args, "%d", pattern_index);
	DO_KNOX_CMD(KNOX_STORE_CROSSPOINT, args, "Unable to store crosspoint pattern!")
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(crosspoint_menu,s,f,h)

MENU_BEGIN(crosspoint)
ADD_CMD( load,	do_knox_recall,	load crosspoint pattern from memory )
ADD_CMD( store,	do_knox_store,	store currently loaded crosspoint pattern )
MENU_END(crosspoint)

static COMMAND_FUNC( do_knox_cross_cmds )
{
	PUSH_MENU(crosspoint);
}

#define MIN_TIME_CYCLE	1
#define MAX_TIME_CYCLE	999

static COMMAND_FUNC( do_knox_timer_start )
{
	char args[LLEN];
	int time_cycle;

	time_cycle = (int)HOW_MANY("time cycle");
	CHECK_RANGE("time cycle",time_cycle, MIN_TIME_CYCLE, MAX_TIME_CYCLE)
	if( ! range_ok ) return;

	/*sprintf(args, "%03d", time_cycle); */
	sprintf(args, "%02d", time_cycle);
	DO_KNOX_CMD(  KNOX_SET_TIMER, args, "Unable to set timed sequencer!")
}

static COMMAND_FUNC( do_knox_timer_stop )
{
	DO_KNOX_CMD(KNOX_STOP_TIMER, NULL, "Unable to stop timed sequencer!")
}



#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(timer_menu,s,f,h)

MENU_BEGIN(timer)
ADD_CMD( start,	do_knox_timer_start,	set time cycle time interval and start timer )
ADD_CMD( stop,	do_knox_timer_stop,	stop time cycle )
MENU_END(timer)

static COMMAND_FUNC( do_knox_time_cmds )
{
	PUSH_MENU(timer);
}

static COMMAND_FUNC( do_fetch_map )
{
	Data_Obj *dp;

	dp = PICK_OBJ("object for routing results");

	if( dp == NO_OBJ ) return;

	if( OBJ_COLS(dp) != 8 ){
		sprintf(ERROR_STRING,
	"Object %s should have 8 columns for knox report",
			OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_COMPS(dp) != 2 ){
		sprintf(ERROR_STRING,
	"Object %s should have 2 components for knox report",
			OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_PREC(dp) != PREC_UBY ){
		sprintf(ERROR_STRING,
	"Object %s should have %s precision for knox report",
			OBJ_NAME(dp),PREC_NAME(PREC_FOR_CODE(PREC_UBY)));
		WARN(ERROR_STRING);
		return;
	}

	/* do we care about the precision? */

	DO_KNOX_CMD(  KNOX_MAP_REPORT, NULL, "Unable to get routing status map report!")

#ifdef HAVE_KNOX
	{
	u_char *p;
	int i;
	p=(u_char *)OBJ_DATA_PTR(dp);
	for(i=0;i<8;i++){
		*(p + i * OBJ_PXL_INC(dp)) = knox_state.ks_video[i];
		*(p + i * OBJ_PXL_INC(dp) + OBJ_COMP_INC(dp) ) = knox_state.ks_audio[i];
	}
	}
#endif // HAVE_KNOX
}

static COMMAND_FUNC( do_show_map )
{
	DO_KNOX_CMD(KNOX_MAP_REPORT, NULL, "Unable to get routing status map report!")

#ifdef HAVE_KNOX
	show_routing_map(SINGLE_QSP_ARG);
#endif // HAVE_KNOX
}

static COMMAND_FUNC( do_inq_knox_cond )
{
	/* The (newer?) switcher in the main rack doesn't seem to recognize the D0 command */
	if( USES_NEWER_FIRMWARE(curr_kdp) ){
		WARN("Sorry, newer switchers don't seem to recognized the condensed map command");
		do_show_map( SINGLE_QSP_ARG );
		return;
	}
advise("seems to be using older firmware?");

	DO_KNOX_CMD(KNOX_CONDENSE_REPORT, NULL, "Unable to get condensed routing status map report!")

#ifdef HAVE_KNOX
	prt_msg("\nCondensed Routing Map Status Report:");
	show_routing_map(SINGLE_QSP_ARG);
#endif // HAVE_KNOX
}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(status_menu,s,f,h)

MENU_BEGIN(status)
ADD_CMD( fetch_map,		do_fetch_map,		store routing map to data vector )
ADD_CMD( show_map,		do_show_map,		display routing map )
ADD_CMD( condense,		do_inq_knox_cond,	get condensed routing map status report )
MENU_END(status)

static COMMAND_FUNC( do_knox_status_cmds )
{
	PUSH_MENU(status);
}

static COMMAND_FUNC( do_lamp_test )
{
	/* Lamp test command has no args! */
	/*if( GET_KNOX_ARGS(args) < 0 ) return; */

	DO_KNOX_CMD(  KNOX_LAMP_TEST, NULL, "Unable to do lamp test!")
}

#ifdef HAVE_KNOX
static void open_knox_device(QSP_ARG_DECL  const char *s)
{
	int fd;
	Knox_Device *kdp;

	kdp = knox_dev_of(QSP_ARG  s);
	if( kdp != NULL ){
		sprintf(ERROR_STRING,
	"open_knox_device:  knox device %s is already open",s);
		WARN(ERROR_STRING);
		return;
	}

	if( (fd = open_serial_device(QSP_ARG  s)) < 0 ){ 
		sprintf(ERROR_STRING,"Unable to open knox device %s",s);
		WARN(ERROR_STRING);
		return;
	}

	/* Here is where we should do any necessary stty settings, baud rate etc */
	/* open_serial_device() puts the device into "raw" mode */

	/* Set the baud rate to 19200 */

	/* The switcher's baud rate is controlled by DIP switches 1 & 2;
	 * cycle power after changing!
	 */
	set_baud(fd,B19200);

	kdp = new_knox_dev(QSP_ARG  s);
	kdp->kd_fd=fd;

	kdp->kd_sbp = (Serial_Buffer *)getbuf(sizeof(*kdp->kd_sbp));
	kdp->kd_sbp->sb_fd = fd;
	reset_buffer(kdp->kd_sbp);

	curr_kdp = kdp;

	identify_firmware(SINGLE_QSP_ARG);

	/*
	identify_firmware();
	*/
	/*
	if( !strcmp(kdp->kd_name,"/dev/knox") ){
		kdp->kd_fwp = &newer_fw;
		advise("Assuming newer firmware with /dev/knox");
	} else {
		kdp->kd_fwp = &older_fw;
		sprintf(ERROR_STRING,"Assuming older firmware with %s",kdp->kd_name);
		advise(ERROR_STRING);
	}
*/
}
#endif // HAVE_KNOX


static COMMAND_FUNC( do_select_device )
{
	const char *s;
#ifdef HAVE_KNOX
	Knox_Device *kdp;
#endif // HAVE_KNOX

	s=NAMEOF("knox device");

#ifdef HAVE_KNOX
	kdp = knox_dev_of(QSP_ARG  s);
	if( kdp != NULL ){
		curr_kdp = kdp;
		return;
	}

	open_knox_device(QSP_ARG  s);
#else // ! HAVE_KNOX
	NO_KNOX_MSG
#endif // ! HAVE_KNOX

}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(knox_menu,s,f,h)

MENU_BEGIN(knox)
ADD_CMD( device,	do_select_device,	select knox device )
ADD_CMD( command,	do_knox_main_cmds,	routing switcher commands )
ADD_CMD( crosspoint,	do_knox_cross_cmds,	crosspoint pattern commands )
ADD_CMD( timer,		do_knox_time_cmds,	timed sequencer commands )
ADD_CMD( status,	do_knox_status_cmds,	switcher status commands )
ADD_CMD( lamp,		do_lamp_test,		perform lamp test )
MENU_END(knox)

/* This was /dev/ttyS0 on fourier, but we insist on using a symlink
 * /dev/knox to make this portable to other systems which might use
 * a different port.
 */

#define KNOX_TTY_DEV	"/dev/knox"

COMMAND_FUNC( do_knox_menu )
{
#ifdef HAVE_KNOX
	if( curr_kdp == NULL ) {
		open_knox_device(QSP_ARG  KNOX_TTY_DEV);
		if( curr_kdp == NULL )
			ERROR1("Unable to open default knox device");
	}
#endif // HAVE_KNOX

	PUSH_MENU(knox);
}

