#include "quip_config.h"

char VersionId_knox_knox[] = QUIP_VERSION_STRING;

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "debug.h"
#include "data_obj.h"
#include "submenus.h"
#include "knox.h"
#include "serial.h"
#include "serbuf.h"		/* expected_response() */
#include "version.h"
#include "ttyctl.h"		/* keyhit() */

#define TIMER_STOPPED	1
#define TIMER_RUNNING	2

static int timer_state=TIMER_STOPPED;

static Knox_State knox_state;	/* not initialized! */

static COMMAND_FUNC( do_knox_timer_stop );

static int doing_command_set = 0;				/* flag for sending set of commands */

static int range_ok;

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

ITEM_INTERFACE_DECLARATIONS(Knox_Device,knox_dev)

#define PICK_KNOX_DEV(pmpt)		pick_knox_dev(QSP_ARG  pmpt)


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


#define GET_WITH_LIMITS( var, string, minval, maxval )				\
										\
	var = HOW_MANY(string);							\
	CHECK_RANGE(string,var, minval, maxval)					\
	if( !range_ok ) ret_stat=(-1);



#define GET_PATTERN( var, string )						\
										\
	GET_WITH_LIMITS( var, string, MIN_CROSSPOINT_PATTERN, MAX_CROSSPOINT_PATTERN)


#define GET_SIGNAL( var, string )						\
										\
	GET_WITH_LIMITS( var, string, MIN_SIGNAL_NUMBER, MAX_SIGNAL_NUMBER )



#define MIN_SIGNAL_NUMBER	1
#define MAX_SIGNAL_NUMBER	8

#define KNOX_DONE_MSG		"DONE"
#define KNOX_ERROR_MSG		"ERROR"

static int get_knox_args(QSP_ARG_DECL   char* arg_buf);
#define GET_KNOX_ARGS(arg_buf)		get_knox_args(QSP_ARG  arg_buf)

static void show_routing_map(void)
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

static COMMAND_FUNC( do_route_both )
{
	char knox_args[LLEN];

	if( GET_KNOX_ARGS(knox_args) < 0 ) return;

	if( do_knox_cmd(QSP_ARG  KNOX_SET_BOTH,knox_args) < 0 ) {
		WARN("Unable to route audio and video!");
	}
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

	if( do_knox_cmd(QSP_ARG  KNOX_SET_DIFF,knox_args) < 0 ) {
		WARN("Unable to route audio and video from different inputs!");
	}
}

static COMMAND_FUNC( do_route_video )
{
	char knox_args[LLEN];

	if( GET_KNOX_ARGS(knox_args) < 0 ) return;

	if( do_knox_cmd(QSP_ARG  KNOX_SET_VIDEO,knox_args) < 0 ) {
		WARN("Unable to route video alone!");
	}
}

static COMMAND_FUNC( do_route_audio )
{
	char knox_args[LLEN];

	if( GET_KNOX_ARGS(knox_args) < 0 ) return;
	if( do_knox_cmd(QSP_ARG  KNOX_SET_AUDIO,knox_args) < 0 ) {
		WARN("Unable to route audio alone!");
	}
}

static Command knox_route_ctbl[] = {
{ "both",	do_route_both,	"route both audio and video (same input)"	},
{ "diff",	do_route_diff,	"route both audio and video (different inputs)" },	
{ "video",	do_route_video,	"route video alone"				},
{ "audio",	do_route_audio,	"route audio alone"				},
{ "quit",	popcmd,		"exit route submenu"				},
{ NULL_COMMAND									}
};

#ifdef UNIMPLEMENTED
static Command knox_salvo_ctbl[] = {
{ "both",	do_salvo_both,	"route both audio and video (same input)"	},
{ "video",	do_salvo_video,	"route video alone"				},
{ "audio",	do_salvo_audio,	"route audio alone"				},
{ "quit",	popcmd,		"exit salvo submenu"				},
{ NULL_COMMAND									}
};

static Command knox_conf_ctbl[] = {
{ "both",	do_conf_both,	"route both audio and video (same input)"	},
{ "video",	do_conf_video,	"route video alone"				},
{ "audio",	do_conf_audio,	"route audio alone"				},
{ "quit",	popcmd,		"exit conference submenu"			},
{ NULL_COMMAND									}
};

static Command knox_cmd_ctbl[] = {
{ "both",	do_cmd_both,	"route both audio and video (same input)"	},
{ "video",	do_cmd_video,	"route video alone"				},
{ "audio",	do_cmd_audio,	"route audio alone"				},
{ "quit",	popcmd,		"exit cmd submenu"				},
{ NULL_COMMAND									}
};
#endif

static COMMAND_FUNC( do_knox_route_cmds )
{
	PUSHCMD(knox_route_ctbl, "route");	
}	

static COMMAND_FUNC( do_knox_salvo_cmds )
{
	if( doing_command_set ) {
		WARN("command set begun, end command set first for salvo commands");
		return;
	}
	/* current_mode = KNOX_SALVO_MODE; */
	PUSHCMD(knox_route_ctbl, "salvo");
}

	
static COMMAND_FUNC( do_knox_conf_cmds )
{
	if( doing_command_set ) {
		WARN("begin command set issued, end command set first for conference commands");
		return;
	}
	/* current_mode = KNOX_CONFERENCE_MODE; */
	PUSHCMD(knox_route_ctbl, "conference");
}

static COMMAND_FUNC( do_take_set_cmds )
{
	if( !doing_command_set ) {
		advise("CAUTIOUS: command set not begun to execute commands!?");
		return;
	}

	if( do_knox_cmd(QSP_ARG  KNOX_TAKE_COMMAND, NULL) < 0 ) {
		WARN("Unable to take current commands sent!");
	}
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

static Command knox_cmds_ctbl[] = {
{ "route",	do_knox_route_cmds,	"regular route audio and video"		},
{ "salvo",	do_knox_salvo_cmds,	"salvo route audio and video"		},
{ "conference",	do_knox_conf_cmds,	"conference route audio and video"	},
{ "begin",	do_begin_set_cmds,	"begin a set of commands (route only)"	},
{ "end_set",	do_end_set_cmds,	"end set of commands and send"		},
{ "take",	do_take_set_cmds,	"execute current set of commands sent"	},
{ "quit",	popcmd,			"exit submenu"				},
{ NULL_COMMAND									}
};

static COMMAND_FUNC( do_knox_main_cmds )
{
	PUSHCMD(knox_cmds_ctbl, "command");
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
	if( do_knox_cmd(QSP_ARG  KNOX_RECALL_CROSSPOINT, args) < 0 ) {
		WARN("Unable to load crosspoint pattern!");
	}
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
	if( do_knox_cmd(QSP_ARG  KNOX_STORE_CROSSPOINT, args) < 0 ) {
		WARN("Unable to store crosspoint pattern!");
	}
}

static Command knox_cross_ctbl[] = {
{ "load",	do_knox_recall,	"load crosspoint pattern from memory"		},
{ "store",	do_knox_store,	"store currently loaded crosspoint pattern" 	},
{ "quit",	popcmd,		"exit submenu" 					},
{ NULL_COMMAND									}
};

static COMMAND_FUNC( do_knox_cross_cmds )
{
	PUSHCMD(knox_cross_ctbl, "crosspoint");
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
	if( do_knox_cmd(QSP_ARG  KNOX_SET_TIMER, args) < 0 ) {
		WARN("Unable to set timed sequencer!");
	}
}

static COMMAND_FUNC( do_knox_timer_stop )
{
	if( do_knox_cmd(QSP_ARG  KNOX_STOP_TIMER, NULL) < 0 ) {
		WARN("Unable to stop timed sequencer!");
	}
}

static Command knox_timer_ctbl[] = {
{ "start",	do_knox_timer_start,	"set time cycle time interval and start timer"	},
{ "stop",	do_knox_timer_stop,	"stop time cycle"				},
{ "quit",	popcmd,			"exit submenu"					},
{ NULL_COMMAND										}
};

static COMMAND_FUNC( do_knox_time_cmds )
{
	PUSHCMD(knox_timer_ctbl, "timer");
}

static COMMAND_FUNC( do_fetch_map )
{
	Data_Obj *dp;
	u_char *p;
	int i;

	dp = PICK_OBJ("object for routing results");

	if( dp == NO_OBJ ) return;

	if( dp->dt_cols != 8 ){
		sprintf(ERROR_STRING,
	"Object %s should have 8 columns for knox report",
			dp->dt_name);
		WARN(ERROR_STRING);
		return;
	}
	if( dp->dt_comps != 2 ){
		sprintf(ERROR_STRING,
	"Object %s should have 2 components for knox report",
			dp->dt_name);
		WARN(ERROR_STRING);
		return;
	}
	if( dp->dt_prec != PREC_UBY ){
		sprintf(ERROR_STRING,
	"Object %s should have %s precision for knox report",
			dp->dt_name,name_for_prec(PREC_UBY));
		WARN(ERROR_STRING);
		return;
	}

	/* do we care about the precision? */

	if( do_knox_cmd(QSP_ARG  KNOX_MAP_REPORT, NULL) < 0 ) {
		WARN("Unable to get routing status map report!");
		return;
	}	

	p=(u_char *)dp->dt_data;
	for(i=0;i<8;i++){
		*(p + i * dp->dt_pinc) = knox_state.ks_video[i];
		*(p + i * dp->dt_pinc + dp->dt_cinc ) = knox_state.ks_audio[i];
	}
}

static COMMAND_FUNC( do_show_map )
{
	if( do_knox_cmd(QSP_ARG  KNOX_MAP_REPORT, NULL) < 0 ) {
		WARN("Unable to get routing status map report!");
		return;
	}	

	show_routing_map();
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

	if( do_knox_cmd(QSP_ARG  KNOX_CONDENSE_REPORT, NULL) < 0 ) {
		WARN("Unable to get condensed routing status map report!");
		return;
	}	

	prt_msg("\nCondensed Routing Map Status Report:");
	show_routing_map();
}

static Command knox_status_ctbl[] = {
{ "fetch_map",		do_fetch_map,		"store routing map to data vector"			},
{ "show_map",		do_show_map,		"display routing map"			},
{ "condense",		do_inq_knox_cond,	"get condensed routing map status report"	},
{ "quit",		popcmd,			"exit submenu"					},
{ NULL_COMMAND											}
};

static COMMAND_FUNC( do_knox_status_cmds )
{
	PUSHCMD(knox_status_ctbl, "status");
}

static COMMAND_FUNC( do_lamp_test )
{
	/* Lamp test command has no args! */
	/*if( GET_KNOX_ARGS(args) < 0 ) return; */

	if( do_knox_cmd(QSP_ARG  KNOX_LAMP_TEST, NULL) < 0 ) {
		WARN("Unable to do lamp test!");
	}
}

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

static COMMAND_FUNC( do_select_device )
{
	const char *s;
	Knox_Device *kdp;

	s=NAMEOF("knox device");

	kdp = knox_dev_of(QSP_ARG  s);
	if( kdp != NULL ){
		curr_kdp = kdp;
		return;
	}

	open_knox_device(QSP_ARG  s);
}

static Command knox_main_ctbl[] = {
{ "device",	do_select_device,	"select knox device"			},
{ "command",	do_knox_main_cmds,	"routing switcher commands"		},
{ "crosspoint",	do_knox_cross_cmds,	"crosspoint pattern commands"		},
{ "timer",	do_knox_time_cmds,	"timed sequencer commands"		},
{ "status",	do_knox_status_cmds,	"switcher status commands"		},
{ "lamp",	do_lamp_test,		"perform lamp test"			},
{ "quit",	popcmd,			"exit submenu"				},
{ NULL_COMMAND									},
};

/* This was /dev/ttyS0 on fourier, but we insist on using a symlink
 * /dev/knox to make this portable to other systems which might use
 * a different port.
 */

#define KNOX_TTY_DEV	"/dev/knox"

COMMAND_FUNC( knoxmenu )
{
	if( curr_kdp == NULL ) {
		auto_version(QSP_ARG  "KNOX","VersionId_knox");
		open_knox_device(QSP_ARG  KNOX_TTY_DEV);
		if( curr_kdp == NULL )
			ERROR1("Unable to open default knox device");
	}

	PUSHCMD(knox_main_ctbl, "knox");
}

