#include "quip_config.h"

char VersionId_interpreter_builtin[] = QUIP_VERSION_STRING;

/*
 * Possible #define's
 *
 * MY_INTR	allow user to specify interrupt handler (UNIX only)
 * DYNAMIC_LOAD	allow loading of code files
 * HELPFUL	include help submenu
 * SUBMENU	allow addition of submenus
 */

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>		/* system() */
#endif

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#ifdef HAVE_TIME_H
#include <time.h>		/* time(), ctime() */
#endif

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>		/* gettimeofday */
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>		/* getuid(), usleep() */
#endif

#include "debug.h"
#include "query.h"
#include "macros.h"
#include "nexpr.h"
#include "rn.h"
#include "savestr.h"
#include "sigpush.h"
#include "submenus.h"		/* call_event_funcs() */
#include "history.h"
#include "chewtext.h"
#include "bi_cmds.h"
#include "filerd.h"
#include "verjbm.h"
#include "menuname.h"
#include "submenus.h"
#include "rt_sched.h"
#include "callback_api.h"

/* local prototypes */
static void whileloop(QSP_ARG_DECL   const char *exp_str);
static COMMAND_FUNC( do_advise );
static COMMAND_FUNC( do_openloop );
static COMMAND_FUNC( echo );
static COMMAND_FUNC( do_system );
static COMMAND_FUNC( do_foreloop );
static COMMAND_FUNC( do_pmpttext );
static COMMAND_FUNC( do_pushtext );
static COMMAND_FUNC( do_while );
static COMMAND_FUNC( do_if );
static COMMAND_FUNC( bi_help );
static COMMAND_FUNC( curr_help );
static COMMAND_FUNC( do_output_redir );
static COMMAND_FUNC( do_append );
static COMMAND_FUNC( do_error_redir );
static COMMAND_FUNC( do_debug );
static COMMAND_FUNC( do_verbose );
static COMMAND_FUNC( do_pop_it );
static COMMAND_FUNC( do_exit );
static COMMAND_FUNC( do_error_exit );
static COMMAND_FUNC( do_warn );
static COMMAND_FUNC( do_nop );
static COMMAND_FUNC( do_seed );
static COMMAND_FUNC( opendoloop );


#ifdef DYNAMIC_LOAD
#include "bind.h"
#endif /* DYNAMIC_LOAD */

#include "submenus.h"

#include "async.h"

#ifdef DEBUG
#include "items.h"	/* extern void dump_items(); */
#endif /* DEBUG */

#ifdef HELPFUL
#include "help.h"
#endif /* HELPFUL */

#ifdef DEBUG
static u_long bi_debug=0;
#endif /* DEBUG */

static const char *builtin_p="builtins";
static const char *help_p="builtin_help";

#ifdef MY_INTR

#include <signal.h>
static const char *intr_str="";

/* when we get an interrupt, do we associate it with a particular query_stream? */

void my_onintr(int asig /* what is this arg??? */)
{
	if( *intr_str == 0 ) nice_exit(0);

	/* Don't pop the stack - just push the top menu! */
	//while( cmd_depth(SGL_DEFAULT_QSP_ARG) > MIN_CMD_DEPTH ) popcmd(SGL_DEFAULT_QSP_ARG);

#ifdef DEBUG
if( debug & bi_debug ){
sprintf(DEFAULT_ERROR_STRING,"my_onintr:  pushing string \"%s\"",intr_str);
advise(DEFAULT_ERROR_STRING);
}
#endif /* DEBUG */

	//push_input_file(DEFAULT_QSP_ARG  "intr_handler");
	//pushtext(DEFAULT_QSP_ARG  intr_str);

	// Use the digest function - it would be nice to have another arg to pass the filename...
	digest(DEFAULT_QSP_ARG  intr_str);

	default_qsp->qs_flags &= ~QS_HAD_INTR;	/* clear flag */
}

static COMMAND_FUNC( set_onintr )
{
	const char *s;

	s=NAMEOF("text to interpret upon interrupt");
	if( *intr_str == 0 ) {
		sigpush((int)SIGINT,my_onintr);
		/*WARN("no previous interrupt action"); */
	} else {
		if( verbose ){
			sprintf(ERROR_STRING,
		"former interrupt action string was \"%s\"",
				intr_str);
			advise(ERROR_STRING);
		}
		rls_str(intr_str);
	}
	intr_str = savestr(s);

#ifdef DEBUG
if( debug & bi_debug ) {
sprintf(ERROR_STRING,"interrupt action string:  \"%s\"",intr_str);
advise(ERROR_STRING);
}
#endif /* DEBUG */

}
#endif /* MY_INTR */

static COMMAND_FUNC( do_advise )
{
	const char *s;

	s=NAMEOF("advisory string");
	advise(s);
}

static COMMAND_FUNC( do_warn )
{
	const char *s;

	s=NAMEOF("warning string");
	WARN(s);
}

static COMMAND_FUNC( do_nop )
{}

static COMMAND_FUNC( echo )
{
	const char *s;

	s=NAMEOF("string to echo");
	prt_msg(s);
}

#ifndef _WINDOWS
static COMMAND_FUNC( do_system )		/** execute a shell command */
{
	const char *s;
	int stat;
	int euid;
	int ruid;

	s=NAMEOF("command");

#ifdef DEBUG
if( debug & bi_debug ) {
sprintf(ERROR_STRING,"command is \"%s\"",s);
advise(ERROR_STRING);
}
#endif /* DEBUG */

	/* BUG put in config testing here... */
	euid = geteuid();
	ruid = getuid();

	if( euid == 0 && ruid != 0 ){
		WARN("Sorry, shell commands not allowed for set-uid root programs");
		return;
	}

	stat=system(s);

	if( stat == -1 )
		tell_sys_error("system");

	else if( verbose ){
		sprintf(ERROR_STRING,"Exit status %d",stat);
		advise(ERROR_STRING);
	}

	sprintf(msg_str,"%d",stat);
	ASSIGN_VAR("exit_status",msg_str);
}
#endif /* !_WINDOWS */

static void whileloop(QSP_ARG_DECL   const char *exp_str)
{
	double value;
	int v;

	value = pexpr(QSP_ARG  exp_str);
	v = value == 0.0 ? 0 : 1;
	_whileloop(QSP_ARG  v);
}

static COMMAND_FUNC( do_while )
{
	const char *s;

	s=NAMEOF("expression");
	whileloop(QSP_ARG  s);
}

static COMMAND_FUNC( do_if )
{
	const char *s;
	const char *c1;
	double value;
	int no_mistake=1;

	s=NAMEOF("expression");
	value = pexpr(QSP_ARG  s);

	/* We used to disable lookahead *after* calling nameof, but that caused
	 * a problem with an If clause at the end of a macro - the lookahead in nameof
	 * caused the macro to be popped, losing the arguments which might have
	 * been needed in the if action.
	 *
	 * Hopefully this is fixed without unpleasant side effects by simply moving
	 * the disable_lookahead up before the call to nameof().
	 *
	 * We are temporarily putting this back the way it was
	 * because of an unintended consequence...
	 *
	 * But I don't see any disable_lookahead here at all???
	 */


	s=NAMEOF("single-word command (or \"Then\" if \"Else\" follows)");

	if( !strcmp(s,"Then") ){
		s=NAMEOF("single-word command");
		c1=savestr(s);
		s=NAMEOF("must enter \"Else\" after \"Then\" clause");
		if( strcmp(s,"Else") != 0 ){
			WARN("must enter \"Else\" after \"Then\" clause");
			no_mistake=0;
		}
		s=NAMEOF("single-word command");
		if( no_mistake ) {
			if( value == 0.0 ){	// Execute 'Else'
				rls_str(c1);	// release Then clause
				c1=savestr(s);	// save Else clause
			}
			push_if(QSP_ARG  c1);	// Need to save line number!
		}
	} else {
		if( value != 0.0 ){
			push_if(QSP_ARG   savestr(s) );
		}
	}
}

static COMMAND_FUNC( do_seed )
{
	int n;

	n=(int)HOW_MANY("value for random number seed");

	sprintf(msg_str,"Using user-supplied seed of %d",n);
	advise(msg_str);

	set_seed(QSP_ARG  n);
}

#ifdef SUBMENU
static COMMAND_FUNC( do_submenu )
{
	const char *sel, *help, *subpmpt;

	sel= savestr( NAMEOF("selector string") );
	help= savestr( NAMEOF("help string") );
	subpmpt= savestr( NAMEOF("prompt for submenu") );
	submenu(cur_pmpt, sel, help, subpmpt );
}
#endif /* SUBMENU */

static COMMAND_FUNC( do_pmpttext )
{
	const char *p;
	const char *t;
	const char *s;

	p=savestr( NAMEOF("prompt string") );
	s=savestr( NAMEOF("variable name") );
	push_input_file(QSP_ARG   "-" );
	redir(QSP_ARG  tfile(SINGLE_QSP_ARG));
	t=savestr( NAMEOF(p) );
	popfile(SINGLE_QSP_ARG);
	ASSIGN_VAR(s,t);
	rls_str(p);
	rls_str(s);
	rls_str(t);
}

static COMMAND_FUNC( do_pushtext )
{
	const char *s;

	s=savestr(NAMEOF("text to push"));
	push_input_file(QSP_ARG  "-");
	fullpush(QSP_ARG  s);
	/* hopefully push'd text is freed automatically? */
}

static COMMAND_FUNC( do_exit_macro )
{
	int i,done_level;
	Macro *mp;

//advise("do_exit_macro BEGIN");
//qdump(SINGLE_QSP_ARG);

	done_level=(-1);	// pointless initialization to quiet compiler
	i=QLEVEL;
	mp = NO_MACRO;
	while( i >= 0 ){
		if( mp != NO_MACRO ){
			if( THIS_QSP->qs_query[i].q_macro != mp ){
				done_level=i;
				i = -1;
			}
		} else if( THIS_QSP->qs_query[i].q_macro != NO_MACRO ){
			/* There is a macro to pop... */
			mp = THIS_QSP->qs_query[i].q_macro;
		}
		i--;
	}
	if( mp == NO_MACRO ){
		WARN("exit_macro:  no macro to exit!?");
		return;
	}
#ifdef CAUTIOUS
	if( done_level == -1 )
		WARN("CAUTIOUS:  do_exit_macro:  done_level not set!?");
#endif /* CAUTIOUS */

	i=QLEVEL;
	while(i>done_level){
		popfile(SINGLE_QSP_ARG);
		i--;
	}
}

static COMMAND_FUNC( do_sim_typing )
{
	const char *s;

	s=NAMEOF("text to push");
	simulate_typing(s);
}

static COMMAND_FUNC( do_pop_it )
{
	int n;

	n=(int)HOW_MANY("number of levels to pop input stream");
	pop_it(QSP_ARG  n);
}

static COMMAND_FUNC( do_exit )
{
	if( verbose ) advise("\nExiting Program");
	nice_exit(0);
}

static COMMAND_FUNC( do_error_exit )
{
	const char *s;

	s=NAMEOF("reason for error exit");
	ERROR1(s);
}

#ifdef MALLOC_DEBUG
static COMMAND_FUNC( do_m_verify )
{
	malloc_verify();
}
#endif /* MALLOC_DEBUG */

#ifndef MAC

static COMMAND_FUNC( do_usleep )
{
	int n;

	n=(int)HOW_MANY("number of microseconds to sleep");
	usleep(n);
}

/* PAS - Do not make this routine static - Called from pcutils lib */
/* pcutilsmenu deleted - jbm */

static COMMAND_FUNC( do_sleep )
{
	int n;

	n=(int)HOW_MANY("number of seconds to sleep");
	sleep(n);
}
#endif /* ! MAC */

static int append_flag=0;
static const char *open_mode_string[2]={"w","a"};

static COMMAND_FUNC( do_append )
{
	if( ASKIF("append output and error redirect files") )
		append_flag=1;
	else
		append_flag=0;
}

static COMMAND_FUNC( do_output_redir )
{
	const char *s;

	s=NAMEOF("output file");
	set_output_file(QSP_ARG  s);
}

static const char *output_file_name=NULL;

void set_output_file(QSP_ARG_DECL  const char *s)
{
	FILE *fp;

	if( output_file_name==NULL ){	/* first time? */
		if( (!strcmp(s,"-")) || (!strcmp(s,"stdout")) ){
			/* stdout should be initially open */
			return;
		}
	} else if( !strcmp(output_file_name,s) ){	/* same file? */
/*
sprintf(ERROR_STRING,"set_output_file %s, doing nothing",s);
advise(ERROR_STRING);
*/
		return;
	}
	/* output_redir will close the current file... */

	if( output_file_name != NULL )
		rls_str(output_file_name);

	output_file_name=savestr(s);

	if( (!strcmp(s,"-")) || (!strcmp(s,"stdout")) )
		fp=stdout;
	else {
		fp=TRYNICE(s,open_mode_string[append_flag]);
	}

	if( !fp ) return;

	output_redir(fp);
}

static COMMAND_FUNC( do_error_redir )
{
	FILE *fp;
	const char *s;

	s=NAMEOF("error file");

	if( (!strcmp(s,"-")) || (!strcmp(s,"stderr")) )
		fp=stderr;
	else {
		fp=TRYNICE(s,open_mode_string[append_flag]);
	}

	if( !fp ) return;

	error_redir(fp);
}

static COMMAND_FUNC( do_debug )
{
	Debug_Module *dbmp;
	const char **db_mod_list;
	Node *np;
	List *lp;
	int i,n;

	lp = dbm_list();
	n=3+eltcount(lp);

	db_mod_list = (const char**) getbuf( n * sizeof(char *) );
	db_mod_list[0] = "all";
	db_mod_list[1] = "yes";
	db_mod_list[2] = "no";
	i=3;

	if( lp != NO_LIST ){
		np=lp->l_head;
		while(np!=NO_NODE){
#ifdef CAUTIOUS
			if( i >= n ) ERROR1("CAUTIOUS:  do_debug:  too many debug modules!?");
#endif /* CAUTIOUS */
			dbmp=(Debug_Module*) np->n_data;
			db_mod_list[i++] = dbmp->db_name;
			np=np->n_next;
		}
	}

	i = WHICH_ONE("debug module",n,db_mod_list);
	givbuf(db_mod_list);
	if( i < 0 ) return;

	switch(i){
		/* BUG this works as long as debug_word is 32 bits... */
		case 0: case 1: debug = 0xffffffff; break;
		case 2:		debug=0; break;
		default:
			np = nth_elt(lp,i-3);
#ifdef CAUTIOUS
			if(np==NO_NODE){
				WARN("CAUTIOUS:  missing node in debug");
				return;
			}
#endif /* CAUTIOUS */
			dbmp = (Debug_Module*) np->n_data;
			set_debug(QSP_ARG  dbmp);
			break;
	}
}

static COMMAND_FUNC( do_verbose )
{
	if( ASKIF("print verbose messages") ){
		set_verbose();
		ASSIGN_VAR("verbose","1");
	} else {
		clr_verbose();
		ASSIGN_VAR("verbose","0");
	}
}

static COMMAND_FUNC( do_openloop )
{
	int n;

	n = (int) HOW_MANY("number of iterations");
	if( n < 1 ){
		sprintf(ERROR_STRING,"loop count (%d) must be positive",n);
		WARN(ERROR_STRING);
		return;
	}
	openloop(QSP_ARG  n);
}

static COMMAND_FUNC( do_foreloop )
{
	Foreloop *frp;

	char delim[LLEN];
	char pmpt[LLEN];
	const char *s;

	frp = (Foreloop*) getbuf( sizeof(*frp) );

	s=NAMEOF("variable name");
	frp->f_varname = savestr(s);

	s=NAMEOF("opening delimiter, usually \"(\"");
	if( !strcmp(s,"(") )
		strcpy(delim,")");
	else
		strcpy(delim,s);

	sprintf(pmpt,"next value, or closing delimiter \"%s\"",delim);

	/* New style doesn't have a limit on the number of items */

	frp->f_lp = new_list();

	while(1){
		s=NAMEOF(pmpt);
		if( !strcmp(s,delim) ){		/* end of list? */
			if( eltcount(frp->f_lp) == 0 ) {		/* no items */
				sprintf(ERROR_STRING,
			"foreach:  no values specified for variable %s",
					frp->f_varname);
				WARN(ERROR_STRING);
				zap_fore(frp);
			} else {
				frp->f_np = frp->f_lp->l_head;
				fore_loop(QSP_ARG  frp);
			}
			return;
		} else {
			Node *np;
			np = mk_node( (void *) savestr(s) );
			addTail(frp->f_lp,np);
		}
	}

}

static COMMAND_FUNC( opendoloop )
{
	openloop(QSP_ARG  -1);
}

static COMMAND_FUNC( do_qtell )
{
	int i;

	i=tell_qlevel(SINGLE_QSP_ARG);
	sprintf(ERROR_STRING,"qlevel=%d",i);
	advise(ERROR_STRING);
}

static COMMAND_FUNC( do_cd_tell )
{
	sprintf(msg_str,"cmd_depth = %d",cmd_depth(SINGLE_QSP_ARG));
	prt_msg(msg_str);
}

/*
static void do_qtrace(VOID)
{
	int i;

	i=HOW_MANY("level");
	show_query_level(i);
}
*/

#ifdef HAVE_HISTORY
static COMMAND_FUNC( set_history )
{
	if( ASKIF("retain input history") )
		history=1;
	else history=0;
}
#endif /* HAVE_HISTORY */

/* was nice_exit, but nice_exit() resets tty, we don't want that when exiting a forked thread... */

static COMMAND_FUNC( my_quick_exit ){ exit(0); }

#ifdef USE_GETBUF
extern COMMAND_FUNC( heap_report );
#endif

static COMMAND_FUNC( get_time )
{
	const char *s;
	time_t t;

	s=NAMEOF("variable name");
	t=time(NULL);
	if( t == (time_t)(-1) ){
		tell_sys_error("time");
		t = (time_t) 0;
	}
	sprintf(msg_str,"%ld",t);
	ASSIGN_VAR(s,msg_str);
}

static COMMAND_FUNC( get_time_of_day )
{
	const char *s1,*s2;
	struct timeval tv;

	s1=NAMEOF("variable name for seconds");
	s2=NAMEOF("variable name for microseconds");

	if( gettimeofday(&tv,NULL) < 0 ){
		perror("gettimeofday");
		WARN("error reading system time");
		return;
	}

	sprintf(msg_str,"%ld",tv.tv_sec);
	ASSIGN_VAR(s1,msg_str);
	// on mac, tv_usec has a wierd type?
	sprintf(msg_str,"%ld",(long)tv.tv_usec);
	ASSIGN_VAR(s2,msg_str);
}

static Data_Obj *ckpt_tbl_dp=NO_OBJ;
static Data_Obj *ckpt_msg_dp=NO_OBJ;
static int n_ckpts=0;
#define MAX_CKPTS	512
#define MAX_MSG_LEN	60
#define CKPT_TBL_NAME	"__ckpt_tbl"
#define CKPT_MSG_NAME	"__ckpt_msg"

static COMMAND_FUNC( do_ckpt )
{
	const char *s;
	struct timeval tv;
	char *ptr;
	time_t *t_ptr;

	s = NAMEOF("tag for this checkpoint");

	if( ckpt_tbl_dp == NO_OBJ ){
		int siz,siz2;
		prec_t prec;

		siz=sizeof(time_t);
		if( siz == 8 ) prec=PREC_ULI;
		else if( siz == 4 ) prec=PREC_UDI;
#ifdef CAUTIOUS
		else ERROR1("CAUTIOUS:  do_ckpt:  unhandled size of time_t");

		siz2 = sizeof(suseconds_t);
		if( siz2 > siz ) {
			sprintf(ERROR_STRING,
"CAUTIOUS:  do_ckpt:  size of suseconds_t (%d) is greater than that of time_t (%d).",siz2,siz);
			ERROR1(ERROR_STRING);
		}
#endif /* CAUTIOUS */

#ifdef HAVE_CUDA
		push_data_area(ram_area);
#endif /* HAVE_CUDA */

		ckpt_tbl_dp = mk_vec(QSP_ARG  CKPT_TBL_NAME, MAX_CKPTS, 2, prec );
		if( ckpt_tbl_dp == NO_OBJ ) ERROR1("Error creating checkpoint table");

		ckpt_msg_dp = mk_img(QSP_ARG  CKPT_MSG_NAME, MAX_CKPTS, MAX_MSG_LEN, 1, PREC_STR );
		if( ckpt_msg_dp == NO_OBJ ) ERROR1("Error creating checkpoint messages");

#ifdef HAVE_CUDA
		pop_data_area();
#endif /* HAVE_CUDA */

	}

	if( strlen(s) >= MAX_MSG_LEN ){
		sprintf(ERROR_STRING,"Sorry, checkpoint tag has too many characters (%ld, max %d), truncating...",
			strlen(s),MAX_MSG_LEN-1);
		WARN(ERROR_STRING);
	}
	if( n_ckpts >= MAX_CKPTS ){
		sprintf(ERROR_STRING,"Sorry, %d checkpoints already placed, can't place '%s'.",n_ckpts,s);
		WARN(ERROR_STRING);
		return;
	}

	if( gettimeofday(&tv,NULL) < 0 ){
		perror("gettimeofday");
		WARN("error reading system time");
		return;
	}

	t_ptr = (time_t *)ckpt_tbl_dp->dt_data;
	t_ptr += n_ckpts * 2 ;
	* t_ptr = tv.tv_sec;
	t_ptr ++;
	* t_ptr = tv.tv_usec;

	ptr = ckpt_msg_dp->dt_data;
	ptr += n_ckpts * MAX_MSG_LEN;

	strncpy(ptr,s,MAX_MSG_LEN-1);
	*(ptr+MAX_MSG_LEN-1) = 0;	// insure null-terminated

	n_ckpts++;
}

static COMMAND_FUNC( do_tell_ckpts )
{
	int i;
	char *mptr;
	time_t *tptr;
	long secs, usecs;
	long secs0, usecs0;
	float delta_ms, cum_ms;

	if( n_ckpts == 0 ){
		WARN("do_tell_ckpts:  no checkpoints set.");
		return;
	}

	tptr = (time_t *) ckpt_tbl_dp->dt_data;
	mptr = (char *) ckpt_msg_dp->dt_data;

	secs0 = (long) *tptr;
	usecs0 = (long) *(tptr+1);

	secs  = (long) *(tptr+  2*(n_ckpts-1));
	usecs = (long) *(tptr+1+2*(n_ckpts-1));
	delta_ms = 1000.0*(secs-secs0) + (usecs-usecs0)/1000.0;
	sprintf(msg_str,"Total HOST time:  %12.3f",delta_ms);
	prt_msg(msg_str);

	cum_ms = 0.0;
	for(i=0;i<n_ckpts;i++){
		secs = (long) *tptr;
		usecs = (long) *(tptr+1);

		delta_ms = 1000.0*(secs-secs0) + (usecs-usecs0)/1000.0;
		cum_ms += delta_ms;

		sprintf(msg_str,"HOST %3d  %12.3f  %12.3f  %s",i+1,delta_ms,
			cum_ms,mptr);
		prt_msg(msg_str);

		tptr += 2;
		mptr += MAX_MSG_LEN;

		secs0 = secs;
		usecs0 = usecs;
	}
	n_ckpts=0;	// clear for next time
}


static COMMAND_FUNC( do_showmaps )
{
#ifdef USE_GETBUF
	showmaps();
#else /* ! USE_GETBUF */
	WARN("do_showmaps:  program not configured with USE_GETBUF, nothing to show.");
#endif /* ! USE_GETBUF */
}

static COMMAND_FUNC( do_report_node_data ){ report_node_data(SINGLE_QSP_ARG); }
static COMMAND_FUNC( do_dump_items ){ dump_items(SINGLE_QSP_ARG); }
/* BUG this is a forward reference to libxsupp...  to eliminate
 * this dependency, we should use a function vector as done for callbacks
 */
static void (*discard_event_func_vec)(SINGLE_QSP_ARG_DECL)=NULL;

void set_discard_func(void (*func)(SINGLE_QSP_ARG_DECL) )
{
	discard_event_func_vec=func;
}

static COMMAND_FUNC( do_flush_events )
{ 
	if( discard_event_func_vec == NULL ){
		WARN("do_flush_events:  no discard function specified!?");
	} else {
		(*discard_event_func_vec)(SINGLE_QSP_ARG);
	}
}

static Command unix_ctbl[]={
#ifdef HAVE_ADJTIMEX
{ "adjtimex",		timex_menu,	"linux clock adjustment submenu"	},
#endif /* HAVE_ADJTIMEX */

{ "get_time",		get_time,	"set var to current time"	},
{ "usecs",		get_time_of_day,"set vars to current seconds and usecs"	},
{ "checkpoint",		do_ckpt,	"record time to checkpoint table"	},
{ "report",		do_tell_ckpts,	"report checkpoints and reset"	},
{ "serial",		ser_menu,	"send/receive serial data"	},
{ "mouse",		mouse_menu,	"talk to mouse on serial port"	},
{ "pipes",		pipemenu,	"subprocess pipe submenu"	},

#ifdef HAVE_HISTORY
{ "history",		set_history,	"enable/disable history"	},
#ifdef TTY_CTL
{ "complete",		set_completion,	"set/clr command completion"	},
#endif /* TTY_CTL */
#endif /* HAVE_HISTORY */

{ "events",		call_event_funcs,"call event handler(s)"	},
{ "flush_events",	do_flush_events,"discard pending events"	},
{ "showmem",		do_showmaps,	"show free heap memory"		},
{ "nodes",		do_report_node_data,"show node statistics"		},

#ifdef DEBUG
{ "dump_items",		do_dump_items,	"list all items"		},
{ "Qstack",		qdump,		"dump state of query stack"	},
{ "tellq",		do_qtell,	"report level of query stack"	},
{ "cmd_depth",		do_cd_tell,	"report level of command stack"	},
#ifdef USE_GETBUF
{ "heaps",		heap_report,	"report free heap mem"		},
#endif
/*
{ "qtrace",		do_qtrace,	"show a single query level"	},
*/
#endif /* DEBUG */

#ifdef MY_INTR
{ "interrupt_action",	set_onintr,	"set interrupt command string"	},
#endif /* MY_INTR */

{ "prompts",		tog_pmpt,	"toggle prompt printing flag"	},
{ "clobber",		togclobber,	"toggle file clobber caution"	},

#ifdef MALLOC_DEBUG
{ "malloc_verify",	do_m_verify,	"heap consistency check"	},
#endif /* MALLOC_DEBUG */

#ifdef DYNAMIC_LOAD
{ "load",		do_load,	"load object module"		},
{ "bind",		do_bind,	"bind new menu item to function"},
#endif /* DYNAMIC_LOAD */

{ "fork",		do_fork,	"fork interpreter"		},
{ "wait",		wait_child,	"wait for child to finish"	},
{ "threads",		thread_menu,	"threads submenu"		},
{ "fast_exit",		my_quick_exit,	"exit without resetting tty"	},

{ "quit",		popcmd,		"exit submenu"			},
{ NULL_COMMAND								}
};

static COMMAND_FUNC( unix_menu ) { PUSHCMD(unix_ctbl,"os"); }

static COMMAND_FUNC( set_max_warn )
{
	int n;

	n=(int)HOW_MANY("max number of warnings");
	set_max_warnings(n);
}

static COMMAND_FUNC( clr_n_warn )
{
	clear_warnings();
}

static COMMAND_FUNC( cnt_n_warn )
{
	int n;
	char str[LLEN];
	const char *s;

	s=NAMEOF("variable name");
	n=count_warnings();
	sprintf(str,"%d",n);
	ASSIGN_VAR(s,str);
}

/*
 * This function was added, so that the output could be redirected
 * to the programs output file (which didn't work w/ "system date")
 */

COMMAND_FUNC( do_date )
{
	time_t timeval;
	char *s;

	time(&timeval);
	s=ctime(&timeval);
	/* ctime includes newline... */
	s[ strlen(s)-1 ] = 0;
	advise(s);
}


static const char *script_warning_text=NULL;

void script_warn(QSP_ARG_DECL  const char *s)
{
	WARN(s);
	if( script_warning_text != NULL )
		chew_text(QSP_ARG  script_warning_text);
}

void set_warning_script(const char *s)
{
	if( script_warning_text != NULL ) rls_str(script_warning_text);
	script_warning_text = savestr(s);

	set_warn_func(script_warn);
}

static COMMAND_FUNC( do_script_warn )
{
	const char *s;

	s=NAMEOF("script fragment to interpret after warning encountered");
	set_warning_script(s);
}

static COMMAND_FUNC( do_identify )
{
	identify_self( ASKIF("print program name before all advisory messages") );
}


/*
 * This command table can be static except on the mac,
 * where it needs to be referenced by a menu init func
 */

static COMMAND_FUNC( get_prompt ) 
{
	if( verbose ){
		sprintf(ERROR_STRING,"Current prompt is \"%s\"",QUERY_PROMPT);
		advise(ERROR_STRING);
	}
	ASSIGN_VAR("prompt",QUERY_PROMPT);
}

static COMMAND_FUNC( do_callbacks )
{
	int yn;

	yn=ASKIF("enable callbacks");
	if( yn )
		callbacks_on(SINGLE_QSP_ARG);
	else
		callbacks_off(SINGLE_QSP_ARG);
}


#ifndef MAC
static
#endif
Command builtin_ctbl[]={

	/* user communication */
{ ECHO_CMD_WORD,	echo,		"echo single word"		},
{ WARN_CMD_WORD,	do_warn,	"give a warning message"	},
{ "script_warn",	do_script_warn,	"specify text to interpret at warning time" },
{ MAX_WARN_CMD_WORD,	set_max_warn,	"specify max warnings before abort"	},
{ CLR_WARN_CMD_WORD,	clr_n_warn,	"clear warning message counter"	},
{ CNT_WARN_CMD_WORD,	cnt_n_warn,	"count warning messages"	},
{ ADVISE_CMD_WORD,	do_advise,	"give an advisory message"	},
{ VERBOSE_CMD_WORD,	do_verbose,	"toggle verbose flag"		},
{ ID_CMD_WORD,		do_identify,	"print program name with all advisories"	},
{ DEBUG_CMD_WORD,	do_debug,	"enable/disable debugging msgs"	},
{ DATE_CMD_WORD,	do_date,	"print current time and date"	},

	/* loop control */
{ RPT_CMD_WORD,		do_openloop,	"open an iterative loop"	},
{ FOR_CMD_WORD,		do_foreloop,	"loop over a list of words"	},
{ DO_CMD_WORD,		opendoloop,	"open a command loop"		},
{ END_CMD_WORD,		closeloop,	"end command loop"		},
{ WHILE_CMD_WORD,	do_while,	"conditionally terminate loop"	},
{ IF_CMD_WORD,		do_if,		"conditionally read a word"	},
{ NOP_CMD_WORD,		do_nop,		"do nothing"			},
{ "exit",		do_exit,	"exit program"			},
{ "error_exit",		do_error_exit,	"exit program after printing error msg"	},

	/* file control */
{ REDIR_CMD_STR,	filerd,		"redirect input from file"	},
{ XSCR_CMD_STR,		copycmd,	"save dialog"			},
{ "output_file",	do_output_redir,"redirect text output"		},
{ "append",		do_append,	"set/clear append flag"		},
{ "error_file",		do_error_redir,	"redirect error messages"	},
{ POP_CMD_WORD,		do_pop_it,	"close current input file"	},
{ "prompt_text",	do_pmpttext,	"get input word from tty"	},
{ "push_text",		do_pushtext,	"push text onto input stream"	},
{ "simulate_typing",	do_sim_typing,	"push text onto keyboard stream"},
{ "exit_macro",		do_exit_macro,	"exit current macro"		},

{ "seed",		do_seed,	"seed random number generator"	},

#ifndef MAC
	/* submenu stuff */
{ "TopMenu",		top_menu,	"push root level menu"		},
{ "macros",		macmenu,	"macro submenu"			},
{ "variables",		varmenu,	"variable control submenu"	},
{ "items",		ittyp_menu,	"item type submenu"		},
{ "versions",		vermenu,	"module version number submenu"	},
{ "os",			unix_menu,	"os-specific builtin commands"	},

#ifndef _WINDOWS
{ "system",		do_system,	"execute a shell command"	},
{ "sched",		sched_menu,	"scheduler submenu"		},
#endif /* _WINDOWS */

#ifdef SUBMENU
{ "submenu",		do_submenu,	"add a submenu to this menu"	},
#endif /* SUBMENU */

	/* various toggles */
#ifndef MAC
{ "sleep",		do_sleep,	"sleep for a while"		},
{ "usleep",		do_usleep,	"sleep for a short while"	},
#endif /* ! MAC */

#ifdef HELPFUL
{ "help",		give_help,	"detailed help facility"	},
{ "hdebug",		help_debug,	"display help directories"	},
#endif /* HELPFUL */

#endif /* ! MAC */

{ "callbacks",          do_callbacks,   "enable/disable callbacks" },
{ "get_prompt",         get_prompt,     "prints out the current prompt" },
{ NULL_COMMAND								}
};

static COMMAND_FUNC( bi_help )
{
	hhelpme( QSP_ARG  builtin_p );
	/* hhelpme( help_p ); */
}

static COMMAND_FUNC( curr_help )
{
	hhelpme( QSP_ARG  NULL );
	/* hhelpme( help_p ); */
}

static Command help_ctbl[]={
{ "??",		bi_help,	"list builtin commands"		},
{ "?",		curr_help,	"print current menu"		},
{ NULL_COMMAND							}
};

void bi_init(SINGLE_QSP_ARG_DECL)
{
	static int bi_inited=0;

	if( bi_inited ) return;

	error_init();
	verjbm(SINGLE_QSP_ARG);

	set_bis(builtin_ctbl,builtin_p,help_ctbl,help_p);

#ifdef HAVE_HISTORY
#ifdef TTY_CTL
	hist_bis(builtin_p);	/* allows get_sel() to complete builtins */
#endif /* TTY_CTL */
#endif /* HAVE_HISTORY */

	bi_inited=1;

#ifdef MAC
	create_menu(builtin_ctbl,CTRL_MENU_NAME);
	macmenu();
	varmenu();
	vermenu();
#endif /* MAC */

#ifdef DEBUG
	bi_debug = add_debug_module(QSP_ARG  "builtins");
#endif /* DEBUG */

	/* we initialize the builtin (reserved) variables here too,
	 * for no good reason except that we know this will get called
	 * early on...
	 */
	

	/* the idea of reserved variables (system variables)
	 * is not really restricted to unix, but for the time
	 * being the only one(s) implemented is a unix-ism.
	 */

	init_reserved_vars(SINGLE_QSP_ARG);

}

