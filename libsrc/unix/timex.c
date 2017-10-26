#include "quip_config.h"

#ifdef HAVE_ADJTIMEX

/* timex.c some routines to adjust the system clock so that we get regular timestamps on
 * events of known frequency (e.g. 60 Hz video interrupts)
 */


#ifdef HAVE_SYS_TIMEX_H
#include <sys/timex.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	/* abs() */
#endif

//#include "jtimex.h"
#include "quip_prot.h"
#include "quip_menu.h"

static void tell_timex_retval(QSP_ARG_DECL  int retval)
{
	switch(retval){
		case TIME_OK:  advise("clock is synchronized"); break;
		case TIME_INS:  advise("insert leap second"); break;
		case TIME_DEL:  advise("delete leap second"); break;
		case TIME_OOP:  advise("leap second in progress"); break;
		case TIME_WAIT:  advise("leap second has occurred"); break;
		case TIME_BAD:  advise("clock not synchronized"); break;
		default:
			NWARN("unexpected return value from adjtimex");
			return; 
			break;
	}
}

static void check_timex(QSP_ARG_DECL  struct timex *tp)
{
	int retval;
	char str[32];

	tp->modes=0;	/* don't set anything */
	retval=adjtimex(tp);
	if( retval < 0 ){
		perror("adjtimex");
		WARN("Unable to read timex values");
		return;
	}
	/* if( verbose ) */ tell_timex_retval(QSP_ARG  retval);

	/* set the script variables tick and freq */
	sprintf(str,"%ld",tp->tick);
	assign_reserved_var("timex_tick",str);
	sprintf(str,"%ld",tp->freq);
	assign_reserved_var("timex_freq",str);
}

static COMMAND_FUNC( do_get_timex )
{
	struct timex tb;

	check_timex(QSP_ARG  &tb);

	sprintf(msg_str,"tick:\t%ld",tb.tick);
	prt_msg(msg_str);
	sprintf(msg_str,"freq:\t%ld",tb.freq);
	prt_msg(msg_str);
}

static void set_tick(QSP_ARG_DECL  int tick)
{
	struct timex tb;

	tb.modes=0;	/* don't set anything */
	if( adjtimex(&tb) < 0 ){
		perror("adjtimex");
		WARN("unable to fetch timex parameters");
		return;
	}
	tb.tick = tick;
	tb.modes |= ADJ_TICK;
	if( adjtimex(&tb) < 0 ){
		perror("adjtimex");
		WARN("unable to set timex tick parameter");
		return;
	}

	/* update the script vars */
	check_timex(QSP_ARG  &tb);
}

static void set_freq(QSP_ARG_DECL  int freq)
{
	struct timex tb;

	tb.modes=0;	/* don't set anything */
	if( adjtimex(&tb) < 0 ){
		perror("adjtimex");
		WARN("unable to fetch timex parameters");
		return;
	}
	tb.freq = freq;
	tb.modes |= ADJ_FREQUENCY;
	if( adjtimex(&tb) < 0 ){
		perror("adjtimex");
		WARN("unable to set timex freq parameter");
		return;
	}

	/* update the script vars */
	check_timex(QSP_ARG  &tb);
}

#define MAX_FREQ		6553600
#define MIN_FREQ		(-6553600)

static COMMAND_FUNC( do_set_freq )
{
	long f;

	f=HOW_MANY("New value for frequency parameter");
	if( f > MAX_FREQ || f < MIN_FREQ ){
		sprintf(ERROR_STRING,"freq parameter (%ld) must be between %d and %d",f,MIN_FREQ,MAX_FREQ);
		WARN(ERROR_STRING);
		return;
	}
	set_freq(QSP_ARG  f);
}

static COMMAND_FUNC( do_set_tick )
{
	long t;

	t=HOW_MANY("New value for tick parameter");
	if( abs(10000-t) > 1 ){
		sprintf(ERROR_STRING,"tick parameter (%ld) usually differs by no more than one from default (10000)",t);
		WARN(ERROR_STRING);
	}
	if( abs(10000-t) > 100 ){
		sprintf(ERROR_STRING,"tick parameter (%ld) may not differ by more than 100 from default (10000)",t);
		WARN(ERROR_STRING);
		return;
	}
	set_tick(QSP_ARG  t);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(timex_menu,s,f,h)

MENU_BEGIN(timex)
ADD_CMD( check,	do_get_timex,	read timex parameters )
ADD_CMD( tick,	do_set_tick,	set tick parameter )
ADD_CMD( freq,	do_set_freq,	set freq paramter )
MENU_END(timex)

COMMAND_FUNC( do_timex_menu )
{
	PUSH_MENU(timex);
}

#endif /* HAVE_ADJTIMEX */
