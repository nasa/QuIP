#include "quip_config.h"
char VersionId_interpreter_my_stty[] = QUIP_VERSION_STRING;

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	/* system() */
#endif

#include "query.h"

#ifdef TTY_CTL
#include "ttyctl.h"
#endif /* TTY_CTL */

#include "my_stty.h"
#include "submenus.h"

static int stty_fd=(-1);

void set_stty_fd(int fd)
{
	stty_fd = fd;
}

#define STTY_FD_CHECK	if( stty_fd < 0 ){ NWARN("need to open a file first"); return; }

static COMMAND_FUNC( do_showall )
{
	STTY_FD_CHECK
	show_all(stty_fd);
}

static COMMAND_FUNC( do_stty_dump )
{
	STTY_FD_CHECK
	dump_all(stty_fd);
}

static COMMAND_FUNC( do_set_par )
{
	if( ASKIF("enable parity") ){
		int odd;

		odd=ASKIF("odd parity");
		STTY_FD_CHECK
		set_parity(stty_fd,1,odd);
	} else {
		STTY_FD_CHECK
		set_parity(stty_fd,0,0);
	}
}

static COMMAND_FUNC( do_set_ndata )
{
	int n;

	n=HOW_MANY("number of data bits");
	STTY_FD_CHECK
	set_ndata(stty_fd,n);
}

static COMMAND_FUNC( do_setflag )
{
	const char *fn;
	int value;

	fn=NAMEOF("flag name");
	value = get_flag_value(QSP_ARG fn);

	STTY_FD_CHECK

	set_tty_flag( fn, stty_fd, value );
}

#ifdef B115200
#define N_BAUD_RATES	12
static const char *baud_rates[N_BAUD_RATES]={"150","200","300","600","1200","2400","4800","9600","19200","38400","57600","115200"};
#else /* undef B115200 */
#ifdef B57600
#define N_BAUD_RATES	11
static const char *baud_rates[N_BAUD_RATES]={"150","200","300","600","1200","2400","4800","9600","19200","38400","57600"};
#else /* undef B57600 */
#define N_BAUD_RATES	10
static const char *baud_rates[N_BAUD_RATES]={"150","200","300","600","1200","2400","4800","9600","19200","38400"};
#endif /* undef B57600 */
#endif /* undef B115200 */

static COMMAND_FUNC( do_set_baud )
{
	int n;

	n=WHICH_ONE("baud rate",N_BAUD_RATES,baud_rates);
	if( n < 0 ) return;

	STTY_FD_CHECK

	switch(n){
		case 0:	set_baud(stty_fd,B150); break;
		case 1:	set_baud(stty_fd,B200); break;
		case 2:	set_baud(stty_fd,B300); break;
		case 3:	set_baud(stty_fd,B600); break;
		case 4:	set_baud(stty_fd,B1200); break;
		case 5:	set_baud(stty_fd,B2400); break;
		case 6:	set_baud(stty_fd,B4800); break;
		case 7:	set_baud(stty_fd,B9600); break;
		case 8:	set_baud(stty_fd,B19200); break;
		case 9:	set_baud(stty_fd,B38400); break;
#ifdef B57600
		case 10:set_baud(stty_fd,B57600); break;
#endif /* B57600 */
#ifdef B115200
		case 11:set_baud(stty_fd,B115200); break;
#endif /* B115200 */
	}
}

Command stty_ctbl[]={
{ "show",	do_showall,	"show current settings"			},
{ "setflag",	do_setflag,	"set/clear a flag"			},
{ "parity",	do_set_par,	"set parity"				},
{ "databits",	do_set_ndata,	"set number of data bits (7/8)"		},
{ "baud",	do_set_baud,	"set baud rate"				},
{ "dump",	do_stty_dump,	"dump commands to restore current state"},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL_COMMAND								}
};

COMMAND_FUNC( stty_menu )
{
	if( stty_fd < 0 )
		WARN("stty_menu:  no valid file descriptor selected!?");

	PUSHCMD(stty_ctbl,"stty");
}


