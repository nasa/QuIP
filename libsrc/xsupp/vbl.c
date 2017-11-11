// To run this code, we need seteuid, getuid, ioperm, and inb...
//#ifdef HAVE_VBL

#include "quip_config.h"

#ifdef HAVE_X11

#ifdef HAVE_SYS_IO_H
#include <sys/io.h>		/* ioperm (glibc) */
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>		/* geteuid */
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>		/* geteuid etc */
#endif

#include "quip_prot.h"		/* error1() */
#include "xsupp.h"
#include "xsupp_prot.h"		/* xdebug */
#include "debug.h"

static int vbl_inited=0;

#define vbl_init() _vbl_init(SINGLE_QSP_ARG)

static void _vbl_init(SINGLE_QSP_ARG_DECL)
{
	uid_t uid;

	if( vbl_inited != 0 ) NERROR1("vbl_init:  already inited!?");

	if( seteuid(0) < 0 ){
		perror("seteuid");	/* if we were suid root */
		warn("vbl_init:  unable to set effective uid to 0 for ioperm");
		advise("Make sure program is suid root.");
		vbl_inited = (-1);
		return;
	}

	if( ioperm(0x3c0,32,1) != 0 ){
		perror("ioperm");
		warn("vbl_init:  unable to map VGA registers.");
		advise("Make sure program is suid root.");
		vbl_inited = (-1);
		return;
	}

	/* now restore uid */
	uid = getuid();
	seteuid(uid);

	vbl_inited = 1;
}

#ifdef QUIP_DEBUG_DEBUG

static void dump_all_regs()
{
	/* the register space is from 0x3c0 to 0x3df, 32 byte registers */
	int addr,a;
	int rv[8];
	int i,j;

	addr=0x3c0;
	for(j=0;j<4;j++){
		a=addr;
		for(i=0;i<8;i++){
			rv[i] = inb(a);
			a++;
		}
		/*               addr   0   1   2   3   4   5   6   7   8 */
		sprintf(msg_str,"0x%x:\t%x\t%x\t%x\t%x\t%x\t%x\t%x\t%x\t%x",addr,rv[0],rv[1],rv[2],rv[3],rv[4],rv[5],rv[6],rv[7],rv[8]);
		addr+=8;
		prt_msg(msg_str);
	}
}
#endif /* QUIP_DEBUG_DEBUG */


/* 0x400  1k
 * 0x100000 1M
 * 0x1000000 16M
 * 0x10000000 256M
 */

// #define MAX_WAITS	0x1000000
#define MAX_WAITS	100000

void _vbl_wait(SINGLE_QSP_ARG_DECL)
{
	int regval;
	int ctr;

	if( !vbl_inited ) vbl_init();
	if( vbl_inited < 0 ) return;	/* not possible */

	regval = inb(0x3da);
	ctr=0;
	/* If we're already in blanking, wait til it finishes... */

#ifdef QUIP_DEBUG
//if( debug & xdebug ){
//sprintf(ERROR_STRING,"vbl_wait:  reg 0x3da = 0x%x",regval);
//advise(ERROR_STRING);
//}
#endif /* QUIP_DEBUG */

	/* on a 70 Hz system (fourier), we seem to sometimes miss a pulse...
	 * The timer count jumps from 9k to 36k, we have approx. 27k loop
	 * iterations per 13 msec, or about 500usecs per iteration.
	 * This suggests that the pulse width is less than 500 usec!?
	 */

	while( (regval & 0x8) == 0x8 && ctr < MAX_WAITS ){
		regval = inb(0x3da);
		ctr++;
#ifdef QUIP_DEBUG
//if( debug & xdebug ){
//sprintf(ERROR_STRING,"vbl_wait:  reg 0x3da = 0x%x   (ctr = %d)",regval,ctr);
//advise(ERROR_STRING);
//}
#endif /* QUIP_DEBUG */
	}
#ifdef QUIP_DEBUG
//if( debug & xdebug ){
//sprintf(ERROR_STRING,"vbl_wait:  bit clear after %d counts",ctr);
//advise(ERROR_STRING);
//}
#endif /* QUIP_DEBUG */

	if( ctr >= MAX_WAITS ){
		sprintf(ERROR_STRING,
"vbl_wait:  polled MAX_WAITS (0x%x) times waiting for VBLANK bit to clear",
			MAX_WAITS);
		warn(ERROR_STRING);
	}

	ctr=0;
	while( (regval & 0x8) == 0 && ctr < MAX_WAITS ){
		regval = inb(0x3da);
		ctr++;
#ifdef QUIP_DEBUG
//if( debug & xdebug ){
//sprintf(ERROR_STRING,"vbl_wait:  reg 0x3da = 0x%x   (ctr = %d)",regval,ctr);
//advise(ERROR_STRING);
//}
#endif /* QUIP_DEBUG */
	}

#ifdef QUIP_DEBUG
if( debug & xdebug ){
sprintf(ERROR_STRING,"vbl_wait:  bit set after %d counts",ctr);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	if( ctr >= MAX_WAITS ){
		sprintf(ERROR_STRING,
"vbl_wait:  polled MAX_WAITS (0x%x) times waiting for VBLANK bit to set",
			MAX_WAITS);
		warn(ERROR_STRING);
	}
}

//#endif /* HAVE_VBL */

#endif /* HAVE_X11 */
