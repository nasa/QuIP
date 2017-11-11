#include "quip_config.h"

#ifdef HAVE_PARPORT

/* test program for parallel port using /dev/parport0
 * This access used to work in the minolta program, but doesn't seem
 * to be working properly now...
 */


/* test program for parallel port
 *
 * This program accesses the hardware directly with inb and outb,
 * must be suid root for ioperm to succeed.
 */

/* #include <asm/io.h> */	/* outb inb */

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

#ifdef HAVE_LINUX_PPDEV_H
#include <linux/ppdev.h>
#endif

#include "quip_prot.h"
#include "data_obj.h"		/* pick_obj */

static int pport_fd=(-1);

/* These defines let us compile on a non-linux system... */
#ifndef PPCLAIM
#define PPCLAIM 0
#endif
#ifndef PPRSTATUS
#define PPRSTATUS 0
#endif

static COMMAND_FUNC(  do_open_pport )
{
	const char *s;

	s=NAMEOF("device name");

	pport_fd = open(s,O_RDWR);
	if( pport_fd < 0 ){
		perror("open");
		WARN("error opening parallel port device");
		return;
	}
	if( ioctl(pport_fd,PPCLAIM,NULL) < 0 ){
		perror("ioctl");
		WARN("error claiming parallel port device");
		return;
	}
}

static int rd_byte(void)
{
	unsigned char b;
	/*
	int n;
	*/

	if( pport_fd < 0 ){
		NWARN("no parallel port device open");
		return(-1);
	}

	/*
	if( (n=read(pport_fd,&b,1)) != 1 ){
		if( n < 0 ){
			perror("read");
			NWARN("error reading byte from parallel port");
			return;
		}
	}
	*/
	if( ioctl(pport_fd,/* PPRDATA */ PPRSTATUS,&b) < 0 ){
		perror("ioctl");
		NWARN("error reading data");
		return(-1);
	}
	//sprintf(msg_str,"%d (0x%x)",b,b);
	//prt_msg(msg_str);

	return( b );
}

static COMMAND_FUNC( do_rd_byte )
{
	const char *s;
	int val;
	char val_string[16];

	s=NAMEOF("variable name");
	val = rd_byte();
	sprintf(val_string,"0x%x",val);
	assign_var(QSP_ARG  s,val_string);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(parport_menu,s,f,h)

MENU_BEGIN(parport)
ADD_CMD( open,	do_open_pport,	open parallel port device )
ADD_CMD( read,	do_rd_byte,	read a byte from parport status lines )
MENU_END(parport)

COMMAND_FUNC( do_parport_menu )
{
	CHECK_AND_PUSH_MENU(parport);
}

#endif /* HAVE_PARPORT */

