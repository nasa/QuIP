#include "quip_config.h"

#ifdef HAVE_PARPORT

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>	/* close */
#endif

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

#ifdef HAVE_LINUX_PPDEV_H
#include <linux/ppdev.h>
#endif

#include "quip_prot.h"
#include "my_parport.h"
//#include "debug.h"

ITEM_INTERFACE_DECLARATIONS(ParPort,parport)

static char *default_parport="/dev/parport0";

ParPort * open_parport(QSP_ARG_DECL  const char *name)
{
	ParPort *ppp;

	if( name == NULL || *name == 0 ){
		name = default_parport;
		if( verbose ){
			sprintf(ERROR_STRING,"open_parport:  using default parallel port %s",name);
			advise(ERROR_STRING);
		}
	}
	ppp = parport_of(QSP_ARG  name);
	if( ppp != NO_PARPORT ){
		sprintf(ERROR_STRING,"ParPort %s is already open",name);
		WARN(ERROR_STRING);
		return NO_PARPORT;
	}

	ppp = new_parport(QSP_ARG  name);
	if( ppp == NO_PARPORT ) {
		sprintf(ERROR_STRING,"Unable to create new parallel port structure %s",name);
		WARN(ERROR_STRING);
		return NO_PARPORT;
	}

	ppp->pp_fd = open(ppp->pp_name,O_RDWR);
	if( ppp->pp_fd < 0 ){
		perror("open");
		WARN("error opening parallel port device");
		return NO_PARPORT;
	}
	if( ioctl(ppp->pp_fd,PPCLAIM,NULL) < 0 ){
		perror("ioctl");
		WARN("error claiming parallel port device");
		close(ppp->pp_fd);
		del_parport(QSP_ARG  ppp);
		return NO_PARPORT;
	}
	return ppp;
}

int read_parport_status(ParPort *ppp)
{
	unsigned char b;

	if( ioctl(ppp->pp_fd,PPRSTATUS,&b) < 0 ){
		perror("ioctl");
		NWARN("error reading parport status");
		return -1;
	}
	return (int) b;
}

/* This function is useful for when we use the parallel port to monitor a pulse generator,
 * such as a TTL video sync signal.
 */

int read_til_transition(ParPort *ppp, int mask)
{
	int original_value, current_value;

	original_value = read_parport_status(ppp) & mask;

	do {
		current_value = read_parport_status(ppp) & mask;
	} while( current_value == original_value );

	return(current_value);
}

#endif /* HAVE_PARPORT */
