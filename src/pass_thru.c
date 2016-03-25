/* serial port pass-through program
 *
 * We use this so that we can plug the polhemus fastrak into a linux box (dirac),
 * and have the output mirrored to the cdti pc next door.
 */

#include "quip_config.h"

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#ifdef HAVE_SYS_FILE_H
#include <sys/file.h>
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#include <stdio.h>
#include <errno.h>

#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif


#include "ttyctl.h"

static pthread_t pass_thru_thr[2];
static int busy=1;

static int open_serial_port(char *s)
{
	int fd;

	fd=open(s,O_RDWR);
	if( fd < 0 ){
		sprintf(error_string,"error opening tty file \"%s\"",s);
		warn(error_string);
		return(fd);
	}

#ifdef HAVE_FLOCK
	if( flock(fd,LOCK_EX|LOCK_NB) < 0 ){
		if( errno == EWOULDBLOCK ){
			sprintf(error_string,"Unable to obtain exclusive lock on tty file %s",s);
			warn(error_string);
			advise("Make sure the port is not in use by another process");
		} else {
			perror("flock");
			sprintf(error_string,"unable to get exclusive lock on serial device %s",s);
			warn(error_string);
		}
		return(-1);
	}
#else
	warn("open_serial_port:  unable to lock for exclusive use (no flock on this system).");
#endif

	return(fd);
}

static void *pass_thru(void *argp)
{
	int *intp;
	int fd_in, fd_out;
	int n;
#define MAX_CHARS	256
	char char_buf[MAX_CHARS];
	char err_str[LLEN];
#ifdef PT_DEBUG
	static int debug=1;
	int serial=1;
#endif /* PT_DEBUG */

	intp = argp;

	fd_in = intp[0];
	fd_out = intp[1];

	do {
		if( (n=keyhit(fd_in)) > 0 ){
			if( n > MAX_CHARS ) n=MAX_CHARS;
			if( read(fd_in,char_buf,n) != n ){
				sprintf(err_str,"Error reading from fd %d",fd_in);
				warn(err_str);
			} else {
				if( write(fd_out,char_buf,n) != n ){
					sprintf(err_str,"Error writing to fd %d", fd_out);
					warn(err_str);
				}
#ifdef PT_DEBUG
				else if( debug ){
					sprintf(err_str,"%5d:  %d chars written to fd %d", serial++,n, fd_out);
					advise(err_str);
				}
#endif /* PT_DEBUG */
			}
		} else if( n == 0 ){
			/* 115000 baud is on the order of 10k chars/sec... */
			usleep(50);
		} else {
			sprintf(err_str,"key_hit returned %d for fd %d",
				n,fd_in);
			warn(err_str);
			return(NULL);
		}
	} while(1);
	return(NULL);
}

int main(int argc, char **av)
{
	int fd1, fd2;
	pthread_attr_t attr1;
	int fd_pair1[2];
	int fd_pair2[2];

	fd1 = open_serial_port("/dev/ttyS14");
	if( fd1 < 0 ) error1("error opening /dev/ttyS14");

	fd2 = open_serial_port("/dev/ttyS15");
	if( fd2 < 0 ) error1("error opening /dev/ttyS15");

	echooff(fd1);
	echooff(fd2);
	ttyraw(fd1);
	ttyraw(fd2);

	//setbaud(fd1,115);
	//setbaud(fd2,115);

	/* fork a couple of threads:
	 * one reads chars from fd1 and writes to fd2,
	 * the other reads from fd2 and writes to fd1.
	 */

	pthread_attr_init(&attr1);	/* initialize to default values */
	pthread_attr_setinheritsched(&attr1,PTHREAD_INHERIT_SCHED);

	fd_pair1[0]=fd1;
	fd_pair1[1]=fd2;
	pthread_create(&pass_thru_thr[0],&attr1,pass_thru,fd_pair1);
	fd_pair2[0]=fd2;
	fd_pair2[1]=fd1;
	pthread_create(&pass_thru_thr[1],&attr1,pass_thru,fd_pair2);

	/* what now?   exit or wait? */
	busy=1;
	while(busy) sleep(1);

	return(0);
}

