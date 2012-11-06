/* emulate the gps3 on a serial port.
 *
 * This can be used to test the recording software by looping
 * one serial port to the other using a null modem cable.
 */

#include "quip_config.h"

char VersionId_gps_gps_emu[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>		/* usleep */
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#include "myerror.h"		/* warn */

#define PACK(n)		(((n)/10)<<4) | ((n)%10)

int main(int ac, char **av)
{
	int n_f=0, n_s=0, n_m=5, n_h=1;
	char packet[5];
	int fd;
	int n;
	unsigned char d=0;

	/*
	fd=open("/dev/ttyS1",O_WRONLY);
	if( fd < 0 ){
		perror("open");
		exit(1);
	}
	*/
	fd = fileno(stdout);

	while(1){
		packet[0] = 0xf7;
		packet[1] = PACK(n_f);
		packet[2] = PACK(n_s);
		packet[3] = PACK(n_m);
		packet[4] = PACK(n_h);

		n_f ++;
		if( n_f >= 30 ){
			n_f = 0;
			n_s++;
			if( n_s >= 60 ){
				n_s = 0;
				n_m++;
				if( n_m >= 60 ){
					n_m = 0;
					n_h ++;
					if( n_h >=24 )
						n_h=0;
				}
			}
		}
		if( (n=write(fd,packet,5)) != 5 ){
			if( n < 0 ){
				perror("write");
				exit(1);
			}
			sprintf(error_string,"5 chars requested, %d written",n);
			warn(error_string);
			exit(1);
		}
		packet[0]=0xff;
		packet[1]=d++;
		packet[2]=d++;
		packet[3]=d++;
		packet[4]=d++;
		if( (n=write(fd,packet,5)) != 5 ){
			if( n < 0 ){
				perror("write");
				exit(1);
			}
			sprintf(error_string,"5 chars requested, %d written",n);
			warn(error_string);
			exit(1);
		}
		usleep(33000);		/* 33 msec */
	}
}

