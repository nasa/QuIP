/* stamps.c   read a stream of characters and time-stamp each one */

#include "quip_config.h"

#include "quip_prot.h"
#include "quip_menu.h"
#include "debug.h"

#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>		/* strlen() */
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#ifdef HAVE_TIME_H
#include <time.h>		// ctime
#endif

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>		/* FIONREAD */
#endif

#include "stamps.h"
#include "list.h"

List *sc_list=NULL;

static struct timeval zero_tv;
static int zero_set=0;

static int input_fd=(-1);
static int output_fd=(-1);
static int stamp_async=0;
static int halting=0;
static int running=0;
static long n_logged=0;

#define DEFAULT_MAX_PER_LINE	5		/* suitable for GPS3 dumps! */

/* this was originally written using dynamic memory allocation, but then I realized hat
 * getbuf() is not thread-safe!?
 */

#define N_SWAP_BUFFERS	2

#define MAX_TIME_STAMPS	512
static Stamped_Char sc_tbl[N_SWAP_BUFFERS][MAX_TIME_STAMPS];
static int next_sc_to_read=0;

/* TMP_BUF_SIZE needs to be larger than MAX_TIME_STAMPS times the average number
 * of chars per packet; for the GPS3, we expect 5 chars, but sometimes get 10, 15 and
 * even 20 chars at once.  Here we have used a factor of 8, which seems to be ok.
 */

#define TMP_BUF_SIZE	4096
static char ts_char_buf[N_SWAP_BUFFERS][TMP_BUF_SIZE];
static int ts_char_offset[N_SWAP_BUFFERS]={0,0};

static int active_buf=0;
static int writing_buf=(-1);

typedef struct char_reader_args {
	Query_Stack *	cra_qsp;
} Char_Reader_Args;

static void read_stamp(SINGLE_QSP_ARG_DECL)
{
	Stamped_Char *scp;
	int n=0;

	if( next_sc_to_read >= MAX_TIME_STAMPS ){
		warn("read_stamp:  out of time stamp buffer space");
		halting=1;
		return;
	}

	while(n<=0){
#ifdef FIONREAD
		if( ioctl(input_fd,FIONREAD,&n) < 0 ){
			perror("ioctl FIONREAD");
			return;
		}
#else // ! FIONREAD
		n=1;
#endif // ! FIONREAD
		
		if( n <= 0 ) usleep(200);
		if( halting ) return;
	}

	/* At this point we know there are n chars available (n positive) */

	scp = &sc_tbl[active_buf][next_sc_to_read];
	if( gettimeofday(&scp->sc_tv,NULL) < 0 ){
		perror("gettimeofday");
		warn("error reading system time");
		return;
	}
	scp->sc_n = n;
	if( ts_char_offset[active_buf]+n > TMP_BUF_SIZE ){
		warn("out of char buf space");
		halting=1;
		scp->sc_n=0;
		return;
	}

	scp->sc_buf = &ts_char_buf[active_buf][ts_char_offset[active_buf]];
	if( read(input_fd,scp->sc_buf,n) != n ){
		sprintf(ERROR_STRING,"Error reading %d chars",n);
		warn(ERROR_STRING);
		return;
	}

	ts_char_offset[active_buf] += n;
	next_sc_to_read++;

	if( next_sc_to_read >= MAX_TIME_STAMPS ){	/* this buf is full */
		if( writing_buf >= 0 )
			error1("read_stamp:  disk writer not keeping up!?");
		writing_buf = active_buf;
		active_buf ++;
		active_buf %= N_SWAP_BUFFERS;
		next_sc_to_read=0;
		ts_char_offset[active_buf]=0;
	}
}

static char fmt_buf[512];

#define HEX_CHAR(n)	((char)( n<10 ? '0' + n : 'a' + n - 10 ))

static char *hex_string( Stamped_Char *scp, int offset, int n )
{
	int i;
	int nibble;

	char *s=fmt_buf;	// BUG check for overflow

	for(i=0;i<n;i++){
		*s++ = ' ';
		*s++ = '0';
		*s++ = 'x';
		nibble = (((scp->sc_buf[offset+i])&0xf0)>>4);
		*s++ = HEX_CHAR(nibble);
		nibble = ((scp->sc_buf[offset+i])&0xf);
		*s++ = HEX_CHAR(nibble);
	}
	*s=0;
	return(fmt_buf);
}

static void show_one(QSP_ARG_DECL  Stamped_Char *scp, int max_per_line)
{
	int i=0;
	long sec, msec;

	if( zero_set ){
		sec = scp->sc_tv.tv_sec - zero_tv.tv_sec;
		msec = (long) floor((scp->sc_tv.tv_usec - zero_tv.tv_usec)/1000.0);
	} else {
		sec = scp->sc_tv.tv_sec ;
		msec = (long) floor(scp->sc_tv.tv_usec/1000);
	}

	do {
		int n_to_print;
		char *s;

		/* format the time */
		s=ctime(&scp->sc_tv.tv_sec);
		s[ strlen(s)-1 ] = 0;
		prt_msg_frag(s);

		sprintf(msg_str,"\t%ld\t%ld",sec,msec);
		prt_msg_frag(msg_str);

		n_to_print = scp->sc_n - i > max_per_line ? max_per_line : scp->sc_n - i;
		sprintf(msg_str,"\t%s",hex_string(scp,i,n_to_print));
		prt_msg(msg_str);
		i += n_to_print;
	} while( i < scp->sc_n );
}

static COMMAND_FUNC( show_stamps )
{
	Node *np;
	Stamped_Char *scp;

	if( sc_list == NULL ) return;

	np = QLIST_HEAD(sc_list);
	while(np!=NULL){
		scp=(Stamped_Char *)np->n_data;
		show_one(QSP_ARG  scp,DEFAULT_MAX_PER_LINE);
		np=np->n_next;
	}
}

static COMMAND_FUNC( set_zero )
{
	if( gettimeofday(&zero_tv,NULL) < 0 ){
		perror("gettimeofday");
		warn("error reading time zero");
		return;
	}
	zero_set=1;
}

#ifdef HAVE_PTHREADS

static pthread_t stamp_thr;
static pthread_t writer_thr;
static int char_reader_done=0;
static Char_Reader_Args cr_args;	// need to be global?

static void *char_reader(void *argp)
{
	int i;
	Query_Stack *qsp;
	Char_Reader_Args *cra_p;

	running=1;
	halting=0;
	char_reader_done=0;
	active_buf=0;
	writing_buf=(-1);
	cra_p = argp;
	qsp = cra_p->cra_qsp;

	next_sc_to_read = 0;
	ts_char_offset[active_buf] = 0;

	// BUG is there a race problem, locking issue here??
	while( ! halting )
		read_stamp(SINGLE_QSP_ARG);

	/* A halt has been received, make sure we flush the chars here now... */
	for(i=next_sc_to_read;i<MAX_TIME_STAMPS;i++)
		sc_tbl[active_buf][i].sc_n = 0;

	while( writing_buf >= 0 )
		usleep(100000);

	writing_buf = active_buf;
	/* need to yield the processor here */
	usleep(200000);		/* make sure there is time for the writer to flush */
	char_reader_done=1;

	return(NULL);
}

/* How often will we need to flush to disk?
 * at 9600 baud, we receive about 1k/sec, which is about 200 chunks?
 * Because we have allocated 512 timestamps per buf, we expect that this
 * will be activated every couple of seconds, so we should be able to easily
 * sleep for 100 msec...
 */

static void *disk_writer(void *argp)
{
	int count;
	Query_Stack *qsp;

	qsp = argp;

	while( !char_reader_done ){
		while( writing_buf < 0 )
			usleep(100000);	/* sleep 100 msec */
		/* first write the table of timestamps */
		count= sizeof(sc_tbl[0][0])*MAX_TIME_STAMPS;
		if( write(output_fd,(char *)&sc_tbl[writing_buf][0],count) != count ){
			tell_sys_error("write");
			warn("error writing timestamp data");
		}
		/* now write the number of characters (variable) */
		count = sizeof(ts_char_offset[writing_buf]);
		if( write(output_fd,(char *)&ts_char_offset[writing_buf],count) != count ){
			tell_sys_error("write");
			warn("error writing char count");
		}
		count = ts_char_offset[writing_buf];
if( verbose ){
sprintf(ERROR_STRING,"writing buffer %d, %d chars",writing_buf,count);
advise(ERROR_STRING);
}
		if( write(output_fd,&ts_char_buf[writing_buf],count) != count ){
			tell_sys_error("write");
			warn("error writing char buffer");
		}

		/* now we're done! */
		writing_buf=(-1);
	}
	return(NULL);
}

#endif // HAVE_PTHREADS

static void decode_file(QSP_ARG_DECL  int fd, int max_per_line)
{
	int reading_file=1;
	int count;
	ssize_t actual;
	Stamped_Char stmp_tbl[MAX_TIME_STAMPS];
	int char_offset;
	static char char_buf[TMP_BUF_SIZE];
	int i,j;
int ntot=0;

	while( reading_file ){
		/* first read the table of timestamps */
		count= sizeof(stmp_tbl[0])*MAX_TIME_STAMPS;
		if( (actual=read(fd,(char *)&stmp_tbl[0],count)) < 0 ){
			tell_sys_error("read");
			warn("error reading timestamp data");
sprintf(ERROR_STRING,"Read error after reading %d total characters",ntot);
advise(ERROR_STRING);
			return;
		} else if( actual == 0 ){
			/* this happens at EOF, we should stat the file first (BUG)
			 * and compare the count to be sure we are really done.
			 * For the time being we just assume all is OK.
			 */
			return;
		} else if( actual != count ){
			sprintf(ERROR_STRING,"Requested %d timestamp chars, %zd actually read",
				count,actual);
			warn(ERROR_STRING);
sprintf(ERROR_STRING,"Read error after reading %d total characters",ntot);
advise(ERROR_STRING);
			return;
		}
ntot+=count;
		/* now read the number of characters (variable) */
		count = sizeof(char_offset);
		if( (actual=read(fd,(char *)&char_offset,count)) < 0 ){
			tell_sys_error("read");
			warn("error reading char count");
sprintf(ERROR_STRING,"Read error after reading %d total characters",ntot);
advise(ERROR_STRING);
			return;
		} else if( actual != count ){
			sprintf(ERROR_STRING,"Requested %d charcount chars, %zd actually read",
				count,actual);
			warn(ERROR_STRING);
sprintf(ERROR_STRING,"Read error after reading %d total characters",ntot);
advise(ERROR_STRING);
			return;
		}
ntot+=count;

		count = char_offset;

		if( (actual=read(fd,&char_buf,count)) < 0 ){
			tell_sys_error("read");
			warn("error reading char buffer");
sprintf(ERROR_STRING,"Read error after reading %d total characters",ntot);
advise(ERROR_STRING);
			return;
		} else if( actual != count ){
			sprintf(ERROR_STRING,"Requested %d data chars, %zd actually read",
				count,actual);
			warn(ERROR_STRING);
sprintf(ERROR_STRING,"Read error after reading %d total characters",ntot);
advise(ERROR_STRING);
			return;
		}
ntot+=count;
		/* Now print this chunk of data */

		i=0;
		j=0;
		while(i<MAX_TIME_STAMPS && stmp_tbl[i].sc_n > 0 ){

			stmp_tbl[i].sc_buf = &char_buf[j];

			///* print # chars */
			//sprintf(msg_str,"\t%d\t",stmp_tbl[i].sc_n);
			//prt_msg_frag(msg_str);

			//s=hex_string(&stmp_tbl[i],n);
			//prt_msg(s);

			show_one(QSP_ARG  &stmp_tbl[i],DEFAULT_MAX_PER_LINE);

			j += stmp_tbl[i].sc_n;
			i++;
		}
	}
}

static COMMAND_FUNC( rd_stamps )
{
	const char *s;

	s=NAMEOF("input filename");
	input_fd = open( s , O_RDONLY );
	if( input_fd < 0 ){
		sprintf(ERROR_STRING,"open(%s)",s);
		tell_sys_error(ERROR_STRING);
	}

	s=NAMEOF("output filename");
	output_fd = open( s , O_WRONLY | O_CREAT | O_TRUNC, 0644 );
	if( output_fd < 0 ){
		sprintf(ERROR_STRING,"open(%s)",s);
		tell_sys_error(ERROR_STRING);
	}

	if( input_fd < 0 || output_fd < 0 ) return;

#ifdef HAVE_PTHREADS
	if( stamp_async ){
		pthread_attr_t attr1;
		pthread_attr_init(&attr1);	/* initialize to default values */
		pthread_attr_setinheritsched(&attr1,PTHREAD_INHERIT_SCHED);
		// BUG what about without thread-safe-query?
		cr_args.cra_qsp = THIS_QSP;
		pthread_create(&stamp_thr,&attr1,char_reader,&cr_args);
		pthread_create(&writer_thr,&attr1,disk_writer,THIS_QSP);
	} else {
		read_stamp(SINGLE_QSP_ARG);
	}
#else // ! HAVE_PTHREADS
	read_stamp(SINGLE_QSP_ARG);
#endif // ! HAVE_PTHREADS
}

static COMMAND_FUNC( do_set_async )
{
	stamp_async = ASKIF("log character data asynchronously");
#ifdef HAVE_PTHREADS
	warn("No support for async timestamps without pthreads!?");
#endif // HAVE_PTHREADS
}

static COMMAND_FUNC( do_stamp_halt )
{
	halting=1;
	usleep(400000);	/* give the threads a chance to finish up */
	/* now call pthread_join? */
}

static COMMAND_FUNC( do_status )
{
	if( ! running ) advise("No async char acquisition in progress");
	else {
		sprintf(ERROR_STRING,"Async char acquisition in progress, %ld chars logged",n_logged);
		advise(ERROR_STRING);
	}
}

static COMMAND_FUNC( do_decode )
{
	int fd;
	const char *s;
	int n;

	s=NAMEOF("timestamped char filename");
	n=(int)HOW_MANY("max. chars per line");

	fd = open( s , O_RDONLY );
	if( fd < 0 ){
		sprintf(ERROR_STRING,"open(%s)",s);
		tell_sys_error(ERROR_STRING);
		return;
	}
	decode_file(QSP_ARG  fd,n);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(stamps_menu,s,f,h)

MENU_BEGIN(stamps)
ADD_CMD( zero,		set_zero,	set time zero )
ADD_CMD( read,		rd_stamps,	read available characters )
ADD_CMD( async,		do_set_async,	set/clear async flag )
ADD_CMD( halt,		do_stamp_halt,	halt async acquisition )
ADD_CMD( status,	do_status,	report status of async acquisition )
ADD_CMD( show,		show_stamps,	display timestamped chars )
ADD_CMD( decode,	do_decode,	decode stored stream file )
MENU_END(stamps)

COMMAND_FUNC( do_stamp_menu )
{
	CHECK_AND_PUSH_MENU(stamps);
}


