// jbm:  Not sure where this program came from originally, but I've been
// using it for years to benchmark disk write speed...
// Use direct I/O!!!

/*
 *  simple diskio test
 *
 *  -c - create file, truncating if preexists.
 *  -d - "duplicate": copy file1 to file2.
 *  -D - use Direct I/O
 *  -g - catch signal SIGUSR1 and terminate
 *  -i - interactive (asks before open or r/w)
 *  -m - just copy memory to memory
 *  -M - pin down the memory
 *  -r - random (instead of sequential)
 *  -t - prints summary on termination.
 *  -v - verbose: prints blocknumber on each I/O
 *	-V - verbose: prints parameters of test
 *  -w - write instead of read
 *  -z - read file backwards
 *
 *  -a - align the transfers on a given memory boundary
 *  -b - blocksize
 *  -e - align to exactly the -a arg & this mask
 *  -f - filename (backward compatibility)
 *  -n - repeats
 *  -o - starting offset in 512-byte blocks.
 *  -p - name of synchronizing pipe from master.
 *  -P - name of synchronizing pipe to master.
 *  -s - file size in 512-byte blocks
 */

#define BYTES_PER_BLOCK	1024	// 512 on some systems - should get from include file!!!

#define __USE_GNU	// must come before fcntl.h and signal.h!

#include    <stdio.h>
#include    <unistd.h>	/* getopt */
#include    <stdlib.h>	/* random */
#include    <malloc.h>	/* memalign */
//#include    <varargs.h>
#define __USE_GNU	// must come before fcntl.h and signal.h!
#include    <fcntl.h>
#include    <time.h>
#include    <errno.h>
#include    <string.h>	// memcpy
#include    <sys/types.h>
#include    <sys/time.h>
#include    <sys/param.h>
#include    <sys/stat.h>
//#include    <sys/signal.h>
#include    <signal.h>

#include    <unistd.h>
/* #include    <aio.h> */
#include    <math.h>	/* random() */

#define MAXASYNC    4

void scat();
//extern void *memalign();

struct timeval tp1;
struct timeval tp2;
double elapsed_time;

#define BUFSIZE	100
char tmp_str[BUFSIZE];
unsigned long where;
/* what is this bsize?  size in blocks!?  */
long blks_per_chunk = 4096;

struct stat sb;
unsigned long    total = 0;


extern int	optind;
extern char *	optarg;

static char error_string[128];

static int			fd, fd2, ipfd, opfd;
static char		*io_buffer, *obuf;
static char	*	fname = NULL;
static char	*	fname2 = NULL;
static unsigned long		file_blocks = 0;
static long		repeats = 1;
#define IS_READ_TEST		((oflags)==O_RDONLY)
#define IS_WRITE_TEST		(oflags & (O_RDWR|O_WRONLY|O_CREAT))
static int		oflags;
static int		oflags2;

static int		offset = 0;
static unsigned long	nchunks;		/* number of I/O chunks ? */
static unsigned long	align = 0;
static unsigned long	alignmask = 0;
static char	*	inpipe = NULL;
static char	*	outpipe = NULL;
/* static unsigned long	before, after; */
static int		i;
//static int		ret;

unsigned long program_flags=0;
#define COPY_DATA		1
#define CREAT_FILE		2
#define INTERACTIVE_MODE	4
#define MEMORY_COPY		8
#define WRITE_TEST		16
#define PIPE_SYNC		32
#define RANDOM			64
#define WILBORN_WYNN		128	// what is this???
#define BACKWARDS		256
#define TIMING			512
#define USING_SIGNAL		1024
#define VERBOSE1		2048
#define VERBOSE2		4096
#define USE_ODIRECT		8192

#define IS_COPYING	(program_flags&COPY_DATA)
#define IS_CREATING	(program_flags&CREAT_FILE)
#define IS_INTERACTIVE	(program_flags&INTERACTIVE_MODE)
#define USING_MEMORY	(program_flags&MEMORY_COPY)
#define IS_WRITING	(program_flags&WRITE_TEST)
#define IS_PIPING	(program_flags&PIPE_SYNC)
#define IS_RANDOM	(program_flags&RANDOM)
#define IS_WILBORN_WYNN	(program_flags&WILBORN_WYNN)
#define IS_BACKWARDS	(program_flags&BACKWARDS)
#define IS_TIMING	(program_flags&TIMING)
#define IS_USING_SIGNAL	(program_flags&USING_SIGNAL)
#define IS_VERBOSE	(program_flags&VERBOSE1)
#define IS_VERY_VERBOSE	(program_flags&VERBOSE2)
#define IS_USING_ODIRECT	(program_flags&USE_ODIRECT)

void set_where()
{
	if( IS_BACKWARDS ){
		where = file_blocks - blks_per_chunk;
		lseek(fd, where * BYTES_PER_BLOCK, 0);
	} else if( offset ){
		where = offset;
		lseek(fd, where * BYTES_PER_BLOCK, 0);
	}
	else
		where = 0;
}

void printwhere()
{
	fprintf(stderr, "block %ld\n", where);
}

void io_failure(op,filename)
char *op, *filename;
{
	perror("");
	fprintf(stderr,"%s failure on %s at ", op, fname);
	printwhere();
	//fprintf(stderr, "Request 0x%lx return 0x%x\n", blks_per_chunk, ret);
	exit(1);
}

void error1(msg)
char *msg;
{
	fprintf(stderr,"ERROR:  %s\n",msg);
	fflush(stderr);
	exit(1);
}

void advise(msg)
char *msg;
{
	fprintf(stderr,"%s\n",msg);
	fflush(stderr);
}

void warn(msg)
char *msg;
{
	fprintf(stderr,"WARNING:  %s\n",msg);
	fflush(stderr);
}

void usage()
{
	fprintf(stderr,
"usage: sio [-cdDgimMrStvVwz] [-a align] [-A aiocnt] [-b blocksize in blocks]\n"
"		[-e alignmask] [-n repeats] [-o start offset in blocks]\n"
"		[-P sio_mgr_pipe -p mgr_sio_pipe] [-s total number of blocks]\n"
"		filename [copyfilename]\n");
	exit(1);
}

int siorand()
{
	return (random());
}


int sigstop = 0;

void
sigcatch(/*int sig, int code, void *sc*/ int arg)
{
	sigstop = 1;
}


void random_io()
{
	ssize_t n;

	if( IS_VERBOSE ) printwhere();

	if( IS_READ_TEST ){
		lseek(fd, where * BYTES_PER_BLOCK, 0);
		n = read(fd, io_buffer, blks_per_chunk*BYTES_PER_BLOCK) / BYTES_PER_BLOCK;
		if( n != blks_per_chunk)
			io_failure("read",fname);
	} else {
		lseek(fd, where * BYTES_PER_BLOCK, 0);
		n = write(fd, io_buffer, blks_per_chunk*BYTES_PER_BLOCK) / BYTES_PER_BLOCK;
		if( n != blks_per_chunk)
			io_failure("write",fname);
	}
	total += blks_per_chunk;
}

void writit(wfd,name)
int wfd; char *name;
{
	ssize_t n;

	n = write(wfd, io_buffer, blks_per_chunk*BYTES_PER_BLOCK) / BYTES_PER_BLOCK;
	if( n != blks_per_chunk)
		io_failure("write",name);
}

void chunk_io()
{
	ssize_t n;

	if( IS_READ_TEST || IS_COPYING ){
		n = read(fd, io_buffer, blks_per_chunk*BYTES_PER_BLOCK);
		if( n == 0 )
			warn("EOF?");
		n /= BYTES_PER_BLOCK;
		if( n != blks_per_chunk)
			io_failure("read",fname);
	}
	if( IS_WRITE_TEST )
		writit(fd,fname);
	if( IS_COPYING)
		writit(fd2,fname2);

	total += blks_per_chunk;
	if( IS_BACKWARDS ){
		where -= blks_per_chunk;
		lseek(fd, where * BYTES_PER_BLOCK, 0);
	} else
		where += blks_per_chunk;
}

static int next_option(int argc, char **argv)
{
	int c;
	c = getopt(argc, argv, "cdDgimMrStvVwWzA:a:e:b:f:n:o:p:P:s:");
//fprintf(stderr,"next_option:  getopt returned 0x%x\n",c);

	return c;
}

static void parse_opt(int c)
{
	switch( c ){
		case 'c': program_flags |= CREAT_FILE;		break;
		case 'd': program_flags |= COPY_DATA;		break;
		case 'D': program_flags |= USE_ODIRECT;		break;
		case 'e':
	alignmask = (unsigned long)strtol(optarg, (char **) 0, 0);
						break;
		case 'g': signal(SIGUSR1, sigcatch);
				program_flags |= USING_SIGNAL;	break;
		case 'i': program_flags |= INTERACTIVE_MODE;	break;
		case 'm': program_flags |= MEMORY_COPY;		break;
		case 'r': program_flags |= RANDOM;
			srandom(time(0) + getpid());		break;
		case 't': program_flags |= TIMING;		break;
		case 'v': program_flags |= VERBOSE1;		break;
		case 'V': program_flags |= VERBOSE2;		break;
		case 'w': program_flags |= WRITE_TEST;		break;
		case 'W': program_flags |= WILBORN_WYNN;	break;
		case 'z': program_flags |= BACKWARDS;		break;
		case 'a':
	align = (unsigned long)strtol(optarg, (char **) 0, 0);
						break;
		case 'b':
	if( (blks_per_chunk = strtol(optarg, (char **) 0, 0)) <= 0) usage();
						break;
		case 'f': fname = optarg;	break;
		case 'n':
	if( (repeats = strtol(optarg, (char **) 0, 0)) <= 0) usage();
						break;
		case 'o':
	if( (offset = strtol(optarg, (char **) 0, 0)) < 0) usage();
						break;
		case 'p': inpipe = optarg;	break;
		case 'P': outpipe = optarg;	break;
		case 's':
	if( (file_blocks = strtol(optarg, (char **) 0, 0)) <= 0) usage();
						break;
		default:
			fprintf(stderr,"Unrecognized flag option '%c'\n",c);
			error1("bad option specification");
			break;
	}
}

void parse_args(argc,argv)
int argc; char **argv;
{
	int c;
	/*
	 * Parse the arguments.
	 */
	while( (c=next_option(argc,argv)) > 0 ){
//fprintf(stderr,"next_option returned 0x%x\n",c);
		parse_opt(c);
	}
fprintf(stderr,"parse_args:  align = 0x%lx\n",align);

}

void check_args(argc,argv)
int argc; char **argv;
{
	if( IS_USING_ODIRECT ){
		if( (alignmask+1) < BYTES_PER_BLOCK && align < BYTES_PER_BLOCK ){
			fprintf(stderr,"alignment must be to at least a block boundary with O_DIRECT\n");
			usage();
		}
	}
	if( alignmask ){
		if( (alignmask + 1) & alignmask ){
			fprintf(stderr, "alignmask + 1 not a power of 2\n");
			usage();
		}
		if( alignmask < align ){
			fprintf(stderr, "alignmask (0x%lx) < align (0x%lx)\n",
				alignmask, align);
			usage();
		}
	}

	if( IS_VERY_VERBOSE ){
		if( IS_RANDOM ) printf("random");
		else printf("sequential");

		if( IS_CREATING) printf(" create,");
		else if( IS_WRITING ) printf(" write,");
		else printf(" read,");

		printf(" %ld total blocks,", blks_per_chunk);

		if( offset) printf(" offset %d blocks,", offset);

		if( IS_RANDOM )
			printf(" %ld reps from %ld blocks\n", repeats, file_blocks);
		else
			printf(" total %ld blocks\n", file_blocks);
	}

	if( fname ){
		if( IS_COPYING ){
			fprintf(stderr,"old-style -f arg incompatible with copy\n");
			exit(1);
		}
		if( inpipe || outpipe ){
			fprintf(stderr, "old-style -f arg incompatible with pipesync\n");
			exit(1);
		}
	} else if( ! USING_MEMORY ){
		argc -= optind;
		if( argc <= 0) usage();
		fname = argv[optind];
		if( IS_COPYING ){
			if( argc < 2 ){
				fprintf(stderr,"Must specify destination file for copy.\n");
				exit(1);
			}
			fname2 = argv[optind + 1];
		}
	}

	if( inpipe || outpipe ){
		if( !inpipe || !outpipe ){
			fprintf(stderr,"Both sync pipes must be specified\n");
			exit(1);
		}
		if( (ipfd = open(inpipe, O_RDONLY)) < 0 ){
			fprintf(stderr,"Can't open %s\n",inpipe);
			exit(1);
		}
		if( (opfd = open(outpipe, O_WRONLY)) < 0 ){
			fprintf(stderr,"Can't open %s\n",outpipe);
			exit(1);
		}
		program_flags |= PIPE_SYNC;
		if( IS_INTERACTIVE ){
			fprintf(stderr,"Interactive mode not useful with pipesync.\n");
			exit(1);
		}
		if( IS_VERBOSE ){
			fprintf(stderr,"Verbose mode not useful with pipesync.\n");
			exit(1);
		}
	}

	if( offset && (IS_CREATING || IS_COPYING || IS_RANDOM )){
		fprintf(stderr,"Offset allowed only on sequential read\n");
		exit(1);
	}

	if( USING_MEMORY ) file_blocks = blks_per_chunk;

	if( !file_blocks ){
		if( IS_CREATING && !IS_COPYING ){
			fprintf(stderr,"Must specify filesize to create.\n");
			exit(1);
		}
		program_flags |= TIMING; /* reasonable default */
	}

	if( ! USING_MEMORY && !(IS_CREATING && !IS_COPYING) ){
		if( stat(fname, &sb) < 0 ){
			fprintf(stderr,"Can't stat %s\n", fname);
			exit(1);
		}
		if( (sb.st_mode & S_IFMT) != S_IFREG ){
			if( !file_blocks ){
				fprintf(stderr, "must specify size if not"
					" regular file.\n");
				exit(1);
			}
		} else /* regular file */ {
			if( file_blocks == 0 ){
				file_blocks = sb.st_size / BYTES_PER_BLOCK;
				program_flags |= TIMING;
			} else if( file_blocks * BYTES_PER_BLOCK > sb.st_size ){
				fprintf(stderr,
	"specified size (%ld blocks) is greater than actual file size (%ld)!\n",
					file_blocks,sb.st_size);
				exit(1);
			}
		}
	}

	if( file_blocks < blks_per_chunk ){
		if( offset)
			fprintf(stderr,"effective ");
		fprintf(stderr,"filesize %ld less than block count %ld\n",
			file_blocks, blks_per_chunk);
		exit(1);
	}

	nchunks = file_blocks / blks_per_chunk;

	if( file_blocks % blks_per_chunk ){
		file_blocks = nchunks * blks_per_chunk;
		fprintf(stderr,
	"Warning: filesize adjusted to %ld blocks to align to %ld block boundary\n",
			file_blocks, blks_per_chunk);
	}

	if( IS_RANDOM  && (IS_CREATING || IS_COPYING) ){
		fprintf(stderr,"Random allowed only on read or write\n");
		exit(1);
	}

	if( IS_COPYING && IS_WRITING ){
		fprintf(stderr,"Incompatible options: copy & write!\n");
		exit(1);
	}

	if( IS_BACKWARDS && (IS_RANDOM  || offset || IS_WRITING  || IS_CREATING || IS_COPYING) ){
		fprintf(stderr,"Backward seek allowed only on sequential read\n");
		exit(1);
	}

	if( (repeats > 1) && !(IS_RANDOM  || USING_MEMORY) ){
		fprintf(stderr, "Repeats only allowed with random or memory\n");
		exit(1);
	}
}

void get_buffers()
{
	unsigned long bufsize;
	int alignment;

//if( alignmask != 0 )
//warn("alignmask is not zero");

	bufsize = (blks_per_chunk*BYTES_PER_BLOCK)+alignmask;
	alignment = align?align:4;

fprintf(stderr,"get_buffers:  alignment = 0x%x\n",alignment);
	if( (io_buffer = memalign(alignment,bufsize) ) == 0 ){
		perror("");
		error1("memalign failed!?\n");
	}

	if( USING_MEMORY ){
		if( (obuf = memalign(alignment,bufsize))
			== 0 ){
			perror("");
			error1("memalign failed!?\n");
		}
		/* why do this copy? */
		/*
		memcpy(obuf, buf, bufsize);
		*/
		return;
	}

	if( align && alignmask ){
		if( ((unsigned long)io_buffer & alignmask) != align)
			io_buffer = (char *)( ((unsigned long)io_buffer + alignmask + 1)
				& (~alignmask | align ) );
		printf("buf=0x%lx\n", (unsigned long)io_buffer);
	}

fprintf(stderr,"io_buffer at 0x%lx\n",(long)io_buffer);
}

static void get_stuff_from_user(void)
{
	printf(
"fname %s, %ld repeats, %ld file_blocks, %ld blks_per_chunk, %ld chunks, %d oflag, %lx buf\n",
		fname, repeats, file_blocks, blks_per_chunk,
		nchunks, oflags, (unsigned long)io_buffer);
	if( IS_RANDOM ) printf("random\n");
	else printf("sequential\n");

	printf("Open file %s for ", fname);
	if( IS_WRITING )
		printf("writing? ");
	else printf("reading? ");
	fflush(stdout);
	if( fgets(tmp_str,BUFSIZE,stdin) == NULL )
		error1("error reading input text");
	if( *tmp_str != 'y') exit(0);
	if( IS_COPYING ){
		if( IS_CREATING) printf("Create destination file %s? ",fname2);
		else printf("Open destination file %s? ",fname2);

		fflush(stdout);
		if( fgets(tmp_str,BUFSIZE,stdin) == NULL )
			error1("error reading input text");
		if( *tmp_str != 'y') exit(0);
	}
}

void open_files()
{

	if( IS_COPYING ){
		oflags = O_RDONLY;
		oflags2 = O_WRONLY;
		if( IS_CREATING)
			oflags2 |= (O_CREAT | O_TRUNC);
	} else {
fprintf(stderr,"open_files:  NOT copying\n");
		if( IS_CREATING ){
fprintf(stderr,"open_files:  creating\n");
			oflags = O_WRONLY | O_CREAT | O_TRUNC;

		} else if( IS_WRITING ){
fprintf(stderr,"open_files:  writing\n");
			oflags = O_WRONLY;
		} else {
fprintf(stderr,"open_files:  reading\n");
			oflags = O_RDONLY;
		}
	}
	if( IS_USING_ODIRECT ){
fprintf(stderr,"using O_DIRECT!!!\n");
		oflags |= O_DIRECT;
		oflags2 |= O_DIRECT;
	}


	get_buffers();

	if( IS_INTERACTIVE ){
		get_stuff_from_user();
	}

fprintf(stderr,"opening %s, flags = 0x%x\n",fname,oflags);
	fd = open(fname, oflags, 0644);
	if(fd < 0 ){
		perror("");
		fprintf(stderr,"Can't open %s\n",fname);
		exit(1);
	}

	if( IS_COPYING ){
		fd2 = open(fname2, oflags2, 0644);
		if( fd2 < 0 ){
			perror("");
			fprintf(stderr,"Can't open %s\n",fname2);
			exit(1);
		}
	}
}

void get_started()
{
	if( IS_INTERACTIVE ){
		printf("Begin ");
		if( IS_WRITING )
			printf("writing ");
		else
			printf("reading ");
		printf("file %s? ",fname);
		fflush(stdout);
		if( fgets(tmp_str,BUFSIZE,stdin) == NULL )
			error1("error reading input text");
		if( *tmp_str != 'y') exit(0);
	}

	if( IS_PIPING ){
		sleep(1);
		if( write(opfd, tmp_str, 1) != 1 ){
			fprintf(stderr,"opfd %d errno %d fname %s\n", 
			opfd, errno, fname);
			perror("");
			error1("Error writing to sync pipe\n");
		}
		if( read(ipfd, tmp_str, 1) != 1 ){
			error1("Error reading from sync pipe\n");
		}
	}
}

void
main(argc, argv)
int	argc;
char		**argv;
{
	/* XXX*/
	signal(SIGPIPE, scat);

	parse_args(argc,argv);
	check_args(argc,argv);
	open_files();


	if( ! USING_MEMORY )
		get_started();

	if( IS_TIMING ) gettimeofday(&tp1, NULL);

	if( USING_MEMORY ){
		while(repeats--){
			memcpy(obuf, io_buffer, blks_per_chunk*BYTES_PER_BLOCK);
			total += blks_per_chunk;
		}
	} else if( IS_RANDOM ){
		while (repeats-- && !sigstop ){
			where = (siorand() % nchunks) * blks_per_chunk;
			random_io();
		}
	} else {	/* sequential */
		set_where();
		for( i = 0; (i < nchunks) && (sigstop == 0); i++ ){
			if( IS_VERBOSE ) printwhere();
			chunk_io();
		}
	}

	if( IS_TIMING ) gettimeofday(&tp2, NULL);

	if( IS_PIPING  & ! IS_USING_SIGNAL ){
		sleep(2);
		if( write(opfd, tmp_str, 1) != 1)
			error1("Error writing to sync pipe\n");
		if( read(ipfd, tmp_str, 1) != 1)
			error1("Error reading from sync pipe\n");
	}

	if( IS_TIMING ){
		elapsed_time = (((double) tp2.tv_usec / 1000000) + tp2.tv_sec) -
			(((double) tp1.tv_usec / 1000000) + tp1.tv_sec);
		if( IS_WILBORN_WYNN )
			printf("%.2f %.1f\n", ((double) total/1000000) / elapsed_time,
			((double) total / blks_per_chunk) / elapsed_time);
		else {
			printf("%.2f s  %s	%lu blocks", elapsed_time, fname, total);
			printf("  %.0f KB/s (%.2f MB/s)  IO/s %.2f\n",
				((double) total * BYTES_PER_BLOCK / 1024) / elapsed_time,
				((double) total * BYTES_PER_BLOCK / (1024*1024)) / elapsed_time,
				((double) total / blks_per_chunk) / elapsed_time);
		}
	}
	if( IS_USING_SIGNAL ){
		elapsed_time = ((double) total / 1953.1) / elapsed_time;
		write(opfd, &elapsed_time, sizeof(elapsed_time));
	}

	exit(0);
}

void scat()
{
	fprintf(stderr,"SIGPIPE!\n");
}

