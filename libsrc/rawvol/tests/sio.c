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

#include    <stdio.h>
#include    <unistd.h>	/* getopt */
#include    <stdlib.h>	/* random */
//#include    <varargs.h>
#include    <fcntl.h>
#include    <errno.h>
#include    <string.h>	// memcpy
#include    <sys/types.h>
#include    <sys/time.h>
#include    <sys/param.h>
#include    <sys/stat.h>
//#include    <sys/signal.h>
#define __USE_GNU
#include    <signal.h>

#include    <unistd.h>
/* #include    <aio.h> */
#include    <math.h>	/* random() */

#define MAXASYNC    4

void scat();
extern void *memalign();

struct timeval tp1;
struct timeval tp2;
double et;

#define BUFSIZE	100
char abuf[BUFSIZE];
unsigned long where;
/* what is this bsize?  size in blocks!?  */
long blks_per_chunk = 4096;

struct stat sb;
unsigned long    total = 0;


extern int	optind;
extern char *	optarg;

static char error_string[128];

static int			c;
static int			fd, fd2, ipfd, opfd;
static char		*buf, *obuf;
static char	*	fname = NULL;
static char	*	fname2 = NULL;
static unsigned long		file_blocks = 0;
static int		repeats = 1;
#define IS_READING		((oflags)==O_RDONLY)
#define IS_WRITING		(oflags & (O_RDWR|O_WRONLY|O_CREAT))
static int		oflags;
static int		oflags2;
static int		is_random = 0;
static int		interact = 0;
static int		verbose = 0;		/* why two verbose's ?? */
static int		Verbose = 0;
static int		timeit = 0;
static int		offset = 0;
static int		backwards = 0;
static int		writing = 0;
static int		create = 0;
static int		memory = 0;
static int		wilborn_wynn = 0;
static int		copy = 0;
static int		pipesync = 0;
static int		sigsio = 0;
static unsigned long	nchunks;		/* number of I/O chunks ? */
static unsigned long	align = 0;
static unsigned long	alignmask = 0;
static char	*	inpipe = NULL;
static char	*	outpipe = NULL;
/* static unsigned long	before, after; */
static int		i;
static int		ret;

void set_where()
{
	if (backwards) {
		where = file_blocks - blks_per_chunk;
		lseek(fd, where * 512, 0);
	} else if (offset) {
		where = offset;
		lseek(fd, where * 512, 0);
	}
	else
		where = 0;
}

void printwhere()
{
	fprintf(stderr, "block %d\n", where);
}

void io_failure(op,filename)
char *op, *filename;
{
	perror("");
	fprintf(stderr,"%s failure on %s at ", op, fname);
	printwhere();
	fprintf(stderr, "Request 0x%lx return 0x%x\n", blks_per_chunk, ret);
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
	if (verbose) printwhere();

	if( IS_READING ) {
		lseek(fd, where * 512, 0);
		ret = read(fd, buf, blks_per_chunk*512) / 512;
		if (ret != blks_per_chunk)
			io_failure("read",fname);
	} else {
		lseek(fd, where * 512, 0);
		ret = write(fd, buf, blks_per_chunk*512) / 512;
		if (ret != blks_per_chunk)
			io_failure("write",fname);
	}
	total += blks_per_chunk;
}

void writit(wfd,name)
int wfd; char *name;
{
	ret = write(wfd, buf, blks_per_chunk*512) / 512;
	if (ret != blks_per_chunk)
		io_failure("write",name);
}

void chunk_io()
{
	if ( IS_READING || copy) {
		ret = read(fd, buf, blks_per_chunk*512);
		if( ret == 0 )
			warn("EOF?");
		ret /= 512;
		if (ret != blks_per_chunk)
			io_failure("read",fname);
	}
	if ( ! IS_READING )
		writit(fd,fname);
	if (copy)
		writit(fd2,fname2);

	total += blks_per_chunk;
	if (backwards) {
		where -= blks_per_chunk;
		lseek(fd, where * 512, 0);
	} else
		where += blks_per_chunk;
}

void parse_args(argc,argv)
int argc; char **argv;
{
	/*
	 * Parse the arguments.
	 */
	while ((c = getopt(argc, argv, "cdDgimMrStvVwWzA:a:e:b:f:n:o:p:P:s:")) != EOF) {
	switch (c) {
	case 'c': create = 1;							break;
	case 'd': copy = 1;							break;
	case 'e': alignmask = (unsigned long)strtol(optarg, (char **) 0, 0);		break;
	case 'g': signal(SIGUSR1, sigcatch); sigsio = 1;			break;
	case 'i': interact = 1;							break;
	case 'm': memory = 1;							break;
	case 'r': is_random = 1; srandom(time(0) + getpid());			break;
	case 't': timeit = 1;							break;
	case 'v': verbose = 1;							break;
	case 'V': Verbose = 1;							break;
	case 'w': writing = 1;							break;
	case 'W': wilborn_wynn = 1;						break;
	case 'z': backwards = 1;						break;
	case 'a': align = (unsigned long)strtol(optarg, (char **) 0, 0);		break;
	case 'b': if ((blks_per_chunk = strtol(optarg, (char **) 0, 0)) <= 0)
								usage();	break;
	case 'f': fname = optarg;						break;
	case 'n': if ((repeats = strtol(optarg, (char **) 0, 0)) <= 0) usage(); break;
	case 'o': if ((offset = strtol(optarg, (char **) 0, 0)) < 0) usage();	break;
	case 'p': inpipe = optarg;						break;
	case 'P': outpipe = optarg;						break;
	case 's': if ((file_blocks = strtol(optarg, (char **) 0, 0)) <= 0)
								usage();	break;
	}
	}
}

void check_args(argc,argv)
int argc; char **argv;
{
	if (alignmask) {
		if ( (alignmask + 1) & alignmask) {
			fprintf(stderr, "alignmask + 1 not a power of 2\n");
			usage();
		}
		if (alignmask < align) {
			fprintf(stderr, "alignmask (0x%x) < align (0x%x)\n",
				alignmask, align);
			usage();
		}
	}

	if (Verbose) {
		if (is_random) printf("random");
		else printf("sequential");

		if (create) printf(" create,");
		else if (writing) printf(" write,");
		else printf(" read,");

		printf(" %ld total blocks,", blks_per_chunk);

		if (offset) printf(" offset %d blocks,", offset);

		if (is_random)
			printf(" %d reps from %d blocks\n", repeats, file_blocks);
		else
			printf(" total %d blocks\n", file_blocks);
	}

	if (fname) {
		if (copy) {
			fprintf(stderr,"old-style -f arg incompatible with copy\n");
			exit(1);
		}
		if (inpipe || outpipe) {
			fprintf(stderr, "old-style -f arg incompatible with pipesync\n");
			exit(1);
		}
	} else if (!memory) {
		argc -= optind;
		if (argc <= 0) usage();
		fname = argv[optind];
		if (copy) {
			if (argc < 2) {
				fprintf(stderr,"Must specify destination file for copy.\n");
				exit(1);
			}
			fname2 = argv[optind + 1];
		}
	}

	if (inpipe || outpipe) {
		if (!inpipe || !outpipe) {
			fprintf(stderr,"Both sync pipes must be specified\n");
			exit(1);
		}
		if ((ipfd = open(inpipe, O_RDONLY)) < 0) {
			fprintf(stderr,"Can't open %s\n",inpipe);
			exit(1);
		}
		if ((opfd = open(outpipe, O_WRONLY)) < 0) {
			fprintf(stderr,"Can't open %s\n",outpipe);
			exit(1);
		}
		pipesync = 1;
		if (interact) {
			fprintf(stderr,"Interactive mode not useful with pipesync.\n");
			exit(1);
		}
		if (verbose) {
			fprintf(stderr,"Verbose mode not useful with pipesync.\n");
			exit(1);
		}
	}

	if (offset && (create || copy || is_random)) {
		fprintf(stderr,"Offset allowed only on sequential read\n");
		exit(1);
	}

	if (memory) file_blocks = blks_per_chunk;

	if (!file_blocks) {
		if (create && !copy) {
			fprintf(stderr,"Must specify filesize to create.\n");
			exit(1);
		}
		timeit = 1;	/* reasonable default */
	}

	if (!memory && !(create && !copy)) {
		if (stat(fname, &sb) < 0) {
			fprintf(stderr,"Can't stat %s\n", fname);
			exit(1);
		}
		if ((sb.st_mode & S_IFMT) != S_IFREG) {
			if (!file_blocks) {
				fprintf(stderr, "must specify size if not"
					" regular file.\n");
				exit(1);
			}
		} else /* regular file */ {
			if (file_blocks == 0) {
				file_blocks = sb.st_size / 512;
				timeit = 1;
			} else if (file_blocks * 512 > sb.st_size) {
				fprintf(stderr,
	"specified size (%d blocks) is greater than actual file size (%ld)!\n",
					file_blocks,sb.st_size);
				exit(1);
			}
		}
	}

	if (file_blocks < blks_per_chunk) {
		if (offset)
			fprintf(stderr,"effective ");
		fprintf(stderr,"filesize %d less than block count %ld\n",
			file_blocks, blks_per_chunk);
		exit(1);
	}

	nchunks = file_blocks / blks_per_chunk;

	if (file_blocks % blks_per_chunk) {
		file_blocks = nchunks * blks_per_chunk;
		fprintf(stderr,
	"Warning: filesize adjusted to %d blocks to align to %ld block boundary\n",
			file_blocks, blks_per_chunk);
	}

	if (is_random && (create || copy)) {
		fprintf(stderr,"Random allowed only on read or write\n");
		exit(1);
	}

	if (copy && writing) {
		fprintf(stderr,"Incompatible options: copy & write!\n");
		exit(1);
	}

	if (backwards && (is_random || offset || writing || create || copy)) {
		fprintf(stderr,"Backward seek allowed only on sequential read\n");
		exit(1);
	}

	if ((repeats > 1) && !(is_random || memory)) {
		fprintf(stderr, "Repeats only allowed with random or memory\n");
		exit(1);
	}
}

void get_buffers()
{
	unsigned long bufsize;
	int alignment;

if( alignmask != 0 )
warn("alignmask is not zero");

	bufsize=(blks_per_chunk*512)+alignmask;
	alignment= align?align:4;

	if (  (buf = memalign(alignment,bufsize) ) == 0) {
		perror("");
		error1("memalign failed!?\n");
	}

	if (memory) {
		if( (obuf = memalign(alignment,bufsize))
			== 0) {
			perror("");
			error1("memalign failed!?\n");
		}
		/* why do this copy? */
		/*
		memcpy(obuf, buf, bufsize);
		*/
		return;
	}

	if (align && alignmask) {
		if (((unsigned long)buf & alignmask) != align)
			buf = (char *)( ((unsigned long)buf + alignmask + 1)
				& (~alignmask | align ) );
		printf("buf=0x%lx\n", (unsigned long)buf);
	}

}

void open_files()
{

	if (copy) {
		oflags = O_RDONLY;
		oflags2 = O_WRONLY;
		if (create)
			oflags2 |= (O_CREAT | O_TRUNC);
	} else {
		if (create)
			oflags = O_WRONLY | O_CREAT | O_TRUNC;

		else if (writing)
			oflags = O_WRONLY;
		else
			oflags = O_RDONLY;
	}


	get_buffers();

	if (interact) {
		printf(
"fname %s, %d repeats, %d file_blocks, %ld blks_per_chunk, %d chunks, %d oflag, %x buf\n",
			fname, repeats, file_blocks, blks_per_chunk,
			nchunks, oflags, (unsigned long)buf);
		if (is_random) printf("random\n");
		else printf("sequential\n");

		printf("Open file %s for ", fname);
		if( IS_WRITING )
			printf("writing? ");
		else printf("reading? ");
		fflush(stdout);
		//gets(abuf);
		if( fgets(abuf,BUFSIZE,stdin) == NULL )
			error1("error reading input text");
		if (*abuf != 'y') exit(0);
		if (copy) {
			if (create) printf("Create destination file %s? ",fname2);
			else printf("Open destination file %s? ",fname2);

			fflush(stdout);
			//gets(abuf);
			if( fgets(abuf,BUFSIZE,stdin) == NULL )
				error1("error reading input text");
			if (*abuf != 'y') exit(0);
		}
	}

	fd = open(fname, oflags, 0644);
	if(fd < 0) {
		perror("");
		fprintf(stderr,"Can't open %s\n",fname);
		exit(1);
	}

	if (copy ){
		fd2 = open(fname2, oflags2, 0644);
		if( fd2 < 0 ) {
			perror("");
			fprintf(stderr,"Can't open %s\n",fname2);
			exit(1);
		}
	}
}

void get_started()
{
	if (interact) {
		printf("Begin ");
		if ( IS_WRITING )
			printf("writing ");
		else
			printf("reading ");
		printf("file %s? ",fname);
		fflush(stdout);
		//gets(abuf);
		if( fgets(abuf,BUFSIZE,stdin) == NULL )
			error1("error reading input text");
		if (*abuf != 'y') exit(0);
	}

	if (pipesync) {
		sleep(1);
		if (write(opfd, abuf, 1) != 1) {
			fprintf(stderr,"opfd %d errno %d fname %s\n", 
			opfd, errno, fname);
			perror("");
			error1("Error writing to sync pipe\n");
		}
		if (read(ipfd, abuf, 1) != 1) {
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


	if( !memory )
		get_started();

	if (timeit) gettimeofday(&tp1, NULL);

	if (memory) {
		while(repeats--) {
			memcpy(obuf, buf, blks_per_chunk*512);
			total += blks_per_chunk;
		}
	} else if (is_random) {
		while (repeats-- && !sigstop) {
			where = (siorand() % nchunks) * blks_per_chunk;
			random_io();
		}
	} else {	/* sequential */
		set_where();
		for (i = 0; (i < nchunks) && (sigstop == 0); i++) {
			if (verbose) printwhere();
			chunk_io();
		}
	}

	if (timeit) gettimeofday(&tp2, NULL);

	if (pipesync & !sigsio) {
		sleep(2);
		if (write(opfd, abuf, 1) != 1)
			error1("Error writing to sync pipe\n");
		if (read(ipfd, abuf, 1) != 1)
			error1("Error reading from sync pipe\n");
	}

	if (timeit) {
		et = (((double) tp2.tv_usec / 1000000) + tp2.tv_sec) -
			(((double) tp1.tv_usec / 1000000) + tp1.tv_sec);
		if (wilborn_wynn)
			printf("%.2f %.1f\n", ((double) total/1000000) / et,
			((double) total / blks_per_chunk) / et);
		else {
			printf("%.2f s  %s	%u blocks", et, fname, total);
			printf("  %.0f KB/s (%.2f MB/s)  IO/s %.2f\n",
				((double) total / 2) / et,
				((double) total / 1953.1) / et,
				((double) total / blks_per_chunk) / et);
		}
	}
	if (sigsio) {
		et = ((double) total / 1953.1) / et;
		write(opfd, &et, sizeof(et));
	}

	exit(0);
}

void scat()
{
	fprintf(stderr,"SIGPIPE!\n");
}

