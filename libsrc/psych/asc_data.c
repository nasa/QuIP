#include "quip_config.h"

/*
 *	asciidata.c	routines to read and write experimental data
 *			in human readable form
 *			jbm 8-28-90
 */

#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>		/* mkstemp */
#endif

#include "quip_prot.h"		/* verbose */
#include "query_bits.h"		// LLEN - BUG

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include "stc.h"
#include "getbuf.h"
#include "list.h"


/* printf/scanf format strings */
static const char *topline="%d classes, %d x values\n";
static const char *xvline="\t%g\n";
static const char *class_header="Trial_Class %d, %d data points\n";
static const char *pointline="\t%d\t%d\t%d\n";

static const char *summline="Summary data\n";
static const char *dribline="Raw data\n";

static char input_line[LLEN];
static int have_input_line;

static const char *dm_arg_str=NULL;

FILE *drib_file=NULL;		/* trial by trial dribble */

//static int read_data_preamble(QSP_ARG_DECL  FILE *fp);
//static int rd_dribble(QSP_ARG_DECL  FILE *fp);
//static void xform_xvals(SINGLE_QSP_ARG_DECL);

static void rls_data_tbl(Trial_Class *tcp)
{
	Data_Tbl *dtp;

	dtp = CLASS_DATA_TBL(tcp);
	givbuf( DTBL_DATA(dtp) );
	givbuf(dtp);
	SET_CLASS_DATA_TBL(tcp,NULL);
}

Data_Tbl *alloc_data_tbl( Trial_Class *tcp, int size )
{
	Data_Tbl *dtp;

	assert( CLASS_DATA_TBL(tcp) == NULL );

	dtp = getbuf(sizeof(Data_Tbl));
	SET_CLASS_DATA_TBL(tcp,dtp);
	SET_DTBL_DATA(dtp, getbuf(size * sizeof(Datum) ) );
	SET_DTBL_SIZE(dtp,size);
	SET_DTBL_N(dtp,0);
	// BUG?  Should we zero the table here?

	return dtp;
}

static int next_input_line(FILE *fp)
{
	if( fgets(input_line,LLEN,fp) == NULL ){
		have_input_line=0;
		return(-1);
	}
	have_input_line=1;
	return(0);
}

void mark_drib(FILE *fp)
{
	fputs(dribline,fp);
	fflush(fp);
}

static void write_class_data(Trial_Class *tcp,FILE *fp)
{
	int j;
	Data_Tbl *dtp;

	//SET_DTBL_N(CLASS_DATA_TBL(tcp),0);

	dtp = CLASS_DATA_TBL(tcp);
	for(j=0;j<DTBL_SIZE(dtp);j++)
		if( DATUM_NTOTAL(DTBL_ENTRY(dtp,j)) != 0 )
			SET_DTBL_N(dtp,1+DTBL_N(dtp));
	fprintf(fp,class_header,CLASS_INDEX(tcp),DTBL_N(dtp));
	for(j=0;j<DTBL_SIZE(dtp);j++)
		if( DATUM_NTOTAL(DTBL_ENTRY(dtp,j)) != 0 )
			fprintf(fp,pointline, j,
				DATUM_NTOTAL(DTBL_ENTRY(dtp,j)),
				DATUM_NCORR(DTBL_ENTRY(dtp,j))
				);
}

/* this is separate so we can include it at the top of dribble files */
/* write header w/ #classes & xvalues */

static void write_data_preamble(QSP_ARG_DECL  FILE *fp)
{
	int i;
	int nclasses;

	assert(xval_array != NULL);

	nclasses = eltcount( class_list(SINGLE_QSP_ARG) );

	fprintf(fp,topline,nclasses,_nvals);
	for(i=0;i<_nvals;i++)
		fprintf(fp,xvline,xval_array[i]);
	fflush(fp);
}

void write_exp_data(QSP_ARG_DECL  FILE *fp)	/* replaces routine formerly in stair.c */
{
	List *lp;
	Node *np;
	Trial_Class *tcp;

	/* new ascii format */

	write_data_preamble(QSP_ARG  fp);
	fputs(summline,fp);

	lp = class_list(SINGLE_QSP_ARG);
	np=QLIST_HEAD(lp);
	while(np!=NULL){
		tcp=(Trial_Class *)np->n_data;
		write_class_data(tcp,fp);
		np=np->n_next;
	}
	fflush(fp);
}

static int read_class_summary(QSP_ARG_DECL  FILE *fp)
{
	short index,np;
	Trial_Class *tcp;
	Data_Tbl *dtp;
	int j;

	if( fscanf(fp,class_header,&index,&np) != 2 ){
		WARN("error reading class header");
		return(-1);
	}
	tcp=index_class(QSP_ARG  index);
	assert( tcp != NO_CLASS );

	dtp = CLASS_DATA_TBL(tcp);
	// BUG?  make sure that np <= size
	if( np > DTBL_SIZE(dtp) ){
		rls_data_tbl(tcp);
		dtp = alloc_data_tbl(tcp,np);
	}

	SET_DTBL_N(dtp,np);
	for(j=0;j<DTBL_N(dtp);j++){
		short di,nt,nc;

		if( fscanf(fp,pointline,&di,&nt,&nc) != 3 ){
			WARN("error reading data line");
			return(-1);
		} else {
			SET_DATUM_NTOTAL( DTBL_ENTRY(dtp,j), nt );
			SET_DATUM_NCORR( DTBL_ENTRY(dtp,j), nc );
		}
	}
	if( feof(fp) ) have_input_line=0;
	return(0);
}

static int read_class_summaries(QSP_ARG_DECL  int n_classes,FILE *fp)		/** read data in summary format */
{
	while(n_classes--){
		if( read_class_summary(QSP_ARG  fp) < 0 )
			return(-1);
	}
	// Now we should be at the end-of-file...
	return(0);
}

/* Read a dribble file (sequential list of trials).
 */
static int rd_dribble(QSP_ARG_DECL  FILE *fp)
{
	int i_class,i_val,resp,crct,i_stair;
	int n;

	while( next_input_line(fp) >= 0 ){
		n=sscanf(input_line,"%d\t%d\t%d\t%d\t%d\n",&i_class,&i_stair,&i_val,&resp,&crct);
		if( n == 5 ){
			Trial_Class *tcp;
			tcp=index_class(QSP_ARG  i_class);
			if( tcp == NULL ){
				fprintf(stderr,"rd_dribble:  didn't find class for index %d!?\n",i_class);
				return -1;
			}
			//note_trial(tcp,i_val,resp,crct);
			note_trial(tcp,i_val,resp,crct);
		} else {
			if( feof(fp) )
				have_input_line=0;
			/* We're not at the end of file, but this line is not part of the dribble data */
			/* Maybe it's the start of a concatenated file??? */
			return(0);
		}
	}
	/* what should n be?? */
	if( feof(fp) ){
		have_input_line=0;
		return(0);
	}
	return(-1);
}

/* Read the bottom half of a data file.
 * This may be either a dribble file or a summary file.
 */

static int read_class_data(QSP_ARG_DECL  FILE *fp,int n_classes)
{
	char tstr[32];

	/* determine data format type */
	/* fscanf(fp,"%s data\n",tstr); */
	if( next_input_line(fp) < 0 ){
		WARN("read_class_data:  premature eof on input file");
		return(-1);
	}
	if( sscanf(input_line,"%s data\n",tstr) != 1 ){
		sprintf(ERROR_STRING,"read_class_data:  unexpected data description:  %s",tstr);
		WARN(ERROR_STRING);
		return(-1);
	}
	if( !strcmp(tstr,"Summary") ){
		if( verbose ) advise("Reading data in summary format");
		return( read_class_summaries(QSP_ARG  n_classes,fp) );
	} else if( !strcmp(tstr,"Raw") ){
		if( verbose ) advise("Reading data in raw format");
		return( rd_dribble(QSP_ARG  fp) );
	} else {
		sprintf(ERROR_STRING,"bizarre data format:  \"%s\"\n",tstr);
		WARN(ERROR_STRING);
		return(-1);
	}
	/* NOTREACHED */
}

static void xform_xvals(SINGLE_QSP_ARG_DECL)
{
	char tmpfilename[80];
	char line[128];
	int i;
	FILE *fp;
	int fd;

	/*
	tmpfilename=tmpnam(NULL);
	if( tmpfilename == NULL ){
		WARN("error creating temporary name, NOT transforming");
		return;
	}
	*/
	strcpy(tmpfilename,"/tmp/dmdata_XXXXXX");
	fd = mkstemp(tmpfilename);
	if( fd < 0 ){
		NWARN("xform_xvals:  unable to create temporary file");
		return;
	}
	if( close(fd) < 0 ){
		perror("close");
		sprintf(ERROR_STRING,"xform_xvals:  unable to close temp file %s",tmpfilename);
		NWARN(ERROR_STRING);
		return;
	}
	sprintf(line,"dm %s > %s",dm_arg_str,tmpfilename);
	fp = popen(line,"w");
	if( !fp ){
		NWARN("error opening dm pipe");
		return;
	}
	for(i=0;i<_nvals;i++)
		fprintf(fp,"%g\n",xval_array[i]);
	fclose(fp);
	sleep(1);		/* wait for system to close file... */

#define MAX_RETRIES	3

	i=0;
again:
	if( i > MAX_RETRIES ){
		sprintf(ERROR_STRING,
			"giving up after %d retries to read file %s",
			MAX_RETRIES,tmpfilename);
		NWARN(ERROR_STRING);
		return;
	}
	fp=try_open(DEFAULT_QSP_ARG  tmpfilename,"r");
	if( !fp ) {
		advise("retrying");
		sleep(1);
		i++;
		goto again;
	}

	for(i=0;i<_nvals;i++)
		if( fscanf(fp,"%f",&xval_array[i]) != 1 )
			NWARN("error scanning transformed x value");
	fclose(fp);

	unlink(tmpfilename);
}

/* Read the top half of a data file.  This tells us the number of x values,
 * and the number of classes.
 * We return the number of classes, or -1 on error.
 */

static int read_data_preamble(QSP_ARG_DECL  FILE *fp)
{
	int i;
	int n;		/* number of classes */

	if( xval_array != NULL ){
		givbuf(xval_array);
		_nvals = 0;
	}

	if( ! have_input_line ){
		/* The first time through we will need to read
		 * a line of input here.  If we have multiple
		 * files concatenated, then we will have a buffered
		 * line and can skip this step.
		 */
		if( next_input_line(fp) < 0 ){
			WARN("read_data_preamble:  no top line");
			return(-1);
		}
	}

	if( sscanf(input_line,topline,&n,&_nvals) != 2 ){
		sprintf(ERROR_STRING,"read_data_preamble:  bad top line:  %s",input_line);
		WARN(ERROR_STRING);
		return(-1);
	}

	if( _nvals < 2 || _nvals > MAX_X_VALUES ){
		sprintf(ERROR_STRING,"read_data_preamble:  ridiculous number of x values (%d)!?",_nvals);
		WARN(ERROR_STRING);
		return(-1);
	}

	/* Make sure that there are at least n classes.
	 * If they already exist, that is ok, we can read in
	 * concatenated data files and lump the data that way.
	 */

	setup_classes(QSP_ARG  n);

	xval_array = (float *) getbuf( _nvals * sizeof(float) );
	/* Read the x values.  If we are reading concatenated files,
	 * we have to insist that all the xval arrays be the same.
	 * BUG we have to put in a check for this!
	 */
	for(i=0;i<_nvals;i++)
		if( fscanf(fp,xvline,&xval_array[i]) != 1 ){
			WARN("error reading an x value");
			return(-1);
		}

	if( dm_arg_str != NULL )
		xform_xvals(SINGLE_QSP_ARG);

	return(n);
}

int read_exp_data(QSP_ARG_DECL  FILE *fp)
{
	int n_classes;
	int status;

	have_input_line=0;

	if( (n_classes=read_data_preamble(QSP_ARG  fp)) < 0 ){
		WARN("Error in data file preamble!?");
		return -1;
	}
	status=read_class_data(QSP_ARG  fp,n_classes);
	if( status < 0 ) return(status);
	if( ! have_input_line ) return(0);
	return(-1);
}

void setup_classes(QSP_ARG_DECL  int n)
{
	int i;
	char name[32];

	/* We don't delete old classes if they already exist;
	 * This allows us to read multiple files if they are concatenated...
	 */
	for(i=0;i<n;i++){
		Trial_Class *tcp;

		sprintf(name,"class%d",i);
		tcp = trial_class_of(name);
		if( tcp == NO_CLASS ){
			/*
			if(verbose){
				sprintf(ERROR_STRING,"setup_classes:  class %s not found, creating a new class",
						name);
				advise(ERROR_STRING);
			}
			*/
			new_class(SINGLE_QSP_ARG);
		}
	}
}

void set_xval_xform(const char *s)
{
	if( dm_arg_str != NULL ) givbuf((void *)dm_arg_str);
	dm_arg_str = savestr(s);
}

void init_dribble_file(SINGLE_QSP_ARG_DECL)
{
	FILE *fp;

	// BUG - we should only keep trying if interactive!?
	while( (fp=TRYNICE(NAMEOF("dribble data file"),"w")) == NULL )
		;
	set_dribble_file(fp);
	write_data_preamble(QSP_ARG  fp);
	mark_drib(fp);		/* identify this as dribble data */
}

int dribbling(void)
{
	if( drib_file != NULL ) return 1;
	return 0;
}

void dribble( Staircase *stc_p, int rsp )
{
	assert( drib_file != NULL );

	fprintf(drib_file,"%d\t%d\t%d\t%d\t%d\n",
		CLASS_INDEX(STAIR_CLASS(stc_p)),
		STAIR_INDEX(stc_p),
		STAIR_VAL(stc_p),
		rsp,
		STAIR_CRCT_RSP(stc_p));

	fflush(drib_file);
}

void close_dribble(void)
{
	fclose(drib_file);
	drib_file=NULL;
	//CLEAR_EXP_FLAGS(DRIBBLING);
}

void set_dribble_file(FILE *fp) { drib_file=fp; }

