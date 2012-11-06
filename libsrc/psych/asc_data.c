#include "quip_config.h"

char VersionId_psych_asciidata[] = QUIP_VERSION_STRING;

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

#include "debug.h"		/* verbose */

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include "stc.h"

/* these includes were added for xvalue transformations... */
#include "savestr.h"
#include "getbuf.h"



/* printf/scanf format strings */
static const char *topline="%d classes, %d x values\n";
static const char *xvline="\t%g\n";
static const char *classheader="Trial_Class %d, %d data points\n";
static const char *pointline="\t%d\t%d\t%d\n";

static const char *summline="Summary data\n";
static const char *dribline="Raw data\n";

static char input_line[128];
static int have_input_line;

static const char *dm_arg_str=NULL;

static int rd_summ(QSP_ARG_DECL  int n, FILE *fp);
static int rd_top(QSP_ARG_DECL  FILE *fp);
static int rd_dribble(QSP_ARG_DECL  FILE *fp);
static void xform_xvals(SINGLE_QSP_ARG_DECL);

static int next_input_line(FILE *fp)
{
	if( fgets(input_line,LLEN,fp) == NULL ){
		have_input_line=0;
		return(-1);
	}
	have_input_line=1;
	return(0);
}

void markdrib(FILE *fp)
{
	fputs(dribline,fp);
	fflush(fp);
}

void wtdata(QSP_ARG_DECL  FILE *fp)	/* replaces routine formerly in stair.c */
{
	List *lp;
	Node *np;
	Trial_Class *clp;

	/* new ascii format */

	wt_top(QSP_ARG  fp);
	fputs(summline,fp);

	lp = class_list(SINGLE_QSP_ARG);
	np=lp->l_head;
	while(np!=NO_NODE){
		clp=(Trial_Class *)np->n_data;
		wtclass(clp,fp);
		np=np->n_next;
	}
	fflush(fp);
}

void wtclass(Trial_Class *clp,FILE *fp)
{
	int j;

	clp->cl_dtp->d_npts=0;
	for(j=0;j<MAXVALS;j++)
		if( clp->cl_dtp->d_data[j].ntotal != 0 )
			clp->cl_dtp->d_npts++;
	fprintf(fp,classheader,clp->cl_index,clp->cl_dtp->d_npts);
	for(j=0;j<MAXVALS;j++)
		if( clp->cl_dtp->d_data[j].ntotal != 0 )
			fprintf(fp,pointline, j,
				clp->cl_dtp->d_data[j].ntotal,
				clp->cl_dtp->d_data[j].ncorr );
}

/* Read the bottom half of a data file.
 * This may be either a dribble file or a summary file.
 */

static int rd_bot(QSP_ARG_DECL  FILE *fp,int n_classes)
{
	char tstr[32];

	/* determine data format type */
	/* fscanf(fp,"%s data\n",tstr); */
	if( next_input_line(fp) < 0 ){
		WARN("rd_bot:  premature eof on input file");
		return(-1);
	}
	if( sscanf(input_line,"%s data\n",tstr) != 1 ){
		sprintf(error_string,"rddata:  unexpected data description:  %s",tstr);
		WARN(error_string);
		return(-1);
	}
	if( !strcmp(tstr,"Summary") ){
		if( verbose ) advise("Reading data in summary format");
		return( rd_summ(QSP_ARG  n_classes,fp) );
	} else if( !strcmp(tstr,"Raw") ){
		if( verbose ) advise("Reading data in raw format");
		return( rd_dribble(QSP_ARG  fp) );
	} else {
		sprintf(error_string,"bizarre data format:  \"%s\"\n",tstr);
		WARN(error_string);
		return(-1);
	}
	/* NOTREACHED */
}

int rddata(QSP_ARG_DECL  FILE *fp)
{
	int n;
	int status;

	have_input_line=0;

	while( (n=rd_top(QSP_ARG  fp)) >= 0 ){
		status=rd_bot(QSP_ARG  fp,n);
		if( status < 0 ) return(status);
		if( ! have_input_line ) return(0);
	}
	return(-1);
}

static int rd_summ(QSP_ARG_DECL  int n,FILE *fp)		/** read data in summary format */
{
	while(n--){
		if( rd_one_summ(QSP_ARG  fp) < 0 )
			return(-1);
	}
	return(0);
}

int rd_one_summ(QSP_ARG_DECL  FILE *fp)
{
	int index,np;
	Trial_Class *clp;
	int j;

	if( fscanf(fp,classheader,&index,&np) != 2 ){
		WARN("error reading class header");
		return(-1);
	}
	clp=index_class(QSP_ARG  index);
#ifdef CAUTIOUS
	if( clp == NO_CLASS ){
		WARN("CAUTIOUS:  missing class");
		return(-1);
	}
#endif

	clp->cl_dtp->d_npts=np;
	for(j=0;j<clp->cl_dtp->d_npts;j++){
		int di,nt,nc;

		if( fscanf(fp,pointline,&di,&nt,&nc) != 3 ){
			WARN("error reading data line");
			return(-1);
		} else {
			clp->cl_dtp->d_data[di].ntotal=nt;
			clp->cl_dtp->d_data[di].ncorr = nc;
		}
	}
	return(0);
}

/* this is separate so we can include it at the top of dribble files */

void wt_top(QSP_ARG_DECL  FILE *fp)
{
	int i;
	int nclasses;

	nclasses = eltcount( class_list(SINGLE_QSP_ARG) );

	fprintf(fp,topline,nclasses,_nvals);
	for(i=0;i<_nvals;i++)
		fprintf(fp,xvline,xval_array[i]);
	fflush(fp);
}

void setup_classes(QSP_ARG_DECL  int n)
{
	int i;
	char name[32];

	/* We don't delete old classes if they already exist;
	 * This allows us to read multiple files if they are concatenated...
	 */
	for(i=0;i<n;i++){
		Trial_Class *clp;

		sprintf(name,"class%d",i);
		clp = trial_class_of(QSP_ARG  name);
		if( clp == NO_CLASS ){
			if(verbose){
				sprintf(error_string,"setup_classes:  class %s not found, creating a new class",
						name);
				advise(error_string);
			}
			new_class(SINGLE_QSP_ARG);
		}
	}
}

/* Read the top half of a data file.  This tells us the number of x values,
 * and the number of classes.
 * We return the number of classes, or -1 on error.
 */

static int rd_top(QSP_ARG_DECL  FILE *fp)
{
	int i;
	int n;		/* number of classes */

	if( ! have_input_line ){
		/* The first time through we will need to read
		 * a line of input here.  If we have multiple
		 * files concatenated, then we will have a buffered
		 * line and can skip this step.
		 */
		if( next_input_line(fp) < 0 ){
			WARN("rd_top:  no top line");
			return(-1);
		}
	}

	if( sscanf(input_line,topline,&n,&_nvals) != 2 ){
		sprintf(error_string,"rd_top:  bad top line:  %s",input_line);
		WARN(error_string);
		return(-1);
	}

	/* Make sure that there are at least n classes.
	 * If they already exist, that is ok, we can read in
	 * concatenated data files and lump the data that way.
	 */

	setup_classes(QSP_ARG  n);

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

void wt_dribble(FILE *fp,Trial_Class *clp,int index,int val,int rsp,int crct)
{
	fprintf(fp,"%d\t%d\t%d\t%d\t%d\n",clp->cl_index,index,val,rsp,crct);
	fflush(fp);
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
			Trial_Class *clp;
			clp=index_class(QSP_ARG  i_class);
			note_trial(clp,i_val,resp,crct);
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

void set_xval_xform(const char *s)
{
	if( dm_arg_str != NULL ) givbuf(dm_arg_str);
	dm_arg_str = savestr(s);
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
		sprintf(error_string,"xform_xvals:  unable to close temp file %s",tmpfilename);
		NWARN(error_string);
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
		sprintf(error_string,
			"giving up after %d retries to read file %s",
			MAX_RETRIES,tmpfilename);
		NWARN(error_string);
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



