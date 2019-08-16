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
static const char *class_header_format_str="Trial_Class %d, %d data points\n";
static const char *pointline="\t%d\t%d\t%d\n";

static const char *summline="Summary data\n";
static const char *dribline="Raw data\n";

static char input_line[LLEN];
static int have_input_line;


FILE *drib_file=NULL;		/* trial by trial dribble */
Data_Obj *global_xval_dp=NULL;


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

#define write_class_summ_data(tc_p,_fp) _write_class_summ_data(QSP_ARG  tc_p,_fp)

static void _write_class_summ_data(QSP_ARG_DECL  Trial_Class *tc_p,void *_fp)
{
	FILE *fp;
	fp = (FILE *) _fp;
	assert(fp!=NULL);
	assert(CLASS_SUMM_DTBL(tc_p)!=NULL);
	write_summary_data( CLASS_SUMM_DTBL(tc_p), fp );
}

void _print_class_summary(QSP_ARG_DECL  Trial_Class * tcp)
{
	if( verbose ){
		sprintf(msg_str,"class = %s, %d points",
			CLASS_NAME(tcp),SUMM_DTBL_N(CLASS_SUMM_DTBL(tcp)));
		prt_msg(msg_str);
		sprintf(msg_str,"val\txval\t\tntot\tncorr\t%% corr\n");
		prt_msg(msg_str);
	}

	write_class_summ_data(tcp,tell_msgfile());
}

static void write_sequence_datum(Sequence_Datum *qd_p, FILE *fp)
{
	fprintf(fp,"%d\t%d\t%d\t%d\t%d\n",
		SEQ_DATUM_CLASS_IDX(qd_p),
		SEQ_DATUM_STAIR_IDX(qd_p),
		SEQ_DATUM_XVAL_IDX(qd_p),
		SEQ_DATUM_RESPONSE(qd_p),
		SEQ_DATUM_CRCT_RSP(qd_p) );

	fflush(fp);
}

#define print_sequence_datum(qd_p) _print_sequence_datum(QSP_ARG  qd_p)

static void _print_sequence_datum(QSP_ARG_DECL  Sequence_Datum *qd_p)
{
	write_sequence_datum(qd_p, tell_msgfile());
}

void _print_class_sequence(QSP_ARG_DECL  Trial_Class *tcp)
{
	List *lp;
	Node *np;

	assert( CLASS_SEQ_DTBL(tcp) != NULL );
	lp = SEQ_DTBL_LIST( CLASS_SEQ_DTBL(tcp) );
	assert(lp!=NULL);

	np = QLIST_HEAD(lp);
	while( np != NULL ){
		Sequence_Datum *qd_p;
		qd_p = (Sequence_Datum *) NODE_DATA(np);
		assert(qd_p!=NULL);
		print_sequence_datum(qd_p);
		np = NODE_NEXT(np);
	}
}

#ifdef FOOBAR
void print_class_summary(QSP_ARG_DECL  Trial_Class * tcp)
{
        int j;
        Summary_Data_Tbl *dtp;
	int n_xvals;

	assert( tcp != NULL );

	dtp=CLASS_SUMM_DTBL(tcp);
	assert( dtp != NULL );

	assert(CLASS_XVAL_OBJ(tcp)!=NULL);
	n_xvals = OBJ_COLS( CLASS_XVAL_OBJ(tcp) );
	assert(n_xvals>1);

	if( verbose ){
		sprintf(msg_str,"class = %s, %d points",
			CLASS_NAME(tcp),SUMM_DTBL_N(CLASS_SUMM_DTBL(tcp)));
		prt_msg(msg_str);
		sprintf(msg_str,"val\txval\t\tntot\tncorr\t%% corr\n");
		prt_msg(msg_str);
	}
	//j=0;
	for(j=0;j<n_xvals;j++){
		if( DATUM_NTOTAL(SUMM_DTBL_ENTRY(dtp,j)) > 0 ){
			float *xv_p;
			xv_p = indexed_data(CLASS_XVAL_OBJ(tcp),j);
			assert(xv_p!=NULL);
			sprintf(msg_str,"%d\t%f\t%d\t%d\t%f",
				j,
				*xv_p,
				DATUM_NTOTAL(SUMM_DTBL_ENTRY(dtp,j)),
				DATUM_NCORR(SUMM_DTBL_ENTRY(dtp,j)),
				(double) DATUM_NCORR(SUMM_DTBL_ENTRY(dtp,j)) /
					(double) DATUM_NTOTAL(SUMM_DTBL_ENTRY(dtp,j)) );
			prt_msg(msg_str);
		}
	}
	prt_msg("\n");
}
#endif // FOOBAR

void write_summary_data( Summary_Data_Tbl *sdt_p, FILE *fp )
{
	int j;

	// count the number of points with at least one trial
	for(j=0;j<SUMM_DTBL_SIZE(sdt_p);j++)
		if( DATUM_NTOTAL(SUMM_DTBL_ENTRY(sdt_p,j)) != 0 )
			SET_SUMM_DTBL_N(sdt_p,1+SUMM_DTBL_N(sdt_p));

	assert(SUMM_DTBL_CLASS(sdt_p)!=NULL);
	fprintf(fp,class_header_format_str,CLASS_INDEX(SUMM_DTBL_CLASS(sdt_p)),SUMM_DTBL_N(sdt_p));

	for(j=0;j<SUMM_DTBL_SIZE(sdt_p);j++)
		if( DATUM_NTOTAL(SUMM_DTBL_ENTRY(sdt_p,j)) != 0 )
			fprintf(fp,pointline, j,
				DATUM_NTOTAL(SUMM_DTBL_ENTRY(sdt_p,j)),
				DATUM_NCORR(SUMM_DTBL_ENTRY(sdt_p,j))
				);
}

void _write_sequential_data(QSP_ARG_DECL  Sequential_Data_Tbl *qdt_p, FILE *fp )
{
	Node *np;
	List *lp;

	assert(qdt_p!=NULL);
	lp = SEQ_DTBL_LIST(qdt_p);
	assert(lp!=NULL);
	np = QLIST_HEAD(lp);
	if( np == NULL ){
		prt_msg("\tNo sequence data.");
		return;
	}

	sprintf(MSG_STR,"\t%s\t%s\t%s\t%s\t%s",
		"class","stair","val","resp","correct_resp");
	prt_msg(MSG_STR);

	while(np!=NULL){
		Sequence_Datum *qd_p;
		qd_p = NODE_DATA(np);
		
		sprintf(MSG_STR,"\t%d\t%d\t%d\t%d\t%d",
			qd_p->sqd_class_idx,
			qd_p->sqd_stair_idx,
			qd_p->sqd_xval_idx,
			qd_p->sqd_response,
			qd_p->sqd_correct_response
			);
		prt_msg(MSG_STR);
        	np = NODE_NEXT(np);
	}
}

/* this is separate so we can include it at the top of dribble files */
/* write header w/ #classes & xvalues */
// This assumes that there is only one x-value object??? BUG?

#define write_data_preamble(fp) _write_data_preamble(QSP_ARG  fp)

static void _write_data_preamble(QSP_ARG_DECL  FILE *fp)
{
	int i;
	int nclasses;
	Trial_Class *tc_p;
	Node *np;
	int n_xvals;
	Data_Obj *xv_dp;

	nclasses = eltcount( trial_class_list() );
	assert(nclasses>=1);

	np = QLIST_HEAD( trial_class_list() );
	assert(np!=NULL);
	tc_p = NODE_DATA(np);
	assert(tc_p!=NULL);
	xv_dp = CLASS_XVAL_OBJ(tc_p);
	assert(xv_dp!=NULL);
	n_xvals = OBJ_COLS(xv_dp);

	fprintf(fp,topline,nclasses,n_xvals);
	for(i=0;i<n_xvals;i++){
		float *xv_p;
		xv_p = indexed_data(xv_dp,i);
		assert(xv_p!=NULL);
		fprintf(fp,xvline,*xv_p);
	}
	fflush(fp);
}

#define iterate_over_classes(func, arg) _iterate_over_classes( QSP_ARG  func, arg)

void _iterate_over_classes( QSP_ARG_DECL  void (*func)(QSP_ARG_DECL  Trial_Class *, void *), void *arg)
{
	List *lp;
	Node *np;
	Trial_Class *tc_p;

	lp = trial_class_list();
	np=QLIST_HEAD(lp);
	while(np!=NULL){
		tc_p=(Trial_Class *)np->n_data;
		assert(tc_p!=NULL);
		(*func)(QSP_ARG  tc_p,arg);
		np=np->n_next;
	}
}

void _write_exp_data(QSP_ARG_DECL  FILE *fp)	/* replaces routine formerly in stair.c */
{
	/* new ascii format */

	write_data_preamble(fp);
	fputs(summline,fp);

	iterate_over_classes(_write_class_summ_data,fp);

	fflush(fp);
}

#define read_class_summary(fp) _read_class_summary(QSP_ARG  fp)

static int _read_class_summary(QSP_ARG_DECL  FILE *fp)
{
	short index,np;
	Trial_Class *tc_p;
	Summary_Data_Tbl *sdt_p;
	int j;

	if( fscanf(fp,class_header_format_str,&index,&np) != 2 ){
		warn("error reading class header");
		return(-1);
	}
	tc_p = find_class_from_index(index);
	assert( tc_p != NULL );

	sdt_p = CLASS_SUMM_DTBL(tc_p);
	assert( np <= SUMM_DTBL_SIZE(sdt_p) );
	// we used to reallocate the data table here???

	SET_SUMM_DTBL_N(sdt_p,np);
	for(j=0;j<SUMM_DTBL_N(sdt_p);j++){
		short di,nt,nc;

		if( fscanf(fp,pointline,&di,&nt,&nc) != 3 ){
			warn("error reading data line");
			return(-1);
		} else {
			SET_DATUM_NTOTAL( SUMM_DTBL_ENTRY(sdt_p,j), nt );
			SET_DATUM_NCORR( SUMM_DTBL_ENTRY(sdt_p,j), nc );
		}
	}
	if( feof(fp) ) have_input_line=0;
	return(0);
}

#define read_class_summaries(n_classes,fp) _read_class_summaries(QSP_ARG  n_classes,fp)

static int _read_class_summaries(QSP_ARG_DECL  int n_classes,FILE *fp)		/** read data in summary format */
{
	while(n_classes--){
		if( read_class_summary(fp) < 0 )
			return(-1);
	}
	// Now we should be at the end-of-file...
	return(0);
}

/* Read a dribble file (sequential list of trials).
 */

#define rd_dribble(fp) _rd_dribble(QSP_ARG  fp)

static int _rd_dribble(QSP_ARG_DECL  FILE *fp)
{
	int i_class,i_val,resp,crct,i_stair;
	int n;

	while( next_input_line(fp) >= 0 ){
		n=sscanf(input_line,"%d\t%d\t%d\t%d\t%d\n",&i_class,&i_stair,&i_val,&resp,&crct);
		if( n == 5 ){
			Trial_Class *tc_p;
			Staircase *st_p;

			tc_p = find_class_from_index(i_class);
			if( tc_p == NULL ){
				fprintf(stderr,"rd_dribble:  didn't find class for index %d!?\n",i_class);
				return -1;
			}

			st_p = find_stair_from_index(i_stair);
			if( st_p == NULL ){
				fprintf(stderr,"rd_dribble:  didn't find stair for index %d!?\n",i_stair);
				return -1;
			}

			assert( STAIR_CLASS(st_p) == tc_p );
			assert( STAIR_CRCT_RSP(st_p) == crct );

			SET_STAIR_VAL(st_p,i_val);

			update_summary(CLASS_SUMM_DTBL(tc_p),st_p,resp);

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
} // rd_dribble

/* Read the bottom half of a data file.
 * This may be either a dribble file or a summary file.
 */

#define read_class_data(fp,n_classes) _read_class_data(QSP_ARG  fp,n_classes)

static int _read_class_data(QSP_ARG_DECL  FILE *fp,int n_classes)
{
	char tstr[32];

	/* determine data format type */
	/* fscanf(fp,"%s data\n",tstr); */
	if( next_input_line(fp) < 0 ){
		warn("read_class_data:  premature eof on input file");
		return(-1);
	}
	if( sscanf(input_line,"%s data\n",tstr) != 1 ){
		sprintf(ERROR_STRING,"read_class_data:  unexpected data description:  %s",tstr);
		warn(ERROR_STRING);
		return(-1);
	}
	if( !strcmp(tstr,"Summary") ){
		if( verbose ) advise("Reading data in summary format");
		return( read_class_summaries(n_classes,fp) );
	} else if( !strcmp(tstr,"Raw") ){
		if( verbose ) advise("Reading data in raw format");
		return( rd_dribble(fp) );
	} else {
		sprintf(ERROR_STRING,"bizarre data format:  \"%s\"\n",tstr);
		warn(ERROR_STRING);
		return(-1);
	}
	/* NOTREACHED */
}

#define setup_classes(n) _setup_classes(QSP_ARG  n)

static void _setup_classes(QSP_ARG_DECL  int n)
{
	int i;
	char name[32];

	/* We don't delete old classes if they already exist;
	 * This allows us to read multiple files if they are concatenated...
	 */
	for(i=0;i<n;i++){
		Trial_Class *tc_p;

		sprintf(name,"class%d",i);
		tc_p = trial_class_of(name);
		if( tc_p == NULL ){
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


/* Read the top half of a data file.  This tells us the number of x values,
 * and the number of classes.
 * We return the number of classes, or -1 on error.
 */

#define read_data_preamble(fp) _read_data_preamble(QSP_ARG  fp)

static int _read_data_preamble(QSP_ARG_DECL  FILE *fp)
{
	int i;
	int n;		/* number of classes */
	int n_xvals;

	if( global_xval_dp != NULL )
		delvec(global_xval_dp);

	if( ! have_input_line ){
		/* The first time through we will need to read
		 * a line of input here.  If we have multiple
		 * files concatenated, then we will have a buffered
		 * line and can skip this step.
		 */
		if( next_input_line(fp) < 0 ){
			warn("read_data_preamble:  no top line");
			return(-1);
		}
	}

	if( sscanf(input_line,topline,&n,&n_xvals) != 2 ){
		sprintf(ERROR_STRING,"read_data_preamble:  bad top line:  %s",input_line);
		warn(ERROR_STRING);
		return(-1);
	}

	if( n_xvals < 2 || n_xvals > MAX_X_VALUES ){
		sprintf(ERROR_STRING,"read_data_preamble:  ridiculous number of x values (%d)!?",n_xvals);
		warn(ERROR_STRING);
		return -1;
	}

	global_xval_dp = mk_vec("file_x_values",n_xvals,1,prec_for_code(PREC_SP));
	if(global_xval_dp==NULL){
		warn("error creating object for file x values!?");
		return -1;
	}

	/* Make sure that there are at least n classes.
	 * If they already exist, that is ok, we can read in
	 * concatenated data files and lump the data that way.
	 */

	setup_classes(n);

	/* Read the x values.  If we are reading concatenated files,
	 * we have to insist that all the xval arrays be the same.
	 * BUG we have to put in a check for this!
	 */
	for(i=0;i<n_xvals;i++){
		float *xv_p;
		xv_p = ((float *)OBJ_DATA_PTR(global_xval_dp)) +
			i * OBJ_PXL_INC(global_xval_dp);
		if( fscanf(fp,xvline,&xv_p) != 1 ){
			warn("error reading an x value");
			return(-1);
		}
	}

	return(n);
}

int _read_exp_data(QSP_ARG_DECL  FILE *fp)
{
	int n_classes;
	int status;

	have_input_line=0;

	if( (n_classes=read_data_preamble(fp)) < 0 ){
		warn("Error in data file preamble!?");
		return -1;
	}
	status=read_class_data(fp,n_classes);
	if( status < 0 ) return(status);
	if( ! have_input_line ) return(0);
	return(-1);
}

void init_dribble_file(SINGLE_QSP_ARG_DECL)
{
	FILE *fp;

	// BUG - we should only keep trying if interactive!?
	while( (fp=try_nice(nameof("dribble data file"),"w")) == NULL )
		;
	set_dribble_file(fp);
	write_data_preamble(fp);
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
}

void set_dribble_file(FILE *fp) { drib_file=fp; }

