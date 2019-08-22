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

//static const char *topline="%d classes, %d x values\n";
//static const char *xvline="\t%g\n";

#define XVAL_LINE_FMT	"\t%g\n"

#define CLASS_SUMM_DATA_HEAD_FMT	"Trial_Class %d, %d data points\n"
//static const char *class_header_format_str="Trial_Class %d, %d data points\n";
//static const char *pointline="\t%d\t%d\t%d\n";
#define SUMM_DATA_POINT_FMT	"\t%d\t%d\t%d\n"

//static const char *dribline="Raw data\n";

//static char input_line[LLEN];
//static int have_input_line;


//FILE *drib_file=NULL;		/* trial by trial dribble */
Data_Obj *global_xval_dp=NULL;


#ifdef FOOBAR
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
#endif // FOOBAR

#define write_class_summ_data(tc_p,_fp) _write_class_summ_data(QSP_ARG  tc_p,_fp)

static void _write_class_summ_data(QSP_ARG_DECL  Trial_Class *tc_p,void *_fp)
{
	FILE *fp;
	fp = (FILE *) _fp;
	assert(fp!=NULL);
	assert(CLASS_SUMM_DTBL(tc_p)!=NULL);
	write_summary_data( CLASS_SUMM_DTBL(tc_p), fp );
}

static void tabulate_class( Trial_Class *tc_p )
{
	List *lp;
	Node *np;

	lp = SEQ_DTBL_LIST( EXPT_SEQ_DTBL( CLASS_EXPT(tc_p) ) );
	assert(lp!=NULL);

	np = QLIST_HEAD(lp);
	while(np!=NULL){
		Sequence_Datum *qd_p;
		qd_p = NODE_DATA(np);

		update_summary(CLASS_SUMM_DTBL(tc_p),qd_p);

		np = NODE_NEXT(np);
	}
}

/*

	int			sdt_size;	// number of allocated entries (x values)
	int			sdt_npts;	// number that have non-zero n trials
	Data_Obj *		sdt_data_dp;
	Summary_Datum *		sdt_data_ptr;	// points to data in the object...
	Trial_Class *		sdt_tc_p;	// may be invalid if lumped...
	Data_Obj *		sdt_xval_dp;	// should match class
	int			sdt_flags;
	*/

#define init_class_summary( tc_p) _init_class_summary( QSP_ARG  tc_p)

static void _init_class_summary( QSP_ARG_DECL  Trial_Class *tc_p)
{
	Summary_Data_Tbl *sdt_p;
	Data_Obj *dp;

	assert(tc_p!=NULL);
	assert(CLASS_SUMM_DTBL(tc_p)==NULL);

	sdt_p = getbuf(sizeof(Summary_Data_Tbl));
	SET_CLASS_SUMM_DTBL(tc_p, sdt_p );

	dp = EXPT_XVAL_OBJ(CLASS_EXPT(tc_p));
	assert(dp!=NULL);

	SET_SUMM_DTBL_SIZE(sdt_p, OBJ_COLS(dp) );
	SET_SUMM_DTBL_DATA_PTR(sdt_p, getbuf( SUMM_DTBL_SIZE(sdt_p) * sizeof(Summary_Datum) ) );
	SET_SUMM_DTBL_CLASS(sdt_p,tc_p);

	SET_SUMM_DTBL_N(sdt_p,0);
	SET_SUMM_DTBL_DATA_OBJ(sdt_p,NULL);

	clear_summary_data( sdt_p );

	if( EXPT_SEQ_DTBL( CLASS_EXPT(tc_p) ) == NULL ){
		warn("init_class_summary:  experiment has no data!?");
		return;
	}
	tabulate_class(tc_p);
}

void _print_class_summary(QSP_ARG_DECL  Trial_Class * tc_p)
{
	if( CLASS_SUMM_DTBL(tc_p) == NULL ){
fprintf(stderr,"print_class_summary calling init_class_summary\n");
		init_class_summary(tc_p);
	}

	if( verbose ){
		sprintf(msg_str,"class = %s, %d points",
			CLASS_NAME(tc_p),SUMM_DTBL_N(CLASS_SUMM_DTBL(tc_p)));
		prt_msg(msg_str);
		sprintf(msg_str,"val\txval\t\tntot\tncorr\t%% corr\n");
		prt_msg(msg_str);
	}

	write_class_summ_data(tc_p,tell_msgfile());
}

void write_summary_data( Summary_Data_Tbl *sdt_p, FILE *fp )
{
	int j;

	// count the number of points with at least one trial
	for(j=0;j<SUMM_DTBL_SIZE(sdt_p);j++)
		if( DATUM_NTOTAL(SUMM_DTBL_ENTRY(sdt_p,j)) != 0 )
			SET_SUMM_DTBL_N(sdt_p,1+SUMM_DTBL_N(sdt_p));

	assert(SUMM_DTBL_CLASS(sdt_p)!=NULL);
	fprintf(fp,CLASS_SUMM_DATA_HEAD_FMT,CLASS_INDEX(SUMM_DTBL_CLASS(sdt_p)),SUMM_DTBL_N(sdt_p));

	for(j=0;j<SUMM_DTBL_SIZE(sdt_p);j++)
		if( DATUM_NTOTAL(SUMM_DTBL_ENTRY(sdt_p,j)) != 0 )
			fprintf(fp,SUMM_DATA_POINT_FMT, j,
				DATUM_NTOTAL(SUMM_DTBL_ENTRY(sdt_p,j)),
				DATUM_NCORR(SUMM_DTBL_ENTRY(sdt_p,j))
				);
}

#define SEQ_DATA_HEADER_FMT	"Sequential Data, %d records\n"
#define SEQ_DATA_COL_HEADS	"\ttrial\tclass\tstair\tval\tresp\tcrct_r\trt_msecs\n"

static int write_sequential_header( Sequential_Data_Tbl *qdt_p, FILE *fp )
{
	int n;
	n = eltcount( SEQ_DTBL_LIST(qdt_p) );
	if( n > 0 ){
		fprintf(fp,SEQ_DATA_HEADER_FMT,n);
		fprintf(fp,SEQ_DATA_COL_HEADS);
	} else {
		// Print a warning here???
		fprintf(fp,"No sequence data.\n");
	}
	return n;
}

#define SEQ_DATUM_FMT	"\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n"

static void write_sequence_datum(Sequence_Datum *qd_p, FILE *fp)
{
	fprintf(fp,SEQ_DATUM_FMT,
		SEQ_DATUM_TRIAL_IDX(qd_p),
		SEQ_DATUM_CLASS_IDX(qd_p),
		SEQ_DATUM_STAIR_IDX(qd_p),
		SEQ_DATUM_XVAL_IDX(qd_p),
		SEQ_DATUM_RESPONSE(qd_p),
		SEQ_DATUM_CRCT_RSP(qd_p),
		SEQ_DATUM_RT_MSECS(qd_p)
	);

	fflush(fp);
}

#define CLASS_NAME_FORMAT	"\t%d\t%s\n"

#define write_class_names( exp_p, fp ) _write_class_names( QSP_ARG  exp_p, fp )

static void _write_class_names( QSP_ARG_DECL  Experiment *exp_p, FILE *fp )
{
	Node *np;

	assert(exp_p!=NULL);
	assert(EXPT_CLASS_LIST(exp_p)!=NULL);
	np = QLIST_HEAD( EXPT_CLASS_LIST(exp_p) );
	while(np!=NULL){
		Trial_Class *tc_p;
		tc_p = NODE_DATA(np);
		fprintf(fp,CLASS_NAME_FORMAT,CLASS_INDEX(tc_p),CLASS_NAME(tc_p));
		np = NODE_NEXT(np);
	}


}

#define CLASS_SUMMARY_FORMAT	"%d classes\n"

#define write_class_preamble( exp_p, fp ) _write_class_preamble( QSP_ARG  exp_p, fp )

static void _write_class_preamble( QSP_ARG_DECL  Experiment *exp_p, FILE *fp )
{
	int n_classes;

	assert(exp_p!=NULL);
	assert(EXPT_CLASS_LIST(exp_p)!=NULL);

	n_classes = eltcount( EXPT_CLASS_LIST(exp_p) );
	assert(n_classes>=1);

	fprintf(fp,CLASS_SUMMARY_FORMAT,n_classes);

	write_class_names(exp_p,fp);
}

#define XVAL_SUMM_FORMAT	"%d x-values\n"

static void write_xval_summary(Experiment *exp_p,FILE *fp)
{
	assert( EXPT_XVAL_OBJ(exp_p) != NULL );
	fprintf(fp,XVAL_SUMM_FORMAT,OBJ_COLS( EXPT_XVAL_OBJ(exp_p) ) );
}

static void write_xval_values(Experiment *exp_p,FILE *fp)
{
	int i;
	int n_xvals;
	Data_Obj *xv_dp;

	xv_dp = EXPT_XVAL_OBJ(exp_p);
	assert(xv_dp!=NULL);
	n_xvals = OBJ_COLS(xv_dp);
	assert( OBJ_PREC(xv_dp) == PREC_SP );

	for(i=0;i<n_xvals;i++){
		float *xv_p;
		xv_p = indexed_data(xv_dp,i);
		assert(xv_p!=NULL);
		fprintf(fp,XVAL_LINE_FMT,*xv_p);
	}
}

static void write_xval_preamble( Experiment *exp_p, FILE *fp )
{
	write_xval_summary(exp_p,fp);
	write_xval_values(exp_p,fp);
}

/* write header w/ #classes & xvalues */
// This assumes that there is only one x-value object??? BUG?

#define write_data_preamble(exp_p,fp) _write_data_preamble(QSP_ARG  exp_p, fp)

static void _write_data_preamble(QSP_ARG_DECL  Experiment *exp_p, FILE *fp)
{
	write_class_preamble( exp_p, fp );
	write_xval_preamble( exp_p, fp );
}

void _write_sequential_data(QSP_ARG_DECL  Sequential_Data_Tbl *qdt_p, FILE *fp )
{
	int n;
	Node *np;

	assert(qdt_p!=NULL);
	assert( SEQ_DTBL_LIST(qdt_p) != NULL );

	n = write_sequential_header(qdt_p,fp);
	if( n <= 0 ) return;

	np = QLIST_HEAD( SEQ_DTBL_LIST(qdt_p) );
	while(np!=NULL){
		Sequence_Datum *qd_p;
		qd_p = NODE_DATA(np);
		write_sequence_datum(qd_p,fp);
        	np = NODE_NEXT(np);
	}
}

#define iterate_over_classes(func, arg) _iterate_over_classes( QSP_ARG  func, arg)

void _iterate_over_classes( QSP_ARG_DECL  void (*func)(QSP_ARG_DECL  Trial_Class *, void *), void *arg)
{
	List *lp;
	Node *np;
	Trial_Class *tc_p;

	lp = EXPT_CLASS_LIST(&expt1);
	np=QLIST_HEAD(lp);
	while(np!=NULL){
		tc_p=(Trial_Class *)np->n_data;
		assert(tc_p!=NULL);
		(*func)(QSP_ARG  tc_p,arg);
		np=np->n_next;
	}
}

#ifdef FOOBAR
#define read_class_summary(fp) _read_class_summary(QSP_ARG  fp)

static int _read_class_summary(QSP_ARG_DECL  FILE *fp)
{
	int index,np;
	Trial_Class *tc_p;
	Summary_Data_Tbl *sdt_p;
	int j;

	if( fscanf(fp,CLASS_SUMM_DATA_HEAD_FMT,&index,&np) != 2 ){
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
		int di,nt,nc;

		if( fscanf(fp,SUMM_DATA_POINT_FMT,&di,&nt,&nc) != 3 ){
			warn("error reading data line");
			return(-1);
		} else {
			SET_DATUM_NTOTAL( SUMM_DTBL_ENTRY(sdt_p,j), nt );
			SET_DATUM_NCORR( SUMM_DTBL_ENTRY(sdt_p,j), nc );
		}
	}
	//if( feof(fp) ) have_input_line=0;
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
 *
 * No header???
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

			//update_summary(CLASS_SUMM_DTBL(tc_p),st_p,resp);

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
#endif // FOOBAR

#define CHECK_FOR_EOF(whence)							\
										\
	if( ret_val == EOF ){							\
		warn("Unexpected end-of-file reading " #whence "!?");		\
		return -1;							\
	}

#define CHECK_FOR_PARSE(n_expected,whence)						\
	if( ret_val != n_expected ){							\
		warn("Error parsing input file while reading " #whence "!?");		\
		return -1;								\
	}


#define read_class_name( exp_p, fp ) _read_class_name( QSP_ARG  exp_p, fp )

static int _read_class_name( QSP_ARG_DECL  Experiment *exp_p, FILE *fp )
{
	char buf[128];
	int index;
	int ret_val;
	Trial_Class *tc_p;
	Node *np;

	ret_val = fscanf(fp,CLASS_NAME_FORMAT,&index,buf);	// BUG possible buffer overrun
	CHECK_FOR_EOF(class name)
	CHECK_FOR_PARSE(2,class name)

	// Now create a class and add it to the list
	// BUG we should validate the indices and that the names are unique!?!?
	tc_p = create_named_class(buf);
	// Under normal circumstances, the index should be correct already
	if( CLASS_INDEX(tc_p) != index ){
		sprintf(ERROR_STRING,"Automatic index (%d) for class %s does not match specified index %d!?",
			CLASS_INDEX(tc_p),CLASS_NAME(tc_p),index);
		warn(ERROR_STRING);
	}
	SET_CLASS_INDEX(tc_p,index);

	np = mk_node(tc_p);
	addTail( EXPT_CLASS_LIST(exp_p), np );
	return 0;
}

#define read_class_names( n, exp_p, fp ) _read_class_names( QSP_ARG  n, exp_p, fp )

static int _read_class_names( QSP_ARG_DECL  int n_classes, Experiment *exp_p, FILE *fp )
{
	int i;

	assert( EXPT_CLASS_LIST(exp_p) != NULL );
	assert( eltcount( EXPT_CLASS_LIST(exp_p) ) == 0 );

	for(i=0;i<n_classes;i++){
		if( read_class_name(exp_p,fp) < 0 )
			return -1;
	}
	return 0;
}


#define read_class_preamble( exp_p, fp ) _read_class_preamble( QSP_ARG  exp_p, fp )

static int _read_class_preamble( QSP_ARG_DECL  Experiment *exp_p, FILE *fp )
{
	int n_classes;
	int ret_val;

	ret_val = fscanf(fp,CLASS_SUMMARY_FORMAT,&n_classes);
	CHECK_FOR_EOF(class summary)
	CHECK_FOR_PARSE(1,class summary)

	if( n_classes <= 0 ){
		sprintf(ERROR_STRING,"Number of classes (%d) must be positive!?",n_classes);
		warn(ERROR_STRING);
		return -1;
	}

	return read_class_names(n_classes,exp_p,fp);
}

#define read_xval_summary(exp_p, fp ) _read_xval_summary(QSP_ARG  exp_p, fp )

static int _read_xval_summary(QSP_ARG_DECL  Experiment *exp_p, FILE *fp )
{
	int n_xv;
	int ret_val;
	Data_Obj *dp;

	ret_val = fscanf(fp,XVAL_SUMM_FORMAT,&n_xv);
	CHECK_FOR_EOF(x-value summary)
	CHECK_FOR_PARSE(1,x-value summary)

	if( n_xv <= 0 ){
		sprintf(ERROR_STRING,"Number of x-values (%d) must be positive!?",n_xv);
		warn(ERROR_STRING);
		return -1;
	}
	if( n_xv > MAX_X_VALUES ){
		sprintf(ERROR_STRING,"Number of x-values (%d) must be less than %d!?",n_xv,MAX_X_VALUES);
		warn(ERROR_STRING);
		return -1;
	}

	// Now it's OK
	assert( EXPT_XVAL_OBJ(exp_p) == NULL );
	dp = mk_vec("experiment_x_values",n_xv,1,prec_for_code(PREC_SP));
	assert(dp!=NULL);
	SET_EXPT_XVAL_OBJ(exp_p,dp);
	return 0;
}

#define read_xval_values( exp_p, fp ) _read_xval_values( QSP_ARG  exp_p, fp )

static int _read_xval_values( QSP_ARG_DECL  Experiment *exp_p, FILE *fp )
{
	int i;
	int n_xvals;
	Data_Obj *xv_dp;

	xv_dp = EXPT_XVAL_OBJ(exp_p);
	assert(xv_dp!=NULL);
	n_xvals = OBJ_COLS(xv_dp);
	assert( OBJ_PREC(xv_dp) == PREC_SP );

	for(i=0;i<n_xvals;i++){
		float *xv_p;
		int ret_val;

		xv_p = indexed_data(xv_dp,i);
		assert(xv_p!=NULL);
		ret_val = fscanf(fp,XVAL_LINE_FMT,xv_p);
		if( ret_val == EOF ){
			warn("Unexpected end-of-file reading x-values!?");
			return -1;
		}
		if( ret_val != 1 ){
			warn("File parsing error reading x-values!?");
			return -1;
		}
	}
	return 0;
}

#define read_xval_preamble( exp_p, fp ) _read_xval_preamble( QSP_ARG  exp_p, fp )

static int _read_xval_preamble( QSP_ARG_DECL  Experiment *exp_p, FILE *fp )
{
	if( read_xval_summary(exp_p,fp) < 0 ) return -1;
	if( read_xval_values(exp_p,fp) < 0 ) return -1;
	return 0;
}

/* Read the top half of a data file.  This tells us the number of x values,
 * and the number of classes.
 * We return the number of classes, or -1 on error.
 */

#define read_data_preamble(exp_p,fp) _read_data_preamble(QSP_ARG  exp_p,fp)

static int _read_data_preamble(QSP_ARG_DECL  Experiment *exp_p, FILE *fp)
{
	if( read_class_preamble( exp_p, fp ) < 0 )
		return -1;
	return read_xval_preamble( exp_p, fp );
}

// This will say "1 records" but it's so much easier to parse with scanf!?

#define SEQ_DATA_HEADER_FMT	"Sequential Data, %d records\n"

#define read_sequential_header( exp_p, fp ) _read_sequential_header( QSP_ARG  exp_p, fp )

static int _read_sequential_header( QSP_ARG_DECL  Experiment *exp_p, FILE *fp )
{
	int ret_val;
	int n_records;

	ret_val = fscanf(fp,SEQ_DATA_HEADER_FMT,&n_records);
	// is it OK for plural str to be empty???
	CHECK_FOR_EOF(sequential data summary)
	CHECK_FOR_PARSE(1,sequential data summary)

	ret_val = fscanf(fp,SEQ_DATA_COL_HEADS);
	CHECK_FOR_EOF(sequential data column heads)
	CHECK_FOR_PARSE(0,sequential data summary)

	// BUG do what now?
	return n_records;
}

#define read_sequential_datum( exp_p, fp ) _read_sequential_datum( QSP_ARG  exp_p, fp )

static int _read_sequential_datum( QSP_ARG_DECL  Experiment *exp_p, FILE *fp )
{ 
	int ret_val;
	Sequence_Datum qd, *qd_p;

	ret_val = fscanf(fp,SEQ_DATUM_FMT,
		& SEQ_DATUM_TRIAL_IDX(&qd),
		& SEQ_DATUM_CLASS_IDX(&qd),
		& SEQ_DATUM_STAIR_IDX(&qd),
		& SEQ_DATUM_XVAL_IDX(&qd),
		& SEQ_DATUM_RESPONSE(&qd),
		& SEQ_DATUM_CRCT_RSP(&qd),
		& SEQ_DATUM_RT_MSECS(&qd)
		);
	CHECK_FOR_EOF(sequential datum)
	CHECK_FOR_PARSE(7,sequential datum)

	qd_p = getbuf( sizeof(qd) );
	*qd_p = qd;

	save_datum(exp_p,qd_p);

	return 0;
}

#define read_sequential_data( exp_p, fp ) _read_sequential_data( QSP_ARG  exp_p, fp )

static int _read_sequential_data( QSP_ARG_DECL  Experiment *exp_p, FILE *fp )
{
	int i,n_records;

	if( (n_records=read_sequential_header(exp_p,fp)) < 0 )
		return -1;

	if( n_records == 0 ){
		warn("No data!?");
		return -1;
	}

	for(i=0;i<n_records;i++){
		if( read_sequential_datum(exp_p,fp) < 0 )
			return -1;
	}

	return 0;
}

int _read_exp_data(QSP_ARG_DECL  FILE *fp)
{
	if( read_data_preamble(&expt1,fp) < 0 )
		return -1;

	return read_sequential_data( &expt1, fp );
}


void _save_data( QSP_ARG_DECL  Experiment *exp_p, FILE *fp )
{
	if( EXPT_SEQ_DTBL(exp_p) == NULL ){
		sprintf(ERROR_STRING,"save_data:  experiment has no sequential data!?");
		warn(ERROR_STRING);
		return;
	}

	write_data_preamble(exp_p,fp);
	write_sequential_data( EXPT_SEQ_DTBL(exp_p), fp );
	fclose(fp);

	// mark the data table as written!
	CLEAR_QDT_FLAG_BITS( EXPT_SEQ_DTBL(exp_p), SEQUENTIAL_DATA_DIRTY );
}

