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

#define XVAL_LINE_FMT	"\t%g\n"
#define CLASS_SUMM_DATA_HEAD_FMT	"Trial_Class %d, %d data points\n"
// old was just index, ntotal, ncorrect
//#define SUMM_DATA_POINT_FMT	"\t%d\t%d\t%d\n"
#define SUMM_DATA_POINT_FMT	"\t%d\t%g\t%d\t%d\t%g\n"

Data_Obj *global_xval_dp=NULL;

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

	assert( CLASS_SUMM_DTBL(tc_p) != NULL );
	clear_summary_data( CLASS_SUMM_DTBL(tc_p) );

	lp = SEQ_DTBL_LIST( EXPT_SEQ_DTBL( CLASS_EXPT(tc_p) ) );
	assert(lp!=NULL);

	np = QLIST_HEAD(lp);
	while(np!=NULL){
		Sequence_Datum *qd_p;
		qd_p = NODE_DATA(np);

		update_class_summary(tc_p,qd_p);

		np = NODE_NEXT(np);
	}
}

void _init_class_summary( QSP_ARG_DECL  Trial_Class *tc_p)
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

	if( EXPT_SEQ_DTBL( CLASS_EXPT(tc_p) ) == NULL ){
		warn("init_class_summary:  experiment has no data!?");
		return;
	}
	tabulate_class(tc_p);
}

void _retabulate_one_class( QSP_ARG_DECL  Trial_Class *tc_p, void *arg )
{
	if( CLASS_SUMM_DTBL(tc_p) == NULL ){
		init_class_summary(tc_p);
	} else {
		tabulate_class(tc_p);
	}
}

void _retabulate_classes(SINGLE_QSP_ARG_DECL)
{
	// iterate over classes...
	iterate_over_classes( _retabulate_one_class, NULL );
}

void _print_class_summary(QSP_ARG_DECL  Trial_Class * tc_p)
{
	if( CLASS_SUMM_DTBL(tc_p) == NULL ){
fprintf(stderr,"print_class_summary calling init_class_summary\n");
		init_class_summary(tc_p);
	}

	/*
	if( verbose ){
		sprintf(msg_str,"class = %s, %d points",
			CLASS_NAME(tc_p),SUMM_DTBL_N(CLASS_SUMM_DTBL(tc_p)));
		prt_msg(msg_str);
		sprintf(msg_str,"val\txval\t\tntot\tncorr\t%% corr\n");
		prt_msg(msg_str);
	}
	*/

	write_class_summ_data(tc_p,tell_msgfile());
}

void write_summary_data( Summary_Data_Tbl *sdt_p, FILE *fp )
{
	int j;
	Data_Obj *xv_dp;

	// We used to set SUMM_DTBL_N here, but now that is done when
	// we convert from sequential data...

	assert(SUMM_DTBL_CLASS(sdt_p)!=NULL);
	fprintf(fp,CLASS_SUMM_DATA_HEAD_FMT,CLASS_INDEX(SUMM_DTBL_CLASS(sdt_p)),SUMM_DTBL_N(sdt_p));

	xv_dp = SUMM_DTBL_XVAL_OBJ(sdt_p);
	assert(xv_dp!=NULL);

	for(j=0;j<SUMM_DTBL_SIZE(sdt_p);j++){
		int ntotal,ncorr;
		float pc, *xv_p;
		ntotal = DATUM_NTOTAL(SUMM_DTBL_ENTRY(sdt_p,j));
		if( ntotal != 0 ){
			ncorr = DATUM_NCORR(SUMM_DTBL_ENTRY(sdt_p,j));
			pc = ncorr;
			pc /= ntotal;
			xv_p = indexed_data(xv_dp,j);
			assert(xv_p!=NULL);
			fprintf(fp,SUMM_DATA_POINT_FMT, j, *xv_p, ntotal, ncorr, pc);
		}
	}
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

// This function for now just prints trial number and val (level),
// but we can imagine making what is printed be user-selectable
// by providing different versions and using a function vector...

#define emit_datum(qd_p) _emit_datum(QSP_ARG  qd_p)
static void _emit_datum(QSP_ARG_DECL  Sequence_Datum *qd_p)
{
	sprintf(MSG_STR,"%d\t%d",SEQ_DATUM_TRIAL_IDX(qd_p),SEQ_DATUM_XVAL_IDX(qd_p));
	prt_msg(MSG_STR);
}

void _print_class_seq(QSP_ARG_DECL  Trial_Class *tc_p)
{
	Node *np;

	assert(tc_p!=NULL);
	assert(CLASS_EXPT(tc_p)!=NULL);
	assert(EXPT_SEQ_DTBL(CLASS_EXPT(tc_p))!=NULL);
	assert(SEQ_DTBL_LIST(EXPT_SEQ_DTBL(CLASS_EXPT(tc_p)))!=NULL);

	np = QLIST_HEAD( SEQ_DTBL_LIST(EXPT_SEQ_DTBL(CLASS_EXPT(tc_p))) );
	while( np != NULL ){
		Sequence_Datum *qd_p;
		qd_p = NODE_DATA(np);
		if( SEQ_DATUM_CLASS_IDX(qd_p) == CLASS_INDEX(tc_p) ){
			emit_datum(qd_p);
		}
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

	if( EXPT_XVAL_OBJ(exp_p) != NULL ){
		// This can happen when we load a second file...
		delvec( EXPT_XVAL_OBJ(exp_p) );
		SET_EXPT_XVAL_OBJ(exp_p,NULL);
	}
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


