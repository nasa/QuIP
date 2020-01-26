#include "quip_config.h"

#include "quip_config.h"
#include <stdio.h>
#include "quip_prot.h"
#include "stc.h"
#include "variable.h"
#include "quip_menu.h"

static Fit_Data fit_data1;
static float user_chance_rate=0.0;
static int fit_log_flag=0;
static int the_slope_constraint;
//SET_FIT_SLOPE_CONSTRAINT(fdp, ctype - 1 );	// -1, 0, or +1

static COMMAND_FUNC( do_clear_data )
{
	Sequential_Data_Tbl *qdt_p;
	qdt_p = EXPT_SEQ_DTBL(&expt1);

	// This suppresses the warning about clearing unsaved data
	CLEAR_QDT_FLAG_BITS(qdt_p,SEQUENTIAL_DATA_DIRTY);
	
	clear_sequential_data( qdt_p );
}

static COMMAND_FUNC( do_read_data )	/** read a data file */
{
	FILE *fp;
	const char *filename;
	char num_str[16];
	int n_have_classes;

	filename=nameof("data file");
	fp=try_open( filename, "r" );
	if( !fp ) return;

	// We used to clear the data tables here,
	// but now they are dynamically allocated
	// and cleared at that time...
	//

	// Clear old classes
	// But this doesn't clear the sequential data, which belongs to the experiment...
	delete_all_trial_classes();

	clear_sequential_data( EXPT_SEQ_DTBL(&expt1) );

	if( read_exp_data(fp) != 0 ){
		fclose(fp);
		sprintf(ERROR_STRING,"do_read_data:  error return from read_exp_data, file %s",filename);
		WARN(ERROR_STRING);
		return;
	}
	fclose(fp);

	// The data is created using the same routines as during collection,
	// but it does not need to be marked dirty because it has already been saved!
	CLEAR_QDT_FLAG_BITS(EXPT_SEQ_DTBL(&expt1),SEQUENTIAL_DATA_DIRTY);

	n_have_classes = eltcount(trial_class_list());

	sprintf(num_str,"%d",n_have_classes);	// BUG?  buffer overflow
						// if n_have_classes too big???
	assign_reserved_var( "n_classes" , num_str );
	
	if( verbose ){
		assert(EXPT_XVAL_OBJ(&expt1)!=NULL);
		sprintf(ERROR_STRING,"File %s read, %d classes, %d x-values",
			filename,n_have_classes,OBJ_COLS(EXPT_XVAL_OBJ(&expt1)));
		advise(ERROR_STRING);
	}
}

#ifdef QUIK
static COMMAND_FUNC( prquic )
{
	FILE *fp;
	int i;
	int in_db;
	Trial_Class *tcp;

	fp=try_nice( nameof("quic file"), "w");
	tcp = pick_trial_class("");
	in_db = askif("transform x values to decibels");

	if( fp == NULL || tcp == NULL ) return;

	if( no_data(SINGLE_QSP_ARG,"prquic") ) return;

	fprintf(fp,"%c\n",004);		/* EOT */
	pntquic(fp,tcp,in_db);
	fclose(fp);
}
#endif /* QUIK */



static COMMAND_FUNC( do_print_summ )
{
	Trial_Class *tcp;

	tcp = pick_trial_class("");
	if( tcp == NULL ) return;
	print_class_summary(tcp);
}

#define print_psychometric_pts(fp, tcp) _print_psychometric_pts(QSP_ARG  fp, tcp)

static void _print_psychometric_pts(QSP_ARG_DECL  FILE *fp, Trial_Class * tcp)
{
        int j;
        Summary_Data_Tbl *dtp;

	assert(CLASS_XVAL_OBJ(tcp)!=NULL);

	if( CLASS_SUMM_DTBL(tcp) == NULL ){
		init_class_summary(tcp);
	}
	assert(CLASS_SUMM_DTBL(tcp)!=NULL);

	dtp=CLASS_SUMM_DTBL(tcp);
	for(j=0;j<SUMM_DTBL_SIZE(dtp);j++){
		if( DATUM_NTOTAL(SUMM_DTBL_ENTRY(dtp,j)) > 0 ){
			float *xv_p;
			xv_p = indexed_data( CLASS_XVAL_OBJ(tcp), j);
			assert(xv_p!=NULL);
			sprintf(MSG_STR,"%f\t", *xv_p);
			fputs(MSG_STR,fp);
			sprintf(MSG_STR,"%f\n",DATUM_FRACTION(SUMM_DTBL_ENTRY(dtp,j)));
			fputs(MSG_STR,fp);
			//fflush(fp);
		}
	}
}

static COMMAND_FUNC( pntgrph )
{
	FILE *fp;
	Trial_Class *tcp;

	tcp = pick_trial_class("");

	// just print to outfile and let user redirect from script if desired...
	fp = qs_msg_file();

	if( fp == NULL || tcp == NULL ) return;

	print_psychometric_pts(fp,tcp);
}

#define print_old_ogive_terse(fdp, msg) _print_old_ogive_terse(QSP_ARG  fdp, msg)

static void _print_old_ogive_terse(QSP_ARG_DECL  Fit_Data *fdp, const char *msg)
{
	// BUG move fcflag into fit_data struct
	if( !fc_flag ) 
		sprintf(msg_str,"%s\t%d\t%f\t%f\t%f",msg,FIT_CLASS_INDEX(fdp),FIT_R(fdp),FIT_THRESH(fdp),FIT_SIQD(fdp));
	else
		sprintf(msg_str,"%s\t%d\t%f\t%f",msg,FIT_CLASS_INDEX(fdp),FIT_R(fdp), FIT_THRESH(fdp));

	prt_msg(msg_str);
}

#define print_new_ogive_terse(fdp, msg) _print_new_ogive_terse(QSP_ARG  fdp, msg)

static void _print_new_ogive_terse(QSP_ARG_DECL  Fit_Data *fdp, const char *msg)
{
	// BUG - should print sum of log likelihood too?
	sprintf(msg_str,"%s\tclass %d\t\tthresh %f\tsiqd %f",msg,FIT_CLASS_INDEX(fdp),FIT_THRESH(fdp),FIT_SIQD(fdp));
	prt_msg(msg_str);
}

#define print_weibull_terse(fdp, msg) _print_weibull_terse(QSP_ARG  fdp, msg)

static void _print_weibull_terse(QSP_ARG_DECL  Fit_Data *fdp, const char *msg)
{
	prt_msg("Oops - print_weibull_terse not implemented!?");
}

#define print_data_terse(fdp,msg) _print_data_terse(QSP_ARG  fdp,msg)

static void _print_data_terse(QSP_ARG_DECL  Fit_Data *fdp, const char *msg)
{
	if( FIT_TYPE(fdp) == FIT_OLD_OGIVE ){
		print_old_ogive_terse(fdp,msg);
	} else if( FIT_TYPE(fdp) == FIT_NEW_OGIVE ){
		print_new_ogive_terse(fdp,msg);
	} else if( FIT_TYPE(fdp) == FIT_WEIBULL ){
		print_weibull_terse(fdp,msg);
	} else {
		sprintf(ERROR_STRING,"print_data_terse:  unexpected fit type (%d)!?",FIT_TYPE(fdp));
		warn(ERROR_STRING);
	}
}

#define print_weibull_verbose(fdp) _print_weibull_verbose(QSP_ARG  fdp)

static void _print_weibull_verbose(QSP_ARG_DECL  Fit_Data *fdp)	// verbose analysis report
{
	warn("print_weibull_verbose:  not implemented!?");
}

#define print_ogive_verbose(fdp) _print_ogive_verbose(QSP_ARG  fdp)

static void _print_ogive_verbose(QSP_ARG_DECL  Fit_Data *fdp)	// verbose analysis report
{
        sprintf(msg_str,"\nTrial_Class %s\n",CLASS_NAME(FIT_CLASS(fdp)));
	prt_msg(msg_str);
        sprintf(msg_str,"initial correlation:\t\t\t%f", FIT_R_INITIAL(fdp) );
	prt_msg(msg_str);
        sprintf(msg_str,"final correlation:\t\t\t%f", FIT_R(fdp) );
	prt_msg(msg_str);

        if(!fc_flag) {
                sprintf(msg_str,"x value at inflection pt:\t\t%f", FIT_THRESH(fdp) );
		prt_msg(msg_str);
                sprintf(msg_str,"semi-interquartile difference:\t\t%f",FIT_SIQD( fdp ) );
		prt_msg(msg_str);
        } else {
		sprintf(msg_str,"x value for 75%%:\t%f", FIT_THRESH(fdp) );
		prt_msg(msg_str);
	}
}

#define print_data_verbose(fdp) _print_data_verbose(QSP_ARG  fdp)

static void _print_data_verbose(QSP_ARG_DECL  Fit_Data *fdp )
{
	if( FIT_TYPE(fdp) == FIT_NEW_OGIVE || FIT_TYPE(fdp) == FIT_OLD_OGIVE ){
		print_ogive_verbose(fdp);
	} else if( FIT_TYPE(fdp) == FIT_WEIBULL ){
		print_weibull_verbose(fdp);
	} else {
		sprintf(ERROR_STRING,"print_data_verbose:  unexpected fit type (%d)!?",FIT_TYPE(fdp));
		warn(ERROR_STRING);
	}
}

static void init_fit_data( Fit_Data *fdp, Fit_Type type, Trial_Class *tcp )
{
	SET_FIT_TYPE(fdp,type);
	SET_FIT_CLASS(fdp,tcp);
	SET_FIT_CHANCE_RATE(fdp,user_chance_rate);
	SET_FIT_LOG_FLAG(fdp,fit_log_flag);

	memset(&(fdp->fd_u),0,sizeof(fdp->fd_u));

	if( IS_OGIVE_FIT(fdp) ){
		SET_FIT_SLOPE_CONSTRAINT(fdp, the_slope_constraint);	// -1, 0, or +1
	}
}

#define fit_func_for_type(type) _fit_func_for_type(QSP_ARG  type)

static void (*_fit_func_for_type(QSP_ARG_DECL  Fit_Type type))(QSP_ARG_DECL  Fit_Data *)
{
	void (*func)(QSP_ARG_DECL  Fit_Data *fdp);

	assert( type >= 0 && type < N_FIT_TYPES );

	switch(type){
		case FIT_OLD_OGIVE:
			func = _old_ogive_fit;
			break;
		case FIT_NEW_OGIVE:
			func = _new_ogive_fit;
			break;
		case FIT_WEIBULL:
			func = _w_analyse;
			break;
		case N_FIT_TYPES:
			error1("invalid fit type!?");
			func = NULL;
			break;
	}
	return func;
}

#define show_fit_terse( func ) _show_fit_terse( QSP_ARG   func )

static void _show_fit_terse( QSP_ARG_DECL   Fit_Type type )
{
	Trial_Class *tcp;
	const char *msg;
	void (*func)(QSP_ARG_DECL  Fit_Data *fdp);

	tcp = pick_trial_class("");
	msg = nameof("output tag string");

	if( tcp == NULL ) return;

	func = fit_func_for_type(type);
	init_fit_data(&fit_data1,type,tcp);
	(*func)(QSP_ARG  &fit_data1);
	print_data_terse(&fit_data1,msg);
}

#define show_fit_verbose(type) _show_fit_verbose(QSP_ARG  type)

static void _show_fit_verbose(QSP_ARG_DECL  Fit_Type type)
{
	Trial_Class *tcp;
	void (*func)(QSP_ARG_DECL  Fit_Data *fdp);

	tcp = pick_trial_class("");
	if( tcp == NULL ) return;

	func = fit_func_for_type(type);
	init_fit_data(&fit_data1,type,tcp);
	(*func)(QSP_ARG  &fit_data1);
	print_data_verbose(&fit_data1);
}


static COMMAND_FUNC( terse_weibull_fit ) { show_fit_terse(FIT_WEIBULL); }
static COMMAND_FUNC( do_terse_old_ogive_fit ) { show_fit_terse(FIT_OLD_OGIVE); }
static COMMAND_FUNC( do_terse_new_ogive_fit ) { show_fit_terse(FIT_NEW_OGIVE); }

static COMMAND_FUNC( weibull_fit ) { show_fit_verbose(FIT_WEIBULL); } 
static COMMAND_FUNC( do_old_ogive_fit ) { show_fit_verbose(FIT_OLD_OGIVE); } 
static COMMAND_FUNC( do_new_ogive_fit ) { show_fit_verbose(FIT_NEW_OGIVE); }

static COMMAND_FUNC( setfc ) { set_fcflag( askif("do analysis relative to 50% chance") ); }

static COMMAND_FUNC( do_set_chance_rate )
{
	float r = how_much("Probability of correct response due to guessing");
	if( r < 0 || r > 1 ){
		sprintf(ERROR_STRING,"Chance rate (%g) should be between 0 and 1!?",r);
		warn(ERROR_STRING);
		return;
	}
	user_chance_rate = r;
}

static COMMAND_FUNC( do_fit_logs )
{
	fit_log_flag = askif("Take logarithm of x value before fitting");
}

static COMMAND_FUNC( do_split )
{
	int wu;
	Trial_Class *tcp;

	tcp = pick_trial_class("");

	wu = askif("retain upper half");

	if( tcp == NULL ) return;

	split(tcp,wu);
}

static const char *clist[]={"negative","unconstrained","positive"};

static COMMAND_FUNC( constrain_slope )
{
	int ctype;

	ctype = which_one("constraint for slope",3,clist);
	if( ctype < 0 ) return;

	the_slope_constraint = ctype - 1;
}


#define ADD_CMD(s,f,h)	ADD_COMMAND(old_ogive_menu,s,f,h)

MENU_BEGIN(old_ogive)
ADD_CMD( analyse,	do_old_ogive_fit,			analyse data )
ADD_CMD( summarize,	do_terse_old_ogive_fit,		analyse data (terse output) )
//ADD_CMD( class,		setcl,			select new stimulus class )
ADD_CMD( 2afc,		setfc,			set forced-choice flag )
ADD_CMD( fit_logs,	do_fit_logs,		specify fitting against log or linear x values)
ADD_CMD( chance_rate, 	do_set_chance_rate,	specify chance P(correct) )
ADD_CMD( constrain,	constrain_slope,	constrain regression slope )
MENU_END(old_ogive)

static COMMAND_FUNC( do_old_ogive )
{
	CHECK_AND_PUSH_MENU(old_ogive);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(new_ogive_menu,s,f,h)

MENU_BEGIN(new_ogive)
ADD_CMD( analyse,	do_new_ogive_fit,			analyse data )
ADD_CMD( summarize,	do_terse_new_ogive_fit,		analyse data (terse output) )
//ADD_CMD( class,		setcl,			select new stimulus class )
ADD_CMD( fit_logs,	do_fit_logs,		specify fitting against log or linear x values)
ADD_CMD( chance_rate, 	do_set_chance_rate,	specify chance P(correct) )
MENU_END(new_ogive)


static COMMAND_FUNC( do_new_ogive )
{
	CHECK_AND_PUSH_MENU(new_ogive);
}

static COMMAND_FUNC( seter )
{
	double er;

	er=how_much("finger error rate");
	w_set_error_rate(er);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(weibull_menu,s,f,h)

MENU_BEGIN(weibull)
ADD_CMD( analyse,	weibull_fit,		analyse data )
ADD_CMD( summarize,	terse_weibull_fit,	analyse data (terse output) )
//ADD_CMD( class,		setcl,		select new stimulus class )
ADD_CMD( 2afc,		setfc,		set forced-choice flag )
ADD_CMD( error_rate,	seter,		specify finger-error rate )
MENU_END(weibull)


static COMMAND_FUNC( do_weibull )
{
	CHECK_AND_PUSH_MENU(weibull);
}

static COMMAND_FUNC( do_pnt_bars )
{
	FILE *fp;
	Trial_Class *tcp;

	tcp = pick_trial_class("");
	fp=try_nice( nameof("output file"), "w" );
	if( fp == NULL || tcp == NULL ) return;

	print_error_bars( fp, tcp );
}


// print to the standard message stream, we can redirect to a file
// to get for plotting...

static COMMAND_FUNC( do_print_class_seq )
{
	Trial_Class *tcp;

	tcp = pick_trial_class("");
	if( tcp == NULL ) return;

	print_class_seq( tcp );
}

static COMMAND_FUNC( do_print_seq )
{
	write_sequential_data( EXPT_SEQ_DTBL(&expt1), tell_msgfile() );
}

static COMMAND_FUNC( do_clean_seq )
{
	clean_sequential_data( EXPT_SEQ_DTBL(&expt1) );
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(lookit_menu,s,f,h)

MENU_BEGIN(lookit)
ADD_CMD( read,		do_read_data,	read new data file )
ADD_CMD( print_summary,		do_print_summ,	print summary data )
ADD_CMD( print_sequence,	do_print_seq,	print sequential data )
ADD_CMD( clean_sequence,	do_clean_seq,	remove redo's and undo's )
ADD_CMD( print_class_seq,	do_print_class_seq,	print val vs. trial number for a single class)
ADD_CMD( plotprint,	pntgrph,	print data for plotting )
ADD_CMD( errbars,	do_pnt_bars,	print psychometric function with error bars )
ADD_CMD( old_ogive,	do_old_ogive,	do fits with ogive (original method) )
ADD_CMD( ogive,		do_new_ogive,	do fits with ogive (new method) )
ADD_CMD( weibull,	do_weibull,	do fits to weibull function )
ADD_CMD( split,		do_split,	split data at zeroes )
ADD_CMD( lump,		do_lump,	lump data conditions )
ADD_CMD( clear_data,	do_clear_data,	clear all data )
#ifdef QUIK
ADD_CMD( Quick,		prquic,		print data in QUICK format )
#endif /* QUIK */
MENU_END(lookit)

COMMAND_FUNC( lookmenu )
{
	CHECK_AND_PUSH_MENU(lookit);
}

