#include "quip_config.h"

#include "quip_config.h"
#include <stdio.h>
#include "quip_prot.h"
#include "stc.h"
#include "variable.h"
#include "quip_menu.h"

static COMMAND_FUNC( do_read_data )	/** read a data file */
{
	FILE *fp;
	const char *filename;
	char num_str[16];
	int n_have_classes;

	filename=nameof("data file");
	fp=try_open( filename, "r" );
	if( !fp ) return;

	/* We used to clear the data tables here,
	 * but now they are dynamically allocated
	 * and cleared at that time...
	 */

	/* clear old classes */
	delete_all_trial_classes();

	if( read_exp_data(fp) != 0 ){
		fclose(fp);
		sprintf(ERROR_STRING,"do_read_data:  error return from read_exp_data, file %s",filename);
		WARN(ERROR_STRING);
		return;
	}
	fclose(fp);
	n_have_classes = eltcount(trial_class_list());

	sprintf(num_str,"%d",n_have_classes);	// BUG?  buffer overflow
						// if n_have_classes too big???
	assign_reserved_var( "n_classes" , num_str );
	
	if( verbose ){
		assert(global_xval_dp!=NULL);
		sprintf(ERROR_STRING,"File %s read, %d classes, %d x-values",
			filename,n_have_classes,OBJ_COLS(global_xval_dp));
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

static COMMAND_FUNC( do_print_seq )
{
	Trial_Class *tcp;

	tcp = pick_trial_class("");
	if( tcp == NULL ) return;
	print_class_sequence(tcp);
}

#define print_psychometric_pts(fp, tcp) _print_psychometric_pts(QSP_ARG  fp, tcp)

static void _print_psychometric_pts(QSP_ARG_DECL  FILE *fp, Trial_Class * tcp)
{
        int j;
        Summary_Data_Tbl *dtp;

	assert(CLASS_XVAL_OBJ(tcp)!=NULL);
	dtp=CLASS_SUMM_DTBL(tcp);
	for(j=0;j<SUMM_DTBL_SIZE(dtp);j++){
		if( DATUM_NTOTAL(SUMM_DTBL_ENTRY(dtp,j)) > 0 ){
			float *xv_p;
			xv_p = indexed_data( CLASS_XVAL_OBJ(tcp), j);
			assert(xv_p!=NULL);
			fprintf(fp,"%f\t", *xv_p);
			fprintf(fp,"%f\n",DATUM_FRACTION(SUMM_DTBL_ENTRY(dtp,j)));
		}
	}
	fclose(fp);
}

static COMMAND_FUNC( pntgrph )
{
	FILE *fp;
	Trial_Class *tcp;

	tcp = pick_trial_class("");
	fp=try_nice( nameof("output file"), "w" );
	if( fp == NULL || tcp == NULL ) return;

	print_psychometric_pts(fp,tcp);
}


static COMMAND_FUNC( terse_weibull_fit )
{
	Trial_Class *tcp;

	tcp = pick_trial_class("");

	if( tcp == NULL ) return;

	w_analyse(tcp);
	w_tersout(tcp);
}

static COMMAND_FUNC( do_terse_ogive_fit )
{
	Trial_Class *tcp;

	tcp = pick_trial_class("");

	if( tcp == NULL ) return;

	ogive_fit(tcp);
	tersout(tcp);
}

static COMMAND_FUNC( weibull_fit )
{
	Trial_Class *tcp;

	tcp = pick_trial_class("");

	if( tcp == NULL ) return;

	w_analyse(tcp);
	weibull_out(tcp);
}

static COMMAND_FUNC( do_ogive_fit )
{
	Trial_Class *tcp;

	tcp = pick_trial_class("");

	if( tcp == NULL ) return;

	ogive_fit(tcp);
	longout(tcp);
}

static COMMAND_FUNC( setfc ) { set_fcflag( askif("do analysis relative to 50% chance") ); }

static COMMAND_FUNC( do_set_chance_rate )
{
	set_chance_rate( how_much("Probability of correct response due to guessing") );
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

#define ADD_CMD(s,f,h)	ADD_COMMAND(ogive_menu,s,f,h)

MENU_BEGIN(ogive)
ADD_CMD( analyse,	do_ogive_fit,			analyse data )
ADD_CMD( summarize,	do_terse_ogive_fit,		analyse data (terse output) )
//ADD_CMD( class,		setcl,			select new stimulus class )
ADD_CMD( 2afc,		setfc,			set forced-choice flag )
ADD_CMD( chance_rate, 	do_set_chance_rate,	specify chance P(correct) )
ADD_CMD( constrain,	constrain_slope,	constrain regression slope )
MENU_END(ogive)

static COMMAND_FUNC( do_ogive )
{
	CHECK_AND_PUSH_MENU(ogive);
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

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(lookit_menu,s,f,h)

MENU_BEGIN(lookit)
ADD_CMD( read,		do_read_data,	read new data file )
ADD_CMD( print_summary,		do_print_summ,	print summary data )
ADD_CMD( print_sequence,	do_print_seq,	print sequential data )
//ADD_CMD( class,		setcl,		select new stimulus class )
ADD_CMD( plotprint,	pntgrph,	print data for plotting )
ADD_CMD( errbars,	do_pnt_bars,	print psychometric function with error bars )
ADD_CMD( ogive,		do_ogive,	do fits with to ogive )
ADD_CMD( weibull,	do_weibull,	do fits to weibull function )
ADD_CMD( split,		do_split,	split data at zeroes )
ADD_CMD( lump,		do_lump,	lump data conditions )
#ifdef QUIK
ADD_CMD( Quick,		prquic,		print data in QUICK format )
#endif /* QUIK */
MENU_END(lookit)

COMMAND_FUNC( lookmenu )
{
	CHECK_AND_PUSH_MENU(lookit);
}

