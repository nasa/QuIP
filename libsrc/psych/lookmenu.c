#include "quip_config.h"

#include "quip_config.h"
#include <stdio.h>
#include "quip_prot.h"
#include "stc.h"
#include "variable.h"

//static Trial_Class *curr_tcp=NULL;
static int n_have_classes=0;

#ifdef FOOBAR
#define CHECK_CURR_TCP(whence)				\
							\
	if( curr_tcp == NULL ){				\
		sprintf(ERROR_STRING,			\
	"%s:  no condition selected!?",#whence);	\
		WARN(ERROR_STRING);			\
		return;					\
	}
#endif // FOOBAR
		
static COMMAND_FUNC( do_read_data )	/** read a data file */
{
	FILE *fp;
	const char *filename;
	char num_str[16];

	filename=NAMEOF("data file");
	fp=TRY_OPEN( filename, "r" );
	if( !fp ) return;

	/* We used to clear the data tables here,
	 * but now they are dynamically allocated
	 * and cleared at that time...
	 */

	/* clear old classes */
	do_delete_all_classes(SINGLE_QSP_ARG);
	//curr_tcp=NULL;
	n_have_classes=0;
	if( read_exp_data(QSP_ARG  fp) != 0 ){
		fclose(fp);
		sprintf(ERROR_STRING,"do_read_data:  error return from read_exp_data, file %s",filename);
		WARN(ERROR_STRING);
		return;
	}
	fclose(fp);
	n_have_classes = eltcount(class_list(SINGLE_QSP_ARG));

	sprintf(num_str,"%d",n_have_classes);	// BUG?  buffer overflow
						// if n_have_classes too big???
	ASSIGN_RESERVED_VAR( "n_classes" , num_str );
	
	if( verbose ){
		sprintf(ERROR_STRING,"File %s read, %d classes, %d x-values",
			filename,n_have_classes,_nvals);
		advise(ERROR_STRING);
	}
}

static int no_data(QSP_ARG_DECL  const char *whence)
{
	if( n_have_classes <= 0 ){
		sprintf(ERROR_STRING,
			"%s:  must read a data file before this operation!",
			whence);
		WARN(ERROR_STRING);
		return(1);
	}
	return(0);
}

#ifdef QUIK
static COMMAND_FUNC( prquic )
{
	FILE *fp;
	int i;
	int in_db;
	Trial_Class *tcp;

	fp=TRYNICE( NAMEOF("quic file"), "w");
	tcp = PICK_TRIAL_CLASS("");
	in_db = ASKIF("transform x values to decibels");

	if( fp == NULL || tcp == NULL ) return;

	if( no_data(SINGLE_QSP_ARG,"prquic") ) return;

	fprintf(fp,"%c\n",004);		/* EOT */
	pntquic(fp,tcp,in_db);
	fclose(fp);
}
#endif /* QUIK */



static COMMAND_FUNC( do_print_raw )
{
	Trial_Class *tcp;

	tcp = PICK_TRIAL_CLASS("");
	if( tcp == NULL ) return;
	if( no_data(QSP_ARG  "do_print_raw") ) return;
	print_raw_data(QSP_ARG  tcp);
}

static void pntcurve(QSP_ARG_DECL  FILE *fp, Trial_Class * tcp)
{
        int j;
        Data_Tbl *dtp;

	dtp=CLASS_DATA_TBL(tcp);
	for(j=0;j<DTBL_SIZE(dtp);j++){
		if( DATUM_NTOTAL(DTBL_ENTRY(dtp,j)) > 0 ){
			fprintf(fp,"%f\t", xval_array[ j ]);
			fprintf(fp,"%f\n",DATUM_FRACTION(DTBL_ENTRY(dtp,j)));
		}
	}
	fclose(fp);
}

static COMMAND_FUNC( pntgrph )
{
	FILE *fp;
	Trial_Class *tcp;

	tcp = PICK_TRIAL_CLASS("");
	fp=TRYNICE( NAMEOF("output file"), "w" );
	if( fp == NULL || tcp == NULL ) return;

	if( no_data(QSP_ARG  "pntgrph") ) return;

	pntcurve(QSP_ARG  fp,tcp);
}


static COMMAND_FUNC( t_wanal )
{
	Trial_Class *tcp;

	tcp = PICK_TRIAL_CLASS("");

	if( tcp == NULL ) return;
	if( no_data(QSP_ARG  "t_wanal") ) return;

	w_analyse(QSP_ARG  tcp);
	w_tersout(QSP_ARG  tcp);
}

static COMMAND_FUNC( t_danal )
{
	Trial_Class *tcp;

	tcp = PICK_TRIAL_CLASS("");

	if( tcp == NULL ) return;
	if( no_data(QSP_ARG  "t_danal") ) return;

	analyse(QSP_ARG  tcp);
	tersout(QSP_ARG  tcp);
}

static COMMAND_FUNC( wanal )
{
	Trial_Class *tcp;

	tcp = PICK_TRIAL_CLASS("");

	if( tcp == NULL ) return;
	if( no_data(QSP_ARG  "wanal") ) return;

	w_analyse(QSP_ARG  tcp);
	weibull_out(QSP_ARG  tcp);
}

static COMMAND_FUNC( danal )
{
	Trial_Class *tcp;

	tcp = PICK_TRIAL_CLASS("");

	if( tcp == NULL ) return;
	if( no_data(QSP_ARG  "danal") ) return;

	analyse(QSP_ARG  tcp);
	longout(QSP_ARG  tcp);
}

static COMMAND_FUNC( setfc ) { set_fcflag( ASKIF("do analysis relative to 50% chance") ); }

static COMMAND_FUNC( do_set_chance_rate )
{
	set_chance_rate( HOW_MUCH("Probability of correct response due to guessing") );
}

#ifdef FUBAR
static COMMAND_FUNC( setcl )
{
	//classno=(int)HOW_MANY("index of class of interest");
	curr_tcp = PICK_TRIAL_CLASS("name of class of interest");
	if( curr_tcp == NULL ) return;

	if( no_data(QSP_ARG  "setcl") ) return;

#ifdef FOOBAR
	if( classno < 0 || classno >= n_have_classes ){
		sprintf(ERROR_STRING,
	"Ridiculous selection %d, should be in range 0 to %d (inclusive)",
			classno,n_have_classes-1);
		WARN(ERROR_STRING);
		classno=0;
	}
#endif // FOOBAR
}
#endif // FUBAR

static COMMAND_FUNC( _split )
{
	int wu;
	Trial_Class *tcp;

	tcp = PICK_TRIAL_CLASS("");

	wu = ASKIF("retain upper half");

	if( tcp == NULL ) return;
	if( no_data(QSP_ARG  "_split") ) return;

	split(QSP_ARG  tcp,wu);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(ogive_menu,s,f,h)

MENU_BEGIN(ogive)
ADD_CMD( analyse,	danal,			analyse data )
ADD_CMD( summarize,	t_danal,		analyse data (terse output) )
//ADD_CMD( class,		setcl,			select new stimulus class )
ADD_CMD( 2afc,		setfc,			set forced-choice flag )
ADD_CMD( chance_rate, 	do_set_chance_rate,	specify chance P(correct) )
ADD_CMD( constrain,	constrain_slope,	constrain regression slope )
MENU_END(ogive)

static COMMAND_FUNC( do_ogive )
{
	PUSH_MENU(ogive);
}

static COMMAND_FUNC( seter )
{
	double er;

	er=HOW_MUCH("finger error rate");
	w_set_error_rate(er);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(weibull_menu,s,f,h)

MENU_BEGIN(weibull)
ADD_CMD( analyse,	wanal,		analyse data )
ADD_CMD( summarize,	t_wanal,	analyse data (terse output) )
//ADD_CMD( class,		setcl,		select new stimulus class )
ADD_CMD( 2afc,		setfc,		set forced-choice flag )
ADD_CMD( error_rate,	seter,		specify finger-error rate )
MENU_END(weibull)


static COMMAND_FUNC( do_weibull )
{
	PUSH_MENU(weibull);
}

static COMMAND_FUNC( do_pnt_bars )
{
	FILE *fp;
	Trial_Class *tcp;

	tcp = PICK_TRIAL_CLASS("");
	fp=TRYNICE( NAMEOF("output file"), "w" );
	if( fp == NULL || tcp == NULL ) return;

	pnt_bars( QSP_ARG  fp, tcp );
}

static COMMAND_FUNC( do_xv_xform )
{
	const char *s;

	s=NAMEOF("dm expression string for x-value transformation");
	set_xval_xform(s);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(lookit_menu,s,f,h)

MENU_BEGIN(lookit)
ADD_CMD( read,		do_read_data,	read new data file )
ADD_CMD( xform,		do_xv_xform,	set automatic x-value transformation )
ADD_CMD( print,		do_print_raw,	print raw data )
//ADD_CMD( class,		setcl,		select new stimulus class )
ADD_CMD( plotprint,	pntgrph,	print data for plotting )
ADD_CMD( errbars,	do_pnt_bars,	print psychometric function with error bars )
ADD_CMD( ogive,		do_ogive,	do fits with to ogive )
ADD_CMD( weibull,	do_weibull,	do fits to weibull function )
ADD_CMD( split,		_split,		split data at zeroes )
ADD_CMD( lump,		lump,		lump data conditions )
#ifdef QUIK
ADD_CMD( Quick,		prquic,		print data in QUICK format )
#endif /* QUIK */
MENU_END(lookit)

COMMAND_FUNC( lookmenu )
{
	PUSH_MENU(lookit);
}

