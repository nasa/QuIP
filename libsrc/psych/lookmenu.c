#include "quip_config.h"

char VersionId_psych_lookmenu[] = QUIP_VERSION_STRING;

#include "quip_config.h"
#include <stdio.h>
#include "stc.h"
#include "query.h"
#include "version.h"
#include "debug.h"		/* verbose */

static int classno=0;
static int n_have_classes=0;

/* local functions */
static void pntcurve(QSP_ARG_DECL  FILE *fp, int cl);
static COMMAND_FUNC( drdfil );
static int no_data(VOID);
#ifdef QUIK
static COMMAND_FUNC( prquic );
#endif /* QUIK */
static COMMAND_FUNC( pntraw );
static COMMAND_FUNC( pntgrph );
static COMMAND_FUNC( t_wanal );
static COMMAND_FUNC( t_danal );
static COMMAND_FUNC( wanal );
static COMMAND_FUNC( danal );
static COMMAND_FUNC( setfc );
static COMMAND_FUNC( do_set_chance_rate );
static COMMAND_FUNC( setcl );
static COMMAND_FUNC( _split );
static COMMAND_FUNC( do_ogive );
static COMMAND_FUNC( seter );
static COMMAND_FUNC( do_weibull );
static COMMAND_FUNC( do_pnt_bars );
static COMMAND_FUNC( do_xv_xform );

static COMMAND_FUNC( drdfil )	/** read a data file */
{
	FILE *fp;
	const char *filename;

	filename=NAMEOF("data file");
	fp=TRY_OPEN( filename, "r" );
	if( !fp ) return;

	/* We used to clear the data tables here,
	 * but now they are dynamically allocated
	 * and cleared at that time...
	 */

	/* clear old classes */
	delcnds(SINGLE_QSP_ARG);

	classno=0;
	n_have_classes=0;
	if( rddata(QSP_ARG  fp) != 0 ){
		fclose(fp);
		sprintf(error_string,"Error reading file %s",filename);
		WARN(error_string);
		return;
	}
	fclose(fp);
	n_have_classes = eltcount(class_list(SINGLE_QSP_ARG));

	if( verbose ){
		sprintf(error_string,"File %s read, %d classes, %d x-values",
			filename,n_have_classes,_nvals);
		advise(error_string);
	}
}

static int no_data()
{
	if( n_have_classes <= 0 ){
		NWARN("must read a data file before this operation!");
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

	fp=TRYNICE( NAMEOF("quic file"), "w");
	if( !fp ) return;

	in_db = ASKIF("transform x values to decibels");

	if( no_data() ) return;

	fprintf(fp,"%c\n",004);		/* EOT */

	pntquic(fp,classno,in_db);

	fclose(fp);
}
#endif /* QUIK */



static COMMAND_FUNC( pntraw )
{
	if( no_data() ) return;
	pntdata(QSP_ARG  classno);
}

static void pntcurve(QSP_ARG_DECL  FILE *fp, int cl)
{
        int j;
        Data_Tbl *dp;
	Trial_Class *clp;

	clp=index_class(QSP_ARG  cl);
	dp=clp->cl_dtp;
	for(j=0;j<_nvals;j++){
		if( dp->d_data[j].ntotal > 0 ){
			fprintf(fp,"%f\t", xval_array[ j ]);
			fprintf(fp,"%f\n",(double) dp->d_data[j].ncorr /
				(double) dp->d_data[j].ntotal );
		}
	}
	fclose(fp);
}

static COMMAND_FUNC( pntgrph )
{
	FILE *fp;

	fp=TRYNICE( NAMEOF("output file"), "w" );
	if( !fp ) return;

	if( no_data() ) return;

	pntcurve(QSP_ARG  fp,classno);
}


static COMMAND_FUNC( t_wanal )
{

	if( no_data() ) return;

	w_analyse(QSP_ARG  classno);
	w_tersout(classno);
}

static COMMAND_FUNC( t_danal )
{

	if( no_data() ) return;

	analyse(QSP_ARG  classno);
	tersout(classno);
}

static COMMAND_FUNC( wanal )
{

	if( no_data() ) return;

	w_analyse(QSP_ARG  classno);
	weibull_out(classno);
}

static COMMAND_FUNC( danal )
{

	if( no_data() ) return;

	analyse(QSP_ARG  classno);
	longout(classno);
}

static COMMAND_FUNC( setfc ) { set_fcflag( ASKIF("do analysis relative to 50% chance") ); }

static COMMAND_FUNC( do_set_chance_rate )
{
	set_chance_rate( HOW_MUCH("Probability of correct response due to guessing") );
}

static COMMAND_FUNC( setcl )
{
	classno=(int)HOW_MANY("index of class of interest");

	if( no_data() ) return;

	if( classno < 0 || classno >= n_have_classes ){
		sprintf(error_string,
	"Ridiculous selection %d, should be in range 0 to %d (inclusive)",
			classno,n_have_classes-1);
		WARN(error_string);
		classno=0;
	}
}

static COMMAND_FUNC( _split )
{
	int wu;

	wu = ASKIF("retain upper half");

	if( no_data() ) return;

	split(QSP_ARG  classno,wu);
}


static Command og_ctbl[]={
{ "analyse",	danal,			"analyse data"			},
{ "summarize",	t_danal,		"analyse data (terse output)"	},
{ "class",	setcl,			"select new stimulus class"	},
{ "2afc",	setfc,			"set forced-choice flag"	},
{ "chance_rate", do_set_chance_rate,	"specify chance P(correct)"	},
{ "constrain",	constrain_slope,	"constrain regression slope"	},
{ "quit",	popcmd,			"quit"				},
{ NULL_COMMAND								}
};

static COMMAND_FUNC( do_ogive )
{
	PUSHCMD(og_ctbl,"ogive");
}

static COMMAND_FUNC( seter )
{
	double er;

	er=HOW_MUCH("finger error rate");
	w_set_error_rate(er);
}

static Command weib_ctbl[]={
{ "analyse",	wanal,		"analyse data"				},
{ "summarize",	t_wanal,	"analyse data (terse output)"		},
{ "class",	setcl,		"select new stimulus class"		},
{ "2afc",	setfc,		"set forced-choice flag"		},
{ "error_rate",	seter,		"specify finger-error rate"		},
{ "quit",	popcmd,		"quit"					},
{ NULL_COMMAND								}
};


static COMMAND_FUNC( do_weibull )
{
	PUSHCMD(weib_ctbl,"weibull");
}

static COMMAND_FUNC( do_pnt_bars )
{
	FILE *fp;

	fp=TRYNICE( NAMEOF("output file"), "w" );
	if( !fp ) return;

	pnt_bars( QSP_ARG  fp, classno );
}

static COMMAND_FUNC( do_xv_xform )
{
	const char *s;

	s=NAMEOF("dm expression string for x-value transformation");
	set_xval_xform(s);
}

static Command lukctbl[]={
{ "read",	drdfil,		"read new data file"			},
{ "xform",	do_xv_xform,	"set automatic x-value transformation"	},
{ "print",	pntraw,		"print raw data"			},
{ "class",	setcl,		"select new stimulus class"		},
{ "plotprint",	pntgrph,	"print data for plotting"		},
{ "errbars",	do_pnt_bars,	"print psychometric function with error bars"},
{ "ogive",	do_ogive,	"do fits with to ogive"			},
{ "weibull",	do_weibull,	"do fits to weibull function"		},
{ "split",	_split,		"split data at zeroes"			},
{ "lump",	lump,		"lump data conditions"			},
#ifdef QUIK
{ "Quick",	prquic,		"print data in QUICK format"		},
#endif /* QUIK */
{ "quit",	popcmd,		"quit"					},
{ NULL_COMMAND								}
};

COMMAND_FUNC( lookmenu )
{
	static int inited=0;

	if( !inited ){
		auto_version(QSP_ARG  "CSTEPIT","VersionId_cstepit");
		inited=1;
	}
	PUSHCMD(lukctbl,"lookit");
}

