#include "quip_config.h"

/* This file is kind of obsolete...
 * We don't really use parameter tables any more.
 * This is a legacy system derived from old
 * Kropfl code from Bell Labs.
 */

#include <stdio.h>
#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "query_prot.h"
#include "nexpr.h"
#include "history.h"

#define UPARAM
#include "param_api.h"

static void	wtprms(QSP_ARG_DECL  FILE *fp,Param *p);
static void	showparm(QSP_ARG_DECL  Param *);
static void	pparam(QSP_ARG_DECL  Param *,FILE *);
static void	getparm(QSP_ARG_DECL  Param *);
static void	getarrp(QSP_ARG_DECL  Param *);
static void	getarrel(QSP_ARG_DECL  Param *,int);
static int	get_pval(QSP_ARG_DECL  const char *,Param *);

static COMMAND_FUNC( do_disp_prms );
static COMMAND_FUNC( do_chng_one );
static COMMAND_FUNC( do_prm_rd );
static COMMAND_FUNC( do_prm_wt );

static Param *theptbl;
static const char *badpstr="bad paramter type";
float fnum;


static void showparm( QSP_ARG_DECL  Param* p ) /** show p on the screen */
{
#ifndef NO_STDIO
	pparam(QSP_ARG  p,stderr);
#else
	warn("Can't show parameter without stdio!?");
#endif
}

static void pparam(QSP_ARG_DECL  Param* p,FILE* fp)	/** print parameter pted to by p to file fp */
{
	int i;
	int *rip; float *rfp;
	short *rsp;
	int nelts;

	if( IS_ARRAY_PARAM(p) ){
		nelts=(int)(p->p_type & NELTMASK);
		if( IS_INT_ARRAY_PARAM(p) ){
			rip=p->u.ip;
			for(i=0;i<nelts;i++) fprintf(fp,
				"%s[%d]\t%d\n",p->p_name,i,*rip++);
		} else if( IS_FLOAT_ARRAY_PARAM(p) ){
			rfp=p->u.fp;
			for(i=0;i<nelts;i++) fprintf(fp,
				"%s[%d]\t%f\n",p->p_name,i,*rfp++);
		} else if( IS_SHORT_ARRAY_PARAM(p) ){
			rsp=p->u.sp;
			for(i=0;i<nelts;i++) fprintf(fp,
				"%s[%d]\t%d\n",p->p_name,i,*rsp++);
		}
	}
	else if( IS_INT_PARAM(p) )
		fprintf(fp,"%s\t%d\n",p->p_name,*p->u.ip );
	else if( IS_FLOAT_PARAM(p) )
		fprintf(fp,"%s\t%f\n",p->p_name,*p->u.fp);
	else if( IS_SHORT_PARAM(p) )
		fprintf(fp,"%s\t%d\n",p->p_name,*p->u.sp);
	else if( IS_STRING_PARAM(p) ){
		fprintf(fp,"%s\t\"%s\"\n",p->p_name,p->u.strp);
	}
	else {
		sprintf(ERROR_STRING,"parameter type:  0x%x",p->p_type);
		warn(ERROR_STRING);
		error1(badpstr);
	}
}

static void getparm(QSP_ARG_DECL  Param *p)
{
	if( IS_ARRAY_PARAM(p) ) getarrp(QSP_ARG  p);

	else if( IS_FLOAT_PARAM(p) )
		*p->u.fp = fnum= (float)how_much( p->p_comment );
	else if( IS_SHORT_PARAM(p) )
		*p->u.sp = (short)how_many( p->p_comment );
	else if( IS_INT_PARAM(p) )
		*p->u.ip = (int)how_many( p->p_comment );
	else if( IS_STRING_PARAM(p) )
		strcpy( p->u.strp, nameof(p->p_comment) );

	else error1(badpstr);
}

static void getarrp(QSP_ARG_DECL  Param *p)
{
	int i;
	int nelts;

	nelts=(int)(p->p_type&NELTMASK);
	for(i=0;i<nelts;i++) getarrel(QSP_ARG  p,i);
}

static void getarrel(QSP_ARG_DECL  Param *p,int pindex)
{
	Param tprm;
	char pmpt[160];
	int max;

	tprm.p_comment=(&pmpt[0]);
	tprm.p_name=p->p_name;
	max=(int)((p->p_type & NELTMASK) + 1);
	if( pindex < 0 || pindex >= max ){
		sprintf(ERROR_STRING,"legal indices:0 to %d",max);
		warn(ERROR_STRING);
		return;
	}
	if( IS_FLOAT_ARRAY_PARAM(p) ){
		tprm.p_type=FLOATP;
		tprm.u.fp=p->u.fp+pindex;
	} else if( IS_SHORT_ARRAY_PARAM(p) ){
		tprm.p_type=SHORTP;
		tprm.u.sp=p->u.sp+pindex;
	} else if( IS_INT_ARRAY_PARAM(p) ){
		tprm.p_type=INTP;
		tprm.u.ip=p->u.ip+pindex;
	}
	switch(pindex){
		case 0: sprintf(pmpt,"(1st) "); break;
		case 1: sprintf(pmpt,"(2nd) "); break;
		case 2: sprintf(pmpt,"(3rd) "); break;
		default: sprintf(pmpt,"(%dth) ",pindex+1); break;
	}
	strcat(pmpt,p->p_comment);
	getparm(QSP_ARG   &tprm );
}


/* show all parameters */

static COMMAND_FUNC( do_disp_prms )
{
	Param *p;

	p=theptbl;
	while( p->p_type != NULL_P_TYPE )
		showparm(QSP_ARG  p++);
}

#define PNAME_PMPT	"parameter name"

static COMMAND_FUNC( do_chng_one )
{
	Param *p;
	const char **pnlist;
	int nlist=0;
	int i=0;
	const char *s;

	p=theptbl;

	/* count the number of parameters */
	while( p->p_type != NULL_P_TYPE ) {
		nlist++;
		p++;
	}

	pnlist = (const char **) getbuf( (nlist+1) * sizeof(char *) );
	if( pnlist == NULL ) mem_err("do_chng_one");

#ifdef HAVE_HISTORY

	if( intractive(SINGLE_QSP_ARG) && IS_TRACKING_HISTORY(THIS_QSP) ){
		List *lp;
		Node *np;

		lp = new_list();
		for(i=0;i<nlist;i++){
			pnlist[i] = theptbl[i].p_name;
			np = mk_node(&theptbl[i]);
			addTail(lp,np);
		}
		pnlist[i]="all";
		np = mk_node(&pnlist[i]);
		addTail(lp,np);

		if( intractive(SINGLE_QSP_ARG) ){
			const char *pline;
			pline = format_prompt(PROMPT_FORMAT, PNAME_PMPT);
			new_defs(pline);		/* is this needed? */
			init_hist_from_item_list(PNAME_PMPT,lp);
		}

		dellist(lp);
	}

#else /* ! HAVE_HISTORY */
	for(i=0;i<nlist;i++)
		pnlist[i] = theptbl[i].p_name;
#endif /* ! HAVE_HISTORY */

	s=nameof(PNAME_PMPT);
	if( !strcmp(s,"all") ){
		p=theptbl;
		while( p->p_type != NULL_P_TYPE ) {
			getparm(QSP_ARG  p);
			showparm(QSP_ARG  p);
			p++;
		}
		return;
	} else if( get_pval(QSP_ARG  s,theptbl) == -1 ){
		sprintf(ERROR_STRING,"Unknown parameter \"%s\"",s);
		warn(ERROR_STRING);
	}
}

/* ifarr("p[3]") returns 3, and has the following side effects:
 */

#define MAX_INDEX_STRING_LEN	16

static index_t ifarr(QSP_ARG_DECL  const char *s)	/* return index or NOTARR */
{
	Typed_Scalar *tsp;
	char index_string[MAX_INDEX_STRING_LEN];
	int i;

/* need to fix up const stuff */

	while( *s && *s != '[' ){
		s++;	/* skip until end or opening brace */
	}
	if( *s==0 ) return(NOTARR);
	s++;		// skip opening brace

	/* changed to use expression parser jbm 12-10-92 */

	i=0;
	while( *s && *s!=']' ){
		if( i >= (MAX_INDEX_STRING_LEN-1)){
			sprintf(ERROR_STRING,"ifarr:  index string too long (%d chars max)",
				MAX_INDEX_STRING_LEN-1);
			warn(ERROR_STRING);
			return(NOTARR);
		}
		index_string[i++] = *s++;
	}
	index_string[i]=0;

	if( *s != ']' ) {
		warn("ifarr:  no closing brace for index");
		return(NOTARR);
	}
	tsp = pexpr(index_string );
	i =  index_for_scalar( tsp );
	RELEASE_SCALAR(tsp);

	return(i);
}

/* This does a lot of in-place monkeying... */

static int get_pval(QSP_ARG_DECL  const char *name,Param* ptable)
{
	index_t pindex;

	pindex=ifarr(QSP_ARG name);
	while( ptable->p_type != NULL_P_TYPE ){
		if( !strcmp(name,ptable->p_name) ) {
			if( pindex==NOTARR ) getparm(QSP_ARG  ptable);
			else getarrel(QSP_ARG  ptable,pindex);
			return(0);
		} else ptable++;
	}
	sprintf(ERROR_STRING,"no parameter \"%s\"",name);
	warn(ERROR_STRING);
	return(-1);
}

static const char *pfstr="parameter file";

static void wtprms(QSP_ARG_DECL  FILE *fp,Param *p)	/** write parameters to file */
{
	Param *ptr;

	ptr=p;
	while( ptr->p_type != NULL_P_TYPE ){
		pparam(QSP_ARG  ptr,fp);
		ptr++;
	}
	fclose(fp);
}


static void rdprms(QSP_ARG_DECL  Param *p,FILE* fp, const char *filename)
{
	int level;
	const char *s;

	redir(fp, filename);
	level=qs_level(SINGLE_QSP_ARG);
	do {
		s=nameof("name of parameter");
		if( get_pval(QSP_ARG  s,p) == -1 )
			warn("error getting parameter value");
		/* lookahead word should decrement qlevel at EOF */
		lookahead(SINGLE_QSP_ARG);
	} while( level == qs_level(SINGLE_QSP_ARG) );
}


static COMMAND_FUNC( do_prm_rd )
{
	FILE *fp;
	const char *s;

	s=nameof(pfstr);
	fp=try_open( s,"r" );
	if( !fp ) return;
	rdprms(QSP_ARG  theptbl,fp,s);
}

static COMMAND_FUNC( do_prm_wt )
{
	FILE *fp;
	fp=try_nice(nameof(pfstr),"w");
	if( fp== NULL ) return;
	wtprms(QSP_ARG  fp,theptbl);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(change_parameter_menu,s,f,h)

MENU_BEGIN(change_parameter)
ADD_CMD( change,	do_chng_one,	change parameter )
ADD_CMD( display,	do_disp_prms,	display parameters )
ADD_CMD( read,		do_prm_rd,	read parameters from a file )
ADD_CMD( write,		do_prm_wt,	write parameters to a file )
MENU_END(change_parameter)

COMMAND_FUNC( prm_menu )
{
	CHECK_AND_PUSH_MENU(change_parameter);
}

void chngp(QSP_ARG_DECL  Param *p) /** display and modify parameters */
{
	theptbl=p;
	prm_menu(SINGLE_QSP_ARG);
}

