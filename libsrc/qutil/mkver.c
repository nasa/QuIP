#include "quip_config.h"

char VersionId_qutil_mkver[] = QUIP_VERSION_STRING;

#include <stdio.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#include "items.h"
#include "version.h"
#include "node.h"
#include "query.h"

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef DEBUG
static u_long verdebug=0;
#endif /* DEBUG */
static List *file_version_lp=NO_LIST;
static const char *version_string=NULL;	/* git-style commit version */
static List *	ver_req_list=NO_LIST;

static char *sym_file_name=NULL;

/* local prototypes */
static int parse_date_string(Date *date_p, const char *datestring);
static List *version_list(SINGLE_QSP_ARG_DECL);
static Version * creat_version(QSP_ARG_DECL  const char *);
static const char *info_string(FileVersion *);
static void list_version_files(QSP_ARG_DECL  Version *);
static int list_one_file(Version *vp,FileVersion *fvp);
static int date_cmp(void *,void *);
static Date get_lib_date(Version *);
static int get_date_string(char **strp,const char *s);
#ifdef SCCS
static int parse_sgi_date(char *cp,char **strp);
#endif
static int get_version_date(Date *date_p,const char *s);
static void make_sym_file(void);
static void process_version_request(QSP_ARG_DECL  const char *module_name,const char *prefix);
static void note_version_request(const char *module_name,const char *prefix);


ITEM_INTERFACE_DECLARATIONS(Version,version)

static List *version_list(SINGLE_QSP_ARG_DECL)
{
	if( version_itp == NO_ITEM_TYPE ) version_init(SINGLE_QSP_ARG);
	return(item_list(QSP_ARG  version_itp));
}

#ifdef HAVE_HISTORY
void init_version_hist(QSP_ARG_DECL  const char *prompt)
{
	if( version_itp == NO_ITEM_TYPE ) version_init(SINGLE_QSP_ARG);
	init_item_hist(QSP_ARG  version_itp,prompt);
}
#endif /* HAVE_HISTORY */


static Version * creat_version(QSP_ARG_DECL  const char *name)
{
	Version *vp;

	vp=new_version(QSP_ARG  name);
	vp->file_list = new_list();
	return(vp);
}

#ifdef FOOBAR

// Old DOS version...

/*
 * mkver arguments are:
 * name 		= Name of toolkit, library, etc.
 * fvp			= Array of version string pointers
 * entries		= # of string pointers
 */

void mkver (char * name, FileVersion* fvp, int n_entries)
{
	Version *vp;
	Node *np;
	int i;

#ifdef DEBUG
	if( verdebug == 0 )
		verdebug=add_debug_module("version");
#endif /* DEBUG */

	vp = version_of(name);

	if( vp == NULL ){		/* first time, create */
		vp = creat_version(name);
		for(i=0;i<n_entries;i++){
			np = mk_node(fvp++);
			addTail(vp->file_list,np);
		}
	}
}

#else		/* ! FOOBAR */

/* get_progfile() and get_symname() are also used in bind.c */

const char *get_progfile()
{
	char *path;
	register char *s;
	const char *pn;
	static char dirbuf[256];

	pn=tell_progname();	/* argv[0] */

	/* here we used to just check for an absolute pathname */
	/* if( *pn == '/' ) return(pn); */

	/* Now we return unchanged any name that contains a slash */

	if( strchr(pn,'/') != NULL ) return(pn);

	path=getenv("PATH");
	if( path == (char *) NULL ){
		NWARN("get_progfile:  no PATH in environment");
		return(NULL);
	}

	/* now parse the path */

	while( *path ){
		struct stat sbuf;

		if( *path==':' ) path++;
		s=dirbuf;
		while( *path && *path != ':' )
			*s++ = *path++;
		*s=0;
		strcat(dirbuf,"/");
		strcat(dirbuf,pn);

		/* now try to open this one */

		if( stat( dirbuf, &sbuf ) != 0 )
			continue;

		/* file exists; check for a.out */

		/* is reg. file? */
		if( (sbuf.st_mode & S_IFREG) == 0 )
			continue;

		return(dirbuf);
	}
	return(NULL);
}

const char * get_symname()
{
	static const char *sym_name=NULL, *prog_file=NULL;

	if( sym_name==NULL ){
		if( prog_file==NULL ) prog_file=get_progfile();
		if( prog_file == NULL )
			NWARN("Can't determine name of program file");
		sym_name=prog_file;
		if( verbose ){
			sprintf(msg_str,
				"searching for symbols in file \"%s\"",
				sym_name);
			prt_msg(msg_str);
		}
	}
	return(sym_name);
}

static void make_sym_file()
{
	char cmd[128];
	static char filename[128];
	const char *s;

	s=get_symname();
	if( s == NULL ) return;

#define TMPSTEM		"/tmp/sym"

	sprintf(filename,"%s%ld",TMPSTEM,(u_long)getpid());
	sym_file_name = filename;

#ifdef SCCS
#define VERSION_ID_STRING	"SccsId"
#else
#define VERSION_ID_STRING	"VersionId"
#endif

#ifdef SGI
	sprintf(cmd,"nm -B %s | egrep %s > %s",s,VERSION_ID_STRING,filename);
#else
	sprintf(cmd,"nm %s | egrep %s > %s",s,VERSION_ID_STRING,filename);
#endif /* SGI */

	system(cmd);
}

static void process_version_request(QSP_ARG_DECL  const char *module_name,const char *prefix)
{
	u_long addr;
	char typ[4];
	char symname[128];
	char cmd[128];
	FILE *fp;
	Version *vp;
	FileVersion *fvp;

	vp = version_of(QSP_ARG  module_name);

	if( vp==NO_VERSION ) {
		vp=creat_version(QSP_ARG  module_name);
		if( vp==NO_VERSION )  return;
	}

	if( verbose ){
		sprintf(msg_str,
			"Extracting version information for module %s",
			module_name);
		prt_msg(msg_str);
	}

	if( sym_file_name == NULL )
		make_sym_file();

#ifdef HAVE_CUDA
	/* .o files made by nvcc have another symbol which we need to ignore */
	sprintf(cmd,"egrep %s %s | grep -v _GLOBAL__I_",prefix,sym_file_name);
#else /* ! HAVE_CUDA */
	sprintf(cmd,"egrep %s %s",prefix,sym_file_name);
#endif /* ! HAVE_CUDA */
	fp=popen(cmd,"r");
	if( !fp ){
		NWARN("process_version_request:  popen error");
		return;
	}

	/*
	 * on SUN lines of nm output are :
	 * addr type_code symbol_name
	 *
	 * this is also true on SGI if you call nm with -B
	 */

	while( fscanf(fp,"%lx %s %s",&addr,typ,symname) == 3 ){
		Node *np;
#ifdef DEBUG
if( debug & verdebug ){
sprintf(msg_str,"scanned symbol %s at 0x%lx",symname,addr);
prt_msg(msg_str);
}
#endif /* DEBUG */
		fvp = (FileVersion *) getbuf(sizeof(*fvp));
		if( fvp == NULL ) mem_err("process_version_request");
		fvp->vers_string = (char *) addr;
		fvp->vers_symname = savestr(symname);
		np = mk_node(fvp);
		addTail(vp->file_list,np);
	}
	pclose(fp);
} /* end process_version_request() */

static void load_all_versions(void)
{
	u_long addr;
	char typ[4];
	char symname[128];
	char cmd[128];
	FILE *fp;
	FileVersion *fvp;

	if( sym_file_name == NULL )
		make_sym_file();

#ifdef HAVE_CUDA
	/* .o files made by nvcc have another symbol which we need to ignore */
	sprintf(cmd,"egrep VersionId %s | grep -v _GLOBAL__I_",sym_file_name);
#else /* ! HAVE_CUDA */
	sprintf(cmd,"egrep VersionId %s",sym_file_name);
#endif /* ! HAVE_CUDA */
	fp=popen(cmd,"r");
	if( !fp ){
		NWARN("load_all_versions:  popen error");
		return;
	}

	/*
	 * on SUN lines of nm output are :
	 * addr type_code symbol_name
	 *
	 * this is also true on SGI if you call nm with -B
	 */

	while( fscanf(fp,"%lx %s %s",&addr,typ,symname) == 3 ){
		Node *np;
#ifdef DEBUG
if( debug & verdebug ){
sprintf(msg_str,"scanned symbol %s at 0x%lx",symname,addr);
prt_msg(msg_str);
}
#endif /* DEBUG */
		fvp = (FileVersion *) getbuf(sizeof(*fvp));
		if( fvp == NULL ) mem_err("process_version_request");
		fvp->vers_string = (char *) addr;
		fvp->vers_symname = savestr(symname);
#ifdef CAUTIOUS
		/* If an error is caught here, it generally implies
		 * a makefile problem or an incorrect build...
		 */
		if( version_string == NULL )
			version_string = fvp->vers_string;
		else if( strcmp(fvp->vers_string,version_string) ){
			sprintf(DEFAULT_ERROR_STRING,
	"CAUTIOUS:  Version string \"%s\" from %s does not match global version \"%s\"!?",
				fvp->vers_string,fvp->vers_symname,
				version_string);
			NWARN(DEFAULT_ERROR_STRING);
		}
#endif /* CAUTIOUS */

		np = mk_node(fvp);
		if( file_version_lp == NO_LIST )
			file_version_lp = new_list();
		addTail(file_version_lp,np);
	}
	pclose(fp);
} /* end load_all_versions */

static void note_version_request(const char *module_name,const char *prefix)
{
	Node *np;
	Version_Request *vrp;

	/* first make sure that this version has not already been requested */
#ifdef CAUTIOUS
	if( ver_req_list != NO_LIST ){
		np=ver_req_list->l_head;
		while(np!=NO_NODE){
			vrp = (Version_Request *)np->n_data;
			if( !strcmp(vrp->module_name,module_name) ){
				sprintf(DEFAULT_ERROR_STRING,
			"CAUTIOUS:  note_version_request:  module %s already requested!?",
					module_name);
				NWARN(DEFAULT_ERROR_STRING);
				return;
			}
			np=np->n_next;
		}
	}
#endif /* CAUTIOUS */

	vrp = (Version_Request *) getbuf(sizeof(*vrp));
	if( vrp == NULL ) mem_err("note_version_request");
	vrp->module_name = module_name;
	vrp->id_string = prefix;
	np=mk_node(vrp);
	if( ver_req_list == NO_LIST )
		ver_req_list = new_list();
	addTail(ver_req_list,np);
}

void auto_version(QSP_ARG_DECL  const char *module_name,const char *prefix)
{
	Version *vp;

#ifdef DEBUG
	if( verdebug == 0 )
		verdebug=add_debug_module(QSP_ARG  "version");
#endif /* DEBUG */

	vp = version_of(QSP_ARG  module_name);

	if( vp!=NO_VERSION ) {
		WARN("duplicate call to auto_version");
		sprintf(ERROR_STRING,
			"module %s is already loaded",module_name);
		WARN(ERROR_STRING);
		return;
	}

	/*
	 * Don't bother to actually do the work until the user wants
	 * the information.  Just save the information for later.
	 */

	note_version_request(module_name,prefix);
}

void get_deferred_requests(SINGLE_QSP_ARG_DECL)
{
	Node *np;
	Version_Request *vrp;

	if( ver_req_list == NO_LIST ) return;

	np=ver_req_list->l_head;
	while( np != NO_NODE ){
		vrp=(Version_Request*) np->n_data;
		process_version_request(QSP_ARG  vrp->module_name,vrp->id_string);
		np=np->n_next;
	}
	dellist(ver_req_list);
	ver_req_list=NO_LIST;

	if( sym_file_name != NULL ){
		if( unlink(sym_file_name) < 0 )
			tell_sys_error("get_deferred requests (unlink)");
		sym_file_name=NULL;
	}
}

#endif /* ! FOOBAR */

static const char *info_string(FileVersion *fvp)
{
	const char *s;

	s=fvp->vers_string;

#ifdef SCCS
	/* a typical version string looks like:
	 * @(#) file.c ver: 1.28 8/5/96			(sun sccs)
	 * @(#) file.c ver: 1.28 05 Aug 1996		(sgi sccs)
	 * OR
	 * #Z# #M# ver: #I# #G#
	 * where '#' has been substituted for '%'
	 * so that SCCS doesn't trash this comment!
	 */

	if (strchr(s, '%') != NULL) {
		sprintf(ERROR_STRING,"need to check in SCCS file %s!",
			fvp->vers_symname);
#ifdef MAC
		NADVISE(ERROR_STRING);
#else /* ! MAC */
		NWARN(ERROR_STRING);
#endif /* ! MAC */
		s = NULL;
	} else
		s += 4;

#else /* ! SCCS (now CVS) */

	/* do nothing until we figure out what is going on... */

#endif
	return(s);
}

#ifdef SCCS
static int parse_sgi_date(char *cp,char **strp)
{
	char filename[64];
	char vstring[16];
	char dname[32];
	char mname[32];
	char yname[32];
	int d,m,y;

	if( sscanf(cp,"%s ver: %s %s %s %s",filename,vstring,
		dname,mname,yname) != 5 ){
		NWARN("error scanning sgi-style version string");
		return(-1);;
	}
	/* now make the sgi date into a sun-style date */
	d=atoi(dname);
	switch(mname[0]){
		case 'J':
			if (mname[1]=='a') m=1;
			else if(mname[2]=='l') m=7;
			else m=6;
			break;
		case 'F': m=2; break;
		case 'M':
			if( mname[2]=='r' ) m=3;
			else m=5;
			break;
		case 'A':
			if( mname[1]=='p' ) m=4;
			else m=8;
			break;
		case 'S': m=9; break;
		case 'O': m=10; break;
		case 'N': m=11; break;
		case 'D': m=12; break;
		default:
			sprintf(ERROR_STRING,
			"unrecognized month string %s",mname);
			NWARN(ERROR_STRING);
			m=1;
			break;
	}
	y=atoi(yname);
	y-=1900;
	if( y>=100 ) y-=100;	/* plan for future!? */
#ifdef CAUTIOUS
	if( y<0 ){
		sprintf(ERROR_STRING,
		"bad year string %s!?",yname);
		NWARN(ERROR_STRING);
		return(-1);
	}
#endif /* CAUTIOUS */
	sprintf(*strp,"%d/%d/%d",m,d,y);
	return(0);
}
#endif

/* Some useful macros for parsing RCS version strings */

#ifndef SCCS

#ifdef CAUTIOUS

#define CHECK_RCS_STRING(ptr,msg)						\
		if( *ptr ==0 ) {						\
			sprintf(ERROR_STRING,"CAUTIOUS:  null RCS info string, %s", msg);	\
			NWARN(ERROR_STRING);					\
		}

#else

#define CHECK_RCS_STRING(ptr,msg)

#endif /* ! CAUTIOUS */


#ifdef CAUTIOUS
#define SKIP_PAST_CHAR(ptr,key,msg)						\
		while( *ptr != key && *ptr!=0 ) ptr++;				\
		if( *ptr ==0 ){							\
			sprintf(ERROR_STRING,					\
		"CAUTIOUS:  skip_past char:  bad RCS info string, %s",msg);	\
			NWARN(ERROR_STRING);					\
			sprintf(ERROR_STRING,"orig string:  \"%s\"",orig_string);	\
			NADVISE(ERROR_STRING);					\
			return(-1);						\
		}								\
		ptr++;	/* skip target char */
#else
#define SKIP_PAST_CHAR(ptr,key,msg)						\
		while( *ptr != key && *ptr!=0 ) ptr++;				\
		ptr++;	/* skip target char */
#endif /* ! CAUTIOUS */

#endif /* ! SCCS */



static int get_date_string(char **strp,const char* s)
{
#ifdef OLD_VERSION_TRACKING
	const char *orig_string;
#ifdef SCCS
	char filename[64];
	char vstring[16];
	char datestring[16];

	if( sscanf(s,"%s ver: %s %s",filename,vstring,datestring) != 3 )
		return(-1);
	if( strlen(datestring) == 2 ){	/* sgi */
		if( parse_sgi_date(s,strp) < 0 ) return(-1);
	} else strcpy(*strp,datestring);
	return(0);
#else
	char *targ;

	orig_string = s; 

	/* format of RCS versions strings is:
	 * "$RCSfile$ $Revision$ $Date$"
	 */

	SKIP_PAST_CHAR(s,'$',"looking for first dollar sign");
	SKIP_PAST_CHAR(s,'$',"looking for second dollar sign");
	SKIP_PAST_CHAR(s,'$',"looking for third dollar sign");
	SKIP_PAST_CHAR(s,'$',"looking for fourth dollar sign");
	SKIP_PAST_CHAR(s,'$',"looking for fifth dollar sign");
	SKIP_PAST_CHAR(s,' ',"looking for first space");

	targ = *strp;		/* not really a BUG, but could just pass targ instead of strp... */
	while( isdigit(*s) || *s=='/' || *s=='-' )
		*targ++ = *s++;
	*targ++ = ' ';
	SKIP_PAST_CHAR(s,' ',"looking for second space");
	while( isdigit(*s) || *s==':' )
		*targ++ = *s++;
	*targ++ = 0;
/*
sprintf(ERROR_STRING,"found date string '%s' in input string '%s'",
*strp,orig_string);
NADVISE(ERROR_STRING);
*/
	return(0);
#endif /* ! SCCS */
#else /* ! OLD_VERSION_TRACKING */

	/* New version string is version number followed by a tab,
	 * followed by the date.
	 */

	while( *s && *s != '\t' ) s++;
	if( *s != '\t' ) return(-1);
	s++;
	strcpy(*strp,s);
	return(0);

#endif /* ! OLD_VERSION_TRACKING */
}

static void list_version_files(QSP_ARG_DECL  Version *vp)
{
	List *lp;
	Node *np;
	FileVersion *fvp;

	lp = alpha_sort(QSP_ARG  vp->file_list);
	if( lp == NO_LIST ) {
		sprintf(ERROR_STRING,"Version database for module %s has no files!?",vp->lib_name);
		NWARN(ERROR_STRING);
		return;
	}
	np = lp->l_head;
	while(np!=NO_NODE){
		fvp = (FileVersion *) np->n_data;
		if( list_one_file(vp,fvp) < 0 ){
			sprintf(ERROR_STRING,
	"Error listing version information from address %s", fvp->vers_symname );
			NWARN(ERROR_STRING);
			if( fvp->vers_string != NULL ){
				sprintf(ERROR_STRING,"Version info:  %s",fvp->vers_string);
				NADVISE(ERROR_STRING);
			}
		}
		np=np->n_next;
	}
	dellist(lp);
}

static int list_one_file(Version *vp,FileVersion *fvp)
{
	const char *cp;
	char filename[64];
	char vstring[32];	/* was 16 before migration to git */
	char datestring[32];
	char *s;
	const char *orig_string;
	Date vdate;

	if ( (cp = info_string(fvp)) == NULL)
		return(-1);

	orig_string = cp;

#ifdef OLD_VERSION_TRACKING
#ifdef SCCS
	if( sscanf(cp,"%s ver: %s",filename,vstring) != 2 ){
		NWARN("error scanning version string");
		np=np->n_next;
		continue;
	}
	s=datestring;
	if( get_date_string(&s,cp) < 0 ){
		NWARN("error scanning date string");
		np=np->n_next;
		continue;
	}
#else /* ! SCCS */

	/* format of RCS versions strings is:
	 * "$RCSfile$ $Revision$ $Date$"
	 */

	/* process filename */

	/* seek first $ */
	SKIP_PAST_CHAR(cp,'$',"looking for next dollar sign")
	SKIP_PAST_CHAR(cp,' ',"looking for next space")

	s=filename;
	while(*cp!=',' && *cp!=0 )
		*s++ = *cp++;	/* BUG check for overrun */
	*s = 0;
	CHECK_RCS_STRING(cp,"expected opening dollar sign")
	/* skip opening dollar */
	SKIP_PAST_CHAR(cp,'$',"looking for another dollar")

	/* process version string */

	/* skip next dollar */
	SKIP_PAST_CHAR(cp,'$',"looking for yet another dollar")
	SKIP_PAST_CHAR(cp,' ',"looking for yet another space")

	s=vstring;
	while( isdigit(*cp) || *cp=='.' )
		*s++ = *cp++;
	*s=0;
	CHECK_RCS_STRING(cp,"expected first closing dollar sign")
	/* skip closing dollar */
	SKIP_PAST_CHAR(cp,'$',"looking for dollar #N")

	/* process date */
	SKIP_PAST_CHAR(cp,'$',"looking for dollar #N+1")
	SKIP_PAST_CHAR(cp,' ',"looking for space #N+1")
	s=datestring;
	while( isdigit(*cp) || *cp=='/' || *cp =='-' )
		*s++ = *cp++;
	*s++ = ' ';
	CHECK_RCS_STRING(cp,"expected space after date string")
	SKIP_PAST_CHAR(cp,' ',"looking for space N+2")
	while( isdigit(*cp) || *cp==':' )
		*s++ = *cp++;
	*s=0;

#endif /* ! SCCS */
#else /* ! OLD_VERSION_TRACKING */
	s=datestring;
	if( get_date_string(&s,cp) < 0 ){
		NWARN("error scanning date string");
	}
	strcpy(filename,fvp->vers_symname);
	s=vstring;
	while( *cp && *cp != '\t' )
		*s++ = *cp++;
	*s=0;
#endif /* ! OLD_VERSION_TRACKING */

	/* The date string is printed different ways by different versions of cvs, we parse and format
	 * here to make sure all are consistent.
	 */
	if( parse_date_string(&vdate,datestring) < 0 ){
		NWARN("list_one_file:  error parsing date string");
		return(-1);
	}

	if( vp != NULL ){
		sprintf(msg_str,"%-10s %-16s %-8s %4d/%02d/%02d", vp->lib_name,
			filename,vstring,vdate.yr,vdate.mo,vdate.day);
	} else {
		sprintf(msg_str,"%-16s %-8s %4d/%02d/%02d",
			filename,vstring,vdate.yr,vdate.mo,vdate.day);
	}
	prt_msg(msg_str);
	return(0);
}

static int date_cmp(void *p1,void* p2)
{
	Date *d1p, *d2p;

	d1p=(Date*) p1; 
	d2p=(Date*) p2;

	if( d1p->yr > d2p->yr ) return(1);
	else if( d1p->yr < d2p->yr ) return(-1);

	if( d1p->mo > d2p->mo ) return(1);
	else if( d1p->mo < d2p->mo ) return(-1);

	if( d1p->day > d2p->day ) return(1);
	else if( d1p->day < d2p->day ) return(-1);

	return(0);
}

static int parse_date_string(Date *date_p, const char *datestring)
{
#ifdef SCCS
	if( sscanf(datestring,"%d/%d/%d",&date_p->mo,&date_p->day,&date_p->yr)
		!= 3 ){
sprintf(ERROR_STRING,"Date string was \"%s\"",s);
NADVISE(ERROR_STRING);
		NWARN("Error scanning date string");
		return(-1);
	}
#else
	char delim1,delim2;

	/* CVS version strings use / on the older machines, but - on the newer 64 bit versions!? */
	if( sscanf(datestring,"%d%c%d%c%d",&date_p->yr,&delim1,&date_p->mo,&delim2,&date_p->day)
		!= 5 ){
sprintf(DEFAULT_ERROR_STRING,"Date string was \"%s\"",datestring);
NADVISE(DEFAULT_ERROR_STRING);
		NWARN("Error scanning date string");
		return(-1);
	}
	if( delim1 != '/' && delim1 != '-' ){
sprintf(DEFAULT_ERROR_STRING,"Date string was \"%s\"",datestring);
NADVISE(DEFAULT_ERROR_STRING);
		sprintf(DEFAULT_ERROR_STRING,"Date delimiter char 0x%x not hyphen or forward slash",delim1);
		NWARN(DEFAULT_ERROR_STRING);
	} else if( delim1 != delim2 ){
sprintf(DEFAULT_ERROR_STRING,"Date string was \"%s\"",datestring);
NADVISE(DEFAULT_ERROR_STRING);
		sprintf(DEFAULT_ERROR_STRING,"Second date delimiter 0x%x does not match first (0x%x)",
			delim2,delim1);
		NWARN(DEFAULT_ERROR_STRING);
sprintf(DEFAULT_ERROR_STRING,"year:  %d     month:  %d     day:  %d",date_p->yr,date_p->mo,date_p->day);
NADVISE(DEFAULT_ERROR_STRING);
	}

	/* BUG?  should we scan the time too? */

#endif /* ! SCCS */
	return(0);
}

static int get_version_date(Date *date_p,const char* s)
{
	char *ds, datestring[32];

	ds=datestring;
	get_date_string(&ds,s);

	if( parse_date_string(date_p,datestring) < 0 ){
		NWARN("error parsing date string");
		return(-1);
	}
	return(0);
}

static Date get_lib_date(Version *vp)
{
	const char *cp;
	Date d1,dmax;
	Node *np;
	FileVersion *fvp;

	dmax.mo = dmax.day = dmax.yr = 0;

	np=vp->file_list->l_head;
	while( np != NO_NODE ){
		fvp = (FileVersion *) np->n_data;
		if ( (cp = info_string(fvp)) == NULL) {
			np=np->n_next;
			continue;
		}
		if( get_version_date(&d1,cp) < 0 ){
			np=np->n_next;
			continue;
		}
		if( date_cmp(&d1,&dmax) > 0 )
			dmax=d1;
		np=np->n_next;
	}
	return(dmax);
}

void list_build_version(void)
{
	if( version_string == NULL )
		load_all_versions();

	/* PACKAGE_VERSION or VERSION ??? */
	sprintf(msg_str,"QuIP version %s (%s)",PACKAGE_VERSION,version_string);
	prt_msg(msg_str);
}

void list_libs(SINGLE_QSP_ARG_DECL)
{
	List *lp;
	Node *np;
	Version *vp;
	Date d;

	lp=version_list(SINGLE_QSP_ARG);
	lp=alpha_sort(QSP_ARG  lp);
	if( lp == NO_LIST ) return;
	np=lp->l_head;
	while( np!=NO_NODE ){
		vp = (Version *) np->n_data;
		d = get_lib_date(vp);
		sprintf(msg_str,"Module %-16s last revision %4d/%02d/%02d",
			vp->lib_name,d.yr,d.mo,d.day);
		prt_msg(msg_str);
		np=np->n_next;
	}
	dellist(lp);
}

void list_files(QSP_ARG_DECL  const char *name)
{
	Version *vp;

	if( !strcmp(name,"all") ){
		list_all_files(SINGLE_QSP_ARG);
		return;
	}

	vp = get_version(QSP_ARG  name);
	if( vp == NO_VERSION ) return;

	list_version_files(QSP_ARG  vp);
}

void list_all_files(SINGLE_QSP_ARG_DECL)
{
#ifdef FOOBAR
	List *lp;
	Node *np;
	Version *vp;

	lp=version_list(SINGLE_QSP_ARG);
	lp=alpha_sort(lp);
	if( lp == NO_LIST ) return;
	np=lp->l_head;
	while( np!=NO_NODE ){
		vp = (Version *) np->n_data;
		list_version_files(vp);
		np=np->n_next;
	}
	dellist(lp);
#else /* ! FOOBAR */
	Node *np;

	if( file_version_lp == NO_LIST )
		load_all_versions();
#ifdef CAUTIOUS
	if( file_version_lp == NO_LIST )
		NERROR1("CAUTIOUS:  list_all_files:  initialization error");
#endif /* CAUTIOUS */

	np=file_version_lp->l_head;
	while(np!=NO_NODE){
		FileVersion *fvp;
		fvp=(FileVersion *)np->n_data;
		list_one_file(NULL,fvp);
		np=np->n_next;
	}
#endif /* ! FOOBAR */
}

