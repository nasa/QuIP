#include <string.h>
#include <stdlib.h>		// getenv
#include <sys/param.h>		// MAXPATHLEN
#include "quip_config.h"
#include "quip_prot.h"
#include "item_prot.h"
#include "quip_version.h"
#include "variable.h"
#include "list.h"
#include "getbuf.h"

#include "query_stack.h"	// temporary!

#ifdef HAVE_UNISTD_H
#include <unistd.h>	// getcwd
#endif // HAVE_UNISTD_H

#ifdef HAVE_TIME_H
#include <time.h>
#endif // HAVE_TIME_H

static Item_Type * var__itp=NULL;

extern void _list_vars(SINGLE_QSP_ARG_DECL)
{
	list_items(var__itp, tell_msgfile());
fprintf(stderr,"list_vars, item_type %s at 0x%lx\n",ITEM_TYPE_NAME(var__itp),(long)var__itp);
}

const char *_var_value(QSP_ARG_DECL  const char *s)
{
	Variable *vp;
	
	vp=var_of(s);
	if( vp == NULL ) return NULL;
	return var_p_value(vp);
}

const char *_var_p_value(QSP_ARG_DECL  Variable *vp)
{
	if( IS_DYNAMIC_VAR(vp) ){
		return (*(VAR_FUNC(vp)))(SINGLE_QSP_ARG);
	} else {
		return VAR_VALUE(vp);
	}
}

ITEM_INIT_FUNC(Variable,var_,0)
ITEM_NEW_FUNC(Variable,var_)
ITEM_CHECK_FUNC(Variable,var_)
ITEM_PICK_FUNC(Variable,var_)
ITEM_DEL_FUNC(Variable,var_)

Variable *_create_reserved_var(QSP_ARG_DECL  const char *var_name, const char *var_val)
{
	Variable *vp;

	vp=var_of(var_name);
	if( vp != NULL ){
		sprintf(ERROR_STRING,
"create_reserved_var:  variable %s already exists!?",var_name);
		warn(ERROR_STRING);
		return NULL;
	}
	return force_reserved_var(var_name,var_val);
}

Variable *_force_reserved_var(QSP_ARG_DECL  const char *var_name, const char *var_val )
{
	Variable *vp;

	vp = new_var_(var_name);
	SET_VAR_VALUE(vp,save_possibly_empty_str(var_val));
	SET_VAR_FLAGS(vp,VAR_RESERVED);
	return vp;
}

#define insure_variable(name,creat_flags) _insure_variable(QSP_ARG  name,creat_flags)

static Variable *_insure_variable(QSP_ARG_DECL  const char *name, int creat_flags )
{
	Variable *vp;
	const char *val_str;

	if( *name == 0 ) return NULL;
	vp=var_of(name);
	if( vp != NULL ){
		return(vp);
	}
	vp = new_var_(name);

	// if this variable exists in the environment, then
	// import it, and mark it as reserved...

	val_str=getenv(name);
	if( val_str != NULL ) {
		SET_VAR_VALUE(vp, save_possibly_empty_str(val_str) );
		SET_VAR_FLAGS(vp, VAR_RESERVED );
		if( creat_flags != VAR_RESERVED ){
			sprintf(ERROR_STRING,
	"insure_variable:  %s exists in environment, but reserved flag not passed!?",
				name);
			warn(ERROR_STRING);
		}
	} else {
		SET_VAR_FLAGS(vp, creat_flags );
		SET_VAR_VALUE(vp, NULL);
	}
	return vp;
}

Variable *_assign_var(QSP_ARG_DECL  const char *var_name, const char *var_val)
{
	Variable *vp;

	vp = insure_variable(var_name, VAR_SIMPLE );

	if( vp == NULL ) return vp;

	// reserved variables are not assignable from scripts, but
	// are assigned programmatically from code using assign_var

	if( IS_DYNAMIC_VAR(vp) ){
		sprintf(ERROR_STRING,"assign_var:  dynamic variable %s is not assignable!?",
			VAR_NAME(vp));
		warn(ERROR_STRING);
		return NULL;
	} else if( IS_RESERVED_VAR(vp) ){
		sprintf(ERROR_STRING,"assign_var:  reserved variable %s is not assignable!?",
			VAR_NAME(vp));
		warn(ERROR_STRING);
		return NULL;
	}

	if( VAR_VALUE(vp) != NULL ){
		rls_str(VAR_VALUE(vp));
	}
	SET_VAR_VALUE(vp, save_possibly_empty_str(var_val) );
	return vp;
}

// reserved variables are not assignable from scripts, but
// are assigned programmatically from code using assign_reserved_var

Variable *_assign_reserved_var(QSP_ARG_DECL  const char *var_name, const char *var_val)
{
	Variable *vp;

	vp = insure_variable(var_name, VAR_RESERVED );

	if( vp == NULL ) return vp;

	if( IS_DYNAMIC_VAR(vp) ){
		sprintf(ERROR_STRING,
"assign_reserved_var:  dynamic variable %s is not assignable!?",
			VAR_NAME(vp));
		warn(ERROR_STRING);
		return NULL;
	} else if( ! IS_RESERVED_VAR(vp) ){
		sprintf(ERROR_STRING,
"assign_reserved_var:  variable %s already exists but is not reserved!?",
			VAR_NAME(vp));
		advise(ERROR_STRING);
abort();
		return NULL;
	}

	if( VAR_VALUE(vp) != NULL ){
		rls_str(VAR_VALUE(vp));
	}
	SET_VAR_VALUE(vp, save_possibly_empty_str(var_val) );
	return vp;
}

Variable *_get_var(QSP_ARG_DECL  const char *name)
{
	Variable *vp;

	vp=var_of(name);
	if( vp == NULL ){
		sprintf(ERROR_STRING,"No variable \"%s\"!?",name);
		warn(ERROR_STRING);
	}
	return vp;
}

void _init_dynamic_var(QSP_ARG_DECL  const char *name, const char *(*func)(SINGLE_QSP_ARG_DECL) )
{
	Variable *vp;

	vp=var_of(name);
	assert( vp == NULL );

	vp = new_var_(name);
	SET_VAR_FLAGS(vp, VAR_DYNAMIC | VAR_RESERVED );
	SET_VAR_FUNC(vp, func);
}

static const char *my_getcwd(SINGLE_QSP_ARG_DECL)
{
#ifdef HAVE_GETCWD
	static char buf[MAXPATHLEN];

	if( getcwd(buf,MAXPATHLEN) == NULL ){
		tell_sys_error("getcwd");
		return ".";
	}
	return buf;
#else
	return ".";
#endif // ! HAVE_GETCWD
}

// on the mac, a pid is a 32 bit integer...
// something in excess of 4,000,000,000,000
// So 13 digits should be enough...

#define MAX_PID_DIGITS	15

static const char *get_pid_string(SINGLE_QSP_ARG_DECL)
{
#ifdef HAVE_GETPID
	static char buf[MAX_PID_DIGITS+1];
	int n_needed;

	pid_t pid;

	pid = getpid();
	n_needed=snprintf(buf,MAX_PID_DIGITS+1,"%d",pid);
	if( n_needed > (MAX_PID_DIGITS+1) ){
		warn("get_pid_string:  Need to increase MAX_PID_DIGITS!?");
	}

	return buf;
#else
	return 0;
#endif // ! HAVE_GETPID
}

// Code for the next two functions was lifted from bi_menu.c

static const char *get_local_date(SINGLE_QSP_ARG_DECL)
{
	time_t timeval;
	char *s;
#ifdef HAVE_CTIME_R
	// using a static string here means all threads share the same time
	static char buf[32];	// must be at least 26 (why?)
#endif // HAVE_CTIME_R

	time(&timeval);

#ifdef HAVE_CTIME_R
	s=ctime_r(&timeval,buf);
#else // ! HAVE_CTIME_R
#ifdef HAVE_CTIME
	s=ctime(&timeval);
#else // ! HAVE_CTIME
#error No time formatting function!?
#endif // ! HAVE_CTIME
#endif // ! HAVE_CTIME_R

	/* erase trailing newline... */
	assert( s[ strlen(s)-1 ] == '\n' );
	s[ strlen(s)-1 ] = '\0';

	return s;
}

static const char *get_utc_date(SINGLE_QSP_ARG_DECL)
{
	time_t timeval;
	char *s;
	static char buf[32];	// must be at least 26
	struct tm tm1, *tm_p;

	time(&timeval);

#ifdef HAVE_GMTIME_R
	tm_p = gmtime_r(&timeval,&tm1);
#else // ! HAVE_GMTIME_R
#ifdef HAVE_GMTIME
	tm_p = gmtime(&timeval);
#else // ! HAVE_GMTIME
#error NO gmtime_r or gmtime function!?
#endif // ! HAVE_GMTIME
#endif // ! HAVE_GMTIME_R

#ifdef HAVE_ASCTIME_R
	s = asctime_r(tm_p,buf);
#else // ! HAVE_ASCTIME_R
#ifdef HAVE_ASCTIME
#else // ! HAVE_ASCTIME
#error NO asctime_r or asctime function!?
#endif // ! HAVE_ASCTIME
#endif // ! HAVE_ASCTIME_R

	/* erase trailing newline... */
	s[ strlen(s)-1 ] = '\0';

	return s;
}

static const char *get_verbose(SINGLE_QSP_ARG_DECL)
{
	if( verbose ) return "1";
	return "0";
}

const char *tell_version(void)
{
	return QUIP_VERSION_STRING;
}

void init_variables(SINGLE_QSP_ARG_DECL)
{
	// used by system builtin in ../unix
	create_reserved_var("exit_status","0");
	create_reserved_var("mouse_data","0");
	create_reserved_var("n_readable","0");
	create_reserved_var("last_line","0");
	create_reserved_var("serial_response","0");
	create_reserved_var("timex_tick","0");
	create_reserved_var("timex_freq","0");
	create_reserved_var("git_version",QUIP_VERSION_STRING);

	init_dynamic_var("verbose",get_verbose);
	init_dynamic_var("cwd",my_getcwd);
	// BUG pid shouldn't change, but might after a fork?
	init_dynamic_var("pid",get_pid_string);
	init_dynamic_var("local_date",get_local_date);
	init_dynamic_var("utc_date",get_utc_date);
	assign_var("program_name",tell_progname());
	assign_var("program_version",tell_version());
}

void _find_vars(QSP_ARG_DECL  const char *s)
{
	List *lp;

	lp=find_items(var__itp,s);
	if( lp==NULL ) return;
	print_list_of_items(lp, tell_msgfile());
}

#define N_EXTRA_CHARS	20

void _search_vars(QSP_ARG_DECL  const char *frag)
{
	List *lp;
	Node *np;
	Variable *vp;
	char *lc_frag;
	char *str1=NULL;
	int str1_size=0;

	lp=item_list(var__itp);
	if( lp == NULL ) return;

	np=QLIST_HEAD(lp);
	lc_frag = getbuf(strlen(frag)+1);
	decap(lc_frag,frag);
	while(np!=NULL){
		vp = (Variable *) NODE_DATA(np);
		if( str1 == NULL ){
			str1_size = (int) strlen(VAR_VALUE(vp)) + 1 + N_EXTRA_CHARS ;
			str1 = getbuf( str1_size );
		} else {
			if( str1_size < strlen(VAR_VALUE(vp))+1 ){
				givbuf(str1);
				str1_size = (int) strlen(VAR_VALUE(vp)) + 1 + N_EXTRA_CHARS ;
				str1 = getbuf( str1_size );
			}
		}

		/* make the match case insensitive */
		decap(str1,VAR_VALUE(vp));
		if( strstr(str1,lc_frag) != NULL ){
			sprintf(msg_str,"%s:\t%s",VAR_NAME(vp),VAR_VALUE(vp));
			prt_msg(msg_str);
		}
		np=NODE_NEXT(np);
	}
	if( str1 != NULL ) givbuf(str1);
	givbuf(lc_frag);
}

void _reserve_variable(QSP_ARG_DECL  const char *name)
{
	Variable *vp;

	vp = insure_variable(name,VAR_RESERVED);
	if( IS_DYNAMIC_VAR(vp) ){
		sprintf(ERROR_STRING,
	"reserve_variable:  no need to reserve dynamic variable %s",
			name);
		warn(ERROR_STRING);
		return;
	}
	if( IS_RESERVED_VAR(vp) ){
		sprintf(ERROR_STRING,
	"reserve_variable:  redundant call to reserve variable %s",
			name);
		warn(ERROR_STRING);
		return;
	}
	SET_VAR_FLAG_BITS(vp,VAR_RESERVED);
}

// Replace the value of a variable with a string with backslashes
// inserted before each quote (single or double)

void _replace_var_string(QSP_ARG_DECL  Variable *vp, const char *find,
						const char *replace )
{
	const char *s, *start;
	String_Buf *sbp;
	int nr;

//fprintf(stderr,"replace_var_string:  find = \"%s\"\n",find);
//fprintf(stderr,"replace_var_string:  replace = \"%s\"\n",replace);
	start=VAR_VALUE(vp);
	s=strstr(start,find);
	if( s == NULL ) return;		// not found - nothing to do

	sbp = new_stringbuf();
	enlarge_buffer(sbp,(int)(strlen(start)+strlen(replace)-strlen(find)+1));
	nr = (int)strlen(find);
	do {
		cat_string_n(sbp,start,(int)(s-start));
		cat_string(sbp,replace);
		start = s + nr;
        assert(start!=NULL);
		if( *start == 0 ) continue;

		s=strstr(start,find);
		if( s == NULL ){
			cat_string(sbp,start);
			start+=strlen(start);
		}
	} while(*start);

	assign_var(VAR_NAME(vp), sb_buffer(sbp) );
	rls_stringbuf(sbp);
}

void _show_var(QSP_ARG_DECL  Variable *vp)
{
	if( IS_SIMPLE_VAR(vp) ){
		sprintf(MSG_STR,"$%s = %s",VAR_NAME(vp),VAR_VALUE(vp));
	} else if( IS_DYNAMIC_VAR(vp) ){
		sprintf(MSG_STR,"$%s = %s (dynamic value, function at 0x%lx)",
			VAR_NAME(vp),var_p_value(vp), (long)VAR_FUNC(vp));
	} else if( IS_RESERVED_VAR(vp) ){
		sprintf(MSG_STR,"$%s = %s (reserved)",VAR_NAME(vp),VAR_VALUE(vp));
	}
	prt_msg(MSG_STR);
}

#define MAX_INT_STRING_LEN	80	// BUG should check...

void _set_script_var_from_int(QSP_ARG_DECL  const char *varname, long val )
{
	char str[MAX_INT_STRING_LEN];

	sprintf(str,"%ld",val);	// BUG possible buffer overrun???

	// BUG should make this a reserved var?
	assign_var(varname,str);
}

