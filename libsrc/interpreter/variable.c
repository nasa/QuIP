
#include "quip_config.h"
#include "quip_prot.h"
#include "quip_version.h"

#ifdef HAVE_UNISTD_H
#include <unistd.h>	// getcwd
#endif // HAVE_UNISTD_H

#ifdef HAVE_TIME_H
#include <time.h>
#endif // HAVE_TIME_H

static Item_Type * var__itp=NULL;

extern void list_vars(SINGLE_QSP_ARG_DECL)
{
	list_items(QSP_ARG  var__itp);
}

const char *var_value(QSP_ARG_DECL  const char *s)
{
	Variable *vp;
	
	vp=VAR_OF(s);
	if( vp == NO_VARIABLE ) return NULL;
	return var_p_value(QSP_ARG  vp);
}

const char *var_p_value(QSP_ARG_DECL  Variable *vp)
{
	if( IS_DYNAMIC_VAR(vp) ){
		return (*(VAR_FUNC(vp)))(SINGLE_QSP_ARG);
	} else {
		return VAR_VALUE(vp);
	}
}

ITEM_INIT_FUNC(Variable,var_)
ITEM_NEW_FUNC(Variable,var_)
ITEM_CHECK_FUNC(Variable,var_)
ITEM_PICK_FUNC(Variable,var_)

Variable *create_reserved_var(QSP_ARG_DECL  const char *var_name, const char *var_val)
{
	Variable *vp;

	vp=var_of(QSP_ARG  var_name);
	if( vp != NO_VARIABLE ){
		sprintf(ERROR_STRING,
"create_reserved_var:  variable %s already exists!?",var_name);
		WARN(ERROR_STRING);
		return NO_VARIABLE;
	}
	return force_reserved_var(QSP_ARG  var_name,var_val);
}

Variable *force_reserved_var(QSP_ARG_DECL  const char *var_name, const char *var_val )
{
	Variable *vp;

	vp = new_var_(QSP_ARG  var_name);
	SET_VAR_VALUE(vp,savestr(var_val));
	SET_VAR_FLAGS(vp,VAR_RESERVED);
	return vp;
}

static Variable *insure_variable(QSP_ARG_DECL  const char *name, int creat_flags )
{
	Variable *vp;
	const char *val_str;

	if( *name == 0 ) return NO_VARIABLE;
	vp=VAR_OF(name);
	if( vp != NO_VARIABLE ){
		return(vp);
	}
	vp = new_var_(QSP_ARG  name);

	// if this variable exists in the environment, then
	// import it, and mark it as reserved...

	val_str=getenv(name);
	if( val_str != NULL ) {
		SET_VAR_VALUE(vp, savestr(val_str) );
		SET_VAR_FLAGS(vp, VAR_RESERVED );
		if( creat_flags != VAR_RESERVED ){
			sprintf(ERROR_STRING,
	"insure_variable:  %s exists in environment, but reserved flag not passed!?",
				name);
			WARN(ERROR_STRING);
		}
	} else {
		SET_VAR_FLAGS(vp, creat_flags );
		SET_VAR_VALUE(vp, NULL);
	}
	return vp;
}

Variable *assign_var(QSP_ARG_DECL  const char *var_name, const char *var_val)
{
	Variable *vp;

	vp = insure_variable(QSP_ARG  var_name, VAR_SIMPLE );

	if( vp == NO_VARIABLE ) return vp;

	// reserved variables are not assignable from scripts, but
	// are assigned programmatically from code using assign_var

	if( IS_DYNAMIC_VAR(vp) ){
		sprintf(ERROR_STRING,"assign_var:  dynamic variable %s is not assignable!?",
			VAR_NAME(vp));
		WARN(ERROR_STRING);
		return NULL;
	} else if( IS_RESERVED_VAR(vp) ){
		sprintf(ERROR_STRING,"assign_var:  reserved variable %s is not assignable!?",
			VAR_NAME(vp));
		WARN(ERROR_STRING);
		return NULL;
	}

	if( VAR_VALUE(vp) != NULL ){
		rls_str(VAR_VALUE(vp));
	}
	SET_VAR_VALUE(vp, savestr(var_val) );
	return vp;
}

// reserved variables are not assignable from scripts, but
// are assigned programmatically from code using assign_reserved_var

Variable *assign_reserved_var(QSP_ARG_DECL  const char *var_name, const char *var_val)
{
	Variable *vp;

	vp = insure_variable(QSP_ARG  var_name, VAR_RESERVED );

	if( vp == NO_VARIABLE ) return vp;

	if( IS_DYNAMIC_VAR(vp) ){
		sprintf(ERROR_STRING,
"assign_reserved_var:  dynamic variable %s is not assignable!?",
			VAR_NAME(vp));
		WARN(ERROR_STRING);
		return NULL;
	} else if( ! IS_RESERVED_VAR(vp) ){
		sprintf(ERROR_STRING,
"assign_reserved_var:  variable %s is not reserved!?",
			VAR_NAME(vp));
		WARN(ERROR_STRING);
abort();
		return NULL;
	}

	if( VAR_VALUE(vp) != NULL ){
		rls_str(VAR_VALUE(vp));
	}
	SET_VAR_VALUE(vp, savestr(var_val) );
	return vp;
}

Variable *get_var(QSP_ARG_DECL  const char *name)
{
	Variable *vp;

	vp=VAR_OF(name);
	if( vp == NO_VARIABLE ){
		sprintf(ERROR_STRING,"No variable \"%s\"!?",name);
		WARN(ERROR_STRING);
	}
	return vp;
}

void init_dynamic_var(QSP_ARG_DECL  const char *name, const char *(*func)(SINGLE_QSP_ARG_DECL) )
{
	Variable *vp;

	vp=VAR_OF(name);
//#ifdef CAUTIOUS
//	if( vp != NO_VARIABLE ){
//		sprintf(ERROR_STRING,
//		"CAUTIOUS:  init_dynamic_var:  variable %s already exists!?",
//			VAR_NAME(vp));
//		WARN(ERROR_STRING);
//		return;
//	}
//#endif /* CAUTIOUS */

	assert( vp == NO_VARIABLE );

	vp = new_var_(QSP_ARG  name);
	SET_VAR_FLAGS(vp, VAR_DYNAMIC | VAR_RESERVED );
	SET_VAR_FUNC(vp, func);
}

static const char *my_getcwd(SINGLE_QSP_ARG_DECL)
{
	static char buf[LLEN];	// BUG should be MAXPATHLEN?

#ifdef HAVE_GETCWD
	if( getcwd(buf,LLEN) == NULL ){
		tell_sys_error("getcwd");
		return ".";
	}
	return buf;
#else
	return ".";
#endif // ! HAVE_GETCWD
}

static const char *my_getpid(SINGLE_QSP_ARG_DECL)
{
#ifdef HAVE_GETPID
	static char buf[16];	// BUG what is the largest pid?
	pid_t pid;

	pid = getpid();
	sprintf(buf,"%d",pid);
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
	// BUG - using a static string here means not thread-safe!?
	static char buf[32];	// must be at least 26

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
	s = asctime_r(tm_p,buf);
#else // ! HAVE_GMTIME_R
#ifdef HAVE_CTIME
	// BUG - this is not gmtime!!!
	s=ctime(&timeval);
#else // ! HAVE_CTIME
#error NO time formatting function!?
#endif // ! HAVE_CTIME
#endif // ! HAVE_GMTIME_R

	/* erase trailing newline... */
	s[ strlen(s)-1 ] = '\0';

	return s;
}

static const char *get_verbose(SINGLE_QSP_ARG_DECL)
{
	if( verbose ) return "1";
	return "0";
}

void init_variables(SINGLE_QSP_ARG_DECL)
{
	// used by system builtin in ../unix
	create_reserved_var(QSP_ARG  "exit_status","0");
	create_reserved_var(QSP_ARG  "mouse_data","0");
	create_reserved_var(QSP_ARG  "n_readable","0");
	create_reserved_var(QSP_ARG  "last_line","0");
	create_reserved_var(QSP_ARG  "serial_response","0");
	create_reserved_var(QSP_ARG  "timex_tick","0");
	create_reserved_var(QSP_ARG  "timex_freq","0");
	create_reserved_var(QSP_ARG  "git_version",QUIP_VERSION_STRING);

	init_dynamic_var(QSP_ARG  "verbose",get_verbose);
	init_dynamic_var(QSP_ARG  "cwd",my_getcwd);
	// BUG pid shouldn't change, but might after a fork?
	init_dynamic_var(QSP_ARG  "pid",my_getpid);
	init_dynamic_var(QSP_ARG  "local_date",get_local_date);
	init_dynamic_var(QSP_ARG  "utc_date",get_utc_date);
	ASSIGN_VAR("program_name",tell_progname());
	ASSIGN_VAR("program_version",tell_version());
}

void find_vars(QSP_ARG_DECL  const char *s)
{
	List *lp;

	lp=find_items(QSP_ARG  var__itp,s);
	if( lp==NO_LIST ) return;
	print_list_of_items(QSP_ARG  lp);
}

void search_vars(QSP_ARG_DECL  const char *frag)
{
	List *lp;
	Node *np;
	Variable *vp;
	char lc_frag[LLEN];

	lp=item_list(QSP_ARG  var__itp);
	if( lp == NO_LIST ) return;

	np=lp->l_head;
	decap(lc_frag,frag);
	while(np!=NO_NODE){
		char str1[LLEN];
		vp = (Variable *) NODE_DATA(np);
		/* make the match case insensitive */
		decap(str1,VAR_VALUE(vp));
		if( strstr(str1,lc_frag) != NULL ){
			sprintf(msg_str,"%s:\t%s",VAR_NAME(vp),VAR_VALUE(vp));
			prt_msg(msg_str);
		}
		np=NODE_NEXT(np);
	}
}

void reserve_variable(QSP_ARG_DECL  const char *name)
{
	Variable *vp;

	vp = insure_variable(QSP_ARG  name,VAR_RESERVED);
	if( IS_DYNAMIC_VAR(vp) ){
		sprintf(ERROR_STRING,
	"reserve_variable:  no need to reserve dynamic variable %s",
			name);
		WARN(ERROR_STRING);
		return;
	}
	if( IS_RESERVED_VAR(vp) ){
		sprintf(ERROR_STRING,
	"reserve_variable:  redundant call to reserve variable %s",
			name);
		WARN(ERROR_STRING);
		return;
	}
	SET_VAR_FLAG_BITS(vp,VAR_RESERVED);
}

// Replace the value of a variable with a string with backslashes
// inserted before each quote (single or double)

void replace_var_string(QSP_ARG_DECL  Variable *vp, const char *find,
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
		if( *start == 0 ) continue;

		s=strstr(start,find);
		if( s == NULL ){
			cat_string(sbp,start);
			start+=strlen(start);
		}
	} while(*start);

	assign_var(QSP_ARG  VAR_NAME(vp), SB_BUF(sbp) );
	rls_stringbuf(sbp);
}

void show_var(QSP_ARG_DECL  Variable *vp)
{
	if( IS_SIMPLE_VAR(vp) ){
		sprintf(MSG_STR,"$%s = %s",VAR_NAME(vp),VAR_VALUE(vp));
	} else if( IS_DYNAMIC_VAR(vp) ){
		sprintf(MSG_STR,"$%s = %s (dynamic value, function at 0x%lx)",
			VAR_NAME(vp),var_p_value(QSP_ARG  vp), (long)VAR_FUNC(vp));
	} else if( IS_RESERVED_VAR(vp) ){
		sprintf(MSG_STR,"$%s = %s (reserved)",VAR_NAME(vp),VAR_VALUE(vp));
	}
	prt_msg(MSG_STR);
}

void set_script_var_from_int(QSP_ARG_DECL  const char *varname, long val )
{
	char str[LLEN];

	sprintf(str,"%ld",val);	// BUG possible buffer overrun???

	// BUG should make this a reserved var?
	ASSIGN_VAR(varname,str);
}
