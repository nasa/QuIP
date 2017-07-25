
#include "quip_config.h"
#include "quip_version.h"

#ifdef HAVE_UNISTD_H
#include <unistd.h>		// sleep, usleep
#endif /* HAVE_UNISTD_H */

#ifdef HAVE_TIME_H
#include <time.h>
#endif // HAVE_TIME_H

#include <math.h>
#include <string.h>

#include "quip_prot.h"
#include "query_prot.h"
#include "query_stack.h"
#include "macro.h"	// Need to make macro private file!
#include "nexpr.h"
#include "rn.h"	// set_seed

#include <fcntl.h>
#include <dirent.h>
#include <sys/stat.h>
#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>		// MAXPATHLEN
#endif /* HAVE_SYS_PARAM_H */

#ifdef BUILD_FOR_OBJC
#include "ios_prot.h"
#endif /* BUILD_FOR_OBJC */


static Item_Type *curr_itp=NULL;

static COMMAND_FUNC( list_types )
{
	list_items(QSP_ARG  ittyp_itp, tell_msgfile(SINGLE_QSP_ARG));
}

static COMMAND_FUNC( select_type )
{
	//const char *s;
	Item_Type *itp;
    
	/*
	s=NAMEOF("item type name");
	itp = get_item_type(QSP_ARG  s);
	*/
	itp = (Item_Type *) pick_item(QSP_ARG  ittyp_itp, "item type name");
	if( itp != NULL )
		curr_itp = itp;
}

static COMMAND_FUNC( do_list_items )
{
	list_items(QSP_ARG  curr_itp, tell_msgfile(SINGLE_QSP_ARG));
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(items_menu,s,f,h)

MENU_BEGIN(items)
ADD_CMD( list_types,	list_types,	list item types)
ADD_CMD( select_type,	select_type,	select item type)
ADD_CMD( list,		do_list_items,	list items of selected type)
MENU_END(items)


static COMMAND_FUNC( do_items ) { PUSH_MENU(items); }

//////////////////////////////////////

// This whole submenu is really just for testing.

#include "rbtree.h"

static qrb_tree *the_tree_p=NULL;

static COMMAND_FUNC( do_rbt_add )
{
	const char *s;
	Item *ip;

	s=NAMEOF("new item name");

	// first look for this name in the tree

	ip = getbuf(sizeof(Item));
	ip->item_name = savestr(s);

	rb_insert_item( the_tree_p, ip );
}

static void rb_item_print( qrb_node *np, qrb_tree *tree_p )
{
	const Item *ip;

	ip = np->data;
	printf("\t%s\n",ip->item_name);
	fflush(stdout);
}

static COMMAND_FUNC( do_rbt_print )
{
	rb_traverse( the_tree_p->root, rb_item_print, the_tree_p );
}

static COMMAND_FUNC( do_rbt_del )
{
	const char *s;

	s = NAMEOF("item name");

	// find the item in the tree
	rb_delete_key(the_tree_p,s);
}

#ifdef RB_TREE_DEBUG
static COMMAND_FUNC( do_rbt_check )
{
	rb_check(the_tree_p);
}
#endif // RB_TREE_DEBUG

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(rbt_menu,s,f,h)
MENU_BEGIN(rbt)
ADD_CMD( add,		do_rbt_add,	add an item to the tree )
ADD_CMD( print,		do_rbt_print,	show contents of the tree )
#ifdef RB_TREE_DEBUG
ADD_CMD( check,		do_rbt_check,	check consistency of the tree )
#endif // RB_TREE_DEBUG
ADD_CMD( delete,	do_rbt_del,	delete an item from the tree )
MENU_END(rbt)

//// this routine should compare keys...
//static int test_compare( const void *kp1, const void *kp2 )	// assume these are items
//{
//	return( strcmp( (const char *)kp1, (const char *)kp2) );
//}

//static void test_destroy_key( void *ip ) {}
//static void test_destroy_data( void *ip ) {}

static COMMAND_FUNC( do_rbtree_test )
{
	if( the_tree_p == NULL ){
		//the_tree_p = create_rb_tree( test_compare, test_destroy_key, test_destroy_data );
		the_tree_p = create_rb_tree();
	}
	PUSH_MENU(rbt);
}

static COMMAND_FUNC( do_list_macs )
{
	list_items(QSP_ARG  macro_itp, tell_msgfile(SINGLE_QSP_ARG));
}

static COMMAND_FUNC( do_find_mac )
{
	const char *s;
	List *lp;

	s=NAMEOF("macro name fragment");

	lp=find_items(QSP_ARG  macro_itp, s);
	if( lp==NULL ) return;

	print_list_of_items(QSP_ARG  lp, tell_msgfile(SINGLE_QSP_ARG));

	// Need to release the list!
	dellist(lp);	// releases nodes and list struct but not data
}


/* search macros for those whose text contains the fragment */

static List *search_macros(QSP_ARG_DECL  const char *frag)
{
	List *lp, *newlp=NULL;
	Node *np, *newnp;
	Macro *mp;
	char *mbuf;
	char *lc_frag;

	if( macro_itp == NULL ) return NULL;
	lp=item_list(QSP_ARG  macro_itp);
	if( lp == NULL ) return lp;

	np=QLIST_HEAD(lp);
	lc_frag = getbuf( strlen(frag) + 1 );
	decap(lc_frag,frag);
	while(np!=NULL){
		mp = (Macro*) NODE_DATA(np);
		if( MACRO_TEXT(mp) != NULL ){	/* NULL if no macro text... */
			mbuf = getbuf( strlen(MACRO_TEXT(mp))+1 );
			/* make the match case insensitive */
			decap(mbuf,MACRO_TEXT(mp));
			if( strstr(mbuf,lc_frag) != NULL ){
				if( newlp == NULL )
					newlp=new_list();
				newnp=mk_node(mp);
				addTail(newlp,newnp);
			}
			givbuf((void *)mbuf);
		}
		np=NODE_NEXT(np);
	}
	givbuf(lc_frag);
	return newlp;
}

static COMMAND_FUNC( do_search_macs )
{
	const char *s;
	List *lp;

	s=NAMEOF("macro fragment");
	lp=search_macros(QSP_ARG  s);
	if( lp == NULL ) return;

	sprintf(msg_str,"Fragment \"%s\" occurs in the following macros:",s);
	prt_msg(msg_str);

	print_list_of_items(QSP_ARG  lp, tell_msgfile(SINGLE_QSP_ARG));
}

static COMMAND_FUNC( do_show_mac )
{
	Macro *mp;

	mp=PICK_MACRO("macro name");
	if( mp!= NULL )
		show_macro(QSP_ARG  mp);
}

static COMMAND_FUNC( do_dump_mac )
{
	Macro *mp;

	mp=PICK_MACRO("macro name");
	if( mp!= NULL )
		dump_macro(QSP_ARG  mp);
}

static COMMAND_FUNC( do_dump_invoked )
{
	List *lp;
	Node *np;
	Macro *mp;

	if( macro_itp == NULL ){
no_macros:
		WARN("do_dump_invoked:  no macros!?");
		return;
	}
	lp=item_list(QSP_ARG  macro_itp);
	if( lp == NULL ) goto no_macros;

	np=QLIST_HEAD(lp);
	while( np != NULL ){
		mp = (Macro *) NODE_DATA(np);
		if( macro_is_invoked(mp) ){
			dump_macro(QSP_ARG  mp);
		}
		np = NODE_NEXT(np);
	}
}

static COMMAND_FUNC( do_info_mac )
{
	Macro *mp;

	mp=PICK_MACRO("macro name");
	if( mp!= NULL )
		macro_info(QSP_ARG  mp);
}

static COMMAND_FUNC( do_allow_macro_recursion )
{
	Macro *mp;
	
	mp=PICK_MACRO("");
	if( mp == NULL ) return;
	allow_recursion_for_macro(mp);
}

static COMMAND_FUNC( do_set_max_warnings )
{
	int n;
	
	n=(int)HOW_MANY("maximum number of warnings (negative for unlimited)");
	SET_QS_MAX_WARNINGS(THIS_QSP,n);
}

static COMMAND_FUNC( do_def_mac )
{
	const char *name;
	Macro *mp;
	Macro_Arg **ma_tbl;
	int n;
	String_Buf *sbp;
	int lineno;

	name=NAMEOF("macro name");
	n=(int)HOW_MANY("number of arguments");

	if( check_adequate_return_strings(QSP_ARG  n+2) < 0 ){
		sprintf(ERROR_STRING,
"define:  %d arguments requested for macro %s exceeds system limit!?\nRebuild application with increased value of N_QRY_RETSTRS.",
			n-2,name);
		ERROR1(ERROR_STRING);
	}

	ma_tbl = setup_macro_args(QSP_ARG  n);
	// We want to store the line number of the file where the macro
	// is declared...  We can read it now from the query stream...
	//lineno = QRY_LINES_READ(CURR_QRY(THIS_QSP));
	lineno = current_line_number(SINGLE_QSP_ARG);

	sbp = read_macro_body(SINGLE_QSP_ARG);

	// Now make sure this macro doesn't already exist
	mp = macro_of(QSP_ARG  name);
	if( mp != NULL ){
		sprintf(ERROR_STRING,"Macro \"%s\" already exists!?",name);
		WARN(ERROR_STRING);
		// Report where the macro was declared...
		sprintf(ERROR_STRING,"Macro \"%s\" defined in file %s, line %d",
			name,macro_filename(mp),macro_lineno(mp) );
		advise(ERROR_STRING);
	} else {
		mp=create_macro(QSP_ARG  name,n,ma_tbl,sbp,lineno);
	}

	rls_stringbuf(sbp);

	// for compatibility with quip - maybe should change this?
	POP_MENU;

} // do_def_mac

static COMMAND_FUNC( do_del_mac )
{
	Macro *mp;

	mp = PICK_MACRO("");

	if( mp == NULL ) return;

	rls_macro(QSP_ARG  mp);
}

static COMMAND_FUNC( do_if )
{
	//double v;
	Typed_Scalar *tsp;
	const char *s;

	tsp=pexpr(QSP_ARG  NAMEOF("condition") );

	s=NAMEOF("Command word or 'Then'");

	// Used to use push_text here, but that didn't push macro args...

	// We don't store the text with savestr -
	// will the text be safe?

	if( !strcmp(s,"Then") ){
		const char *s2;
		const char *s3;
		s=NAMEOF("Command word");
		s2=NAMEOF("'Else'");
		if( strcmp(s2,"Else") ){
			WARN("Then clause must be followed by 'Else'");
			return;
		}
		s3=NAMEOF("Command word");

		//if( v != 0.0 )
		if( ! has_zero_value(tsp) )
			push_if(QSP_ARG  s);
		else
			push_if(QSP_ARG  s3);
	} else {
		//if( v != 0.0 )
		if( ! has_zero_value(tsp) )
			push_if(QSP_ARG  s);
	}
	RELEASE_SCALAR(tsp);
}

static COMMAND_FUNC( do_push_text )
{
	const char *s;

	s=NAMEOF("command string");

	// We have used push_text here for a long time, but if we want
	// to push a foreach command, then it needs to be at the same level...
	PUSH_TEXT(s,"-");
}


static COMMAND_FUNC( do_redir )
{
	const char *s;
	FILE *fp;

	s = NAMEOF("name of file");
	// BUG use NS function...
	// what does that comment mean?  For mac version?
	fp=fopen(s,"r");
	if( fp == NULL ){
		sprintf(ERROR_STRING,"Error opening file %s!?",s );
		WARN(ERROR_STRING);
	} else {
		redir(QSP_ARG  fp, s );
	}
}

static COMMAND_FUNC( do_warn )
{
	const char *s;
	s = NAMEOF("warning message");
	//[THIS_QSP warn : [THIS_QSP nameOf : @"warning message" ] ];
	WARN( s );
}

static COMMAND_FUNC( do_expect_warning )
{
	const char *s;

	s=NAMEOF("Beginning of warning message");
	expect_warning(QSP_ARG  s);
}

static COMMAND_FUNC( do_check_expected_warning )
{
	check_expected_warning(SINGLE_QSP_ARG);
}

/******************** variables menu ***********************/

static COMMAND_FUNC( do_set_var )
{
	Quip_String *name, *value;
	Variable *vp;
    
	name=NAMEOF("variable name");
	value=NAMEOF("variable value");

	vp = var_of(QSP_ARG  name);
	if( vp != NULL && IS_RESERVED_VAR(vp) ){
		sprintf(ERROR_STRING,"Sorry, variable \"%s\" is reserved.",name);
		WARN(ERROR_STRING);
		return;
	}

	assign_var(QSP_ARG  name,value);
}

// on 64 bit architecture, long is 64 bits,
// but what about 32bit (iOS)?

static const char *def_gfmt_str="%.7g";

static void init_default_formats(SINGLE_QSP_ARG_DECL)
{
	assert( QS_NUMBER_FMT(THIS_QSP) == NULL );

	SET_QS_GFORMAT(THIS_QSP, def_gfmt_str );
	SET_QS_DFORMAT(THIS_QSP, "%"   PRId64 );
	SET_QS_XFORMAT(THIS_QSP, "0x%" PRIx64 );
	SET_QS_OFORMAT(THIS_QSP, "0%"  PRIo64 );

	SET_QS_NUMBER_FMT( THIS_QSP, QS_DFORMAT(THIS_QSP) );
}

#ifdef SOLVE_FOR_MAX_ROUNDABLE

// enable this just for system calibration
//#define SOLVE_FOR_MAX_ROUNDABLE
static void solve_for_max_roundable(void)
{
	double d,e;
	d=32000;
	do {
		d*=2;
		e=d+0.6;
	} while( round(e) != e );
fprintf(stderr,"round( %g + 0.6 ) == %g !?\n",d,e);
	do {
		d=round(d*0.9);
		e=d+0.6;
	} while( round(e) == e );
fprintf(stderr,"round( %g + 0.6 ) != %g !?\n",d,e);
	do {
		d*=1.02;
		e=d+0.6;
	} while( round(e) != e );
fprintf(stderr,"round( %g + 0.6 ) == %g !?\n",d,e);
	do {
		d=round(d*0.99);
		e=d+0.6;
	} while( round(e) == e );
fprintf(stderr,"round( %g + 0.6 ) != %g !?\n",d,e);
exit(1);
}
#endif // SOLVE_FOR_MAX_ROUNDABLE

static inline void ensure_assign_var_stringbuf(SINGLE_QSP_ARG_DECL)
{
	// Make sure we have a free string buffer
	if( QS_AV_STRINGBUF(THIS_QSP) == NULL ){
		SET_QS_AV_STRINGBUF( THIS_QSP, new_stringbuf() );
		enlarge_buffer(QS_AV_STRINGBUF(THIS_QSP),LLEN);
	}
}

static inline void assign_var_stringbuf_from_string(QSP_ARG_DECL  Typed_Scalar *tsp)
{
	// It is a string...
	assert( tsp->ts_value.u_vp != NULL );
	copy_string(QS_AV_STRINGBUF(THIS_QSP),(char *)tsp->ts_value.u_vp);
}

#define DEST	sb_buffer(QS_AV_STRINGBUF(THIS_QSP))

static inline void assign_integer_from_double(QSP_ARG_DECL  Typed_Scalar *tsp)
{
	/* We used to cast the value to integer if
	 * the format string is an integer format -
	 * But now we don't do this if the number has
	 * a fractional part...
	 */
	double d;
	d=tsp->ts_value.u_d;

	// We used to convert to integer if the number
	// was equal to the rounded version...  but that
	// fails for very big numbers...

	// can call solve_for_max_roundable here?

// Problems on iOS, but running the solve code above suggests
// that 1e+14 is OK??

#define MAX_ROUNDABLE_DOUBLE	1e+14

#ifdef HAVE_ROUND

	if( fabs(d) < MAX_ROUNDABLE_DOUBLE && d == round(d) ){
		/* Why cast to unsigned?  What if signed??? */
		/* We want to cast to unsigned to get the largest integer? */
		// does the sign of the cast really matter?
		if( d > 0 )
			sprintf(DEST,QS_NUMBER_FMT(THIS_QSP),(uint64_t)d);
		else
			sprintf(DEST,QS_NUMBER_FMT(THIS_QSP),(int64_t)d);
	} else {
		sprintf(DEST,QS_GFORMAT(THIS_QSP),d);
	}

#else /* ! HAVE_ROUND */

#ifdef HAVE_FLOOR
	if( d != floor(d) )
		sprintf(DEST,QS_GFORMAT(THIS_QSP),d);
	else {
		//sprintf(DEST,QS_NUMBER_FMT(THIS_QSP),(unsigned long)d);
		sprintf(DEST,QS_NUMBER_FMT(THIS_QSP),(long)d);
	}
#else /* ! HAVE_FLOOR */
	//sprintf(DEST,QS_NUMBER_FMT(THIS_QSP),(unsigned long)d);
	sprintf(DEST,QS_NUMBER_FMT(THIS_QSP),(long)d);
#endif /* ! HAVE_FLOOR */
#endif /* ! HAVE_ROUND */
}

#define SCALAR_IS_DOUBLE(tsp)		( SCALAR_MACH_PREC_CODE(tsp) == PREC_DP )


#define IS_INTEGER_FMT( f )		( (f) == QS_XFORMAT(THIS_QSP) ||	\
					  (f) == QS_DFORMAT(THIS_QSP) ||	\
					  (f) == QS_OFORMAT(THIS_QSP) )

static inline void assign_var_stringbuf_from_number(QSP_ARG_DECL  Typed_Scalar *tsp)
{
	// If the format is hex, decimal, or octal...
	if( IS_INTEGER_FMT(QS_NUMBER_FMT(THIS_QSP)) ){
		if( SCALAR_IS_DOUBLE(tsp) ){
			assign_integer_from_double(QSP_ARG  tsp);
		} else {	// integer format, integer scalar
			assert( SCALAR_MACH_PREC_CODE(tsp) == PREC_LI );
			sprintf(DEST,QS_NUMBER_FMT(THIS_QSP),tsp->ts_value.u_ll);
		}
	} else	{
		// Not integer format, print decimal places
		double d;
		d=double_for_scalar(tsp);
		sprintf(DEST,QS_NUMBER_FMT(THIS_QSP),d);
	}
}

#define CHECK_FMT_STRINGS					\
	if( QS_NUMBER_FMT(THIS_QSP) == NULL )			\
		init_default_formats(SINGLE_QSP_ARG);

static COMMAND_FUNC( do_assign_var )
{
	Quip_String *namestr, *estr;
	Typed_Scalar *tsp;
	namestr=NAMEOF("variable name" );
	estr=NAMEOF("expression" );
    
	CHECK_FMT_STRINGS

	tsp=pexpr(QSP_ARG  estr);
	if( tsp == NULL ) return;

	ensure_assign_var_stringbuf(SINGLE_QSP_ARG);

	// See if the expression is a string expression
	if( tsp->ts_prec_code == PREC_STR ){
		assign_var_stringbuf_from_string(QSP_ARG  tsp);
	} else {
		assign_var_stringbuf_from_number(QSP_ARG  tsp);
	}

	RELEASE_SCALAR(tsp);

	assign_var(QSP_ARG  namestr, DEST );
}

static COMMAND_FUNC( do_show_var )
{
	Variable *vp;
    
	vp = PICK_VAR("");
	if( vp == NULL ) return;

	show_var(QSP_ARG  vp);
}

static COMMAND_FUNC( do_del_var )
{
	Variable *vp;
    
	vp = PICK_VAR("");
	if( vp == NULL ) return;

	// remove from history list?

	del_var_(QSP_ARG  vp);
}

static COMMAND_FUNC( do_list_vars )
{
	//[Variable list];
	list_vars(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_replace_var )
{
	Variable *vp;
	const char *find, *replace;

	vp = PICK_VAR("");
	find = NAMEOF("substring to replace");
	replace = NAMEOF("replacement text");

	if( vp == NULL ) return;

	replace_var_string(QSP_ARG  vp,find,replace);
}

#define MIN_SIG_DIGITS	4
#define MAX_SIG_DIGITS	24


static COMMAND_FUNC( do_set_nsig )
{
	int n;

	n = (int) HOW_MANY("number of digits to print in numeric variables");
	if( n<MIN_SIG_DIGITS || n>MAX_SIG_DIGITS ){
		sprintf(ERROR_STRING,
	"Requested number of digits (%d) should be between %d and %d, using %d",
			n,MIN_SIG_DIGITS,MAX_SIG_DIGITS,MAX_SIG_DIGITS);
		WARN(ERROR_STRING);
		n=MAX_SIG_DIGITS;
	}
	CHECK_FMT_STRINGS

	// we insist first that we are using the g (float) format?
	if( QS_NUMBER_FMT(THIS_QSP) != QS_GFORMAT(THIS_QSP) ){
		WARN("do_set_nsig:  Changing variable format to float (g)");
		SET_QS_NUMBER_FMT(THIS_QSP,QS_GFORMAT(THIS_QSP));
	}

	sprintf(QS_NUMBER_FMT(THIS_QSP),"%%.%dg",n);
}

static const char **var_fmt_list=NULL;

static void init_fmt_choices(SINGLE_QSP_ARG_DECL)
{
	var_fmt_list = (const char **) getbuf( N_PRINT_FORMATS * sizeof(char *) );

	var_fmt_list[ FMT_DECIMAL ] = "decimal";
	var_fmt_list[ FMT_HEX ] = "hex";
	var_fmt_list[ FMT_OCTAL ] = "octal";
	var_fmt_list[ FMT_UDECIMAL ] = "unsigned_decimal";
	var_fmt_list[ FMT_FLOAT ] = "float";
	var_fmt_list[ FMT_POSTSCRIPT ] = "postscript";

	assert( N_PRINT_FORMATS == 6 );
}

static void set_fmt(QSP_ARG_DECL  Number_Fmt i)
{
	switch(i){
		case FMT_FLOAT:  SET_QS_NUMBER_FMT(THIS_QSP,QS_GFORMAT(THIS_QSP)); break;

		case FMT_UDECIMAL:	/* do something special for unsigned? */
		case FMT_DECIMAL:  SET_QS_NUMBER_FMT(THIS_QSP,QS_DFORMAT(THIS_QSP)); break;

		case FMT_HEX:  SET_QS_NUMBER_FMT(THIS_QSP,QS_XFORMAT(THIS_QSP)); break;

		case FMT_OCTAL:  SET_QS_NUMBER_FMT(THIS_QSP,QS_OFORMAT(THIS_QSP)); break;
		case FMT_POSTSCRIPT:
			/* does this make sense? */
WARN("set_fmt:  not sure what to do with FMT_POSTSCRIPT - using decimal.");
			SET_QS_NUMBER_FMT(THIS_QSP,QS_DFORMAT(THIS_QSP));
			break;
		default:
			assert( AERROR("set_fmt:  unexpected format code!?") );
			break;
	}
}

static COMMAND_FUNC( do_set_fmt )
{
	Number_Fmt i;

	if( var_fmt_list == NULL ) init_fmt_choices(SINGLE_QSP_ARG);

	i=(Number_Fmt)WHICH_ONE("print format for variable evaluation",
		N_PRINT_FORMATS,var_fmt_list);
	if( ((int)i) < 0 ) return;

	set_fmt(QSP_ARG  i);
}

static COMMAND_FUNC( do_push_fmt )
{
	Number_Fmt i;

	if( var_fmt_list == NULL ) init_fmt_choices(SINGLE_QSP_ARG);

	i=(Number_Fmt)WHICH_ONE("print format for variable evaluation",
		N_PRINT_FORMATS,var_fmt_list);
	if( ((int)i) < 0 ) return;

	CHECK_FMT_STRINGS

	if( QS_VAR_FMT_STACK(THIS_QSP) == NULL ){
		SET_QS_VAR_FMT_STACK(THIS_QSP,new_stack());
	}
	PUSH_TO_STACK( QS_VAR_FMT_STACK(THIS_QSP), QS_NUMBER_FMT(THIS_QSP) );

	set_fmt(QSP_ARG  i);
}

static COMMAND_FUNC( do_pop_fmt )
{
	//assert( QS_VAR_FMT_STACK(THIS_QSP) != NULL );
	if( QS_VAR_FMT_STACK(THIS_QSP) == NULL || STACK_IS_EMPTY(QS_VAR_FMT_STACK(THIS_QSP)) ){
		WARN("No variable format has been pushed, can't pop!?");
		return;
	}

	SET_QS_NUMBER_FMT( THIS_QSP, POP_FROM_STACK( QS_VAR_FMT_STACK(THIS_QSP) ) );
}

static COMMAND_FUNC( do_despace )
{
	char buf[LLEN];
	const char *vn;
	const char *src;
	int i=0;

	vn=NAMEOF("name for destination variable");
	src=NAMEOF("string to de-space");

	while( i<(LLEN-1) && *src ){
		if( isspace(*src) )
			buf[i++] = '_';
		else
			buf[i++] = *src;
		src++;
	}
	buf[i]=0;
	if( *src )
		WARN("despace:  input string too long, truncating.");

	ASSIGN_VAR(vn,buf);
}

static COMMAND_FUNC( do_find_vars )
{
	const char *s;

	s=NAMEOF("name fragment");

	find_vars(QSP_ARG  s);
}

static COMMAND_FUNC( do_search_vars )
{
	const char *s;

	s=NAMEOF("value fragment");

	search_vars(QSP_ARG  s);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(variables_menu,s,f,h)

MENU_BEGIN(variables)
ADD_CMD( set,		do_set_var,	set a variable	)
ADD_CMD( assign,	do_assign_var,	assign a variable from an expression	)
ADD_CMD( show,		do_show_var,	show the value of a variable	)
ADD_CMD( delete,	do_del_var,	delete a variable	)
ADD_CMD( list,		do_list_vars,	list all variables	)
ADD_CMD( digits,	do_set_nsig,	specify number of significant digits )
ADD_CMD( format,	do_set_fmt,	format for numeric vars )
ADD_CMD( push_fmt,	do_push_fmt,	push numeric format to stack )
ADD_CMD( pop_fmt,	do_pop_fmt,	pop numeric format from stack )
ADD_CMD( despace,	do_despace,	set variable to de-spaced string )
ADD_CMD( find,		do_find_vars,	find named variables	)
ADD_CMD( search,	do_search_vars,	search variable values	)
ADD_CMD( replace,	do_replace_var,	find and replace variable substring )
MENU_END(variables)



/************************ macros menu **************************/

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(macros_menu,s,f,h)

MENU_BEGIN(macros)
ADD_CMD( define,	do_def_mac,	define a macro	)
ADD_CMD( list,	do_list_macs,	list all macros	)
ADD_CMD( show,	do_show_mac,	show macro definition	)
ADD_CMD( dump,	do_dump_mac,	print executable macro definition 	)
ADD_CMD( info,	do_info_mac,	show macro information	)
ADD_CMD( find,	do_find_mac,	find named macros	)
ADD_CMD( search,	do_search_macs,	search macro bodies	)
ADD_CMD( delete,	do_del_mac,	delete macro )
ADD_CMD( nest,	do_allow_macro_recursion,	allow recursion for specified macro	)
ADD_CMD( invoked,	do_dump_invoked,	dump definitions of used macros )
MENU_END(macros)


static COMMAND_FUNC( do_var_menu )
{ PUSH_MENU(variables); }
static COMMAND_FUNC( do_mac_menu )
{ PUSH_MENU(macros); }

static COMMAND_FUNC( do_repeat )
{
	int n;
	n=(int)HOW_MANY("number of iterations");
	if( n <= 0 ){
		WARN("do_repeat:  number of repetitions must be positive!?");
		return;
	}

	OPEN_LOOP(n);
}

static COMMAND_FUNC( do_close_loop )
{
	//[THIS_QSP closeLoop ];
	CLOSE_LOOP;
}

static void whileloop(QSP_ARG_DECL   const char *exp_str)
{
	Typed_Scalar *tsp;

	tsp = pexpr(QSP_ARG  exp_str);
	if( has_zero_value(tsp) )
		_whileloop(QSP_ARG  0);
	else
		_whileloop(QSP_ARG  1);
	RELEASE_SCALAR(tsp);
}

static COMMAND_FUNC( do_while )
{
	const char *s;

	s=NAMEOF("expression");
	whileloop(QSP_ARG  s);
}

static COMMAND_FUNC( do_fore_loop )
{
	Foreach_Loop *frp;

	char delim[LLEN];
	char pmpt[LLEN];
	const char *s;

	frp = NEW_FOREACH_LOOP;

	s=NAMEOF("variable name");
	SET_FL_VARNAME(frp, savestr(s) );

	s=NAMEOF("opening delimiter, usually \"(\"");
	if( !strcmp(s,"(") )
		strcpy(delim,")");
	else
		strcpy(delim,s);

	sprintf(pmpt,"next value, or closing delimiter \"%s\"",delim);

	/* New style doesn't have a limit on the number of items */

	SET_FL_LIST(frp, new_list());

	while(1){
		s=NAMEOF(pmpt);
		if( !strcmp(s,delim) ){		/* end of list? */
			if( eltcount(FL_LIST(frp)) == 0 ) {		/* no items */
				sprintf(ERROR_STRING,
			"foreach:  no values specified for variable %s",
					FL_VARNAME(frp));
				WARN(ERROR_STRING);
				zap_fore(frp);
			} else {
				SET_FL_NODE(frp, QLIST_HEAD(FL_LIST(frp)) );
				fore_loop(QSP_ARG  frp);
			}
			return;
		} else {
			Node *np;
			np = mk_node( (void *) savestr(s) );
			addTail(FL_LIST(frp),np);
		}
	}

}

static COMMAND_FUNC( do_do_loop )
{
	OPEN_LOOP(-1);
}

static COMMAND_FUNC( do_error_exit )
{
	const char *s;

	s=NAMEOF("error message");
	ERROR1(s);
}

static COMMAND_FUNC( do_abort_prog )
{
	abort();	// used mainly for debugging
}


COMMAND_FUNC( do_show_prompt )
{
	sprintf(MSG_STR,"Current prompt is:  %s", QS_CMD_PROMPT_STR(THIS_QSP));
	prt_msg(MSG_STR);
}

static COMMAND_FUNC( do_advise )
{
	Quip_String *s;
	s=NAMEOF("word to echo");
	ADVISE(s);
}

static COMMAND_FUNC( do_log_message )
{
	Quip_String *s;
	s=NAMEOF("message");
	log_message(s);
}

static COMMAND_FUNC( do_debug )
{
	Debug_Module *dmp;

	dmp = pick_debug(QSP_ARG  "");
	if( dmp != NULL )
		set_debug(QSP_ARG  dmp);
}

static COMMAND_FUNC( do_echo )
{
	Quip_String *s;

	s=NAMEOF("word to echo");
	prt_msg(s);
}

static COMMAND_FUNC( do_list_current_menu )
{
	list_menu( QSP_ARG  TOP_OF_STACK( QS_MENU_STACK(THIS_QSP) ) );
	list_menu( QSP_ARG  QS_HELP_MENU(THIS_QSP) );
}

static COMMAND_FUNC( do_list_builtin_menu )
{
	list_menu( QSP_ARG  QS_BUILTIN_MENU(THIS_QSP) );
	list_menu( QSP_ARG  QS_HELP_MENU(THIS_QSP) );
}

static COMMAND_FUNC(do_nop)
{

}

static COMMAND_FUNC( do_exit_file )
{
	exit_current_file(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_exit_macro )
{
	exit_current_macro(SINGLE_QSP_ARG);
}

// We used to call assign_var here, but now verbose is a "dynamic" var

static COMMAND_FUNC( do_verbose )
{
	if( ASKIF("print verbose messages") ){
		set_verbose(SINGLE_QSP_ARG);
	} else {
		clr_verbose(SINGLE_QSP_ARG);
	}
}

// For now use these in unix for ios emulation...

static COMMAND_FUNC( do_cd )
{
	const char *s;

	s=NAMEOF("directory");
	if( s == NULL || strlen(s)==0 ) return;

	if( chdir(s) < 0 ){
		tell_sys_error("chdir");
		sprintf(ERROR_STRING,"Failed to chdir to %s",s);
		WARN(ERROR_STRING);
	}
	// should we cache in a variable such as $cwd?
}

static COMMAND_FUNC( do_ls )
{
	DIR *dir_p;
	struct dirent *di_p;

	dir_p = opendir(".");
	if( dir_p == NULL ){
		WARN("Failed to open directory.");
		return;
	}
	di_p = readdir(dir_p);
	while( di_p != NULL ){
		advise(di_p->d_name);
		di_p = readdir(dir_p);
	}
	if( closedir(dir_p) < 0 ){
		WARN("Error closing directory!?");
		return;
	}
}

// like do_ls, but stores filenames in a data object...
// This is not needed on unix, where we can use popen with ls...

static COMMAND_FUNC( do_get_filenames )
{
	const char *dir;
	const char *objname;
	DIR *dir_p;
	struct dirent *di_p;
	Data_Obj *dp;
	int i,n;
	int maxlen=0;
	Dimension_Set ds1;

	objname = NAMEOF("name for new object in which to store filenames");
	dir = NAMEOF("directory");

	dp = dobj_of(QSP_ARG  objname);
	if( dp != NULL ){
		sprintf(ERROR_STRING,
	"get_filenames:  object %s already exists!?",OBJ_NAME(dp));
		advise(ERROR_STRING);

		//return;
		// This used to be a warning and an error,
		// but here we provide some garbage collection...
		delvec( QSP_ARG  dp);
	}

	dir_p = opendir(dir);
	if( dir_p == NULL ){
		sprintf(ERROR_STRING,
	"get_filenames:  failed to open directory '%s'",dir);
		WARN(ERROR_STRING);
		return;
	}

	di_p = readdir(dir_p);
	n=0;
	while( di_p != NULL ){
		// exclude . .. and hidden files
		if( *(di_p->d_name) != '.' ){
			if( strlen(di_p->d_name) >= maxlen )
				maxlen = (int)strlen(di_p->d_name) + 1;
			n++;
		}
		di_p = readdir(dir_p);
	}
//fprintf(stderr,"directory has %d entries, maxlen = %d\n",n,maxlen);

	if( n == 0 ) goto finish;

	// now rewind and store the names
	SET_DIMENSION(&ds1,4,1);
	SET_DIMENSION(&ds1,3,1);
	SET_DIMENSION(&ds1,2,1);
	SET_DIMENSION(&ds1,1,n);
	SET_DIMENSION(&ds1,0,maxlen);

	dp = make_dobj(QSP_ARG  objname, &ds1,PREC_FOR_CODE(PREC_STR));
	if( dp == NULL ) goto finish;

	rewinddir(dir_p);
	for(i=0;i<n;i++){
		char *dst;

		di_p = readdir(dir_p);
		// Can readdir ever fail here?
		assert( di_p != NULL );

		if( *(di_p->d_name) != '.' ){
			dst = (char *) OBJ_DATA_PTR(dp);
			dst += i * OBJ_TYPE_INC(dp,1);
			strcpy(dst,di_p->d_name);
		} else {
			i--;
		}
	}

finish:
	if( closedir(dir_p) < 0 ){
		WARN("Error closing directory!?");
		return;
	}
}

static COMMAND_FUNC( do_mkdir )
{
	const char *s;
	mode_t mode;

	s=NAMEOF("name for new directory");

	if( s == NULL || *s==0 ) return;

	mode = S_IRWXU | S_IROTH | S_IXOTH | S_IRGRP | S_IXGRP ;

	if( mkdir(s,mode) < 0 ){
		tell_sys_error("mkdir");
		WARN("Error creating new directory!?");
		return;
	}
}

static COMMAND_FUNC( do_pwd )
{
	char buf[LLEN];

	if( getcwd(buf,LLEN) == NULL ){
		tell_sys_error("getcwd");
		return;
	}

	// cwd is now a dynamic variable!?
	//ASSIGN_VAR("cwd",savestr(buf));
	prt_msg(buf);
}

static COMMAND_FUNC( do_rmdir )
{
	const char *s;

	s=NAMEOF("name of directory");

	if( s == NULL || *s==0 ) return;

	if( rmdir(s) < 0 ){
		tell_sys_error("rmdir");
		sprintf(ERROR_STRING,"Error removing directory %s!?",s);
		WARN(ERROR_STRING);
		return;
	}
}

static COMMAND_FUNC( do_rm )
{
	const char *s;

	s=NAMEOF("name of file");

	if( s == NULL || *s==0 ) return;

	if( unlink(s) < 0 ){
		tell_sys_error("unlink");
		sprintf(ERROR_STRING,"Error removing file %s!?",s);
		WARN(ERROR_STRING);
		return;
	}
}

static COMMAND_FUNC( do_rm_all )
{
	DIR *dir_p;
	struct dirent *di_p;

	// BUG For security, we should make sure that the
	// current directory is a subdirectory of $BUNDLE_DIR/Documents...

	dir_p = opendir(".");
	if( dir_p == NULL ){
		WARN("Failed to open directory.");
		return;
	}

	di_p = readdir(dir_p);
	while( di_p != NULL ){
		const char *fn;

		fn=di_p->d_name;
		if( strcmp(fn,".") && strcmp(fn,"..") ){
			if( verbose ){
				sprintf(ERROR_STRING,"Removing %s",fn);
				advise(ERROR_STRING);
			}
			if( unlink(fn) < 0 ){
				tell_sys_error("unlink");
				sprintf(ERROR_STRING,"Error removing file %s!?",fn);
				WARN(ERROR_STRING);
				return;
			}
		}
		di_p = readdir(dir_p);
	}
	if( closedir(dir_p) < 0 ){
		WARN("Error closing directory!?");
		return;
	}

}

static COMMAND_FUNC( do_count_lines )
{
	int c,n;
	FILE *fp;
	const char *fn;
	const char *vn;

	vn=NAMEOF("variable name for result");
	fn=NAMEOF("filename");
	fp = try_open(QSP_ARG  fn,"r");
	if( !fp ) return;

	/* Now count the lines in the file */
	n=0;
	while( (c=getc(fp)) != EOF ){
		if( c == '\n' ) n++;
	}
	fclose(fp);

	// we co-opt error_string as an available buffer...
	sprintf(ERROR_STRING,"%d",n);
	ASSIGN_VAR(vn,ERROR_STRING);
}

// For now use these in unix for ios emulation...


// why does this have to be a global var???
// We have to remember the open file so we can close it
// when we switch...
static const char *open_mode_string[2]={"w","a"};

// This should also probably be per-qsp
//static const char *output_file_name=NULL;

static void set_output_file(QSP_ARG_DECL  const char *s)
{
	FILE *fp;

	if( QS_OUTPUT_FILENAME(qsp) == NULL ){	/* first time? */
		if( (!strcmp(s,"-")) || (!strcmp(s,"stdout")) ){
			/* stdout should be initially open */
			return;
		}
	} else if( !strcmp( QS_OUTPUT_FILENAME(qsp),s) ){	/* same file? */
/*
sprintf(ERROR_STRING,"set_output_file %s, doing nothing",s);
advise(ERROR_STRING);
*/
		return;
	}
	/* output_redir will close the current file... */

	SET_QS_OUTPUT_FILENAME(THIS_QSP,s);

	if( (!strcmp(s,"-")) || (!strcmp(s,"stdout")) )
		fp=stdout;
	else {
		fp=TRYNICE(s,open_mode_string[ APPEND_FLAG ]);
	}

	if( !fp ) return;

	output_redir(QSP_ARG  fp);
}

static COMMAND_FUNC( do_output_redir )
{
	const char *s;

	s=NAMEOF("output file");
	set_output_file(QSP_ARG  s);
}


static COMMAND_FUNC( do_append )
{
	if( ASKIF("append output and error redirect files") )
		SET_APPEND_FLAG(1)
	else
		SET_APPEND_FLAG(0)
}

static COMMAND_FUNC( do_error_redir )
{
	FILE *fp;
	const char *s;

	s=NAMEOF("error file");

	if( (!strcmp(s,"-")) || (!strcmp(s,"stderr")) )
		fp=stderr;
	else {
		fp=TRYNICE(s,open_mode_string[ APPEND_FLAG ]);
	}

	if( !fp ) return;

	error_redir(QSP_ARG  fp);
}

static COMMAND_FUNC( do_usleep )
{
#ifdef HAVE_USLEEP
	int n;

	n=(int)HOW_MANY("number of microseconds to sleep");
	usleep(n);
#else // ! HAVE_USLEEP
    HOW_MANY("number of microseconds to sleep");
#ifdef BUILD_FOR_IOS
	advise("Sorry, no usleep function in this build!?");
#else	// ! BUILD_FOR_IOS
	WARN("Sorry, no usleep function in this build!?");
#endif // ! BUILD_FOR_IOS
#endif // ! HAVE_USLEEP
}

static COMMAND_FUNC( do_sleep )
{
	int n;

	n=(int)HOW_MANY("number of seconds to sleep");
#ifdef HAVE_SLEEP
	sleep(n);
#else // ! HAVE_SLEEP
	WARN("Sorry, no sleep available in this build!?");
#endif // ! HAVE_SLEEP
}

static COMMAND_FUNC( do_alarm )
{
	float f;
	const char *s;

	// Should we have named alarms???

	f=(float)HOW_MUCH("number of seconds before alarm");
	s=NAMEOF("script to execute on alarm");

	set_alarm_script(QSP_ARG  s);
	set_alarm_time(QSP_ARG  f);
}

#ifdef FOOBAR

/* Unix style alarm implementation - put this somewhere else */

static const char *timer_script = NULL;

static void my_alarm(int x)
{
	sprintf(DEFAULT_ERROR_STRING,"my_alarm:  arg is %d",x);
	advise(DEFAULT_ERROR_STRING);
}

static COMMAND_FUNC( do_alarm )
{
	float f;
	const char *s;
	struct itimerval itv;
	int status;

	f=HOW_MUCH("number of seconds before alarm");
	s=NAMEOF("script to execute on alarm");

	if( timer_script != NULL ){
		rls_str(timer_script);
	}

	timer_script = savestr(s);

	if( f <= 0 ){
		// cancel any pending alarm...
		f=0;
	}

	// make it_interval non-zero for recurring alarms
	itv.it_interval.tv_sec = 0;
	itv.it_interval.tv_usec = 0;

	itv.it_value.tv_sec = floor(f);
	itv.it_value.tv_usec = floor( 1000000 * ( f - floor(f) ) );

/*	if( signal(SIGARLM,my_alarm) == SIG_ERR ){
		tell_sys_error("signal");
		return;
	}
*/
	if( (status=setitimer(ITIMER_REAL,&itv,NULL)) < 0 ){
		tell_sys_error("setitimer");
	}
}

#endif /* FOOBAR */

static COMMAND_FUNC( do_copy_cmd )
{
	FILE *fp;

	fp=TRYNICE( nameof(QSP_ARG  "transcript file"), "w" );
	if( fp ) {
		if(dupout(QSP_ARG fp)==(-1))
			fclose(fp);
	}
}

static COMMAND_FUNC( do_report_version )
{
	sprintf(MSG_STR,"%s version:  %s (%s)",tell_progname(),tell_version(),
		QUIP_UTC_BUILD_DATE);
	prt_msg(MSG_STR);
}

// This pushes the top menu - we need another command to clear
// the menu stack if necessary!?

static COMMAND_FUNC( do_top_menu )
{
	push_top_menu( SINGLE_QSP_ARG );
}

/*
 * This function was added, so that the output could be redirected
 * to the programs output file (which didn't work w/ "system date")
 *
 * Should this print to stdout or stderr???
 * We might like to be able to do either!!!
 */

static COMMAND_FUNC( do_date )
{
	const char *s;
	s=get_date_string(SINGLE_QSP_ARG);
	//advise(s);
	prt_msg(s);
}

#define N_TIMEZONE_CHOICES	2
static const char *tz_choices[N_TIMEZONE_CHOICES]={
	"local",
	"UTC"
};

static COMMAND_FUNC( do_timezone )
{
	int i;

	i=WHICH_ONE("time zone",N_TIMEZONE_CHOICES,tz_choices);
	switch(i){
		case 0:
			CLEAR_QS_FLAG_BITS(THIS_QSP,QS_TIME_FMT_UTC);
			break;
		case 1:
			SET_QS_FLAG_BITS(THIS_QSP,QS_TIME_FMT_UTC);
			break;
		default:
			assert( AERROR("do_timezone:  bad timezone !?") );

			break;
	}
}

#ifdef QUIP_DEBUG

static COMMAND_FUNC( do_dump_items ){ dump_items(SINGLE_QSP_ARG); }

static COMMAND_FUNC( do_qtell )
{
	sprintf(ERROR_STRING,"qlevel=%d",QLEVEL);
	advise(ERROR_STRING);
}

static COMMAND_FUNC( do_cd_tell )
{
	sprintf(msg_str,"cmd_depth = %d",STACK_DEPTH(QS_MENU_STACK(THIS_QSP)) );
	prt_msg(msg_str);
}

#endif // QUIP_DEBUG

static COMMAND_FUNC( do_seed )
{
	u_long n;

	n=HOW_MANY("value for random number seed");

	sprintf(msg_str,"Using user-supplied seed of %ld (0x%lx)",n,n);
	advise(msg_str);

	set_seed(QSP_ARG  n);
}

static COMMAND_FUNC( do_pmpttext )
{
	const char *p;
#ifndef BUILD_FOR_OBJC
	const char *t;
#endif // BUILD_FOR_OBJC
	const char *s;

	p=savestr( NAMEOF("prompt string") );
	s=savestr( NAMEOF("variable name") );
#ifndef BUILD_FOR_OBJC
	//push_input_file(QSP_ARG   "-" );
	redir(QSP_ARG  tfile(SINGLE_QSP_ARG), "-" );
	t=savestr( NAMEOF(p) );
	pop_file(SINGLE_QSP_ARG);
	ASSIGN_VAR(s,t);
	rls_str(t);
#else // BUILD_FOR_OBJC
	WARN("Sorry, pmpttext (bi_menu.c) not yet implemented!?");
#endif // BUILD_FOR_OBJC
	rls_str(p);
	rls_str(s);
}

static COMMAND_FUNC( my_quick_exit ){ exit(0); }

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(builtin_menu,s,f,h)
MENU_BEGIN(builtin)
ADD_CMD( echo,		do_echo,	echo a word		)
ADD_CMD( advise,	do_advise,	echo a word to stderr	)
ADD_CMD( log_message,	do_log_message,	print a log message to stderr	)
ADD_CMD( repeat,	do_repeat,	open an iterative loop	)
ADD_CMD( end,		do_close_loop,	close a loop		)
ADD_CMD( foreach,	do_fore_loop,	iterate over a set of words	)
ADD_CMD( do,		do_do_loop,	open a loop		)
ADD_CMD( while,		do_while,	conditionally close a loop	)
ADD_CMD( variables,	do_var_menu,	variables submenu	)
ADD_CMD( macros,	do_mac_menu,	macros submenu		)
ADD_CMD( expect_warning,	do_expect_warning,	specify expected warning	)
ADD_CMD( check_expected_warning,	do_check_expected_warning,	check for expected warning	)
ADD_CMD( warn,		do_warn,	print a warning message	)
ADD_CMD( <,		do_redir,	read commands from a file	)
ADD_CMD( >,		do_copy_cmd,	copy commands to a transcript file	)
ADD_CMD( If,		do_if,		conditionally execute a command	)
ADD_CMD( interpret,	do_push_text,	execute a command string	)
ADD_CMD( prompt_text,	do_pmpttext,	prompt the user for a string )
ADD_CMD( max_warnings,	do_set_max_warnings,	set maximum number of warnings	)
ADD_CMD( output_file,	do_output_redir,	redirect text output )
ADD_CMD( append,	do_append,	set/clear append flag )
ADD_CMD( error_file,	do_error_redir,	redirect error messages )
ADD_CMD( top_menu,	do_top_menu,	push top menu )
ADD_CMD( exit_macro,	do_exit_macro,	exit current macro )
ADD_CMD( exit_file,	do_exit_file,	exit current file )
//ADD_CMD( PopFile,	do_pop_file,	pop input file(s)	)
ADD_CMD( items,		do_items,	item submenu		)
ADD_CMD( rbtree_test,	do_rbtree_test,	test red-black tree functions)
ADD_CMD( version,	do_report_version,	report software version of this build )
ADD_CMD( features,	do_list_features,	list features present in this build )
ADD_CMD( set_seed,	do_seed,	set random number generator seed )
ADD_CMD( date,		do_date,	print date and time )
ADD_CMD( time_zone,	do_timezone,	select local time or UTC )
ADD_CMD( show_prompt,	do_show_prompt,	display the current prompt )
ADD_CMD( pwd,		do_pwd,		display working directory )
ADD_CMD( cd,		do_cd,		change working directory )
ADD_CMD( ls,		do_ls,		list working directory )
ADD_CMD( mkdir,		do_mkdir,	create new directory )
ADD_CMD( rmdir,		do_rmdir,	remove directory )
ADD_CMD( rm,		do_rm,		remove file )
ADD_CMD( rm_all,	do_rm_all,	remove all files from current dir )
ADD_CMD( count_lines,	do_count_lines,	count the lines in a file )
ADD_CMD( get_filenames,	do_get_filenames,	place filenames in a data object )
ADD_CMD( encrypt_string,	do_encrypt_string,	encrypt a string )
ADD_CMD( decrypt_string,	do_decrypt_string,	decrypt a string )
ADD_CMD( encrypt_file,	do_encrypt_file,	encrypt a file )
ADD_CMD( decrypt_file,	do_decrypt_file,	decrypt a file )
ADD_CMD( read_encrypted_file,	do_read_encrypted_file,	read an encrypted file )
ADD_CMD( debug,		do_debug,	enable debug module	)
ADD_CMD( nop,       do_nop,     do nothing )
ADD_CMD( allow_events,	suspend_chewing,	process event actions immediately )
ADD_CMD( disallow_events,	unsuspend_chewing,	process event actions sequentially )
#undef verbose
ADD_CMD( verbose,	do_verbose,	enable/disable verbose messages )
ADD_CMD( exit,		do_exit_prog,	exit program		)
ADD_CMD( fast_exit,	my_quick_exit,	exit without resetting tty )
ADD_CMD( abort,		do_abort_prog,	abort program		)
ADD_CMD( error_exit,	do_error_exit,	print error message and exit )
ADD_CMD( sleep,		do_sleep,	sleep for a while )
ADD_CMD( usleep,	do_usleep,	sleep for a short while )
ADD_CMD( alarm,		do_alarm,	execute script after delay )
ADD_CMD( os,		do_unix_menu,	OS specific functions )

#ifdef QUIP_DEBUG
ADD_CMD( dump_items,	do_dump_items,	list all items )
ADD_CMD( Qstack,	qdump,		dump state of query stack )
ADD_CMD( tellq,		do_qtell,	report level of query stack )
ADD_CMD( cmd_depth,	do_cd_tell,	report level of command stack )
#ifdef USE_GETBUF
ADD_CMD( heaps,		heap_report,	report free heap mem )
#endif
#endif /* QUIP_DEBUG */

MENU_END(builtin)


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(help_menu,s,f,h)
MENU_BEGIN(help)
ADD_CMD( ?,	do_list_current_menu,	list commands in current menu	)
ADD_CMD( ??,	do_list_builtin_menu,	list commands in builtin menu	)
MENU_SIMPLE_END(help)

void init_builtins(void)
{
	// Normally we do not have to call the init functions,
	// as it is done automatically by the macro PUSH_MENU,
	// but these menus are never pushed, so we do it here.
	init_help_menu();
	init_builtin_menu();

	SET_QS_BUILTIN_MENU(DEFAULT_QSP,builtin_menu);
	SET_QS_HELP_MENU(DEFAULT_QSP,help_menu);
}

// used in os/threads.c

void init_aux_menus(Query_Stack *qsp)
{
	SET_QS_BUILTIN_MENU(qsp,builtin_menu);
	SET_QS_HELP_MENU(qsp,help_menu);
}


/*********************/

#ifdef USED_ONLY_FOR_CALIBRATION

// This short program can be used to determine a useful value
// of MAX_ROUNDABLE_DOUBLE

#include <stdio.h>
#include <math.h>

int main(int ac, char **av)
{
	double e,d=1;

	do {
		d = floor(d*2);
		d += 0.1;

		e = d - round(d);

		fprintf(stderr,"d = %g, e = %g\n",d,e);
	} while( ! isinf(d) && e > 0 );
}

#endif // USED_ONLY_FOR_CALIBRATION
