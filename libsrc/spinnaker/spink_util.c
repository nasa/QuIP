/* Jeff's interface to the 1394 subsystem usign PGR's libflycap_c */

#include "quip_config.h"

#include "quip_prot.h"
#include "function.h"
#include "data_obj.h"

#include <stdio.h>
#include <string.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>		/* usleep */
#endif

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

#include "spink.h"

#ifdef HAVE_LIBSPINNAKER

// some globals...
static Spink_Map *current_map=NULL;
#define MAX_TREE_DEPTH	5
static Spink_Node *current_parent_p[MAX_TREE_DEPTH]={NULL,NULL,NULL,NULL};
static int use_display_names=0;
static size_t numInterfaces = 0;

static spinSystem hSystem = NULL;
static spinInterfaceList hInterfaceList = NULL;
spinCameraList hCameraList = NULL;

#endif // HAVE_LIBSPINNAKER

int current_node_idx; 


size_t numCameras = 0;

ITEM_INTERFACE_DECLARATIONS(Spink_Interface,spink_interface,RB_TREE_CONTAINER)
ITEM_INTERFACE_DECLARATIONS(Spink_Cam,spink_cam,RB_TREE_CONTAINER)
ITEM_INTERFACE_DECLARATIONS(Spink_Map,spink_map,RB_TREE_CONTAINER)
ITEM_INTERFACE_DECLARATIONS(Spink_Node,spink_node,RB_TREE_CONTAINER)
ITEM_INTERFACE_DECLARATIONS(Spink_Node_Type,spink_node_type,RB_TREE_CONTAINER)
ITEM_INTERFACE_DECLARATIONS(Spink_Category,spink_cat,RB_TREE_CONTAINER)
ITEM_INTERFACE_DECLARATIONS(Chunk_Data,chunk_data,RB_TREE_CONTAINER)

#define UNIMP_FUNC(name)						\
	sprintf(ERROR_STRING,"Function %s is not implemented!?",name);	\
	warn(ERROR_STRING);


#ifdef HAVE_LIBSPINNAKER

Item_Context * _pop_spink_node_context(SINGLE_QSP_ARG_DECL)
{
	Item_Context *icp;
	if( spink_node_itp == NULL ) init_spink_nodes();
	icp = pop_item_context(spink_node_itp);
	return icp;
}

void _push_spink_node_context(QSP_ARG_DECL  Item_Context *icp)
{
	if( spink_node_itp == NULL ) init_spink_nodes();
	push_item_context(spink_node_itp,icp);
}

Item_Context * _pop_spink_cat_context(SINGLE_QSP_ARG_DECL)
{
	Item_Context *icp;
	if( spink_cat_itp == NULL ) init_spink_cats();
	icp = pop_item_context(spink_cat_itp);
	return icp;
}

void _push_spink_cat_context(QSP_ARG_DECL  Item_Context *icp)
{
	if( spink_cat_itp == NULL ) init_spink_cats();
	push_item_context(spink_cat_itp,icp);
}

// Don't we already have this???

static void substitute_char(char *buf,char find, char replace)
{
	char *s;

	s=buf;
	while( *s ){
		if( *s == find )
			*s = replace;
		s++;
	}
}

#define get_unique_cam_name(buf, buflen) _get_unique_cam_name(QSP_ARG  buf, buflen)

static int _get_unique_cam_name(QSP_ARG_DECL  char *buf, int buflen)
{
	int i=2;
	int orig_len;
	Spink_Cam *skc_p;

	orig_len = strlen(buf);
	if( orig_len+3 > buflen ){
		sprintf(ERROR_STRING,
			"Camera name buffer needs to be enlarged to accomodate multiple instances of '%s'!?",
			buf);
		error1(ERROR_STRING);
	}
	buf[orig_len]='_';
	buf[orig_len+2]=0;
	while(i<10){
		buf[orig_len+1]='0'+i;	// 2-9
		skc_p = spink_cam_of(buf);
		if( skc_p == NULL ) return 0;
		i++;
	}
	return -1;
}

#define INVALID_SET_FUNC(name)									\
static void _set_##name##_node_from_script(QSP_ARG_DECL  Spink_Node *skn_p)					\
{												\
	sprintf(ERROR_STRING,"set_%s_node:  %s nodes should never be set!?",#name,#name);	\
	error1(ERROR_STRING);									\
}

INVALID_SET_FUNC(category)
INVALID_SET_FUNC(register)
INVALID_SET_FUNC(port)
INVALID_SET_FUNC(base)
INVALID_SET_FUNC(unknown)
INVALID_SET_FUNC(enum_entry)

// These need to be implemented...
INVALID_SET_FUNC(value)

//get_node_max_value_float		FloatGetMax
//get_node_max_value_int		IntegerGetMax
//set_node_value_float		FloatSetValue
//set_node_value_int		IntegerSetValue
//set_node_value_string		StringSetValue

//set_node_value_string		StringSetValue

//command_is_done		CommandIsDone
//exec_spink_command		CommandExecute


#define set_command_node(skn_p) _set_command_node(QSP_ARG  skn_p)

static void _set_command_node(QSP_ARG_DECL  Spink_Node *skn_p)
{
	spinNodeHandle hNode;
	bool8_t done;

	assert(skn_p->skn_type_p->snt_type == CommandNode);

	if( lookup_spink_node(skn_p, &hNode) < 0 || exec_spink_command(hNode) < 0 ){
		sprintf(ERROR_STRING,"Error executing %s",skn_p->skn_name);
		warn(ERROR_STRING);
	}
	// wait here for command to finish
	do {
		if( command_is_done(hNode,&done) < 0 ) return;
	} while( ! done );
}

static void _set_command_node_from_script(QSP_ARG_DECL  Spink_Node *skn_p)
{
	set_command_node(skn_p);
}

#define set_string_node(skn_p, s) _set_string_node(QSP_ARG  skn_p, s)

static void _set_string_node(QSP_ARG_DECL  Spink_Node *skn_p, const char *s)
{
	spinNodeHandle hNode;

	assert(skn_p->skn_type_p->snt_type == StringNode);

	// Does StringSetValue make a deep copy???  If not, we need to save the string before passing!?
	if( lookup_spink_node(skn_p, &hNode) < 0 || set_node_value_string(hNode,s) < 0 ){
		sprintf(ERROR_STRING,"Error setting %s",skn_p->skn_name);
		warn(ERROR_STRING);
	}
}

static void _set_string_node_from_script(QSP_ARG_DECL  Spink_Node *skn_p)
{
	const char *s;

	s = nameof(skn_p->skn_name);

	set_string_node(skn_p,s);
}

#define get_float_range(skn_p, min_p, max_p) _get_float_range(QSP_ARG  skn_p, min_p, max_p)

static void _get_float_range(QSP_ARG_DECL  Spink_Node *skn_p, double *min_p, double *max_p)
{
	spinNodeHandle hNode;

	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( get_node_min_value_float(hNode,min_p) < 0 ) return;
	if( get_node_max_value_float(hNode,max_p) < 0 ) return;
}

#define get_int_range(skn_p, min_p, max_p) _get_int_range(QSP_ARG  skn_p, min_p, max_p)

static void _get_int_range(QSP_ARG_DECL  Spink_Node *skn_p, int64_t *min_p, int64_t *max_p)
{
	spinNodeHandle hNode;

	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( get_node_min_value_int(hNode,min_p) < 0 ) return;
	if( get_node_max_value_int(hNode,max_p) < 0 ) return;
}

#define set_float_node(skn_p, dval) _set_float_node(QSP_ARG  skn_p, dval)

static void _set_float_node(QSP_ARG_DECL  Spink_Node *skn_p, double dval)
{
	spinNodeHandle hNode;

	assert(skn_p->skn_type_p->snt_type == FloatNode);

	if( lookup_spink_node(skn_p, &hNode) < 0 || set_node_value_float(hNode,dval) < 0 ){
		sprintf(ERROR_STRING,"Error setting %s",skn_p->skn_name);
		warn(ERROR_STRING);
	}
}

static void _set_float_node_from_script(QSP_ARG_DECL  Spink_Node *skn_p)
{
	double minv, maxv;
	double dval;
	char pmpt[LLEN];

	assert(skn_p->skn_type_p->snt_type == FloatNode);

	get_float_range(skn_p,&minv,&maxv);
	sprintf(pmpt,"%s (%g-%g)",skn_p->skn_name,minv,maxv);
	dval = how_much(pmpt);

	set_float_node(skn_p,dval);
}

#define set_integer_node(skn_p, ival) _set_integer_node(QSP_ARG  skn_p, ival)

static void _set_integer_node(QSP_ARG_DECL  Spink_Node *skn_p, int64_t ival)
{
	spinNodeHandle hNode;

	assert(skn_p->skn_type_p->snt_type == IntegerNode);

	if( lookup_spink_node(skn_p, &hNode) < 0 || set_node_value_int(hNode,ival) < 0 ){
		sprintf(ERROR_STRING,"Error setting %s",skn_p->skn_name);
		warn(ERROR_STRING);
	}
}

static void _set_integer_node_from_script(QSP_ARG_DECL  Spink_Node *skn_p)
{
	int64_t minv, maxv;
	int64_t ival;
	char pmpt[LLEN];

	assert(skn_p->skn_type_p->snt_type == IntegerNode);

	get_int_range(skn_p,&minv,&maxv);
	sprintf(pmpt,"%s (%ld-%ld)",skn_p->skn_name,minv,maxv);
	ival = how_many(pmpt);

	set_integer_node(skn_p,ival);
}

#define set_boolean_node(skn_p, flag) _set_boolean_node(QSP_ARG  skn_p, flag)

static void _set_boolean_node(QSP_ARG_DECL  Spink_Node *skn_p, bool8_t flag)
{
	spinNodeHandle hNode;

	assert(skn_p->skn_type_p->snt_type == BooleanNode);
	
	if( lookup_spink_node(skn_p, &hNode) < 0 || set_node_value_bool(hNode,flag) < 0 ){
		sprintf(ERROR_STRING,"Error setting %s",skn_p->skn_name);
		warn(ERROR_STRING);
	}
}

static void _set_boolean_node_from_script(QSP_ARG_DECL  Spink_Node *skn_p)
{
	char pmpt[LLEN];
	bool8_t flag;

	sprintf(pmpt,"%s",skn_p->skn_name);
	if( askif(pmpt) )
		flag = TRUE;
	else
		flag = FALSE;

	set_boolean_node(skn_p,flag);
}


// The enumeration entry nodes have names like EnumerationEntry_GainAuto_Once - but we
// want to make the choice be "Once", so we skip ahead past the second underscore...

static void make_enumeration_choices(const char ***tbl_ptr, int *nc_p, Spink_Node *skn_p)
{
	int n;
	Node *np;
	const char **tbl;
	const char *s;

	assert(skn_p!=NULL);
	assert(skn_p->skn_children!=NULL);
	assert(skn_p->skn_type_p->snt_type == EnumerationNode);

	n = eltcount(skn_p->skn_children);
	assert(n>0);
	*nc_p = n;
	tbl = getbuf( n * sizeof(char *) );
	*tbl_ptr = tbl;
	np = QLIST_HEAD(skn_p->skn_children);
	
	while(np!=NULL){
		Spink_Node *child;
		child = NODE_DATA(np);
		s = child->skn_name;
		while( *s && *s != '_' ) s++;
		assert(*s=='_');
		s++;	// skip first
		while( *s && *s != '_' ) s++;
		assert(*s=='_');
		s++;	// skip second
		assert(*s!=0);
		*tbl = s;
		tbl++;
		np = NODE_NEXT(np);
	}
}

#define set_enumeration_node(skn_p, child) _set_enumeration_node(QSP_ARG  skn_p, child)

static void _set_enumeration_node(QSP_ARG_DECL  Spink_Node *skn_p, Spink_Node *child)
{
	spinNodeHandle hNode;

//fprintf(stderr,"set_enumeration_node:  child = %s, type = %s\n", child->skn_name,child->skn_type_p->snt_name);
	assert(skn_p->skn_type_p->snt_type == EnumerationNode);
	assert(child->skn_type_p->snt_type == EnumEntryNode);
	assert( child->skn_enum_ival != INVALID_ENUM_INT_VALUE );

	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( set_enum_int_val(hNode,child->skn_enum_ival) < 0 )
		warn("Error setting enum value!?");
}

static void _set_enumeration_node_from_script(QSP_ARG_DECL  Spink_Node *skn_p)
{
	int idx;
	const char **choices;
	int n_choices;
	Spink_Node *child;
	Node *np;

	make_enumeration_choices(&choices,&n_choices,skn_p);
	idx = which_one(skn_p->skn_name,n_choices,choices);
	givbuf(choices);

	if( idx < 0 ) return;

	// now find the enum node
	np = nth_elt(skn_p->skn_children,idx);
	assert(np!=NULL);
	child = NODE_DATA(np);

	set_enumeration_node(skn_p,child);
}



#define INVALID_PRINT_VALUE_FUNC(name)								\
static void _print_##name##_node_value(QSP_ARG_DECL  Spink_Node *skn_p)				\
{												\
	sprintf(ERROR_STRING,"print_%s_node_value:  %s nodes cannot be printed!?",#name,#name);	\
	error1(ERROR_STRING);									\
}


//INVALID_PRINT_VALUE_FUNC(register)
//INVALID_PRINT_VALUE_FUNC(enum_entry)
INVALID_PRINT_VALUE_FUNC(port)
INVALID_PRINT_VALUE_FUNC(base)
INVALID_PRINT_VALUE_FUNC(unknown)

static void _print_register_node_value(QSP_ARG_DECL  Spink_Node *skn_p)
{
	sprintf(MSG_STR,STRING_NODE_FMT_STR,"(unhandled case!?)");
	prt_msg_frag(MSG_STR);
}

static void _print_enum_entry_node_value(QSP_ARG_DECL  Spink_Node *skn_p)
{
	sprintf(MSG_STR,STRING_NODE_FMT_STR,"");
	prt_msg_frag(MSG_STR);
}

static void _print_category_node_value(QSP_ARG_DECL  Spink_Node *skn_p)
{
	sprintf(MSG_STR,STRING_NODE_FMT_STR,"");
	prt_msg_frag(MSG_STR);
}

static void _print_value_node_value(QSP_ARG_DECL  Spink_Node *skn_p)
{
	char val_buf[MAX_BUFF_LEN];
	size_t buf_len = MAX_BUFF_LEN;
	spinNodeHandle hNode;
	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( get_node_value_string(val_buf,&buf_len,hNode) < 0 ) return;
	sprintf(MSG_STR,STRING_NODE_FMT_STR,val_buf);
	prt_msg_frag(MSG_STR);
}

static void _print_string_node_value(QSP_ARG_DECL  Spink_Node *skn_p)
{
	char val_buf[MAX_BUFF_LEN];
	size_t buf_len = MAX_BUFF_LEN;
	spinNodeHandle hNode;
	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( get_string_node_string(val_buf,&buf_len,hNode) < 0 ) return;
	sprintf(MSG_STR,STRING_NODE_FMT_STR,val_buf);
	prt_msg_frag(MSG_STR);
}

static void _print_integer_node_value(QSP_ARG_DECL  Spink_Node *skn_p)
{
	int64_t integerValue = 0;
	spinNodeHandle hNode;
	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( get_int_value(hNode, &integerValue) < 0 ) return;
	sprintf(MSG_STR,INT_NODE_FMT_STR, integerValue);
	prt_msg_frag(MSG_STR);
}

static void _print_float_node_value(QSP_ARG_DECL  Spink_Node *skn_p)
{
	double floatValue = 0.0;
	spinNodeHandle hNode;
	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( get_float_value(hNode,&floatValue) < 0 ) return;
	sprintf(MSG_STR,FLT_NODE_FMT_STR, floatValue);
	prt_msg_frag(MSG_STR);
}

static void _print_boolean_node_value(QSP_ARG_DECL  Spink_Node *skn_p)
{
	bool8_t booleanValue = False;
	spinNodeHandle hNode;
	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( get_bool_value(hNode,&booleanValue) < 0 ) return;
	sprintf(MSG_STR,STRING_NODE_FMT_STR, (booleanValue ? "true" : "false"));
	prt_msg_frag(MSG_STR);
}

static void _print_command_node_value(QSP_ARG_DECL  Spink_Node *skn_p)
{
	char val_buf[MAX_BUFF_LEN];
	size_t buf_len = MAX_BUFF_LEN;
	spinNodeHandle hNode;
	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( get_tip_value(hNode,val_buf,&buf_len) < 0 ) return;
	if( buf_len > MAX_NODE_VALUE_CHARS_TO_PRINT) {
		int i;
		for (i = 0; i < MAX_NODE_VALUE_CHARS_TO_PRINT-3; i++) {
			MSG_STR[i] = val_buf[i];
		}
		MSG_STR[i++]='.';
		MSG_STR[i++]='.';
		MSG_STR[i++]='.';
		MSG_STR[i++]=0;
	} else {
		sprintf(MSG_STR,STRING_NODE_FMT_STR, val_buf);
	}
	prt_msg_frag(MSG_STR);
}

static void _print_enumeration_node_value(QSP_ARG_DECL  Spink_Node *skn_p)
{
	char val_buf[MAX_BUFF_LEN];
	size_t buf_len = MAX_BUFF_LEN;
	spinNodeHandle hCurrentEntryNode = NULL;
	spinNodeHandle hNode;
	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( get_current_entry(hNode,&hCurrentEntryNode) < 0 ) return;
	if( get_entry_symbolic(hCurrentEntryNode, val_buf, &buf_len) < 0 ) return;
	sprintf(MSG_STR,STRING_NODE_FMT_STR,val_buf);
	prt_msg_frag(MSG_STR);
}

#define INIT_NODE_TYPE(name,code)					\
	snt_p = new_spink_node_type(#name);				\
	snt_p->snt_type = code;						\
	snt_p->snt_set_func = _set_##name##_node_from_script;			\
	snt_p->snt_print_value_func = _print_##name##_node_value;	\


#define init_default_node_types() _init_default_node_types(SINGLE_QSP_ARG)

static void _init_default_node_types(SINGLE_QSP_ARG_DECL)
{
	Spink_Node_Type *snt_p;

	INIT_NODE_TYPE(category,CategoryNode)
	INIT_NODE_TYPE(register,RegisterNode)
	INIT_NODE_TYPE(port,PortNode)
	INIT_NODE_TYPE(base,BaseNode)
	INIT_NODE_TYPE(unknown,UnknownNode)
	INIT_NODE_TYPE(value,ValueNode)
	INIT_NODE_TYPE(string,StringNode)
	INIT_NODE_TYPE(integer,IntegerNode)
	INIT_NODE_TYPE(float,FloatNode)
	INIT_NODE_TYPE(boolean,BooleanNode)
	INIT_NODE_TYPE(command,CommandNode)
	INIT_NODE_TYPE(enumeration,EnumerationNode)
	INIT_NODE_TYPE(enum_entry,EnumEntryNode)
}

Spink_Node_Type *_find_type_by_code(QSP_ARG_DECL  spinNodeType type)
{
	List *lp;
	Node *np;

	if( spink_node_type_itp == NULL )
		init_spink_node_types();

	lp = spink_node_type_list();
	if( lp == NULL || eltcount(lp) == 0 ){
		init_default_node_types();
		lp = spink_node_type_list();
	}
	assert( lp != NULL && eltcount(lp) != 0 );

	np = QLIST_HEAD(lp);
	while( np != NULL ){
		Spink_Node_Type *snt_p;
		snt_p = NODE_DATA(np);
		if( snt_p->snt_type == type ) return snt_p;
		np = NODE_NEXT(np);
	}
	// Should we create the new type here???
	warn("Node type not found!?");
	return NULL;
}

// This helper function deals with output indentation, of which there is a lot.

#define indent(level) _indent(QSP_ARG  level)

static void _indent(QSP_ARG_DECL  int level)
{
	int i = 0;

	for (i = 0; i < level; i++) {
		prt_msg_frag("   ");
	}
}

#define print_display_name(hNode) _print_display_name(QSP_ARG  hNode)

static void _print_display_name(QSP_ARG_DECL  Spink_Node * skn_p)
{
	char fmt_str[16];
	char name_buf[MAX_BUFF_LEN];
	size_t name_len = MAX_BUFF_LEN;
	spinNodeHandle hNode;

	assert(max_display_name_len>0);
	if( lookup_spink_node(skn_p, &hNode) < 0 ) return;
	if( use_display_names ){
		if( get_display_name(name_buf,&name_len,hNode) < 0 ) return;
	} else {
		if( get_node_name(name_buf,&name_len,hNode) < 0 ) return;
	}

	sprintf(fmt_str,"%%-%ds",max_display_name_len+3);
	sprintf(MSG_STR,fmt_str,name_buf);
	prt_msg_frag(MSG_STR);
}

#define print_node_type(snt_p) _print_node_type(QSP_ARG  snt_p)

static void _print_node_type(QSP_ARG_DECL  Spink_Node_Type * snt_p)
{
	sprintf(MSG_STR,"%-16s",snt_p->snt_name);
	prt_msg_frag(MSG_STR);
}

#define show_rw_status(hNode) _show_rw_status(QSP_ARG  hNode)

static void _show_rw_status(QSP_ARG_DECL  Spink_Node *skn_p)
{
	if( NODE_IS_READABLE(skn_p) ){
		if( NODE_IS_WRITABLE(skn_p) ){
			prt_msg("   (read/write)");
		} else {
			prt_msg("   (read-only)");
		}
	} else if( NODE_IS_WRITABLE(skn_p) ){
			prt_msg("   (write-only)");
	} else {
		prt_msg("   (no read or write access!?)");
	}
}

#define MAX_LEVEL	3

void _print_spink_node_info(QSP_ARG_DECL  Spink_Node *skn_p, int level)
{
	Spink_Node_Type *snt_p;

	//assert(level>0);	// don't print root node
	indent(level-1);
	print_display_name(skn_p);
	indent(MAX_LEVEL-level);
	snt_p = skn_p->skn_type_p;
	assert(snt_p!=NULL);
	print_node_type(snt_p);
	(*(snt_p->snt_print_value_func))(QSP_ARG  skn_p);
	show_rw_status(skn_p);
}

static void _print_node_from_tree(QSP_ARG_DECL  Spink_Node *skn_p)
{
	if( skn_p->skn_level == 0 ){
		return;	// don't print root node
	}
	print_spink_node_info(skn_p,skn_p->skn_level);
}

#define traverse_node_tree(skn_p, func) _traverse_node_tree(QSP_ARG  skn_p, func)

static void _traverse_node_tree(QSP_ARG_DECL  Spink_Node *skn_p, void (*func)(QSP_ARG_DECL  Spink_Node *))
{
	Node *np;
	Spink_Node *child_p;

	assert(skn_p!=NULL);
	(*func)(QSP_ARG  skn_p);
	if( skn_p->skn_children == NULL ) return;
	np = QLIST_HEAD(skn_p->skn_children);
	if( np == NULL ) return;
	while(np!=NULL){
		child_p = NODE_DATA(np);
		traverse_node_tree(child_p,func);
		np = NODE_NEXT(np);
	}
}

void _print_map_tree(QSP_ARG_DECL  Spink_Map *skm_p)
{
	assert(skm_p!=NULL);
	assert(skm_p->skm_root_p!=NULL);
	sprintf(MSG_STR,"\n%s\n",skm_p->skm_name);
	prt_msg(MSG_STR);
	traverse_node_tree(skm_p->skm_root_p,_print_node_from_tree);
}

void _print_cat_tree(QSP_ARG_DECL  Spink_Category *sct_p)
{
	assert(sct_p!=NULL);
	assert(sct_p->sct_root_p!=NULL);
	traverse_node_tree(sct_p->sct_root_p,_print_node_from_tree);
}

static int _register_one_node(QSP_ARG_DECL  spinNodeHandle hNode, int level)
{
	char name[LLEN];
	size_t l=LLEN;
	Spink_Node *skn_p;
	spinNodeType type;
	int n;

	if( get_node_name(name,&l,hNode) < 0 )
		error1("register_one_node:  error getting node name!?");
	assert(strlen(name)<(LLEN-1));

	skn_p = new_spink_node(name);
	assert(skn_p!=NULL);

	assert(current_map!=NULL);
	skn_p->skn_skm_p = current_map;
	if( level == 0 ){
		assert(current_map->skm_root_p == NULL);
		assert(!strcmp(name,"Root"));
		current_map->skm_root_p = skn_p;
		skn_p->skn_parent = NULL;
		skn_p->skn_idx = (-1);	// because no parent
		//assert(current_parent_p==NULL);
	} else {
		int idx;
		idx=level-1;
		assert(idx>=0&&idx<MAX_TREE_DEPTH);
		skn_p->skn_parent = current_parent_p[idx];
		skn_p->skn_idx = current_node_idx;	// set in traverse...
	}
	assert(level>=0&&level<MAX_TREE_DEPTH);
	current_parent_p[level] = skn_p;

	skn_p->skn_flags = 0;
	if( spink_node_is_readable(hNode) ){
		skn_p->skn_flags |= NODE_READABLE;
	}
	if( spink_node_is_writable(hNode) ){
		skn_p->skn_flags |= NODE_WRITABLE;
	}

	n = get_display_name_len(hNode);
	if( n > max_display_name_len )
		max_display_name_len = n;

	if( get_node_type(hNode,&type) < 0 ) return -1;
	skn_p->skn_type_p = find_type_by_code(type);
	assert(skn_p->skn_type_p!=NULL);

	// don't make a category for the root node
	if( level > 0 && type == CategoryNode ){
		Spink_Category *sct_p;
		sct_p = spink_cat_of(skn_p->skn_name);
		//assert(sct_p==NULL); 
		// This should always be NULL the first time we initialize the system,
		// but currently we aren't cleaning up when we shut it down and reinitialize...
		if( sct_p == NULL ){
			sct_p = new_spink_cat(skn_p->skn_name);
		}
		assert(sct_p!=NULL); 
		sct_p->sct_root_p = skn_p;
	}

	skn_p->skn_enum_ival = INVALID_ENUM_INT_VALUE;
	if( type == EnumEntryNode ){
		int64_t ival;
		if( get_enum_int_value(hNode,&ival) < 0 ){
			skn_p->skn_enum_ival = INVALID_ENUM_INT_VALUE;
		} else {
			skn_p->skn_enum_ival = ival;
			assert(ival>=0);
		}
	}

	skn_p->skn_children = NULL;
	skn_p->skn_level = level;

//fprintf(stderr,"register_one_node:  %s   flags = %d\n",skn_p->skn_name,skn_p->skn_flags);

	//skn_p->skn_handle = hNode;
	return 0;
}

// We are duplicating the node tree with our own structures,
// but it is difficult to link the structs as we build the tree,
// so we do it after the fact using the parent pointers.

static void add_child_to_parent(Spink_Node *skn_p)
{
	Node *np;

	np = mk_node(skn_p);
	if( skn_p->skn_parent->skn_children == NULL )
		skn_p->skn_parent->skn_children = new_list();
	assert( skn_p->skn_parent->skn_children != NULL );

	addHead(skn_p->skn_parent->skn_children,np);
}

#define cleanup_node_and_children(skn_p) _cleanup_node_and_children(QSP_ARG   skn_p)

static void _cleanup_node_and_children(QSP_ARG_DECL   Spink_Node *skn_p)
{
	if( skn_p->skn_children != NULL ){
		Node *np;
		np = remHead(skn_p->skn_children);
		while( np != NULL ){
			Spink_Node *child;
			child = NODE_DATA(np);
			assert(child!=NULL);
			cleanup_node_and_children(child);
			rls_node(np);
			np = remHead(skn_p->skn_children);
		}
		rls_list(skn_p->skn_children);
	}
	del_spink_node(skn_p);
}

#define build_child_lists() _build_child_lists(SINGLE_QSP_ARG)

static void _build_child_lists(SINGLE_QSP_ARG_DECL)
{
	List *lp;
	Node *np;
	
	lp = spink_node_list();
	assert(lp!=NULL);
	assert(eltcount(lp)>0);

	np = QLIST_HEAD(lp);
	while(np!=NULL){
		Spink_Node *skn_p;
		skn_p = NODE_DATA(np);
		if( skn_p->skn_parent != NULL ){
			add_child_to_parent(skn_p);
		} else {
			assert( skn_p->skn_idx == (-1) );	// root node
		}
		np = NODE_NEXT(np);
	}
}

void _pop_map_contexts(SINGLE_QSP_ARG_DECL)
{
	pop_spink_node_context();
	pop_spink_cat_context();
}

void _push_map_contexts(QSP_ARG_DECL  Spink_Map *skm_p)
{
	push_spink_node_context(skm_p->skm_node_icp);
	push_spink_cat_context(skm_p->skm_cat_icp);
}

#define register_map_nodes(hMap,skm_p) _register_map_nodes(QSP_ARG  hMap,skm_p)

static void _register_map_nodes(QSP_ARG_DECL  spinNodeMapHandle hMap, Spink_Map *skm_p)
{
	spinNodeHandle hRoot=NULL;

//fprintf(stderr,"register_map_nodes BEGIN   hMap = 0x%lx\n",(u_long)hMap);

	//push_spink_node_context(skm_p->skm_icp);
	push_map_contexts(skm_p);

//fprintf(stderr,"register_map_nodes fetching root node   hMap = 0x%lx\n",(u_long)hMap);
	if( fetch_spink_node(hMap, "Root", &hRoot) < 0 )
		error1("register_map_nodes:  error fetching map root node");

//fprintf(stderr,"register_map_nodes:  root node fetched, traversing...\n");
	current_map = skm_p;
	//current_parent_p = NULL;
	skm_p->skm_root_p = NULL;
	if( traverse_by_node_handle(hRoot,0,_register_one_node) < 0 )
		error1("error traversing node map");
	current_map = NULL;

	// Do this before popping the context!!!
	build_child_lists();

	pop_map_contexts();

}

#define cleanup_map_nodes(skm_p) _cleanup_map_nodes(QSP_ARG  skm_p)

static void _cleanup_map_nodes(QSP_ARG_DECL  Spink_Map *skm_p)
{
	push_map_contexts(skm_p);
	assert(skm_p->skm_root_p!=NULL);
	cleanup_node_and_children(skm_p->skm_root_p);
	pop_map_contexts();
}


#define cleanup_map_categories(skm_p) _cleanup_map_categories(QSP_ARG  skm_p)

static void _cleanup_map_categories(QSP_ARG_DECL  Spink_Map *skm_p)
{
	Spink_Category *sct_p;
	List *lp;
	Node *np;

	push_map_contexts(skm_p);
	lp = spink_cat_list();
	if( lp == NULL ) return;

	while((np=remHead(lp))!=NULL){
		sct_p = NODE_DATA(np);
		del_spink_cat(sct_p);
	}
	pop_map_contexts();
}


#define register_one_map(skc_p, code, name) _register_one_map(QSP_ARG  skc_p, code, name)

static Spink_Map * _register_one_map(QSP_ARG_DECL  Spink_Cam *skc_p, Node_Map_Type type, const char *name)
{
	Spink_Map *skm_p;
	spinNodeMapHandle hMap = NULL;

	insure_current_camera(skc_p);
	assert( skc_p->skc_current_handle != NULL );
//fprintf(stderr,"register_one_map:  %s has current handle 0x%lx\n", skc_p->skc_name,(u_long)skc_p->skc_current_handle);

	skm_p = new_spink_map(name);
	if( skm_p == NULL ) error1("Unable to create map struct!?");
//fprintf(stderr,"Created new map struct %s at 0x%lx\n", skm_p->skm_name,(u_long)skm_p);


	if( spink_node_itp == NULL ) init_spink_nodes();
	skm_p->skm_node_icp = create_item_context(spink_node_itp,name);
	assert(skm_p->skm_node_icp!=NULL);

	if( spink_cat_itp == NULL ) init_spink_cats();
	skm_p->skm_cat_icp = create_item_context(spink_cat_itp,name);
	assert(skm_p->skm_cat_icp!=NULL);

	// do we need to push the context too???

	//skm_p->skm_handle = NULL;
	skm_p->skm_type = type;
	skm_p->skm_skc_p = skc_p;

//	fetch_map_handle(skm_p);
	get_node_map_handle(&hMap,skm_p,"register_one_map");	// first time just sets
//fprintf(stderr,"register_one_map:  hMap = 0x%lx, *hMap = 0x%lx \n",(u_long)hMap, (u_long)*((void **)hMap));

	register_map_nodes(hMap,skm_p);

	return skm_p;
} // register_one_map

#define cleanup_one_map(skm_p) _cleanup_one_map(QSP_ARG  skm_p)

static void _cleanup_one_map(QSP_ARG_DECL  Spink_Map *skm_p)
{
	cleanup_map_nodes(skm_p);
	cleanup_map_categories(skm_p);
	// delete the contexts???
	del_spink_map(skm_p);
}

#define register_cam_nodemaps(skc_p) _register_cam_nodemaps(QSP_ARG  skc_p)

static void _register_cam_nodemaps(QSP_ARG_DECL  Spink_Cam *skc_p)
{
//fprintf(stderr,"register_cam_nodemaps BEGIN\n");
//fprintf(stderr,"register_cam_nodemaps registering device map\n");
	sprintf(MSG_STR,"%s.stream_TL",skc_p->skc_name);
	skc_p->skc_stream_map = register_one_map(skc_p,STREAM_NODE_MAP,MSG_STR);

	sprintf(MSG_STR,"%s.device_TL",skc_p->skc_name);
	skc_p->skc_dev_map = register_one_map(skc_p,DEV_NODE_MAP,MSG_STR);
//fprintf(stderr,"register_cam_nodemaps registering camera map\n");

	sprintf(MSG_STR,"%s.genicam",skc_p->skc_name);
	skc_p->skc_cam_map = register_one_map(skc_p,CAM_NODE_MAP,MSG_STR);

	// Now get width and height from the map...

//fprintf(stderr,"register_cam_nodemaps DONE\n");
}

/*
static void _unregister_cam_nodemaps(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	unregister_one_map(skc_p->skc_cam_map);
	unregister_one_map(skc_p->skc_dev_map);
	unregister_one_map(skc_p->skc_stream_map);
}
*/

#define int_node_value(s) _int_node_value(QSP_ARG  s)

static int64_t _int_node_value(QSP_ARG_DECL  const char *s)
{
	int64_t integerValue = 0;
	spinNodeHandle hNode;
	Spink_Node *skn_p;
	Spink_Node_Type *snt_p;

	skn_p = get_spink_node(s);
	if( skn_p == NULL ){
		sprintf(ERROR_STRING,"int_node_value:  Node '%s' not found!?",s);
		warn(ERROR_STRING);
		return 0;
	}
	snt_p = skn_p->skn_type_p;
	assert(snt_p!=NULL);
	if(snt_p->snt_type != IntegerNode){
		sprintf(ERROR_STRING,"int_node_value:  Node '%s' is not an integer node!?",s);
		warn(ERROR_STRING);
		return 0;
	}
	if( lookup_spink_node(skn_p, &hNode) < 0 ) return 0;
	if( get_int_value(hNode, &integerValue) < 0 ) return 0;

	return integerValue;
}


#define get_cam_dimensions(skc_p) _get_cam_dimensions(QSP_ARG  skc_p)

static void _get_cam_dimensions(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	select_spink_map(skc_p->skc_cam_map);
	skc_p->skc_cols = int_node_value("Width");
	skc_p->skc_rows = int_node_value("Height");
	skc_p->skc_depth = 1;	// BUG should determine based on pixel mode!
	skc_p->skc_bytes_per_image = skc_p->skc_cols * skc_p->skc_rows * skc_p->skc_depth;
	select_spink_map(NULL);
fprintf(stderr,"get_cam_dimensions:  %s has %d rows and %d columns\n",
skc_p->skc_name,skc_p->skc_rows,skc_p->skc_cols);
}

#define init_one_spink_cam(idx) _init_one_spink_cam(QSP_ARG  idx)

static int _init_one_spink_cam(QSP_ARG_DECL  int idx)
{
	spinCamera hCam;
//	spinNodeMapHandle hNodeMap;
	Spink_Cam *skc_p;
	char buf[MAX_BUFF_LEN];
	size_t len = MAX_BUFF_LEN;

	if( get_cam_from_list(hCameraList,idx,&hCam) < 0 )
		return -1;
//fprintf(stderr,"init_one_spink_cam:  get_cam_from_list returned 0x%lx\n",(u_long)hCam);

	if( get_camera_model_name(buf,len,hCam) < 0 ) return -1;
	substitute_char(buf,' ','_');
	// Check and see if another camera of this type has already
	// been detected...
	skc_p = spink_cam_of(buf);
	if( skc_p != NULL ){
		if( get_unique_cam_name(buf,MAX_BUFF_LEN) < 0 ){
			return -1;
		}
	}
	skc_p = new_spink_cam(buf);
	if( skc_p == NULL ) return -1;
	skc_p->skc_current_handle = hCam;
//fprintf(stderr,"init_one_spink_cam:  setting current handle to 0x%lx\n",(u_long)hCam);

	skc_p->skc_flags = 0;
	//skc_p->skc_handle = hCam;
	skc_p->skc_sys_idx = idx;
	skc_p->skc_iface_idx = -1;	// invalid value
	skc_p->skc_n_buffers = 0;

	// register_cam_nodemaps will get the camera handle again...
//	if( release_spink_cam(hCam) < 0 )
//		return -1;

	//skc_p->skc_TL_dev_node_map = hNodeMapTLDevice;
	//skc_p->skc_genicam_node_map = hNodeMap;

	skc_p->skc_dev_map=NULL;
	skc_p->skc_cam_map=NULL;
	skc_p->skc_stream_map=NULL;

	// The camera has to be connected to get the genicam node map!
	register_cam_nodemaps(skc_p);

	get_cam_dimensions(skc_p);

	// We have to explicitly release here, as we weren't able to call
	// insure_current_camera at the beginning...
	//spink_release_cam(skc_p);

	release_current_camera(1);

	return 0;
} // init_one_spink_cam

#define cleanup_cam_nodemaps(skc_p) _cleanup_cam_nodemaps(QSP_ARG  skc_p)

static void _cleanup_cam_nodemaps(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	cleanup_one_map(skc_p->skc_dev_map);
	cleanup_one_map(skc_p->skc_cam_map);
	cleanup_one_map(skc_p->skc_stream_map);
}

#define cleanup_one_cam(skc_p) _cleanup_one_cam(QSP_ARG  skc_p)

static void _cleanup_one_cam(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	cleanup_cam_nodemaps(skc_p);
	del_spink_cam(skc_p);
}

#define cleanup_camera_structs() _cleanup_camera_structs(SINGLE_QSP_ARG)

static void _cleanup_camera_structs(SINGLE_QSP_ARG_DECL)
{
	List *lp;
	Node *np;
	Spink_Cam *skc_p;

	do {
		lp=spink_cam_list();
		if( lp == NULL ) return;
		np = QLIST_HEAD(lp);
		if(np!=NULL){
			skc_p = NODE_DATA(np);
			assert(skc_p!=NULL);
			cleanup_one_cam(skc_p);
		}
	} while(np!=NULL);
}


#define cleanup_interface_structs()	_cleanup_interface_structs(SINGLE_QSP_ARG)

static int _cleanup_interface_structs(SINGLE_QSP_ARG_DECL)
{
	// iterate through the list
	Node *np;
	List *lp;
	Spink_Interface *ski_p;

	lp = spink_interface_list();
	if( lp == NULL ) return 0;

	while( (np=remHead(lp)) != NULL ){
		ski_p = (Spink_Interface *) NODE_DATA(np);
		del_spink_interface(ski_p);
		rls_node(np);
	}
	return 0;
}

#define cleanup_structs() _cleanup_structs(SINGLE_QSP_ARG)

static void _cleanup_structs(SINGLE_QSP_ARG_DECL)
{
	cleanup_camera_structs();
	cleanup_interface_structs();
}

#define create_spink_camera_structs() _create_spink_camera_structs(SINGLE_QSP_ARG)

static int _create_spink_camera_structs(SINGLE_QSP_ARG_DECL)
{
	int i;

	for(i=0;i<numCameras;i++){
		if( init_one_spink_cam(i) < 0 )
			return -1;
	}
	return 0;
} // end create_spink_camera_structs

#define create_spink_interface_structs() _create_spink_interface_structs(SINGLE_QSP_ARG)

static int _create_spink_interface_structs(SINGLE_QSP_ARG_DECL)
{
	int i;
	Spink_Interface *ski_p;
	char buf[MAX_BUFF_LEN];
	size_t len = MAX_BUFF_LEN;


	for(i=0;i<numInterfaces;i++){
		spinInterface hInterface;
		// This call causes releaseSystem to crash!?
		if( get_spink_interface_from_list(&hInterface,hInterfaceList,i) < 0 )
			return -1;

		get_interface_name(buf,len,hInterface);
		substitute_char(buf,' ','_');
		ski_p = new_spink_interface(buf);

		//ski_p->ski_handle = hInterface;
		ski_p->ski_idx = i;

		if( release_interface(hInterface) < 0 )
			return -1;
	}
	return 0;
}

#define INIT_CHUNK_DATUM(name,type)			\
	cd_p = new_chunk_data(#name);			\
	assert(cd_p!=NULL);				\
	cd_p->cd_type = type;

#define init_chunk_data_structs() _init_chunk_data_structs(SINGLE_QSP_ARG)

static void _init_chunk_data_structs(SINGLE_QSP_ARG_DECL)
{
	Chunk_Data *cd_p;

	INIT_CHUNK_DATUM(Width,INT_CHUNK_DATA);
	INIT_CHUNK_DATUM(Timestamp,INT_CHUNK_DATA);
	INIT_CHUNK_DATUM(PixelFormat,INT_CHUNK_DATA);
	INIT_CHUNK_DATUM(OffsetY,INT_CHUNK_DATA);
	INIT_CHUNK_DATUM(OffsetX,INT_CHUNK_DATA);
// Image
	INIT_CHUNK_DATUM(Height,INT_CHUNK_DATA);
	INIT_CHUNK_DATUM(Gain,FLOAT_CHUNK_DATA);
	INIT_CHUNK_DATUM(FrameID,INT_CHUNK_DATA);
	INIT_CHUNK_DATUM(ExposureTime,FLOAT_CHUNK_DATA);
// CRC
	INIT_CHUNK_DATUM(BlackLevel,FLOAT_CHUNK_DATA);
}
#endif // HAVE_LIBSPINNAKER


int _init_spink_cam_system(SINGLE_QSP_ARG_DECL)
{
#ifdef HAVE_LIBSPINNAKER
	assert( hSystem == NULL );

	if( get_spink_system(&hSystem) < 0 )
		return -1;

	if( get_spink_interfaces(hSystem,&hInterfaceList,&numInterfaces) < 0 ) return -1;
	if( create_spink_interface_structs() < 0 ) return -1;

	// We get the cameras from the system, not from individual interfaces...
	if( get_spink_cameras(hSystem,&hCameraList,&numCameras) < 0 ) return -1;
	if( create_spink_camera_structs() < 0 ) return -1;

	init_chunk_data_structs();

	do_on_exit(_release_spink_cam_system);

#endif // HAVE_LIBSPINNAKER
	return 0;
}

#ifdef HAVE_LIBSPINNAKER

#define stop_all_cameras() _stop_all_cameras(SINGLE_QSP_ARG)

static void _stop_all_cameras(SINGLE_QSP_ARG_DECL)
{
	List *lp;
	Node *np;
	Spink_Cam *skc_p;

	lp = spink_cam_list();
	if( lp == NULL ) return;
	np = QLIST_HEAD(lp);
	if( np == NULL ) return;

	while(np!=NULL){
		skc_p = NODE_DATA(np);
		if( IS_CAPTURING(skc_p) ) spink_stop_capture(skc_p);
		np = NODE_NEXT(np);
	}
}

void _release_spink_cam_system(SINGLE_QSP_ARG_DECL)
{
	if( hSystem == NULL ) return;	// may already be shut down?

SPINK_DEBUG_MSG(releast_spink_cam_system BEGIN)

	// make sure that no cameras are running...
	stop_all_cameras();

	release_current_camera(0);

//fprintf(stderr,"release_spink_cam_system releasing interface structs\n");
//	if( release_spink_interfaces() < 0 ) return;

	// must come AFTER interfaces are released...
	cleanup_structs();

	if( release_spink_cam_list(&hCameraList) < 0 ) return;
	if( release_spink_interface_list(&hInterfaceList) < 0 ) return;
	if( release_spink_system(hSystem) < 0 ) return;
	hSystem = NULL;
SPINK_DEBUG_MSG(releast_spink_cam_system DONE)
}

int _spink_release_cam(QSP_ARG_DECL  Spink_Cam *skc_p)
{
	assert(skc_p->skc_current_handle!=NULL);
//fprintf(stderr,"spink_release_cam:  old handle was 0x%lx, will set to NULL\n",(u_long)skc_p->skc_current_handle);
	if( release_spink_cam(skc_p->skc_current_handle) < 0 )
		return -1;
	skc_p->skc_current_handle=NULL;
	return 0;
}

#define node_for_chunk(cd_p) _node_for_chunk(QSP_ARG  cd_p)

static Spink_Node *_node_for_chunk(QSP_ARG_DECL  Chunk_Data *cd_p)
{
	char buf[128];
	Spink_Node *skn_p;
//fprintf(stderr,"node_for_chunk %s BEGIN\n",cd_p->cd_name);
	sprintf(buf,"EnumEntry_ChunkSelector_%s",cd_p->cd_name);
	skn_p = get_spink_node(buf);
	//assert(skn_p!=NULL);
	return skn_p;
}

void _enable_chunk_data(QSP_ARG_DECL  Spink_Cam *skc_p, Chunk_Data *cd_p)
{
	Spink_Map *skm_p;
	Spink_Node *skn_p;
	Spink_Node *val_p;

	skm_p = skc_p->skc_cam_map;
	assert(skm_p!=NULL);
	push_map_contexts(skm_p);
	skn_p = get_spink_node("ChunkModeActive");
	assert( skn_p != NULL );
	set_boolean_node(skn_p,True);
	/*
	get_node_map_handle(&hMap,skm_p,"enable_chunk_data");
	assert(hMap!=NULL);
	if( fetch_spink_node(hMap, "ChunkModeActive", &hNode) < 0 ){
		warn("enable_chunk_data:  error getting ChunkModeActive node!?");
		return;
	}
	if( ! spink_node_is_available(hNode) ){
		//if( verbose )
			report_node_access_error(hNode,"available");
		return;
	}
	if( ! spink_node_is_writable(hNode) ){
		report_node_access_error(hNode,"writable");
		return;
	}
	*/

	skn_p = get_spink_node("ChunkSelector");
	assert( skn_p != NULL );
	assert(skn_p->skn_type_p->snt_type == EnumerationNode );

//fprintf(stderr,"enable_chunk_data:  looking up node for %s\n",cd_p->cd_name);
	val_p = node_for_chunk(cd_p);
	if( val_p != NULL ){
//fprintf(stderr,"enable_chunk_data:  found %s\n",val_p->skn_name);
		set_enumeration_node(skn_p,val_p);

		skn_p = get_spink_node("ChunkEnable");
		set_boolean_node(skn_p,True);
	}

	pop_map_contexts();
}

void _fetch_chunk_data(QSP_ARG_DECL  Chunk_Data *cd_p, Data_Obj *dp)
{
	spinImage hImg;
	char buf[128];

	if( OWNS_DATA(dp) ){
		sprintf(ERROR_STRING,"fetch_chunk_data:  object %s owns its data, not a camera buffer!?",
			OBJ_NAME(dp));
		warn(ERROR_STRING);
		return;
	}

	hImg = OBJ_EXTRA(dp);
	assert(hImg!=NULL);

	sprintf(buf,"Chunk%s",cd_p->cd_name);
//fprintf(stderr,"chunk string is %s\n",buf);
	if(cd_p->cd_type == INT_CHUNK_DATA ){
		if( get_image_chunk_int(hImg,buf,&(cd_p->cd_u.u_intval)) < 0 )
			warn("error fetching int chunk value!?");
	} else if( cd_p->cd_type == FLOAT_CHUNK_DATA ){
		if( get_image_chunk_float(hImg,buf,&(cd_p->cd_u.u_fltval)) < 0 )
			warn("error fetching float chunk value!?");
	} else error1("fetch_chunk_data:  bad chunk data type code!?");
}

void _format_chunk_data(QSP_ARG_DECL  char *buf, Chunk_Data *cd_p)
{
	if(cd_p->cd_type == INT_CHUNK_DATA ){
		sprintf(buf,"%ld",cd_p->cd_u.u_intval);
	} else if( cd_p->cd_type == FLOAT_CHUNK_DATA ){
		sprintf(buf,"%g",cd_p->cd_u.u_fltval);
	} else error1("format_chunk_data:  bad chunk data type code!?");
}

#endif // HAVE_LIBSPINNAKER

