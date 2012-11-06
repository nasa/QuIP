char VersionId_jbm_jbm[] = "$RCSfile$ $Revision$ $Date$";

#include <stdio.h>
#include "debug.h"
#include "tryhard.h"
#include "query.h"
#include "my_param.h"
#include "adjust.h"
#include "version.h"
#include "vars.h"

LOCAL_FUNC COMMAND_FUNC( mychng );
LOCAL_FUNC COMMAND_FUNC( do_read4 );
LOCAL_FUNC COMMAND_FUNC( grab_mem );
COMMAND_FUNC( dotfile );

static char *main_prompt="interpret";

typedef struct my_obj {
	const char *	my_name;
	char *		my_text;
} My_Obj;

#define NO_MY_OBJ	((My_Obj *)NULL)

ITEM_INTERFACE_DECLARATIONS(My_Obj,my_obj)

static void _del_my_obj(My_Obj *mop);

int p1,p2;
int parr[2]={0,1};
float farr[2]={(float)1.1,(float)2.2};

Param myptbl[]={
{ "p1",		"first param",	INTP,		&p1		},
{ "p2",		"second param",	INTP,		&p2		},
{ "arr",	"int array",	IARRP+2,	&parr[0]	},
{ "farr",	"float array",	FARRP+2,	&farr[0]	},
{ NULL_PARAM							}
};

COMMAND_FUNC( mychng )
{
	chngp(QSP_ARG  myptbl);
}

COMMAND_FUNC( do_read4 )
{
	int i,r,g,b;

	i=(int)HOW_MANY("index");
	r=(int)HOW_MANY("r");
	g=(int)HOW_MANY("g");
	b=(int)HOW_MANY("b");
}

COMMAND_FUNC( grab_mem )
{
	char *s;

	s=savestr("ABC");
	s=savestr("abcdefghijk");
	s=savestr("ACK!!!");
	s=savestr("pbffht!!!");
}

COMMAND_FUNC( dotfile )
{
	redir( QSP_ARG tfile(SINGLE_QSP_ARG) );
}

#ifdef DYNAMIC_LOAD
COMMAND_FUNC( tst_sym )
{
	char *s;
	int a;
	extern int find_symbol();

	s=NAMEOF("symbol name");
	a=find_symbol(s);
	printf("symbol %s at address 0x%x\n",s,a);
}
#endif

COMMAND_FUNC( my_info )
{
	My_Obj *mop;

	mop=pick_my_obj(QSP_ARG  "");
	if( mop == NO_MY_OBJ ) return;

	printf("name = %s\n",mop->my_name);
	printf("\tstring = %s\n",mop->my_text);
}

COMMAND_FUNC( do_new_item )
{
	My_Obj *mop;
	const char *s;

	s=NAMEOF("name");
advise("calling new_my_obj");
	mop=new_my_obj(s);
	s=NAMEOF("text");

	if( mop==NO_MY_OBJ ) return;
	mop->my_text=savestr(s);
}

COMMAND_FUNC( do_del_item )
{
	My_Obj *mop;

	mop=pick_my_obj(QSP_ARG  "");
	if( mop == NO_MY_OBJ ) return;

	_del_my_obj(mop);
}

COMMAND_FUNC( do_item_context )
{
	const char *s;
	Item_Context *icp;

	s=NAMEOF("context name");
	if( my_obj_itp==NO_ITEM_TYPE ) my_obj_init();
	icp=create_item_context(my_obj_itp,s);
	if( icp==NO_ITEM_CONTEXT ) return;
	push_item_context(my_obj_itp,icp);
}

COMMAND_FUNC( do_list_contexts )
{
	if( my_obj_itp==NO_ITEM_TYPE ) my_obj_init();
	list_item_contexts(my_obj_itp);
}

static void _del_my_obj(My_Obj *mop)
{
sprintf(error_string,"Deleting object %s",mop->my_name);
	del_my_obj(mop->my_name);	/* remove from database */
	rls_str((char *)mop->my_text);
	rls_str((char *)mop->my_name);
}

static COMMAND_FUNC( do_pop_context )
{
	if( my_obj_itp==NO_ITEM_TYPE ) my_obj_init();
	if( my_obj_itp->it_del_method != (void (*)(Item *)) del_my_obj )
		set_del_method(my_obj_itp,(void (*)(Item *))del_my_obj);

advise("popping context");
	pop_item_context(my_obj_itp);
}

static COMMAND_FUNC( do_list_my_objs ){ list_my_objs(); }

Command it_ctbl[]={
{ "list",	do_list_my_objs,	"list items"		},
{ "info",	my_info,	"info"			},
{ "new",	do_new_item,	"create a new item"	},
{ "delete",	do_del_item,	"delete an item"	},
{ "context",	do_item_context,"push item context"	},
{ "pop",	do_pop_context,	"pop item context"	},
{ "list_contexts",do_list_contexts,"list item contexts" },
{ "quit",	popcmd,		"exit submenu"		},
{ NULL_COMMAND						}
};

COMMAND_FUNC( do_getbuf )
{
	u_long s;
	void *a;
	char str[32];

	s=HOW_MANY("number of bytes");
	a = getbuf(s);

	printf("Obtained memory at 0x%lx\n",(u_long)a);
	sprintf(str,"0x%lx",(u_long)a);
	assign_var("addr",str);
}

COMMAND_FUNC( do_givbuf )
{
	u_long a;

	a=HOW_MANY("address");
	givbuf((void *)a);
}

COMMAND_FUNC( item_menu )
{
	PUSHCMD(it_ctbl,"myitems");
}

static COMMAND_FUNC( do_showmaps ){ showmaps(); }

Command mem_ctbl[]={
{ "maps",	do_showmaps,	"show free memory"			},
{ "getbuf",	do_getbuf,	"get some memory"			},
{ "givbuf",	do_givbuf,	"release some memory"			},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL_COMMAND						}
};

COMMAND_FUNC( mem_menu )
{
	PUSHCMD(mem_ctbl,"memory");
}

COMMAND_FUNC( do_t1 ){ advise("test1"); }
COMMAND_FUNC( do_t2 ){ advise("test2"); }
COMMAND_FUNC( do_t3 ){ advise("test3"); }

/* ansi style defn required to squelch warnings on pc */
static void tst_adj_func(FLOAT_ARG f)
{
	sprintf(error_string,"current adjustment is %g",f);
	advise(error_string);
}

COMMAND_FUNC( tst_adj )
{
	set_adj_func(tst_adj_func);
	setaps(0.0,100.0,50.0,10.0,30.0,0.1,15.0);

	do_adjust(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( cmd11 ){	advise("cmd 1 1"); }
static COMMAND_FUNC( cmd12 ){	advise("cmd 1 2"); }
static COMMAND_FUNC( cmd21 ){	advise("cmd 2 1"); }
static COMMAND_FUNC( cmd22 ){	advise("cmd 2 2"); }

Command sub1[]={
{ "cmd1",	cmd11,	"command 1"	},
{ "cmd2",	cmd12,	"command 2"	},
{ "quit",	popcmd,	"exit submenu"	},
{ NULL_COMMAND				}
};

Command sub2[]={
{ "cmd1",	cmd21,	"command 1"	},
{ "cmd2",	cmd22,	"command 2"	},
{ "quit",	popcmd,	"exit submenu"	},
{ NULL_COMMAND				}
};

COMMAND_FUNC( cmd2 )
{
	PUSHCMD(sub2,"sub2");
}

COMMAND_FUNC( cmd1 )
{
	PUSHCMD(sub1,"sub1");
}

COMMAND_FUNC( prttst )
{
	int n1,n2;

	n1=HOW_MANY("milliseconds");
	n2=HOW_MANY("microseconds");

	sprintf(msg_str,"%3d.%03d",n1,n2);
	prt_msg(msg_str);
}

Command main_ctbl[]={
#ifdef DYNAMIC_LOAD
{ "symbol",	tst_sym,	"find a symbol in the a.out file"	},
#endif
{ "tfile",	dotfile,	"redirect to tfile"			},
{ "params",	mychng,		"change parameters"			},
{ "read4",	do_read4,	"read 4 integers"			},
{ "grabmem",	grab_mem,	"save some strings"			},
{ "Items",	item_menu,	"item test menu"			},
{ "adjust",	tst_adj,	"test adjustment routine"		},
{ "memory",	mem_menu,	"memory allocation test menu"		},
{ "test",	do_t1,		"test cmd"				},
{ "test_2",	do_t2,		"test cmd"				},
{ "test_3",	do_t3,		"test cmd"				},
{ "cmd1",	cmd1,		"submenu 1"				},
{ "cmd2",	cmd2,		"submenu 2"				},
{ "print_test",	prttst,		"test decimal printing"			},
{ "quit",	popcmd,		"quit"					},
{ NULL_COMMAND								}
};

int main(int ac,char **av)
{
	QSP_DECL

	INIT_QSP
	rcfile(QSP_ARG  av[0]);
	/* this isn't defined under UNIX */
	/* mkver ("NON-LIB", nl_files, MAX_NL_FILES); */
	set_args(ac,av);
	PUSHCMD(main_ctbl,main_prompt);

	while(1) do_cmd(SINGLE_QSP_ARG);

	return(0);
}

