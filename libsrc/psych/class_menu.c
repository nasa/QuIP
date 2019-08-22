#include "quip_config.h"

#include "quip_prot.h"
#include "stc.h"

static COMMAND_FUNC( do_new_class )
{
	const char *name, *cmd;
	Trial_Class *tc_p;

	name = nameof("nickname for this class");
	cmd = nameof("string to execute for this stimulus class");

	tc_p = create_named_class(name);
	if( tc_p == NULL ) return;

	SET_CLASS_STIM_CMD(tc_p,savestr(cmd));

	add_class_to_expt(&expt1,tc_p);
}

static COMMAND_FUNC( do_reset_all_classes )
{
	reset_expt_classes(&expt1);
}

static COMMAND_FUNC( do_delete_all_classes )
{
	delete_all_trial_classes();
}

static COMMAND_FUNC( do_list_classes )
{
	list_trial_classs( tell_msgfile() );
}

static COMMAND_FUNC( do_class_info )
{
	Trial_Class *tc_p;

	tc_p = pick_trial_class("");
	if( tc_p == NULL ) return;

	print_class_info(tc_p);
}

static COMMAND_FUNC( do_reset_class )
{
	Trial_Class *tc_p;

	tc_p = pick_trial_class("");
	if( tc_p == NULL ) return;

	reset_class( tc_p );
}

static COMMAND_FUNC(do_set_rsp_cmd)
{
	Trial_Class *tc_p;
	const char *s;

	tc_p = pick_trial_class("");
	s = nameof("commands to execute to collect and process trial responses");
	if( tc_p == NULL ) return;

	set_response_cmd( tc_p, s );
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(class_menu,s,f,h)

MENU_BEGIN(class)
ADD_CMD( new,			do_new_class,		add a stimulus class )
ADD_CMD( list,			do_list_classes,	list all stimulus classes )
ADD_CMD( info,			do_class_info,		print info about a class )
ADD_CMD( set_response_cmd,	do_set_rsp_cmd,		specify response handler )
ADD_CMD( reset,			do_reset_class,		clear class data and reset staircases )
ADD_CMD( reset_all,		do_reset_all_classes,	clear data for all conditions )
ADD_CMD( delete_all,		do_delete_all_classes,	delete all conditions )
MENU_END(class)

//ADD_CMD( summary_data,		do_show_class_summ,	print summary data from a class )
//ADD_CMD( sequential_data,	do_show_class_seq,	print sequential data from a class )

COMMAND_FUNC( do_class_menu )
{
	CHECK_AND_PUSH_MENU(class);
}


