#include "quip_config.h"

#include "quip_prot.h"
#include "stc.h"

static int class_index=0;

static COMMAND_FUNC( do_new_class )
{
	const char *name, *cmd;
	Trial_Class *tc_p;

	name = nameof("nickname for this class");
	cmd = nameof("string to execute for this stimulus class");

	tc_p = create_named_class(name);
	if( tc_p == NULL ) return;

	SET_CLASS_STIM_CMD(tc_p,savestr(cmd));
}

static void set_class_xval_obj( Trial_Class *tc_p, Data_Obj *dp )
{
	if( CLASS_XVAL_OBJ(tc_p) != NULL )
		remove_reference(CLASS_XVAL_OBJ(tc_p));

	SET_CLASS_XVAL_OBJ(tc_p,dp);

	if( dp != NULL )
		add_reference(dp);
}

Trial_Class *_create_named_class(QSP_ARG_DECL  const char *name)
{
	Trial_Class *tc_p;
	Summary_Data_Tbl *sdt_p;

	// Make sure not in use
	tc_p = trial_class_of(name);
	if( tc_p != NULL ){
		sprintf(ERROR_STRING,"Class name \"%s\" is already in use!?",
			name);
		warn(ERROR_STRING);
		return NULL;
	}

	tc_p = new_trial_class(name );
	SET_CLASS_INDEX(tc_p,class_index++);
	SET_CLASS_N_STAIRS(tc_p,0);

	SET_CLASS_XVAL_OBJ(tc_p,NULL);			// so we don't un-reference garbage
	set_class_xval_obj(tc_p,global_xval_dp);	// may be null

	sdt_p = new_summary_data_tbl();
	init_summ_dtbl_for_class(sdt_p,tc_p);

	SET_CLASS_SEQ_DTBL(tc_p,new_sequential_data_tbl());
	SET_SEQ_DTBL_CLASS( CLASS_SEQ_DTBL(tc_p), tc_p );

	SET_CLASS_STIM_CMD(tc_p, NULL);
	SET_CLASS_RESP_CMD(tc_p, NULL);

	assert( CLASS_SUMM_DTBL(tc_p) != NULL );
	clear_summary_data( CLASS_SUMM_DTBL(tc_p) );

	sprintf(MSG_STR,"create_named_class:  created new class '%s' with index %d",CLASS_NAME(tc_p),CLASS_INDEX(tc_p));
	prt_msg(MSG_STR);

	return tc_p;
}

static COMMAND_FUNC( do_clear_all_classes )
{
	clear_all_data_tables();
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

	advise("Sorry, don't know how to print class info yet!?");
}

static COMMAND_FUNC( do_show_class_summ )
{
	Trial_Class *tc_p;

	tc_p = pick_trial_class("");
	if( tc_p == NULL ) return;

	write_summary_data( CLASS_SUMM_DTBL(tc_p), tell_msgfile() );
}

static COMMAND_FUNC( do_show_class_seq )
{
	Trial_Class *tc_p;

	tc_p = pick_trial_class("");
	if( tc_p == NULL ) return;

	write_sequential_data( CLASS_SEQ_DTBL(tc_p), tell_msgfile() );
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(class_menu,s,f,h)

MENU_BEGIN(class)
ADD_CMD( new,		do_new_class,		add a stimulus class )
ADD_CMD( list,		do_list_classes,	list all stimulus classes )
ADD_CMD( info,		do_class_info,		print info about a class )
ADD_CMD( summary_data,	do_show_class_summ,	print summary data from a class )
ADD_CMD( sequential_data,	do_show_class_seq,	print sequential data from a class )
ADD_CMD( clear_all,	do_clear_all_classes,	clear data for all conditions )
ADD_CMD( delete_all,	do_delete_all_classes,	delete all conditions )
MENU_END(class)

COMMAND_FUNC( do_class_menu )
{
	CHECK_AND_PUSH_MENU(class);
}


