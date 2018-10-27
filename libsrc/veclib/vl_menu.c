
#include "quip_config.h"
#include "quip_prot.h"
#include "quip_menu.h"
#include "veclib/vec_func.h"

#include "nvf.h"

static COMMAND_FUNC( do_vf_exec )
{
	Vector_Function *vfp;

	vfp = pick_vec_func("");

	if( vfp == NULL ) return;

	do_vfunc(vfp);
}

static const char *number_type_name[/* N_ARGSET_TYPES */]={
	"unknown",
	"real",
	"complex",
	"mixed_R/C",
	"quaternion",
	"mixed_R/Q"
};

#define vf_info(vfp) _vf_info(QSP_ARG  vfp)

static void _vf_info(QSP_ARG_DECL  Vector_Function *vfp)
{
	int i;
	int n_printed=0;

	sprintf(msg_str,"Vector function %s:",VF_NAME(vfp));
	prt_msg(msg_str);

#define MAX_PER_LINE	4

	prt_msg_frag("\tallowable precisions:");
	for(i=0;i<N_MACHINE_PRECS;i++){
		if( VF_PRECMASK(vfp) & (1<<i) ){
			/* BUG?  is 0 a legal precision code? it is PREC_NONE... */
			sprintf(msg_str,"%s%s",
			(n_printed>=MAX_PER_LINE?",\n\t\t\t": (n_printed>0?", ":"")),
			PREC_NAME(prec_for_code(i)) );
			prt_msg_frag(msg_str);
			if( n_printed >= MAX_PER_LINE ) n_printed=0;
			n_printed++;
		}
	}
	prt_msg("");
	prt_msg_frag("\tallowable types:");
	n_printed=0;
	for(i=0;i<N_ARGSET_TYPES;i++){
		if( VF_TYPEMASK(vfp) & VL_TYPE_MASK(i) ){
			sprintf(msg_str,"%s%s",
			n_printed>0?", ":"",
			number_type_name[i]);
			prt_msg_frag(msg_str);
			n_printed++;
		}
	}
	prt_msg("");
}

static COMMAND_FUNC( do_vf_info )
{
	Vector_Function *vfp;

	vfp=pick_vec_func("");
	if( vfp==NULL ) return;
	vf_info(vfp);
}

static COMMAND_FUNC( do_list_vfs )
{ list_vec_funcs(tell_msgfile()); }

#define ADD_CMD(s,f,h)	ADD_COMMAND(veclib_menu,s,f,h)

MENU_BEGIN(veclib)
ADD_CMD( execute,	do_vf_exec,	execute a vector function	)
ADD_CMD( list,		do_list_vfs,	list all vector functions	)
ADD_CMD( info,		do_vf_info,	vector function info		)
ADD_CMD( chains,	do_chains,	chained operation submenu	)
MENU_END(veclib)

COMMAND_FUNC( do_vl_menu )
{
	static int inited=0;

	if( !inited ){
		vl_init(SINGLE_QSP_ARG);
		//init_vl_menu();
		inited=1;
	}
	CHECK_AND_PUSH_MENU(veclib);
}



