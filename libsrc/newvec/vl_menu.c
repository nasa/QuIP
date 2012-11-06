
#include "quip_config.h"
char VersionId_newvec_vl_menu[] = QUIP_VERSION_STRING;

/* test a function */

#include "nvf.h"
#include "new_chains.h"

/* local prototypes */

static void vf_info(Vec_Func *vfp);

static COMMAND_FUNC( vt_exec )
{
	Vec_Func *vfp;

	vfp = PICK_VF("");

	if( vfp == NO_NEW_VEC_FUNC ) return;

	do_vfunc(QSP_ARG  vfp);
}

static const char *number_type_name[N_NUMBER_TYPES]={
	"unknown",
	"real",
	"complex",
	"quaternion"
};

static void vf_info(Vec_Func *vfp)
{
	int i;

	sprintf(msg_str,"Vector function %s:",vfp->vf_name);
	prt_msg(msg_str);

	prt_msg_frag("\tallowable precisions:");
	for(i=0;i<N_MACHINE_PRECS;i++){
		if( vfp->vf_precmask & (1<<i) ){
			sprintf(msg_str,"  %s",prec_name[i]);
			prt_msg_frag(msg_str);
		}
	}
	prt_msg("");
	prt_msg_frag("\tallowable types:");
	for(i=0;i<N_NUMBER_TYPES;i++){
		if( vfp->vf_typemask & (1<<i) ){
			sprintf(msg_str,"  %s",number_type_name[i]);
			prt_msg_frag(msg_str);
		}
	}
	prt_msg("");
}

static COMMAND_FUNC( do_vf_info )
{
	Vec_Func *vfp;

	vfp=PICK_VF("");
	if( vfp==NO_NEW_VEC_FUNC ) return;
	vf_info(vfp);
}

static COMMAND_FUNC( do_list_vfs )
{ list_vfs(SINGLE_QSP_ARG); }

Command vl_ctbl[]={
{ "execute",	vt_exec,	"execute a vector function"		},
{ "list",	do_list_vfs,	"list all vector functions"		},
{ "info",	do_vf_info,	"vector function info"			},
{ "chains",	do_chains,	"chained operation submenu"		},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL_COMMAND								}
};

COMMAND_FUNC( vl_menu )
{
	vl_init(SINGLE_QSP_ARG);
	PUSHCMD(vl_ctbl,"veclib");
}





