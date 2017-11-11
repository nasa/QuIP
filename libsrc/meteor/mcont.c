
#include "quip_config.h"

#ifdef HAVE_METEOR

/* meteor video controls */

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

#include "ioctl_meteor.h"
#include "quip_prot.h"
#include "mmenu.h"

typedef struct meteor_setting {
	const char *	ms_name;
	int		ms_setcode;
	int		ms_getcode;
} Meteor_Setting;

#define N_SETTINGS	4
static Meteor_Setting setting_tbl[N_SETTINGS]={
{ "hue",	METEORSHUE,	METEORGHUE		},
{ "contrast",	METEORSCONT,	METEORGCONT		},
{ "saturation",	METEORSCSAT,	METEORGCSAT		},
{ "brightness",	METEORSBRIG,	METEORGBRIG		}
};

static const char *setting_choices[N_SETTINGS];

static void meteor_get_setting(QSP_ARG_DECL  Meteor_Setting *msp)
{
	int c;

	if (ioctl(meteor_fd, msp->ms_getcode, &c) < 0){
		perror("get_setting ioctl failed");
		return;
	}
	c &=0xff;

	sprintf(msg_str,"%s is %d, (0x%x)",msp->ms_name,c,c);
	prt_msg(msg_str);
}

static void meteor_set_setting(QSP_ARG_DECL  Meteor_Setting *msp)
{
	int c;

	c=HOW_MANY(msp->ms_name);

	if (ioctl(meteor_fd, msp->ms_setcode, &c) < 0){
		perror("set_setting ioctl failed");
		return;
	}
}

static COMMAND_FUNC( report_setting )
{
	int i;

	for(i=0;i<N_SETTINGS;i++)
		setting_choices[i]=setting_tbl[i].ms_name;
	i=WHICH_ONE("control",N_SETTINGS,setting_choices);
	if( i<0 ) return;
	meteor_get_setting(QSP_ARG  &setting_tbl[i]);
}

static COMMAND_FUNC( update_setting )
{
	int i;

	for(i=0;i<N_SETTINGS;i++)
		setting_choices[i]=setting_tbl[i].ms_name;
	i=WHICH_ONE("control",N_SETTINGS,setting_choices);
	if( i<0 ) return;
	meteor_set_setting(QSP_ARG   &setting_tbl[i]);
}


#define ADD_CMD(s,f,h)	ADD_COMMAND(video_menu,s,f,h)

MENU_BEGIN(video)
ADD_CMD( report,	report_setting,		report a control setting )
ADD_CMD( update,	update_setting,		update a control setting )
MENU_END(video)

COMMAND_FUNC( do_video_controls )
{
	CHECK_AND_PUSH_MENU(video);
}

#endif /* HAVE_METEOR */

