#include "quip_config.h"

#ifdef HAVE_OPENGL

#ifdef HAVE_GL_GLX_H
#include <GL/glx.h>
#endif

#include "quip_prot.h"
#include "dl.h"
#include "gl_viewer.h"

static Display_List *current_dlp=NO_DISPLAY_LIST;
static int next_serial_number=0;

static GLenum dl_mode=GL_COMPILE;

ITEM_INTERFACE_DECLARATIONS(Display_List,dl)
#define PICK_DL(pmpt)	pick_dl(QSP_ARG pmpt)


COMMAND_FUNC( do_del_dl )
{
	Display_List *dlp;

	dlp = PICK_DL("");
	if( dlp == NO_DISPLAY_LIST ) return;

	delete_dl(QSP_ARG  dlp);
}

void delete_dl(QSP_ARG_DECL  Display_List *dlp)
{
	glDeleteLists(dlp->dl_serial,1);

	del_dl( QSP_ARG  dlp );
	givbuf(dlp->dl_name);
	/* object struct itself is put on the free list... */
}

COMMAND_FUNC( do_new_dl )
{
	Display_List *dlp;
	const char *s;

	s=NAMEOF("name for new display list");

	dlp=dl_of(QSP_ARG  s);
	if( dlp != NO_DISPLAY_LIST ){
		sprintf(ERROR_STRING,"A display list named \"%s\" already exists!?",s);
		WARN(ERROR_STRING);
		return;
	}
	new_display_list(QSP_ARG  s);
}

void new_display_list(QSP_ARG_DECL  const char *name)
{
	Display_List *dlp;

	if( current_dlp != NO_DISPLAY_LIST ){
		sprintf(ERROR_STRING,"Display list %s is already open, can't create new display list %s",
			current_dlp->dl_name,name);
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,"End display list %s first",current_dlp->dl_name);
		advise(ERROR_STRING);
		return;
	}

	dlp = new_dl(QSP_ARG  name);
	if( dlp == NO_DISPLAY_LIST ) return;

	/* Here we might do other initialization (e.g. make gl calls to create a display list, set the ptr, etc ) */

	dlp->dl_serial = ++next_serial_number;

	glNewList(dlp->dl_serial,dl_mode);

	current_dlp = dlp;
}

COMMAND_FUNC( do_info_dl )
{
	Display_List *dlp;					\
								\
	dlp = PICK_DL("");					\
	if( dlp == NO_DISPLAY_LIST ) return;			\
								\
	info_dl(QSP_ARG  dlp );
}

//COMMAND_FUNC( do_dump_dl ) { CALL_IF_EXISTS( dump_dl ) } 

COMMAND_FUNC( do_call_dl )
{
	Display_List *dlp;					\
								\
	dlp = PICK_DL("");					\
	if( dlp == NO_DISPLAY_LIST ) return;			\
								\
	call_dl( dlp );
}

void info_dl(QSP_ARG_DECL  Display_List *dlp)
{
	sprintf(msg_str,"Display List \"%s\":",dlp->dl_name);
	prt_msg(msg_str);
	sprintf(msg_str,"\tserial = %d",dlp->dl_serial);
	prt_msg(msg_str);
}

//void dump_dl(Display_List *dlp)
//{
//	sprintf(ERROR_STRING,"Sorry, don't know how to dump a display list yet");
//	advise(ERROR_STRING);
//}

COMMAND_FUNC( do_end_dl )
{
	end_dl();
}

void end_dl(void)
{
	if( current_dlp == NO_DISPLAY_LIST )
		NWARN("No display list currently open, can't end.");
	else {
		glEndList();
		current_dlp = NO_DISPLAY_LIST;
	}
}

void call_dl(Display_List *dlp)
{
	if( current_dlp != NO_DISPLAY_LIST && dlp==current_dlp ){
		sprintf(DEFAULT_ERROR_STRING,"Recursive call to display list %s!?",dlp->dl_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	glCallList(dlp->dl_serial);
}

double display_list_exists(QSP_ARG_DECL  const char *name)
{
	Display_List *dl;
	dl = dl_of(QSP_ARG  name);

	if( dl == NO_DISPLAY_LIST ) return(0);
	return(1.0);
}

#endif /* HAVE_OPENGL */
