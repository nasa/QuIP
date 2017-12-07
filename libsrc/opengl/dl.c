#include "quip_config.h"

#ifdef HAVE_OPENGL

#ifdef BUILD_FOR_OBJC
#include <OpenGL/gl.h>
#else // ! BUILD_FOR_OBJC
#ifdef HAVE_GL_GLX_H
#include <GL/glx.h>
#endif // HAVE_GL_GLX_H
#endif // ! BUILD_FOR_OBJC

#include "quip_prot.h"
#include "dl.h"
#include "gl_viewer.h"

static Display_List *current_dlp=NULL;
static int next_serial_number=0;

static GLenum dl_mode=GL_COMPILE;

ITEM_INTERFACE_DECLARATIONS(Display_List,dl,0)


COMMAND_FUNC( do_del_dl )
{
	Display_List *dlp;

	dlp = pick_dl("");
	if( dlp == NULL ) return;

	delete_dl(QSP_ARG  dlp);
}

void delete_dl(QSP_ARG_DECL  Display_List *dlp)
{
	glDeleteLists(dlp->dl_serial,1);

	del_dl( dlp );
	givbuf(dlp->dl_name);
	/* object struct itself is put on the free list... */
}

COMMAND_FUNC( do_new_dl )
{
	Display_List *dlp;
	const char *s;

	s=NAMEOF("name for new display list");

	dlp=dl_of(s);
	if( dlp != NULL ){
		sprintf(ERROR_STRING,"A display list named \"%s\" already exists!?",s);
		WARN(ERROR_STRING);
		return;
	}
	new_display_list(QSP_ARG  s);
}

void new_display_list(QSP_ARG_DECL  const char *name)
{
	Display_List *dlp;

	if( current_dlp != NULL ){
		sprintf(ERROR_STRING,"Display list %s is already open, can't create new display list %s",
			current_dlp->dl_name,name);
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,"End display list %s first",current_dlp->dl_name);
		advise(ERROR_STRING);
		return;
	}

	dlp = new_dl(name);
	if( dlp == NULL ) return;

	/* Here we might do other initialization (e.g. make gl calls to create a display list, set the ptr, etc ) */

	dlp->dl_serial = ++next_serial_number;

	glNewList(dlp->dl_serial,dl_mode);

	current_dlp = dlp;
}

COMMAND_FUNC( do_info_dl )
{
	Display_List *dlp;					\
								\
	dlp = pick_dl("");					\
	if( dlp == NULL ) return;			\
								\
	info_dl(QSP_ARG  dlp );
}

//COMMAND_FUNC( do_dump_dl ) { CALL_IF_EXISTS( dump_dl ) } 

COMMAND_FUNC( do_call_dl )
{
	Display_List *dlp;					\
								\
	dlp = pick_dl("");					\
	if( dlp == NULL ) return;			\
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

void _end_dl(SINGLE_QSP_ARG_DECL)
{
	if( current_dlp == NULL )
		warn("No display list currently open, can't end.");
	else {
		glEndList();
		current_dlp = NULL;
	}
}

void _call_dl(QSP_ARG_DECL  Display_List *dlp)
{
	if( current_dlp != NULL && dlp==current_dlp ){
		sprintf(ERROR_STRING,"Recursive call to display list %s!?",dlp->dl_name);
		warn(ERROR_STRING);
		return;
	}
	glCallList(dlp->dl_serial);
}

double display_list_exists(QSP_ARG_DECL  const char *name)
{
	Display_List *dl;
	dl = dl_of(name);

	if( dl == NULL ) return(0);
	return(1.0);
}

#endif /* HAVE_OPENGL */
