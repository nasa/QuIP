
typedef struct glfb {
	const char *	fb_name;
	int		fb_width;
	int		fb_height;
#ifndef BUILD_FOR_MACOS
#ifdef HAVE_OPENGL
	GLenum		fb_id;
	GLenum		fb_renderbuffer;
#endif // HAVE_OPENGL
#endif // ! BUILD_FOR_MACOS
} Framebuffer;

ITEM_INTERFACE_PROTOTYPES(Framebuffer,glfb)

#define pick_glfb(s)	_pick_glfb(QSP_ARG  s)
#define list_glfbs(fp)	_list_glfbs(QSP_ARG  fp)
#define del_glfb(s)	_del_glfb(QSP_ARG  s)

extern Framebuffer * create_framebuffer(QSP_ARG_DECL  const char *name, int w, int h);
extern void delete_framebuffer( QSP_ARG_DECL  Framebuffer *fbp );
extern void glfb_info( QSP_ARG_DECL  Framebuffer *fbp );

