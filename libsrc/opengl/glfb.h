
typedef struct glfb {
	const char *	fb_name;
	int		fb_width;
	int		fb_height;
#ifdef HAVE_OPENGL
	GLenum		fb_id;
	GLenum		fb_renderbuffer;
#endif // HAVE_OPENGL
} Framebuffer;

ITEM_INTERFACE_PROTOTYPES(Framebuffer,glfb)

#define PICK_GLFB(s)	pick_glfb(QSP_ARG  s)

extern Framebuffer * create_framebuffer(QSP_ARG_DECL  const char *name, int w, int h);
extern void delete_framebuffer( QSP_ARG_DECL  Framebuffer *fbp );
extern void glfb_info( QSP_ARG_DECL  Framebuffer *fbp );

