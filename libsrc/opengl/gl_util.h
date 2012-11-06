
#include "quip_config.h"

#include "glut_supp.h"
#include "data_obj.h"

typedef struct named_constant {
	const char *	nc_name;
#ifdef HAVE_OPENGL
	GLenum		nc_code;
#endif
} Named_Constant;

extern debug_flag_t gl_debug;


extern void set_texture_image(Data_Obj *);

#ifdef HAVE_OPENGL

#define INVALID_CONSTANT	((GLenum)-1)

extern const char *gl_cap_string(GLenum cap);
extern GLenum choose_viewing_mode(QSP_ARG_DECL const char *);
extern GLenum choose_primitive(QSP_ARG_DECL const char *);
extern GLenum choose_winding_dir(QSP_ARG_DECL const char *);
extern GLenum choose_facing_dir(QSP_ARG_DECL const char *);
extern GLenum choose_cap(QSP_ARG_DECL const char *);
extern GLenum choose_shading_model(QSP_ARG_DECL const char *);
extern const char *primitive_name(GLenum);
extern GLenum choose_light_source(QSP_ARG_DECL const char *);
extern GLenum choose_polygon_mode(QSP_ARG_DECL const char *prompt);
#endif /* HAVE_OPENGL */

#define CHOOSE_VIEWING_MODE(pmpt)		choose_viewing_mode(QSP_ARG pmpt)
#define CHOOSE_PRIMITIVE(pmpt)			choose_primitive(QSP_ARG pmpt)
#define CHOOSE_WINDING_DIR(pmpt)		choose_winding_dir(QSP_ARG pmpt)
#define CHOOSE_FACING_DIR(pmpt)			choose_facing_dir(QSP_ARG pmpt)
#define CHOOSE_CAP(pmpt)			choose_cap(QSP_ARG pmpt)
#define CHOOSE_SHADING_MODEL(pmpt)		choose_shading_model(QSP_ARG pmpt)
#define CHOOSE_LIGHT_SOURCE(pmpt)		choose_light_source(QSP_ARG pmpt)
#define CHOOSE_POLYGON_MODE(pmpt)		choose_polygon_mode(QSP_ARG pmpt)

