#include "quip_config.h"

#ifdef HAVE_OPENGL

// This needs to come first, but on the mac it complains!?
#ifndef BUILD_FOR_OBJC
#ifdef HAVE_GL_GLEW_H
#include <GL/glew.h>
#endif
#endif // BUILD_FOR_OBJC

#ifdef HAVE_GLUT
#include "glut_supp.h"
#endif	/* HAVE_GLUT */

#include "glx_supp.h"

#include "quip_prot.h"
#include "debug.h"

#include "platform.h"
#include "pf_viewer.h"

#include "gl_util.h"
#include "data_obj.h"
#include "gl_info.h"
#include "dl.h"
#include "tile.h"
#include "glfb.h"
#include "opengl_utils.h"

#ifdef BUILD_FOR_MACOS
#include <OpenGL/glu.h>
#endif // BUILD_FOR_MACOS

#define NOT_IMP(s)	{ sprintf(ERROR_STRING,"Sorry, %s not implemented yet.",s); NWARN(ERROR_STRING); }

#include "string.h"

debug_flag_t gl_debug=0;

#define check_gl_error(s)	_check_gl_error(QSP_ARG  s)

static void _check_gl_error(QSP_ARG_DECL  char *s)
{
	GLenum e;

	e=glGetError();
	if( e == GL_NO_ERROR ) return;
	switch(e){
		case GL_INVALID_OPERATION:
			sprintf(ERROR_STRING,
				"%s:  invalid operation",s);
			WARN(ERROR_STRING);
			break;
		default:
			sprintf(ERROR_STRING,"check_gl_error:  unhandled error code after %s",s);
			WARN(ERROR_STRING);
			break;
	}
}

static void get_rgb_triple(QSP_ARG_DECL float *v)
{
	v[0]=(float)HOW_MUCH("red");
	v[1]=(float)HOW_MUCH("green");
	v[2]=(float)HOW_MUCH("blue");
}

static COMMAND_FUNC( set_clear_color )
{
	float v[3];

	get_rgb_triple(QSP_ARG v);

	if( debug & gl_debug ) advise("glClearColor");

	glClearColor(v[0],v[1],v[2],0.0);
}

static COMMAND_FUNC( do_clear_color )
{
	if( debug & gl_debug ) advise("glClear GL_COLOR_BUFFER_BIT");
	glClear(GL_COLOR_BUFFER_BIT);
}

/* other possible values are:
 * GL_ACCUM_BUFFER_BIT
 * GL_STENCIL_BUFFER_BIT
 *
 * corresponding routines:
 * glClearIndex
 * glClearAccum
 * glClearStencil
 */

static COMMAND_FUNC( do_clear_depth )
{
	if( debug & gl_debug ) advise("glClearDepth 1.0");
	glClearDepth(1.0);
	if( debug & gl_debug ) advise("glClear GL_DEPTH_BUFFER_BIT");
	glClear(GL_DEPTH_BUFFER_BIT);
}

static COMMAND_FUNC( set_gl_pen )
{
	float v[3];

	get_rgb_triple(QSP_ARG v);
	if( debug & gl_debug ) advise("glColor3f");
	glColor3f(v[0],v[1],v[2]);
}

#define N_SHADING_MODELS	2
static const char *shading_models[]={
	"flat",
	"smooth"
};

static COMMAND_FUNC( select_shader )
{
	int i;

	i=WHICH_ONE("shading model",N_SHADING_MODELS,shading_models);
	if( i < 0 ) return;

	if( debug & gl_debug ) advise("glShadeModel");
	switch(i){
		case 0:	glShadeModel(GL_FLAT); break;
		case 1:	glShadeModel(GL_SMOOTH); break;
	}
}

static COMMAND_FUNC(do_glFlush)
{
	glFlush();
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(color_menu,s,f,h)

MENU_BEGIN( color )
ADD_CMD( background,	set_clear_color,	set color for clear )
ADD_CMD( clear_color,	do_clear_color,		clear color buffer )
ADD_CMD( clear_depth,	do_clear_depth,		clear depth buffer )
ADD_CMD( color,		set_gl_pen,		set current drawing color )
ADD_CMD( shade,		select_shader,		select shading model )
ADD_CMD( flush,		do_glFlush,		flush graphics pipeline )
MENU_END( color )

static COMMAND_FUNC( do_color_menu )
{
	PUSH_MENU(color);
}

static GLenum current_primitive=INVALID_CONSTANT;

static COMMAND_FUNC( do_gl_begin )
{
	GLenum p;

	p=CHOOSE_PRIMITIVE("type of primitive object");

	if( p == INVALID_CONSTANT ) return;

	if( current_primitive != INVALID_CONSTANT ){
		const char *s;

		s=primitive_name(current_primitive);
		if( s != NULL ){
			sprintf(ERROR_STRING,
		"Can't begin new primitive, already specifying %s", s );
			NWARN(ERROR_STRING);
			return;
		}
	}

	if( debug & gl_debug ) advise("glBegin");
	glBegin(p);
	current_primitive = p;
}

static COMMAND_FUNC( do_gl_end )
{
	if( current_primitive == INVALID_CONSTANT ){
		NWARN("Can't end, no object begun!?");
		return;
	}

	/* BUG check here that vertex count is appropriate for obj type */

	if( debug & gl_debug ) advise("glEnd");
	glEnd();
	current_primitive = INVALID_CONSTANT ;
}

static COMMAND_FUNC(	do_gl_vertex )
{
	float x,y,z;

	x=(float)HOW_MUCH("x coordinate");
	y=(float)HOW_MUCH("y coordinate");
	z=(float)HOW_MUCH("z coordinate");

	if( debug & gl_debug ){
		sprintf(ERROR_STRING,"glVertex3f %g %g %g",x,y,z);
		advise(ERROR_STRING);
	}
	glVertex3f(x,y,z);
}

static COMMAND_FUNC(	do_gl_color )
{
	float r,g,b;

	r=(float)HOW_MUCH("red");
	g=(float)HOW_MUCH("green");
	b=(float)HOW_MUCH("blue");

	if( debug & gl_debug ){
		sprintf(ERROR_STRING,"glColor3f %g %g %g",r,g,b);
		advise(ERROR_STRING);
	}
	glColor3f(r,g,b);
}

static COMMAND_FUNC(	do_gl_color_material )
{
	GLenum face, mode;

	face = CHOOSE_FACING_DIR("facing direction of polygons");
	mode = CHOOSE_LIGHTING_COMPONENT("reflectivity components to link with color commands");
	glColorMaterial(face,mode);
}

static COMMAND_FUNC(	do_gl_normal )
{
	float x,y,z;

	x=(float)HOW_MUCH("x coordinate");
	y=(float)HOW_MUCH("y coordinate");
	z=(float)HOW_MUCH("z coordinate");

	if( debug & gl_debug ){
		sprintf(ERROR_STRING,"glNormal3f %g %g %g",x,y,z);
		advise(ERROR_STRING);
	}
	glNormal3f(x,y,z);
}

static COMMAND_FUNC(	do_gl_tc )
{
	float s,t;

	s=(float)HOW_MUCH("s coordinate");
	t=(float)HOW_MUCH("t coordinate");

	if( debug & gl_debug ) advise("glTexCoord2f");
	glTexCoord2f(s,t);
}

static COMMAND_FUNC(	do_gl_ef )
{
	NOT_IMP("do_gl_ef")
}

#define N_MATERIAL_PROPERTIES	5
static const char *property_names[N_MATERIAL_PROPERTIES]={
	"ambient",
	"diffuse",
	"specular",
	"shininess",
	"emission",
};

// missing here are GL_AMBIENT_AND_DIFFUSE and GL_COLOR_INDICES!

static COMMAND_FUNC(	do_gl_material )
{
	int i;
	float pvec[4];

	i=WHICH_ONE("property",N_MATERIAL_PROPERTIES,property_names);
	if( i < 0 ) return;

	switch(i){
		case 0: {
			pvec[0] = (float)HOW_MUCH("ambient red");
			pvec[1] = (float)HOW_MUCH("ambient green");
			pvec[2] = (float)HOW_MUCH("ambient blue");
			pvec[3] = (float)HOW_MUCH("ambient alpha");
			if( debug & gl_debug ) advise("glMaterialfv GL_FRONT_AND_BACK GL_DIFFUSE (ambient?)");
			/* diffuse or ambient??? */
			glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, pvec);
			break; }
		case 1: {
			pvec[0] = (float)HOW_MUCH("diffuse red");
			pvec[1] = (float)HOW_MUCH("diffuse green");
			pvec[2] = (float)HOW_MUCH("diffuse blue");
			pvec[3] = (float)HOW_MUCH("diffuse alpha");
			if( debug & gl_debug ) advise("glMaterialfv GL_FRONT_AND_BACK GL_DUFFUSE");
			glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, pvec);
			break; }
		case 2: {
			pvec[0] = (float)HOW_MUCH("specular red");
			pvec[1] = (float)HOW_MUCH("specular green");
			pvec[2] = (float)HOW_MUCH("specular blue");
			pvec[3] = (float)HOW_MUCH("specular alpha");
			if( debug & gl_debug ) advise("glMaterialfv GL_FRONT_AND_BACK GL_SPECULAR");
			glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, pvec);
			break; }
		case 3: {
			pvec[0] = (float)HOW_MUCH("shininess");
			if( debug & gl_debug ) advise("glMaterialfv GL_FRONT_AND_BACK GL_SHININESS");
			glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, pvec);
			break; }
		case 4:
			pvec[0] = (float)HOW_MUCH("emission red");
			pvec[1] = (float)HOW_MUCH("emission green");
			pvec[2] = (float)HOW_MUCH("emission blue");
			pvec[2] = (float)HOW_MUCH("emission alpha");
			if( debug & gl_debug ) advise("glMaterialfv GL_FRONT_AND_BACK GL_EMISSION");
			glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, pvec);
			break;
		default:
			ERROR1("do_gl_material:  bad property (shouldn't happen)");
			break;
	}
}

static COMMAND_FUNC(	do_gl_ae )
{
	NOT_IMP("do_gl_ae")
}
static COMMAND_FUNC(	do_gl_ec )
{
	NOT_IMP("do_gl_ec")
}
static COMMAND_FUNC(	do_gl_ep )
{
	NOT_IMP("do_gl_ep")
}

static COMMAND_FUNC( do_fface )
{
	GLenum dir;

	dir = CHOOSE_WINDING_DIR("winding direction of polygon front faces");
	if( dir == INVALID_CONSTANT ) return;

	if( debug & gl_debug ) advise("glFrontFace");
	glFrontFace(dir);
}

static COMMAND_FUNC( do_cface )
{
	GLenum c;

	c = CHOOSE_FACING_DIR("facing direction of polygons to be culled");
	if( c == INVALID_CONSTANT ) return;

	if( debug & gl_debug ) advise("glCullFace");
	glCullFace(c);
}

static COMMAND_FUNC(	do_gl_ptsize )
{
	float size;

	size = (float)HOW_MUCH("size");

	if( debug & gl_debug ) advise("glPointSize");
	glPointSize(size);
}

/* Each hit uses up at least 4 table entries...
 * The number of names, minz, maxz, then one or more names
 *
 * For some reason, all 6 names objects are showing up as hit???
 */

/* This function gets called when there is a mouse click... */
//#define MAX_SELECT 128
#define MAX_SELECT 64
static GLuint selectBuf[MAX_SELECT];

static COMMAND_FUNC( do_slct_obj )
{
	int x, y;
	const char *s;
	GLuint n_names, *ptr;
	GLint hits;
	GLuint this_hit=0;	/* initialize to silence compiler */
	GLuint the_hit;
	float z1,z2;
	float z_this,z_min;
	int clickWindowSize = 1;
	GLint viewport[4];
	GLint i;
	GLuint j;
	char ret_str[32];

	x=(int)HOW_MANY("x location");
	y=(int)HOW_MANY("y location");
	s=NAMEOF("scene render command");
	s=savestr(s);

	//glPushMatrix();

	glSelectBuffer(MAX_SELECT,selectBuf);
	check_gl_error("glSelectBuffer");
	glRenderMode(GL_SELECT);
	check_gl_error("glRenderMode");

	glMatrixMode(GL_PROJECTION);
	check_gl_error("glMatrixMode");
	glPushMatrix();
	check_gl_error("glPushMatrix");
	glLoadIdentity();
	check_gl_error("glLoadIdentity");

	glGetIntegerv(GL_VIEWPORT,viewport);
	check_gl_error("glGetIntegerv");
	gluPickMatrix(x,viewport[3]-y,
			clickWindowSize,clickWindowSize,viewport);
	check_gl_error("gluPickMatrix");
	// Setup camera.
	//drawer.setupCamera(&manager);
	//glMatrixMode(GL_MODELVIEW);
	glInitNames();
	check_gl_error("glInitNames");
	glPushName(0);
	check_gl_error("glPushName");
	//drawer.drawAirspace(&manager, &table);
	
	/* Call the user's draw routine here... */
	chew_text(QSP_ARG  s, "(gl object selection)" );
	rls_str(s);

	// Restoring the original projection matrix.
	glMatrixMode(GL_PROJECTION);
	check_gl_error("glMatrixMode GL_PROJECTION");
	glPopMatrix();
	check_gl_error("glPopMatrix");
	glMatrixMode(GL_MODELVIEW);
	check_gl_error("glMatrixMode GL_MODELVIEW");
	glFlush();
	check_gl_error("glFlush");

	// Returning to normal rendering mode.
	hits = glRenderMode(GL_RENDER);

	//printf ("hits = %d\n", hits);

	// Red book example.
	// When there are multiple objects, the hits are recorded
	// in drawing order, we have to check the z values to know
	// which is the foreground object.
	z_min=100000;
	ptr = (GLuint *) selectBuf;
	for (i = 0; i < hits; i++) {
		n_names = *ptr++;
		z1= ((float)*ptr++)/0x7fffffff;
		z2= ((float)*ptr++)/0x7fffffff;

		if( z1 < z2 ) z_this = z1;
		else z_this=z2;

		if( z_this < z_min ) z_min=z_this;

		for(j=0;j<n_names;j++){
			this_hit=*ptr++;
		}
//		sprintf(ERROR_STRING,"Hit %d, %d from %g to %g, last is %d",
//			i,n_names,z1,z2,this_hit);
//		advise(ERROR_STRING);
	}
	the_hit=0;
	ptr = (GLuint *) selectBuf;
	for (i = 0; i < hits; i++) {
		n_names = *ptr++;
		z1= ((float)*ptr++)/0x7fffffff;
		z2= ((float)*ptr++)/0x7fffffff;
		if( z1 < z2 ) z_this = z1;
		else z_this=z2;
		for(j=0;j<n_names;j++){
			this_hit=*ptr++;
		}
		if( z_this == z_min )
			the_hit=this_hit;
	}
//sprintf(ERROR_STRING,"front-most hit is %d",the_hit);
//advise(ERROR_STRING);
	sprintf(ret_str,"%d",the_hit);
	ASSIGN_RESERVED_VAR("selection_index",ret_str);

	//if (hits != 0) {
	//manager.advanceTrial(processHits());
	//}

	//glPopMatrix();

	/* Set a variable to indicate what happened */
}

static COMMAND_FUNC( do_load_name )
{
	int n;

	n=(int)HOW_MANY("'name' number");
	if( debug & gl_debug ){
		sprintf(ERROR_STRING,"glLoadName %d",n);
		advise(ERROR_STRING);
	}
	glLoadName(n);
	check_gl_error("glLoadName");
}

static COMMAND_FUNC( do_push_name )
{
	int n;

	n=(int)HOW_MANY("'name' number");
	if( debug & gl_debug ){
		sprintf(ERROR_STRING,"glPushName %d",n);
		advise(ERROR_STRING);
	}
	glPushName(n);
}

static COMMAND_FUNC( do_pop_name )
{
	if( debug & gl_debug ){
		sprintf(ERROR_STRING,"glPopName");
		advise(ERROR_STRING);
	}
	glPopName();
}

static COMMAND_FUNC( do_set_buf )
{
	GLenum buf;

	if( debug & gl_debug ){
		sprintf(ERROR_STRING,"glDrawBuffer");
		advise(ERROR_STRING);
	}
	buf=CHOOSE_DRAW_BUFFER("buffer for drawing");
	glDrawBuffer(buf);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(object_menu,s,f,h)

MENU_BEGIN(object)
ADD_CMD( draw_buffer,	do_set_buf,	set drawing buffer )
ADD_CMD( begin_obj,	do_gl_begin,	begin primitive description )
ADD_CMD( end_obj,	do_gl_end,	end primitive description )
ADD_CMD( vertex,	do_gl_vertex,	specify a vertex )
ADD_CMD( color,		do_gl_color,	set current color )
ADD_CMD( color_material,		do_gl_color_material,	specify material properties set by color command (when material_properties enabled) )
ADD_CMD( normal,	do_gl_normal,	set normal vector )
ADD_CMD( tex_coord,	do_gl_tc,	set texture coordinate )
ADD_CMD( edge_flag,	do_gl_ef,	control drawing of edges )
ADD_CMD( material,	do_gl_material,	set material properties )
ADD_CMD( array_elt,	do_gl_ae,	extract vertex array data )
ADD_CMD( eval_coord,	do_gl_ec,	generate coordinates )
ADD_CMD( eval_point,	do_gl_ep,	generate point coordinates )
ADD_CMD( front_face,	do_fface,	specify front face of polygons )
ADD_CMD( cull_face,	do_cface,	specfy cull face of polygons )
ADD_CMD( point_size,	do_gl_ptsize,	set width in pixels of points )
ADD_CMD( select,	do_slct_obj,	select an object with the mouse )
ADD_CMD( load_name,	do_load_name,	load a name )
ADD_CMD( push_name,	do_push_name,	push a name onto the stack )
ADD_CMD( pop_name,	do_pop_name,	pop a name from the stack )
MENU_END(object)

static COMMAND_FUNC( do_gl_obj_menu )
{
	PUSH_MENU(object);
}

static COMMAND_FUNC( do_enable )
{
	GLenum cap;

	cap = CHOOSE_CAP("capability");
	if( cap == INVALID_CONSTANT ){
		WARN("do_enable:	bad capability chosen");
		return;
	}

	if( debug & gl_debug ) {
		sprintf(ERROR_STRING,"glEnable %s",gl_cap_string(cap));
		advise(ERROR_STRING);
	}
	glEnable(cap);
}

static COMMAND_FUNC( do_disable )
{
	GLenum cap;

	cap = CHOOSE_CAP("capability");
	if( cap == INVALID_CONSTANT ) return;

	if( debug & gl_debug ) {
		sprintf(ERROR_STRING,"glDisable %s",gl_cap_string(cap));
		advise(ERROR_STRING);
	}
	glDisable(cap);
}

#define CAP_RESULT_VARNAME	"cap_enabled"

static COMMAND_FUNC( do_cap_q )
{
	GLenum cap;

	cap = CHOOSE_CAP("capability");
	if( cap == INVALID_CONSTANT ){
		ASSIGN_RESERVED_VAR(CAP_RESULT_VARNAME,"-1");
		return;
	}

	if( glIsEnabled(cap) == GL_TRUE ){
		ASSIGN_RESERVED_VAR(CAP_RESULT_VARNAME,"1");
	} else {
		ASSIGN_RESERVED_VAR(CAP_RESULT_VARNAME,"0");
	}
}


static const char * extension_table[]={
	"GL_ARB_color_buffer_float",
	"GL_ARB_depth_buffer_float",
	"GL_ARB_depth_clamp",
	"GL_ARB_depth_texture",
	"GL_ARB_draw_buffers",
	"GL_ARB_draw_elements_base_vertex",
	"GL_ARB_draw_instanced",
	"GL_ARB_fragment_program",
	"GL_ARB_fragment_program_shadow",
	"GL_ARB_fragment_shader",
	"GL_ARB_framebuffer_object",
	"GL_ARB_framebuffer_sRGB",
	"GL_ARB_half_float_pixel",
	"GL_ARB_half_float_vertex",
	"GL_ARB_imaging",
	"GL_ARB_instanced_arrays",
	"GL_ARB_multisample",
	"GL_ARB_multitexture",
	"GL_ARB_occlusion_query",
	"GL_ARB_pixel_buffer_object",
	"GL_ARB_point_parameters",
	"GL_ARB_point_sprite",
	"GL_ARB_provoking_vertex",
	"GL_ARB_seamless_cube_map",
	"GL_ARB_shader_objects",
	"GL_ARB_shader_texture_lod",
	"GL_ARB_shading_language_100",
	"GL_ARB_shadow",
	"GL_ARB_sync",
	"GL_ARB_texture_border_clamp",
	"GL_ARB_texture_compression",
	"GL_ARB_texture_compression_rgtc",
	"GL_ARB_texture_cube_map",
	"GL_ARB_texture_env_add",
	"GL_ARB_texture_env_combine",
	"GL_ARB_texture_env_crossbar",
	"GL_ARB_texture_env_dot3",
	"GL_ARB_texture_float",
	"GL_ARB_texture_mirrored_repeat",
	"GL_ARB_texture_non_power_of_two",
	"GL_ARB_texture_rectangle",
	"GL_ARB_texture_rg",
	"GL_ARB_transpose_matrix",
	"GL_ARB_vertex_array_bgra",
	"GL_ARB_vertex_blend",
	"GL_ARB_vertex_buffer_object",
	"GL_ARB_vertex_program",
	"GL_ARB_vertex_shader",
	"GL_ARB_window_pos",
	"GL_EXT_abgr",
	"GL_EXT_bgra",
	"GL_EXT_bindable_uniform",
	"GL_EXT_blend_color",
	"GL_EXT_blend_equation_separate",
	"GL_EXT_blend_func_separate",
	"GL_EXT_blend_minmax",
	"GL_EXT_blend_subtract",
	"GL_EXT_clip_volume_hint",
	"GL_EXT_depth_bounds_test",
	"GL_EXT_draw_buffers2",
	"GL_EXT_draw_range_elements",
	"GL_EXT_fog_coord",
	"GL_EXT_framebuffer_blit",
	"GL_EXT_framebuffer_multisample",
	"GL_EXT_framebuffer_multisample_blit_scaled",
	"GL_EXT_framebuffer_object",
	"GL_EXT_framebuffer_sRGB",
	"GL_EXT_geometry_shader4",
	"GL_EXT_gpu_program_parameters",
	"GL_EXT_gpu_shader4",
	"GL_EXT_multi_draw_arrays",
	"GL_EXT_packed_depth_stencil",
	"GL_EXT_packed_float",
	"GL_EXT_provoking_vertex",
	"GL_EXT_rescale_normal",
	"GL_EXT_secondary_color",
	"GL_EXT_separate_specular_color",
	"GL_EXT_shadow_funcs",
	"GL_EXT_stencil_two_side",
	"GL_EXT_stencil_wrap",
	"GL_EXT_texture_array",
	"GL_EXT_texture_compression_dxt1",
	"GL_EXT_texture_compression_s3tc",
	"GL_EXT_texture_env_add",
	"GL_EXT_texture_filter_anisotropic",
	"GL_EXT_texture_integer",
	"GL_EXT_texture_lod_bias",
	"GL_EXT_texture_mirror_clamp",
	"GL_EXT_texture_rectangle",
	"GL_EXT_texture_shared_exponent",
	"GL_EXT_texture_sRGB",
	"GL_EXT_texture_sRGB_decode",
	"GL_EXT_timer_query",
	"GL_EXT_transform_feedback",
	"GL_EXT_vertex_array_bgra",
	"GL_APPLE_aux_depth_stencil",
	"GL_APPLE_client_storage",
	"GL_APPLE_element_array",
	"GL_APPLE_fence",
	"GL_APPLE_float_pixels",
	"GL_APPLE_flush_buffer_range",
	"GL_APPLE_flush_render",
	"GL_APPLE_object_purgeable",
	"GL_APPLE_packed_pixels",
	"GL_APPLE_pixel_buffer",
	"GL_APPLE_rgb_422",
	"GL_APPLE_row_bytes",
	"GL_APPLE_specular_vector",
	"GL_APPLE_texture_range",
	"GL_APPLE_transform_hint",
	"GL_APPLE_vertex_array_object",
	"GL_APPLE_vertex_array_range",
	"GL_APPLE_vertex_point_size",
	"GL_APPLE_vertex_program_evaluators",
	"GL_APPLE_ycbcr_422",
	"GL_ATI_separate_stencil",
	"GL_ATI_texture_env_combine3",
	"GL_ATI_texture_float",
	"GL_ATI_texture_mirror_once",
	"GL_IBM_rasterpos_clip",
	"GL_NV_blend_square",
	"GL_NV_conditional_render",
	"GL_NV_depth_clamp",
	"GL_NV_fog_distance",
	"GL_NV_fragment_program_option",
	"GL_NV_fragment_program2",
	"GL_NV_light_max_exponent",
	"GL_NV_multisample_filter_hint",
	"GL_NV_point_sprite",
	"GL_NV_texgen_reflection",
	"GL_NV_vertex_program2_option",
	"GL_NV_vertex_program3",
	"GL_SGIS_generate_mipmap",
	"GL_SGIS_texture_edge_clamp",
	"GL_SGIS_texture_lod"
};

#define N_KNOWN_EXTENSIONS	(sizeof(extension_table)/sizeof(const char *))

static COMMAND_FUNC( do_check_extension )
{
	int i;

	i=WHICH_ONE("GL extension",N_KNOWN_EXTENSIONS,extension_table);
	if( i < 0 ) return;

#ifndef BUILD_FOR_OBJC
	if( check_extension(QSP_ARG  extension_table[i]) ){
		ASSIGN_RESERVED_VAR("extension_present","1");
	} else {
		ASSIGN_RESERVED_VAR("extension_present","0");
	}
#else // ! BUILD_FOR_OBJC
	WARN("Sorry, can't check for extensions in native Apple build...");
#endif // ! BUILD_FOR_OBJC
}

static COMMAND_FUNC( do_tex_image )
{
	Data_Obj *dp = PICK_OBJ("image");
	//im_dim = HOW_MUCH("pixel dimension");

	if( dp == NULL ) return;

	set_texture_image(QSP_ARG  dp);
}

void set_texture_image(QSP_ARG_DECL  Data_Obj *dp)
{
	int code,prec;
	/*glDepthFunc(GL_LEQUAL);
		glPixelStorei(GL_UNPACK_ALIGNMENT,1);*/

	if(OBJ_COMPS(dp)==1) code=GL_LUMINANCE;
	else if( OBJ_COMPS(dp) == 3 ) code=GL_RGB;
	else {
		sprintf(ERROR_STRING,
			"set_texture_image:  Object %s has type dimension %d, expected 1 or 3",
			OBJ_NAME(dp),OBJ_COMPS(dp));
		NWARN(ERROR_STRING);
		return;
	}

	if( OBJ_PREC(dp) == PREC_SP ) prec=GL_FLOAT;
	else if( OBJ_PREC(dp) == PREC_UBY ) prec=GL_UNSIGNED_BYTE;
	else {
		sprintf(ERROR_STRING,"set_texture_image:  Object %s has precision %s, expected %s or %s",
			OBJ_NAME(dp),PREC_NAME(OBJ_PREC_PTR(dp)),
			NAME_FOR_PREC_CODE(PREC_SP),NAME_FOR_PREC_CODE(PREC_UBY));
		NWARN(ERROR_STRING);
		return;
	}

	if( debug & gl_debug ) advise("glTexImage2D");
	glTexImage2D(GL_TEXTURE_2D, 0, OBJ_COMPS(dp), OBJ_COLS(dp),
		OBJ_ROWS(dp), 0, code, prec, OBJ_DATA_PTR(dp));

	if( debug & gl_debug ) advise("glTexParameterf (4)");
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	if( debug & gl_debug ) advise("glTexEnv");
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	/*glEnable(GL_TEXTURE_2D);
		glShadeModel(GL_FLAT);*/
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(capabilities_menu,s,f,h)

MENU_BEGIN(capabilities)
ADD_CMD( enable,	do_enable,	enable capability )
ADD_CMD( disable,	do_disable,	disable capability )
ADD_CMD( query,		do_cap_q,	query capability (result in $cap_enabled) )
ADD_CMD( check_extension,	do_check_extension,	query renderer extension (result in $extension_present) )
ADD_CMD( tex_image,	do_tex_image,	specify a texture image )
MENU_END(capabilities)


static COMMAND_FUNC( do_cap_menu )
{
	PUSH_MENU(capabilities);
}

static COMMAND_FUNC( set_pt_size )
{
	GLfloat s;

	s=(float)HOW_MUCH("width in pixels for rendered points");
	if( s <= 0 ){
		sprintf(ERROR_STRING,"Requested point size (%g) must be positive",s);
		WARN(ERROR_STRING);
		return;
	}
	if( debug & gl_debug ) advise("glPointSize");
	glPointSize(s);
}

static COMMAND_FUNC( set_line_width )
{
	GLfloat w;

	w=(float)HOW_MUCH("width in pixels for rendered lines");
	if( w <= 0 ){
		sprintf(ERROR_STRING,"Requested line width (%g) must be positive",w);
		WARN(ERROR_STRING);
		return;
	}
	if( debug & gl_debug ) advise("glLineWidth");
	glLineWidth(w);
}

static COMMAND_FUNC( set_poly_mode )
{

#ifdef FOOBAR
	char *face = NAMEOF("face");
	char *mode = NAMEOF("polygon mode");
	if((strcmp(face,"frontNback"))&&(strcmp(face,"front"))&&(strcmp(face,"back"))){
		advise("Valid types of faces are:	'frontNback'	'front'	'back'");
		return;
	}

	if((strcmp(mode,"point"))&&(strcmp(mode,"line"))&&(strcmp(mode,"fill"))){
		advise("Valid types of modes are:	'point'	'line'	'fill'");
		return;
	}

	if( debug & gl_debug ) advise("glPolygonMode");
	/* could be more elegant ... */
	if (!((strcmp(face, "frontNback"))||(strcmp(mode, "point"))))
		glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
	if (!((strcmp(face, "frontNback"))||(strcmp(mode, "line"))))
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	if (!((strcmp(face, "frontNback"))||(strcmp(mode, "fill"))))
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	if((!strcmp(face, "front"))&&(!strcmp(mode, "point")))
		glPolygonMode(GL_FRONT, GL_POINT);
	if((!strcmp(face, "front"))&&(!strcmp(mode, "line")))
		glPolygonMode(GL_FRONT, GL_LINE);
	if((!strcmp(face, "front"))&&(!strcmp(mode, "fill")))
		glPolygonMode(GL_FRONT, GL_FILL);
	if((!strcmp(face, "back"))&&(!strcmp(mode, "point")))
		glPolygonMode(GL_BACK, GL_POINT);
	if((!strcmp(face, "back"))&&(!strcmp(mode, "line")))
		glPolygonMode(GL_BACK, GL_LINE);
	if((!strcmp(face, "back"))&&(!strcmp(mode, "fill")))
		glPolygonMode(GL_BACK, GL_FILL);
#endif /* FOOBAR */

	GLenum face_dir;
	GLenum polygon_mode;

	face_dir = CHOOSE_FACING_DIR("faces to render");
	polygon_mode = CHOOSE_POLYGON_MODE("rendering mode for polygons");

	if( face_dir == INVALID_CONSTANT || polygon_mode == INVALID_CONSTANT )
		return;

	if( debug & gl_debug ) advise("glPolygonMode");

	glPolygonMode(face_dir,polygon_mode);
}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(mode_menu,s,f,h)

MENU_BEGIN(mode)
ADD_CMD( point_size,	set_pt_size,	set point size )
ADD_CMD( line_width,	set_line_width,	set line width )
ADD_CMD( polygon_mode,	set_poly_mode,	set polygon mode )
MENU_END(mode)

static COMMAND_FUNC( do_mode_menu )
{
	PUSH_MENU(mode);
}

static COMMAND_FUNC( set_xf_mode )
{
	GLenum m;

	m=CHOOSE_VIEWING_MODE("matrix mode");
	if( m == INVALID_CONSTANT ) return;

	if( debug & gl_debug ) advise("glMatrixMode");
	glMatrixMode( m );
}

static COMMAND_FUNC( do_identity )
{
	if( debug & gl_debug ) advise("glLoadIdentity");
	glLoadIdentity();
}

static COMMAND_FUNC( set_frustum )
{
	GLdouble l,r,b,t,n,f;

	l = HOW_MUCH("left");
	r = HOW_MUCH("right");
	b = HOW_MUCH("bottom");
	t = HOW_MUCH("top");
	n = HOW_MUCH("near");
	f = HOW_MUCH("far");

	if( debug & gl_debug ) advise("glFrustum");
	glFrustum(l,r,b,t,n,f);
}

static COMMAND_FUNC( do_look_at )
{
	float x,y,z;
	float cx,cy,cz;
	float ux,uy,uz;

	x = (float)HOW_MUCH("x camera position");
	y = (float)HOW_MUCH("y camera position");
	z = (float)HOW_MUCH("z camera position");
	cx = (float)HOW_MUCH("x target position");
	cy = (float)HOW_MUCH("y target position");
	cz = (float)HOW_MUCH("z target position");
	ux = (float)HOW_MUCH("x up direction");
	uy = (float)HOW_MUCH("y up direction");
	uz = (float)HOW_MUCH("z up direction");

	if( debug & gl_debug ) advise("gluLookAt");
	gluLookAt(x,y,z,cx,cy,cz,ux,uy,uz);
}

static COMMAND_FUNC( do_scale )
{
	float fx,fy,fz;

	fx=(float)HOW_MUCH("x scale factor");
	fy=(float)HOW_MUCH("y scale factor");
	fz=(float)HOW_MUCH("z scale factor");

	if( debug & gl_debug ) advise("glScalef");
	glScalef(fx,fy,fz);
}

static COMMAND_FUNC( do_xlate )
{
	float tx,ty,tz;

	tx=(float)HOW_MUCH("x translation");
	ty=(float)HOW_MUCH("y translation");
	tz=(float)HOW_MUCH("z translation");

	if( debug & gl_debug ) advise("glTranslatef");
	glTranslatef(tx,ty,tz);
}

static COMMAND_FUNC( do_persp )
{
	GLdouble fovy, aspect, near, far;

	fovy = HOW_MUCH("vertical field of view in degrees (0-180)");
	aspect = HOW_MUCH("aspect ration (H/V)");
	near = HOW_MUCH("distance to near clipping plane");
	far = HOW_MUCH("distance to far clipping plane");

	/* BUG check for errors */

	if( debug & gl_debug ) advise("gluPerspective");
	gluPerspective(fovy,aspect,near,far);
}

static COMMAND_FUNC( do_ortho )
{
	GLdouble l,r,b,t,n,f;

	l = HOW_MUCH("left");
	r = HOW_MUCH("right");
	b = HOW_MUCH("bottom");
	t = HOW_MUCH("top");
	n = HOW_MUCH("near");
	f = HOW_MUCH("far");

	if( debug & gl_debug ) advise("glOrtho");
	glOrtho(l,r,b,t,n,f);
}

static COMMAND_FUNC( do_sv_mv_mat )
{
	Data_Obj *dp;

	const char *matrix = NAMEOF("matrix type");
	dp=PICK_OBJ("matrix object");
	if( dp == NULL ) return;

	/* BUG check size & type here */

	if((strcmp(matrix,"modelview"))&&(strcmp(matrix,"projection"))){
		advise("Valid types of matrices are:	'modelview'	'projection'");
		return;
	}

	if (!(strcmp(matrix, "modelview")))
		glGetFloatv(GL_MODELVIEW_MATRIX,(GLfloat *)OBJ_DATA_PTR(dp));
	if (!(strcmp(matrix, "projection")))
		glGetFloatv(GL_PROJECTION_MATRIX,(GLfloat *)OBJ_DATA_PTR(dp));
}

static COMMAND_FUNC( do_ld_mat )
{
	Data_Obj *dp;

	dp=PICK_OBJ("matrix object");
	if( dp == NULL ) return;

	/* BUG check size & type here */

	if( debug & gl_debug ) advise("glLoadMatrixf");
	glLoadMatrixf((GLfloat *)OBJ_DATA_PTR(dp));
}

static COMMAND_FUNC( do_mul_mat )
{
	Data_Obj *dp;

	dp=PICK_OBJ("matrix object");
	if( dp == NULL ) return;

	/* BUG check size & type here */

	if( debug & gl_debug ) advise("glMultMatrixf");
	glMultMatrixf((GLfloat *)OBJ_DATA_PTR(dp));
}

static COMMAND_FUNC( do_rotate )
{
	float angle;
	float dx,dy,dz;

	angle = (float)HOW_MUCH("angle in degrees");
	dx = (float)HOW_MUCH("rotation axis x");
	dy = (float)HOW_MUCH("rotation axis y");
	dz = (float)HOW_MUCH("rotation axis z");

	if( debug & gl_debug ) advise("glRotatef");
	glRotatef(angle,dx,dy,dz);
}

static int n_pushed_matrices=1;

#define MAX_MATRICES	32
/* set this to -1 to force a check of how many we really have */

static int max_matrices=MAX_MATRICES;

static COMMAND_FUNC( do_push_mat )
{
	if( max_matrices < 0 ){
		glGetIntegerv(GL_MAX_MODELVIEW_STACK_DEPTH,&max_matrices);
		sprintf(ERROR_STRING,"%d matrices max. in stack",max_matrices);
		advise(ERROR_STRING);
	}
	if( n_pushed_matrices >= max_matrices ){
		sprintf(ERROR_STRING,
	"Modelview matrix stack already contains %d items, can't push",n_pushed_matrices);
		WARN(ERROR_STRING);
		return;
	}
	n_pushed_matrices++;
	if( debug & gl_debug ) advise("glPushMatrix");
	glPushMatrix();
	/* no error code?
	 * BUG we should count & limit number of pushes allowed...
	 */
}

static COMMAND_FUNC( do_pop_mat )
{
	if( n_pushed_matrices <= 1 ){
		WARN("Can't pop last matrix from stack");
		return;
	}
	if( debug & gl_debug ) advise("glPopMatrix");
	glPopMatrix();
	n_pushed_matrices--;
}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(xform_menu,s,f,h)

MENU_BEGIN(xform)
ADD_CMD( mode,		set_xf_mode,	set mode for viewing transformation )
ADD_CMD( identity,	do_identity,	initialize viewing matrix )
ADD_CMD( frustum,	set_frustum,	specify viewing frustum )
ADD_CMD( ortho,		do_ortho,	specify orthographic viewing volume )
ADD_CMD( look_at,	do_look_at,	specify viewing position and direction )
ADD_CMD( scale,		do_scale,	specify scaling factor )
ADD_CMD( translate,	do_xlate,	specify translation )
ADD_CMD( rotate,	do_rotate,	specify rotation )
ADD_CMD( perspective,	do_persp,	specify perspective transformation )
ADD_CMD( save_matrix,	do_sv_mv_mat,	save current modelview or projection matrix )
ADD_CMD( load_matrix,	do_ld_mat,	load current matrix from object )
ADD_CMD( mult_matrix,	do_mul_mat,	multiply current matrix by object )
ADD_CMD( push_matrix,	do_push_mat,	push down matrix stack )
ADD_CMD( pop_matrix,	do_pop_mat,	pop top of matrix stack )
MENU_END(xform)

static COMMAND_FUNC( do_xf_menu )
{
	PUSH_MENU(xform);
}

static COMMAND_FUNC( set_shading_model )
{
	GLenum m;

	m = CHOOSE_SHADING_MODEL("shading model");
	// BUG need to install it!!

	fprintf(stderr,"set_shading_model:  m = %d\n",m);
	WARN("set_shading_model:  not implemented!?");
}

static GLenum which_light=INVALID_CONSTANT;

static COMMAND_FUNC( do_sel_light )
{
	GLenum l;

	l=CHOOSE_LIGHT_SOURCE("light source");
	if( l == INVALID_CONSTANT ) return;

	which_light=l;
}

#define CHECK_LIGHT(s)							\
									\
	if( which_light == INVALID_CONSTANT ){				\
		sprintf(ERROR_STRING,					\
			"Must select a light before specifying %s",s);	\
		WARN(ERROR_STRING);					\
		return;							\
	}


static COMMAND_FUNC( set_ambient )
{
	float v[4];

	v[0] = (float)HOW_MUCH("red component");
	v[1] = (float)HOW_MUCH("green component");
	v[2] = (float)HOW_MUCH("blue component");
	v[3] = 1.0;

	CHECK_LIGHT("ambient");

	if( debug & gl_debug ) advise("glLightfv GL_AMBIENT");
	glLightfv(which_light,GL_AMBIENT,v);
}

static COMMAND_FUNC( set_diffuse )
{
	float v[4];

	v[0]=(float)HOW_MUCH("red component");
	v[1]=(float)HOW_MUCH("green component");
	v[2]=(float)HOW_MUCH("blue component");
	v[3]=1.0;

	CHECK_LIGHT("diffuse");

	if( debug & gl_debug ) advise("glLightfv GL_DIFFUSE");
	glLightfv(which_light,GL_DIFFUSE,v);
}

static COMMAND_FUNC( set_specular )
{
	float v[4];

	v[0] = (float)HOW_MUCH("red component");
	v[1] = (float)HOW_MUCH("green component");
	v[2] = (float)HOW_MUCH("blue component");
	v[3] = 1.0;

	CHECK_LIGHT("specular");

	if( debug & gl_debug ) advise("glLightfv GL_SPECULAR");
	glLightfv(which_light, GL_SPECULAR, v);
}

static COMMAND_FUNC( set_light_position )
{
	float v[4];

	v[0]=(float)HOW_MUCH("x position");
	v[1]=(float)HOW_MUCH("y position");
	v[2]=(float)HOW_MUCH("z position");
	v[3]=(float)HOW_MUCH("w position");

	CHECK_LIGHT("position")

	if( debug & gl_debug ) advise("glLightfv GL_POSITION");
	glLightfv(which_light,GL_POSITION,v);
}

static COMMAND_FUNC( set_spot_dir )
{
	float v[3];

	v[0]=(float)HOW_MUCH("x component");
	v[1]=(float)HOW_MUCH("y component");
	v[2]=(float)HOW_MUCH("z component");

	CHECK_LIGHT("spot direction")

	if( debug & gl_debug ) advise("glLightfv GL_POSITION");
	glLightfv(which_light,GL_SPOT_DIRECTION,v);
}

static COMMAND_FUNC( set_global_ambient )
{
	float v[4];

	v[0]=(float)HOW_MUCH("red");
	v[1]=(float)HOW_MUCH("green");
	v[2]=(float)HOW_MUCH("blue");
	v[3]=1.0;

	if( debug & gl_debug ) advise("glLightModelfv");
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT,v);
}

static COMMAND_FUNC( set_local_viewer )
{
	if( debug & gl_debug ) advise("glLightModeli");

	if( ASKIF("use viewer position when computing specular reflections") )
		glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER,GL_TRUE);
	else
		glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER,GL_FALSE);
}

static COMMAND_FUNC( set_two_side )
{
	if( debug & gl_debug ) advise("glLightModeli");

	if( ASKIF("light back faces of polygons") )
		glLightModeli(GL_LIGHT_MODEL_TWO_SIDE,GL_TRUE);
	else
		glLightModeli(GL_LIGHT_MODEL_TWO_SIDE,GL_FALSE);
}

#ifdef FOOBAR
static COMMAND_FUNC( set_color_control )
{
	if( ASKIF("calculate specular color separately") )
		glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL,GL_SEPARATE_COLOR_CONTROL);
	else
		glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL,GL_SINGLE_COLOR);
}
#endif /* FOOBAR */

#define N_ATTENUATIONS 3

static const char *atten_names[N_ATTENUATIONS]={
	"constant",
	"linear",
	"quadratic"
};

static COMMAND_FUNC( set_atten )
{
	int i;
	float f;

	i=WHICH_ONE("attenuation model",N_ATTENUATIONS,atten_names);
	f=(float)HOW_MUCH("attenuation constant");
	if( i < 0 ) return;
	if( debug & gl_debug ) advise("glLightf");
	switch(i){
		case 0: glLightf(which_light,GL_CONSTANT_ATTENUATION,f); break;
		case 1: glLightf(which_light,GL_LINEAR_ATTENUATION,f); break;
		case 2: glLightf(which_light,GL_QUADRATIC_ATTENUATION,f); break;
	}
}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(lighting_menu,s,f,h)

MENU_BEGIN(lighting)
ADD_CMD( shading_model,		set_shading_model,	select shading model )
ADD_CMD( select_light,		do_sel_light,		select light for subsequent operations )
ADD_CMD( ambient,		set_ambient,		set ambient parameters )
ADD_CMD( diffuse,		set_diffuse,		set diffuse parameters )
ADD_CMD( specular,		set_specular,		set specular parameters )
ADD_CMD( position,		set_light_position,	set light position )
ADD_CMD( attenuation,		set_atten,		set light attenuation )
ADD_CMD( spot_direction,	set_spot_dir,		set spotlight direction )
ADD_CMD( global_ambient,	set_global_ambient,	set color of global ambient )
ADD_CMD( local_viewer,		set_local_viewer,	enable/disable use of viewing position in specular reflection calculations )
ADD_CMD( two_side,		set_two_side,		enable/disable two-sided lighting )
	/*
ADD_CMD( separate_specular,	set_color_control,	specular color calculated separately )
*/
MENU_END(lighting)

static COMMAND_FUNC( do_lighting_menu )
{
	PUSH_MENU(lighting);
}

static COMMAND_FUNC(do_list_dls){list_dls(QSP_ARG  tell_msgfile(SINGLE_QSP_ARG));}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(display_list_menu,s,f,h)

MENU_BEGIN(display_list)
ADD_CMD( new_list,	do_new_dl,	create new display list )
ADD_CMD( end_list,	do_end_dl,	end current display list )
ADD_CMD( list,		do_list_dls,	list all display lists )
ADD_CMD( delete,	do_del_dl,	delete a display list )
ADD_CMD( info,		do_info_dl,	give info about a display list )
//ADD_CMD( dump,	do_dump_dl,	dump a display list to the screen )
ADD_CMD( call,		do_call_dl,	call display list )
MENU_END(display_list)


static COMMAND_FUNC( do_dl_menu )
{
	PUSH_MENU(display_list);
}

static COMMAND_FUNC(do_swap_buffers){swap_buffers();}

static COMMAND_FUNC( do_vbl_sync )
{
	int sync;

	sync = ASKIF("sync to vblank");

	if( sync ){
		setenv("__GL_SYNC_TO_VBLANK","1",1);
	} else {
		unsetenv("__GL_SYNC_TO_VBLANK");
	}
}

static COMMAND_FUNC( do_vbl_wait )
{
	int n;

	n = (int)HOW_MANY("number of frames to wait");
	wait_video_sync(n);
}

/************* frame buffers ******************/

static COMMAND_FUNC( do_create_fb )
{
	Framebuffer *fbp;
	const char *s;
	int w,h;

	s = NAMEOF("name for framebuffer");
	w = (int)HOW_MANY("width in pixels");
	h = (int)HOW_MANY("height in pixels");

	fbp = create_framebuffer(QSP_ARG  s,w,h);
	if( fbp == NULL ) {
		sprintf(ERROR_STRING,"Error creating framebuffer %s",s);
		WARN(ERROR_STRING);
	}
}

static COMMAND_FUNC( do_delete_fb )
{
	Framebuffer *fbp;

	fbp = PICK_GLFB("");
	if( fbp == NULL ) return;

	delete_framebuffer(QSP_ARG  fbp);
}

static COMMAND_FUNC( do_list_fbs )
{
	list_glfbs(QSP_ARG  tell_msgfile(SINGLE_QSP_ARG));
}

static COMMAND_FUNC( do_fb_info )
{
	Framebuffer *fbp;

	fbp = PICK_GLFB("");
	if( fbp == NULL ) return;

	glfb_info(QSP_ARG  fbp);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(glfb_menu,s,f,h)

MENU_BEGIN(glfb)
ADD_CMD( new_fb,		do_create_fb,	create a new framebuffer object )
ADD_CMD( del_fb,		do_delete_fb,	delete a framebuffer object )
ADD_CMD( list,			do_list_fbs,	list all framebuffer objects )
ADD_CMD( info,			do_fb_info,	print info about a framebuffer object )
MENU_END(glfb)

static COMMAND_FUNC( do_glfb_menu )
{
	PUSH_MENU(glfb);
}

#ifdef HAVE_OPENGL
int gl_pixel_type(Data_Obj *dp)
{
	int t;

	switch(OBJ_COMPS(dp)){
		case 1: t = GL_LUMINANCE; break;
		/* 2 is allowable, but what do we do with it? */
		case 3: t = GL_BGR; break;
		case 4: t = GL_BGRA; break;
		default:
			t=0;	// quiet compiler
			NERROR1("bad pixel depth!?");
			break;
	}
	return(t);
}

void glew_check(SINGLE_QSP_ARG_DECL)
{
#ifdef HAVE_LIBGLEW
	static int glew_checked=0;

	if( glew_checked ){
		if( verbose )
			NADVISE("glew_check:  glew already checked.");
		return;
	}

	// BUG glewInit will core dump if GL is not already initialized!?
	// We try to fix this by making sure that the cuda viewer is already
	// specified for GL before calling this...

	glewInit();

	if (!glewIsSupported( "GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object" )) {
		/*
		fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
		fprintf(stderr, "This sample requires:\n");
		fprintf(stderr, "  OpenGL version 1.5\n");
		fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
		fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
		*/
		/*
		cudaThreadExit();
		exit(-1);
		*/
NERROR1("glew_check:  Please create a GL window before specifying a cuda viewer.");
	}

	glew_checked=1;
#else // ! HAVE_LIBGLEW
advise("glew_check:  libglew not present, can't check for presence of extensions!?.");
#endif // ! HAVE_LIBGLEW
}

#endif // HAVE_OPENGL

// Does the GL context have to be set when we do this??

static COMMAND_FUNC( do_new_gl_buffer )
{
	const char *s;
	Data_Obj *dp;
	Platform_Device *pdp;
	Compute_Platform *cdp;
	dimension_t d,w,h;
#ifdef HAVE_OPENGL
	Dimension_Set ds;
	int t;
#endif // HAVE_OPENGL

	s = NAMEOF("name for GL buffer object");
	cdp = PICK_PLATFORM("platform");
	if( cdp != NULL )
		push_pfdev_context(QSP_ARG  PF_CONTEXT(cdp) );
	pdp = PICK_PFDEV("device");
	if( cdp != NULL )
		pop_pfdev_context(SINGLE_QSP_ARG);

	w = (int)HOW_MANY("width");
	h = (int)HOW_MANY("height");
	d = (int)HOW_MANY("depth");

	/* what should the depth be??? default to 1 for now... */

	if( pdp == NULL ) return;

	/* Make sure this name isn't already in use... */
	dp = dobj_of(QSP_ARG  s);
	if( dp != NULL ){
		sprintf(ERROR_STRING,"Data object name '%s' is already in use, can't use for GL buffer object.",s);
		NWARN(ERROR_STRING);
		return;
	}

#ifdef HAVE_OPENGL
	// BUG need to be able to set the cuda device.
	// Note, however, that we don't need GL buffers on the Tesla...
	//set_data_area(cuda_data_area[0][0]);
	set_data_area( PFDEV_AREA(pdp,PFDEV_GLOBAL_AREA_INDEX) );

	ds.ds_dimension[0]=d;
	ds.ds_dimension[1]=w;
	ds.ds_dimension[2]=h;
	ds.ds_dimension[3]=1;
	ds.ds_dimension[4]=1;
	dp = _make_dp(QSP_ARG  s,&ds,PREC_FOR_CODE(PREC_UBY));
	if( dp == NULL ){
		sprintf(ERROR_STRING,
			"Error creating data_obj header for %s",s);
		ERROR1(ERROR_STRING);
	}

	SET_OBJ_FLAG_BITS(dp, DT_NO_DATA);	/* can't free this data */
	SET_OBJ_FLAG_BITS(dp, DT_GL_BUF);	/* indicate obj is a GL buffer */

	SET_OBJ_DATA_PTR(dp, NULL);
//fprintf(stderr,"do_new_gl_buffer:  allocating gl_info for %s\n",OBJ_NAME(dp));
	SET_OBJ_GL_INFO(dp, (GL_Info *) getbuf( sizeof(GL_Info) ) );
//fprintf(stderr,"do_new_gl_buffer:  DONE allocating gl_info for %s\n",OBJ_NAME(dp));

	glew_check(SINGLE_QSP_ARG);	/* without this, we get a segmentation
			 * violation on glGenBuffers???
			 */

	// We need an extra field in which to store the GL identifier...
	// AND another extra field in which to store the associated texid.

// Why is this ifdef here?  These don't seem to depend
// on libglew???
// Answer:  We need libglew to bring in openGL extensions like glBindBuffer...

//advise("calling glGenBuffers");
//fprintf(stderr,"OBJ_GL_INFO(%s) = 0x%lx\n",OBJ_NAME(dp),(long)OBJ_GL_INFO(dp));
//fprintf(stderr,"OBJ_BUF_ID_P(%s) = 0x%lx\n",OBJ_NAME(dp),(long)OBJ_BUF_ID_P(dp));
	glGenBuffers(1, OBJ_BUF_ID_P(dp) );	// first arg is # buffers to generate?

//sprintf(ERROR_STRING,"glGenBuffers gave us buf_id = %d",OBJ_BUF_ID(dp));
//advise(ERROR_STRING);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER,  OBJ_BUF_ID(dp) ); 

	// glBufferData will allocate the memory for the buffer,
	// but won't copy unless the pointer is non-null
	// How do we get the gpu memory space address?
	// That must be with map

	glBufferData(GL_PIXEL_UNPACK_BUFFER,
		OBJ_COMPS(dp) * OBJ_COLS(dp) * OBJ_ROWS(dp), NULL, GL_STREAM_DRAW);  

	/* buffer arg set to 0 unbinds any previously bound buffers...
	 * and restores client memory usage.
	 */
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
//#endif // HAVE_LIBGLEW

	glGenTextures(1, OBJ_TEX_ID_P(dp) );		// makes a texture name
	glBindTexture(GL_TEXTURE_2D, OBJ_TEX_ID(dp) );
	t = gl_pixel_type(dp);
	glTexImage2D(	GL_TEXTURE_2D,
			0,			// level-of-detail - is this the same as miplevel???
			OBJ_COMPS(dp),		// internal format, can also be symbolic constant such as
						// GL_RGBA etc
			OBJ_COLS(dp),		// width - must be 2^n+2 (border) for some n???
			OBJ_ROWS(dp),		// height - must be 2^m+2 (border) for some m???
			0,			// border - must be 0 or 1
			t,			// format of pixel data
			GL_UNSIGNED_BYTE,	// type of pixel data
			NULL			// pixel data - null pointer means
						// allocate but do not copy?
						// - offset into PIXEL_UNPACK_BUFFER??
			);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Why was this here?  It would seem to un-bind the target???
	glBindTexture(GL_TEXTURE_2D, 0);
	
	//glFinish();	// necessary or not?

//advise("calling platform-specific buffer registration function");
	if( (*PF_REGBUF_FN(PFDEV_PLATFORM(pdp)))( QSP_ARG  dp ) < 0 ){
		WARN("do_new_gl_buffer:  Error in platform-specific buffer registration!?");
		// BUG? - should clean up here!
	}

	// Leave the buffer mapped by default
	//cutilSafeCall(cudaGLMapBufferObject( &OBJ_DATA_PTR(dp),  OBJ_BUF_ID(dp) ));
//advise("calling platform-specific buffer mapping function");
	if( (*PF_MAPBUF_FN(PFDEV_PLATFORM(pdp)))( QSP_ARG  dp ) < 0 ){
		WARN("do_new_gl_buffer:  Error in platform-specific buffer mapping!?");
		// BUG? - should clean up here!
	}

	SET_OBJ_FLAG_BITS(dp, DT_BUF_MAPPED);
	// propagate change to children and parents
	propagate_flag(dp,DT_BUF_MAPPED);

#else // ! HAVE_OPENGL
	NO_OGL_MSG
#endif // ! HAVE_OPENGL
} /* end do_new_gl_buffer */


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(gl_menu,s,f,h)

MENU_BEGIN(gl)
ADD_CMD( window,		do_render_to,	specify drawing window )
ADD_CMD( gl_buffer,		do_new_gl_buffer,	create a new GL buffer )
ADD_CMD( objects,		do_gl_obj_menu,	object specification submenu )
ADD_CMD( framebuffers,		do_glfb_menu,	off-screen framebuffer submenu )
ADD_CMD( modes,			do_mode_menu,	display mode submenu )
ADD_CMD( capabilities,		do_cap_menu,	rendering capability submenu )
ADD_CMD( color,			do_color_menu,	drawing color submenu )
/* display and setup_view commands used to be here, but just for teapot? */
ADD_CMD( swap_buffers,		do_swap_buffers,	swap display buffers )
ADD_CMD( sync_to_vblank,	do_vbl_sync,	enable/disable vblank synching )
ADD_CMD( vbl_wait,		do_vbl_wait,	wait a specified number of frames )
ADD_CMD( transform,		do_xf_menu,	viewing transformation submenu )
ADD_CMD( lighting,		do_lighting_menu,	lighting submenu )
ADD_CMD( fullscreen,		do_set_fullscreen,	enable/disable fullscreen mode )
ADD_CMD( display_lists,		do_dl_menu,	display list submenu )
ADD_CMD( stereo,		do_stereo_menu,	nVidia shutter glasses submenu )
ADD_CMD( tiles,			do_tile_menu,	tile object submenu )
MENU_END(gl)

COMMAND_FUNC( do_gl_menu )
{
	static int inited=0;
	if( !inited ){
		inited=1;
		gl_debug = add_debug_module(QSP_ARG  "gl");
		DECLARE_STR1_FUNCTION(	display_list_exists,	display_list_exists )
	}
	PUSH_MENU(gl);
}

#endif /* HAVE_OPENGL */

