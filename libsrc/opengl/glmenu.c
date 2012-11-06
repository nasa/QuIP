#include "quip_config.h"

char VersionId_opengl_glmenu[] = QUIP_VERSION_STRING;

#ifdef HAVE_OPENGL

#ifdef HAVE_GLUT
#include "glut_supp.h"
#endif	/* HAVE_GLUT */

#include "glx_supp.h"

#include "gl_util.h"
#include "submenus.h"
#include "query.h"	/* assign_var */
#include "data_obj.h"
#include "dl.h"
#include "version.h"
#include "tile.h"
#include "debug.h"
#include "chewtext.h"

#define NOT_IMP(s)	{ sprintf(ERROR_STRING,"Sorry, %s not implemented yet.",s); NWARN(ERROR_STRING); }

#include "string.h"

debug_flag_t gl_debug=0;

void check_gl_error(char *s)
{
	GLenum e;

	e=glGetError();
	if( e == GL_NO_ERROR ) return;
	switch(e){
		case GL_INVALID_OPERATION:
			sprintf(DEFAULT_ERROR_STRING,
				"%s:  invalid operation",s);
			NWARN(DEFAULT_ERROR_STRING);
			break;
		default:
			sprintf(DEFAULT_ERROR_STRING,"check_gl_error:  unhandled error code after %s",s);
			NWARN(DEFAULT_ERROR_STRING);
			break;
	}
}

static void get_rgb_triple(QSP_ARG_DECL float *v)
{
	v[0]=HOW_MUCH("red");
	v[1]=HOW_MUCH("green");
	v[2]=HOW_MUCH("blue");
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

static Command gl_color_ctbl[]={
{ "background",	set_clear_color,"set color for clear"		},
{ "clear_color",do_clear_color,	"clear color buffer"		},
{ "clear_depth",do_clear_depth,	"clear depth buffer"		},
{ "color",	set_gl_pen,	"set current drawing color"	},
{ "shade",	select_shader,	"select shading model"		},
{ "flush",	do_glFlush,	"flush graphics pipeline"	},
{ "quit",	popcmd,		"exit submenu"			},
{ NULL_COMMAND							}
};

static COMMAND_FUNC( color_menu )
{
	PUSHCMD(gl_color_ctbl,"color");
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

	x=HOW_MUCH("x coordinate");
	y=HOW_MUCH("y coordinate");
	z=HOW_MUCH("z coordinate");

	if( debug & gl_debug ){
		sprintf(ERROR_STRING,"glVertex3f %g %g %g",x,y,z);
		advise(ERROR_STRING);
	}
	glVertex3f(x,y,z);
}

static COMMAND_FUNC(	do_gl_color )
{
	float r,g,b;

	r=HOW_MUCH("red");
	g=HOW_MUCH("green");
	b=HOW_MUCH("blue");

	if( debug & gl_debug ){
		sprintf(ERROR_STRING,"glColor3f %g %g %g",r,g,b);
		advise(ERROR_STRING);
	}
	glColor3f(r,g,b);
}

static COMMAND_FUNC(	do_gl_normal )
{
	float x,y,z;

	x=HOW_MUCH("x coordinate");
	y=HOW_MUCH("y coordinate");
	z=HOW_MUCH("z coordinate");

	if( debug & gl_debug ){
		sprintf(ERROR_STRING,"glNormal3f %g %g %g",x,y,z);
		advise(ERROR_STRING);
	}
	glNormal3f(x,y,z);
}

static COMMAND_FUNC(	do_gl_tc )
{
	float s,t;

	s=HOW_MUCH("s coordinate");
	t=HOW_MUCH("t coordinate");

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

static COMMAND_FUNC(	do_gl_material )
{
	int i;
	float pvec[4];

	i=WHICH_ONE("property",N_MATERIAL_PROPERTIES,property_names);
	if( i < 0 ) return;

	switch(i){
		case 0: {
			pvec[0] = HOW_MUCH("ambient red");
			pvec[1] = HOW_MUCH("ambient green");
			pvec[2] = HOW_MUCH("ambient blue");
			pvec[3] = 1.0;
			if( debug & gl_debug ) advise("glMaterialfv GL_FRONT_AND_BACK GL_DIFFUSE (ambient?)");
			/* diffuse or ambient??? */
			glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, pvec);
			break; }
		case 1: {
			pvec[0] = HOW_MUCH("diffuse red");
			pvec[1] = HOW_MUCH("diffuse green");
			pvec[2] = HOW_MUCH("diffuse blue");
			pvec[3] = 1.0;
			if( debug & gl_debug ) advise("glMaterialfv GL_FRONT_AND_BACK GL_DUFFUSE");
			glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, pvec);
			break; }
		case 2: {
			pvec[0] = HOW_MUCH("specular red");
			pvec[1] = HOW_MUCH("specular green");
			pvec[2] = HOW_MUCH("specular blue");
			pvec[3] = 1.0;
			if( debug & gl_debug ) advise("glMaterialfv GL_FRONT_AND_BACK GL_SPECULAR");
			glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, pvec);
			break; }
		case 3: {
			pvec[0] = HOW_MUCH("shininess");
			if( debug & gl_debug ) advise("glMaterialfv GL_FRONT_AND_BACK GL_SHININESS");
			glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, pvec);
			break; }
		case 4: {
			pvec[0] = HOW_MUCH("emission red");
			pvec[1] = HOW_MUCH("emission green");
			pvec[2] = HOW_MUCH("emission blue");
			pvec[3] = 1.0;
			if( debug & gl_debug ) advise("glMaterialfv GL_FRONT_AND_BACK GL_EMISSION");
			glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, pvec);
			break; }
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

	size = HOW_MUCH("size");

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

	x=HOW_MANY("x location");
	y=HOW_MANY("y location");
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
	DIGEST( s );		/* as in cstepit/cs_supp.c */
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
	ASSIGN_VAR("selection_index",ret_str);

	//if (hits != 0) {
	//manager.advanceTrial(processHits());
	//}

	//glPopMatrix();

	/* Set a variable to indicate what happened */
}

static COMMAND_FUNC( do_load_name )
{
	int n;

	n=HOW_MANY("'name' number");
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

	n=HOW_MANY("'name' number");
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

static Command obj_ctbl[]={
{ "begin_obj",	do_gl_begin,	"begin primitive description"	},
{ "end_obj",	do_gl_end,	"end primitive description"	},
{ "vertex",	do_gl_vertex,	"specify a vertex"		},
{ "color",	do_gl_color,	"set current color"		},
{ "normal",	do_gl_normal,	"set normal vector"		},
{ "tex_coord",	do_gl_tc,	"set texture coordinate"	},
{ "edge_flag",	do_gl_ef,	"control drawing of edges"	},
{ "material",	do_gl_material,	"set material properties"	},
{ "array_elt",	do_gl_ae,	"extract vertex array data"	},
{ "eval_coord",	do_gl_ec,	"generate coordinates"		},
{ "eval_point",	do_gl_ep,	"generate point coordinates"	},
{ "front_face",	do_fface,	"specify front face of polygons"},
{ "cull_face",	do_cface,	"specfy cull face of polygons"	},
{ "point_size",	do_gl_ptsize,	"set width in pixels of points"	},
{ "select",	do_slct_obj,	"select an object with the mouse"	},
{ "load_name",	do_load_name,	"load a name"			},
{ "push_name",	do_push_name,	"push a name onto the stack"	},
{ "pop_name",	do_pop_name,	"pop a name from the stack"	},
{ "quit",	popcmd,		"exit submenu"			},
{ NULL_COMMAND							}
};

static COMMAND_FUNC( obj_menu )
{
	PUSHCMD(obj_ctbl,"object");
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
		ASSIGN_VAR(CAP_RESULT_VARNAME,"-1");
		return;
	}

	if( glIsEnabled(cap) == GL_TRUE ){
advise("cap_enabled = 1");
		ASSIGN_VAR(CAP_RESULT_VARNAME,"1");
	} else {
advise("cap_enabled = 0");
		ASSIGN_VAR(CAP_RESULT_VARNAME,"0");
	}
}


static COMMAND_FUNC( do_tex_image )
{
	Data_Obj *dp = PICK_OBJ("image");
	//im_dim = HOW_MUCH("pixel dimension");

	if( dp == NO_OBJ ) return;

	set_texture_image(dp);
}

void set_texture_image(Data_Obj *dp)
{
	int code,prec;
	/*glDepthFunc(GL_LEQUAL);
		glPixelStorei(GL_UNPACK_ALIGNMENT,1);*/

	if(dp->dt_comps==1) code=GL_LUMINANCE;
	else if( dp->dt_comps == 3 ) code=GL_RGB;
	else {
		sprintf(DEFAULT_ERROR_STRING,
			"set_texture_image:  Object %s has type dimension %d, expected 1 or 3",
			dp->dt_name,dp->dt_comps);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	if( dp->dt_prec == PREC_SP ) prec=GL_FLOAT;
	else if( dp->dt_prec == PREC_UBY ) prec=GL_UNSIGNED_BYTE;
	else {
		sprintf(DEFAULT_ERROR_STRING,"set_texture_image:  Object %s has precision %s, expected %s or %s",
			dp->dt_name,name_for_prec(dp->dt_prec),
			name_for_prec(PREC_SP),name_for_prec(PREC_UBY));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	if( debug & gl_debug ) advise("glTexImage2D");
	glTexImage2D(GL_TEXTURE_2D, 0, dp->dt_comps, dp->dt_cols,
		dp->dt_rows, 0, code, prec, dp->dt_data);

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

static Command cap_ctbl[]={
{ "enable",	do_enable,	"enable capability"				},
{ "disable",	do_disable,	"disable capability"				},
{ "query",	do_cap_q,	"query capability (result in $cap_enabled)"	},
{ "tex_image",	do_tex_image,	"specify a texture image"											},
{ "quit",	popcmd,		"exit submenu"					},
{ NULL_COMMAND									}
};

static COMMAND_FUNC( cap_menu )
{
	PUSHCMD(cap_ctbl,"capabilities");
}

static COMMAND_FUNC( set_pt_size )
{
	GLfloat s;

	s=HOW_MUCH("width in pixels for rendered points");
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

	w=HOW_MUCH("width in pixels for rendered lines");
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

static Command mode_ctbl[]={
	{ "point_size",	set_pt_size,	"set point size"	},
	{ "line_width",	set_line_width,	"set line width"	},
	{ "polygon_mode",	set_poly_mode,	"set polygon mode"	},
	{ "quit",	popcmd,		"exit submenu"		},
	{ NULL_COMMAND						}
};

static COMMAND_FUNC( mode_menu )
{
	PUSHCMD(mode_ctbl,"mode");
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

	x = HOW_MUCH("x camera position");
	y = HOW_MUCH("y camera position");
	z = HOW_MUCH("z camera position");
	cx = HOW_MUCH("x target position");
	cy = HOW_MUCH("y target position");
	cz = HOW_MUCH("z target position");
	ux = HOW_MUCH("x up direction");
	uy = HOW_MUCH("y up direction");
	uz = HOW_MUCH("z up direction");

	if( debug & gl_debug ) advise("gluLookAt");
	gluLookAt(x,y,z,cx,cy,cz,ux,uy,uz);
}

static COMMAND_FUNC( do_scale )
{
	float fx,fy,fz;

	fx=HOW_MUCH("x scale factor");
	fy=HOW_MUCH("y scale factor");
	fz=HOW_MUCH("z scale factor");

	if( debug & gl_debug ) advise("glScalef");
	glScalef(fx,fy,fz);
}

static COMMAND_FUNC( do_xlate )
{
	float tx,ty,tz;

	tx=HOW_MUCH("x translation");
	ty=HOW_MUCH("y translation");
	tz=HOW_MUCH("z translation");

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
	if( dp == NO_OBJ ) return;

	/* BUG check size & type here */

	if((strcmp(matrix,"modelview"))&&(strcmp(matrix,"projection"))){
		advise("Valid types of matrices are:	'modelview'	'projection'");
		return;
	}

	if (!(strcmp(matrix, "modelview")))
		glGetFloatv(GL_MODELVIEW_MATRIX,(GLfloat *)dp->dt_data);
	if (!(strcmp(matrix, "projection")))
		glGetFloatv(GL_PROJECTION_MATRIX,(GLfloat *)dp->dt_data);
}

static COMMAND_FUNC( do_ld_mat )
{
	Data_Obj *dp;

	dp=PICK_OBJ("matrix object");
	if( dp == NO_OBJ ) return;

	/* BUG check size & type here */

	if( debug & gl_debug ) advise("glLoadMatrixf");
	glLoadMatrixf((GLfloat *)dp->dt_data);
}

static COMMAND_FUNC( do_mul_mat )
{
	Data_Obj *dp;

	dp=PICK_OBJ("matrix object");
	if( dp == NO_OBJ ) return;

	/* BUG check size & type here */

	if( debug & gl_debug ) advise("glMultMatrixf");
	glMultMatrixf((GLfloat *)dp->dt_data);
}

static COMMAND_FUNC( do_rotate )
{
	float angle;
	float dx,dy,dz;

	angle = HOW_MUCH("angle in degrees");
	dx = HOW_MUCH("rotation axis x");
	dy = HOW_MUCH("rotation axis y");
	dz = HOW_MUCH("rotation axis z");

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

static Command xf_ctbl[]={
{ "mode",	set_xf_mode,	"set mode for viewing transformation"		},
{ "identity",	do_identity,	"initialize viewing matrix"			},
{ "frustum",	set_frustum,	"specify viewing frustum"			},
{ "ortho",	do_ortho,	"specify orthographic viewing volume"		},
{ "look_at",	do_look_at,	"specify viewing position and direction"	},
{ "scale",	do_scale,	"specify scaling factor"			},
{ "translate",	do_xlate,	"specify translation"				},
{ "rotate",	do_rotate,	"specify rotation"				},
{ "perspective",do_persp,	"specify perspective transformation"		},
{ "save_matrix",do_sv_mv_mat,	"save current modelview or projection matrix"	},
{ "load_matrix",do_ld_mat,	"load current matrix from object"		},
{ "mult_matrix",do_mul_mat,	"multiply current matrix by object"		},
{ "push_matrix",do_push_mat,	"push down matrix stack"			},
{ "pop_matrix",	do_pop_mat,	"pop top of matrix stack"			},
{ "quit",	popcmd,		"exit submenu"					},
{ NULL_COMMAND									}
};

static COMMAND_FUNC( xf_menu )
{
	PUSHCMD(xf_ctbl,"xform");
}

static COMMAND_FUNC( set_shading_model )
{
	GLenum m;

	m = CHOOSE_SHADING_MODEL("shading model");
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

	v[0] = HOW_MUCH("red component");
	v[1] = HOW_MUCH("green component");
	v[2] = HOW_MUCH("blue component");
	v[3] = 1.0;

	CHECK_LIGHT("ambient");

	if( debug & gl_debug ) advise("glLightfv GL_AMBIENT");
	glLightfv(which_light,GL_AMBIENT,v);
}

static COMMAND_FUNC( set_diffuse )
{
	float v[4];

	v[0]=HOW_MUCH("red component");
	v[1]=HOW_MUCH("green component");
	v[2]=HOW_MUCH("blue component");
	v[3]=1.0;

	CHECK_LIGHT("diffuse");

	if( debug & gl_debug ) advise("glLightfv GL_DIFFUSE");
	glLightfv(which_light,GL_DIFFUSE,v);
}

static COMMAND_FUNC( set_specular )
{
	float v[4];

	v[0] = HOW_MUCH("red component");
	v[1] = HOW_MUCH("green component");
	v[2] = HOW_MUCH("blue component");
	v[3] = 1.0;

	CHECK_LIGHT("specular");

	if( debug & gl_debug ) advise("glLightfv GL_SPECULAR");
	glLightfv(which_light, GL_SPECULAR, v);
}

static COMMAND_FUNC( set_position )
{
	float v[4];

	v[0]=HOW_MUCH("x position");
	v[1]=HOW_MUCH("y position");
	v[2]=HOW_MUCH("z position");
	v[3]=HOW_MUCH("w position");

	CHECK_LIGHT("position")

	if( debug & gl_debug ) advise("glLightfv GL_POSITION");
	glLightfv(which_light,GL_POSITION,v);
}

static COMMAND_FUNC( set_spot_dir )
{
	float v[3];

	v[0]=HOW_MUCH("x component");
	v[1]=HOW_MUCH("y component");
	v[2]=HOW_MUCH("z component");

	CHECK_LIGHT("spot direction")

	if( debug & gl_debug ) advise("glLightfv GL_POSITION");
	glLightfv(which_light,GL_SPOT_DIRECTION,v);
}

static COMMAND_FUNC( set_global_ambient )
{
	float v[4];

	v[0]=HOW_MUCH("red");
	v[1]=HOW_MUCH("green");
	v[2]=HOW_MUCH("blue");
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
	f=HOW_MUCH("attenuation constant");
	if( i < 0 ) return;
	if( debug & gl_debug ) advise("glLightf");
	switch(i){
		case 0: glLightf(which_light,GL_CONSTANT_ATTENUATION,f); break;
		case 1: glLightf(which_light,GL_LINEAR_ATTENUATION,f); break;
		case 2: glLightf(which_light,GL_QUADRATIC_ATTENUATION,f); break;
	}
}

static Command lighting_ctbl[]={
{ "shading_model",	set_shading_model,	"select shading model"		},
{ "select_light",	do_sel_light,		"select light for subsequent operations"	},
{ "ambient",		set_ambient,		"set ambient parameters"	},
{ "diffuse",		set_diffuse,		"set diffuse parameters"	},
{ "specular",		set_specular,		"set specular parameters"	},
{ "position",		set_position,		"set light position"		},
{ "attenuation",	set_atten,		"set light attenuation"		},
{ "spot_direction",	set_spot_dir,		"set spotlight direction"	},
{ "global_ambient",	set_global_ambient,	"set color of global ambient"	},
{ "local_viewer",	set_local_viewer,	"enable/disable use of viewing position in specular reflection calc's"	},
{ "two_side",		set_two_side,		"enable/disable two-sided lighting"	},
	/*
{ "separate_specular",	set_color_control,	"specular color calculated separately"	},
*/
{ "quit",		popcmd,			"exit submenu"			},
{ NULL_COMMAND									}
};

static COMMAND_FUNC( lighting_menu )
{
	PUSHCMD(lighting_ctbl,"lighting");
}

static COMMAND_FUNC(do_list_dls){list_dls(SINGLE_QSP_ARG);}

static Command dl_ctbl[]={
{ "new",	do_new_dl,	"create new display list"		},
{ "end",	do_end_dl,	"end current display list"		},
{ "list",	do_list_dls,	"list all display lists"		},
{ "delete",	do_del_dl,	"delete a display list"			},
{ "info",	do_info_dl,	"give info about a display list"	},
//{ "dump",	do_dump_dl,	"dump a display list to the screen"	},
{ "call",	do_call_dl,	"call display list"			},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL_COMMAND								}
};


static COMMAND_FUNC( dl_menu )
{
	PUSHCMD(dl_ctbl,"display_lists");
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

	n = HOW_MANY("number of frames to wait");
	wait_video_sync(n);
}

static Command gl_ctbl[]={
{ "objects",		obj_menu,	"object specification submenu"	},
{ "modes",		mode_menu,	"display mode submenu"		},
{ "capabilities",	cap_menu,	"rendering capability submenu"	},
{ "color",		color_menu,	"drawing color submenu"		},
/* display and setup_view commands used to be here, but just for teapot? */
{ "swap_buffers",	do_swap_buffers,	"swap display buffers"		},
{ "sync_to_vblank",	do_vbl_sync,	"enable/disable vblank synch'ing"},
{ "vbl_wait",		do_vbl_wait,	"wait a specified number of frames"},
{ "transform",		xf_menu,	"viewing transformation submenu"},
{ "lighting",		lighting_menu,	"lighting submenu"		},
{ "window",		do_render_to,	"specify drawing window"	},
{ "display_lists",	dl_menu,	"display list submenu"		},
{ "tiles",		tile_menu,	"tile object submenu"	},
{ "quit",		popcmd,		"exit submenu"			},
{ NULL_COMMAND								}
};

COMMAND_FUNC( gl_menu )
{
	static int inited=0;
	if( !inited ){
		auto_version(QSP_ARG  "GLMENU","VersionId_opengl");
		inited=1;
		gl_debug = add_debug_module(QSP_ARG  "gl");
	}
	PUSHCMD(gl_ctbl,"gl");
}

#endif /* HAVE_OPENGL */

