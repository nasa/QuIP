#include "quip_config.h"

char VersionId_opengl_gl_util[] = QUIP_VERSION_STRING;

#ifdef HAVE_OPENGL

#include "glut_supp.h"
#include "gl_util.h"
#include "query.h"	/* assign_var */

#define CHOOSE_NAMED_CONSTANT(pmpt,choices_p,tbl,n)		choose_named_constant(QSP_ARG pmpt,choices_p,tbl,n)

static GLenum choose_named_constant( QSP_ARG_DECL  const char *prompt, const char ***choices_p, Named_Constant *tbl, int n )
{
	int i;
	const char **choices;

	if( *choices_p == NULL ){
		choices = (const char **)getbuf( sizeof(char *) * n );
		for(i=0;i<n;i++)
			choices[i] = tbl[i].nc_name;
		*choices_p = choices;
	} else {
		choices = *choices_p;
	}

	i = WHICH_ONE(prompt,n,choices);

	if( i < 0 ) return(INVALID_CONSTANT);
	return(tbl[i].nc_code);
}

typedef Named_Constant  Primitive;

static Primitive prim_tbl[]={
{ "points",		GL_POINTS		},
{ "lines",		GL_LINES		},
{ "line_strip",		GL_LINE_STRIP		},
{ "line_loop",		GL_LINE_LOOP		},
{ "triangles",		GL_TRIANGLES		},
{ "triangle_strip",	GL_TRIANGLE_STRIP	},
{ "triangle_fan",	GL_TRIANGLE_FAN		},
{ "quads",		GL_QUADS		},
{ "quad_strip",		GL_QUAD_STRIP		},
{ "polygon",		GL_POLYGON		}
};

#define N_PRIMITIVES	(sizeof(prim_tbl)/sizeof(Primitive))

static const char **prim_choices=NULL;

GLenum choose_primitive(QSP_ARG_DECL const char *prompt)
{
	return CHOOSE_NAMED_CONSTANT( prompt, &prim_choices, prim_tbl, N_PRIMITIVES );
}

const char *primitive_name(GLenum p)
{
	unsigned int i;

	for(i=0;i<N_PRIMITIVES;i++)
		if( p == prim_tbl[i].nc_code ) return(prim_tbl[i].nc_name);
	return(NULL);
}


static Named_Constant polymode_tbl[]={
{ "point",	GL_POINT	},
{ "line",	GL_LINE	},
{ "fill",	GL_FILL	},
};

#define N_POLYGON_MODES	(sizeof(polymode_tbl)/sizeof(Named_Constant))

static const char **polymode_choices=NULL;

GLenum choose_polygon_mode(QSP_ARG_DECL const char *prompt)
{
	return CHOOSE_NAMED_CONSTANT( prompt, &polymode_choices, polymode_tbl, N_POLYGON_MODES );
}


typedef Named_Constant Capability;

static Capability cap_tbl[]={
{ "blend",		GL_BLEND		},
{ "depth_test",		GL_DEPTH_TEST		},
{ "fog",		GL_FOG			},
{ "line_stipple",	GL_LINE_STIPPLE		},
{ "lighting",		GL_LIGHTING		},
{ "culling",		GL_CULL_FACE		},
{ "light0",		GL_LIGHT0		},
{ "light1",		GL_LIGHT1		},
{ "light2",		GL_LIGHT2		},
{ "normalize",		GL_NORMALIZE		},
{ "rescale_normal",	GL_RESCALE_NORMAL	},
{ "material_properties",GL_COLOR_MATERIAL	},
{ "texture_2D" ,	GL_TEXTURE_2D   	},
};

#define N_NAMED_CAPABILITIES	(sizeof(cap_tbl)/sizeof(Capability))

const char *gl_cap_string(GLenum cap)
{
	unsigned int i;
	static char cap_name[100];

	for(i=0;i<N_NAMED_CAPABILITIES;i++){
		if( cap_tbl[i].nc_code == cap ){
			return(cap_tbl[i].nc_name);
		}
	}
	sprintf(cap_name,"unnamed capability (%d)",cap);
	return(cap_name);
}

#define N_CAPABILITIES	(sizeof(cap_tbl)/sizeof(Capability))

static const char **cap_choices=NULL;

GLenum choose_cap(QSP_ARG_DECL const char *prompt)
{
	return CHOOSE_NAMED_CONSTANT( prompt, &cap_choices, cap_tbl, N_CAPABILITIES );
}

typedef Named_Constant Viewing_Mode;

static Viewing_Mode vwmode_tbl[]={
{ "projection",		GL_PROJECTION	},
{ "modelview",		GL_MODELVIEW	},
{ "texture",		GL_TEXTURE	},
};

#define N_VIEWING_MODES	(sizeof(vwmode_tbl)/sizeof(Viewing_Mode))

static const char **vwmode_choices=NULL;

GLenum choose_viewing_mode(QSP_ARG_DECL const char *prompt)
{ return CHOOSE_NAMED_CONSTANT( prompt, &vwmode_choices, vwmode_tbl, N_VIEWING_MODES ); }

/******************************************************************/

typedef Named_Constant Winding_Direction;

static Winding_Direction winddir_tbl[]={
{ "clockwise",		GL_CW	},
{ "counterclockwise",	GL_CCW	}
};

#define N_WINDING_DIRECTIONS	(sizeof(winddir_tbl)/sizeof(Winding_Direction))

static const char **winddir_choices=NULL;

GLenum choose_winding_dir(QSP_ARG_DECL const char *prompt)
{ return CHOOSE_NAMED_CONSTANT( prompt, &winddir_choices, winddir_tbl, N_WINDING_DIRECTIONS ); }

/******************************************************************/

typedef Named_Constant Facing_Direction;

static Facing_Direction facedir_tbl[]={
{ "front",		GL_FRONT		},
{ "back",		GL_BACK			},
{ "front_and_back",	GL_FRONT_AND_BACK	}
};

#define N_FACING_DIRECTIONS	(sizeof(facedir_tbl)/sizeof(Facing_Direction))

static const char **facedir_choices=NULL;

GLenum choose_facing_dir(QSP_ARG_DECL const char *prompt)
{ return CHOOSE_NAMED_CONSTANT( prompt, &facedir_choices, facedir_tbl, N_FACING_DIRECTIONS ); }

/******************************************************************/

typedef Named_Constant Shading_Model;

static Shading_Model shading_tbl[]={
{ "smooth",		GL_SMOOTH		},
};

#define N_SHADING_MODELS	(sizeof(shading_tbl)/sizeof(Shading_Model))

static const char **shading_choices=NULL;

GLenum choose_shading_model(QSP_ARG_DECL const char *prompt)
{ return CHOOSE_NAMED_CONSTANT( prompt, &shading_choices, shading_tbl, N_SHADING_MODELS ); }

/******************************************************************/

typedef Named_Constant Light_Source;

static Light_Source light_tbl[]={
{ "light0",		GL_LIGHT0		},
{ "light1",		GL_LIGHT1		},
{ "light2",		GL_LIGHT2		},
{ "light3",		GL_LIGHT3		},
{ "light4",		GL_LIGHT4		},
{ "light5",		GL_LIGHT5		},
{ "light6",		GL_LIGHT6		},
{ "light7",		GL_LIGHT7		},
};

#define N_LIGHT_SOURCES	(sizeof(light_tbl)/sizeof(Light_Source))

static const char **light_choices=NULL;

GLenum choose_light_source(QSP_ARG_DECL const char *prompt)
{ return CHOOSE_NAMED_CONSTANT( prompt, &light_choices, light_tbl, N_LIGHT_SOURCES ); }

#endif /* HAVE_OPENGL */

