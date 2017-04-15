#include "quip_config.h"

#include "quip_prot.h"
#include "query_stack.h"

typedef enum {
	SWF_DEBUG,
	SWF_CAUTIOUS,
	SWF_HISTORY,
	SWF_HELPFUL,
	SWF_MONITOR_COLLISIONS,
	SWF_THREAD_SAFE_QUERY,
	SWF_TTY_CTL,
	SWF_SUBMENU,

	SWF_JPEG,
	SWF_PNG,
	SWF_TIFF,
	SWF_MPEG,
	SWF_RGB,
	SWF_QUICKTIME,
	SWF_MATIO,
	SWF_OPENCV,

	SWF_CURL,
	SWF_SOUND,
	SWF_MPLAYER,
	SWF_XINE,
	SWF_SSE,
	SWF_NUMREC,
	SWF_CUDA,
	SWF_OPENCL,
	SWF_LIBAVCODEC,
	SWF_X11,
	SWF_X11_EXT,
	SWF_OPENGL,
	SWF_MOTIF,
	SWF_GSL,
	SWF_GCRYPT,
	SWF_FLYCAP,
	SWF_DC1394,
	SWF_V4L2,
	N_SW_FEATURES
} sw_feature_t;

typedef enum {
	UNKNOWN,
	PRESENT,
	ABSENT
} presence_t;

typedef struct sw_feature {
	presence_t	swf_state;
	sw_feature_t	swf_code;
	const char *	swf_desc;
} SW_Feature;

static SW_Feature swf_tbl[N_SW_FEATURES]={
{ UNKNOWN, SWF_DEBUG,		"module debugging"		},
{ UNKNOWN, SWF_CAUTIOUS,	"cautious checking"		},
{ UNKNOWN, SWF_HISTORY,		"response history"		},
{ UNKNOWN, SWF_HELPFUL,		"detailed command help"	},
{ UNKNOWN, SWF_MONITOR_COLLISIONS, "monitoring hash table collisions"	},
{ UNKNOWN, SWF_THREAD_SAFE_QUERY, "thread-safe interpreter"	},
{ UNKNOWN, SWF_TTY_CTL,		"low-level terminal control"	},
{ UNKNOWN, SWF_SUBMENU,		"special submenus"		},

{ UNKNOWN, SWF_JPEG,		"JPEG files"			},
{ UNKNOWN, SWF_PNG,		"PNG files"			},
{ UNKNOWN, SWF_TIFF,		"TIFF files"			},
{ UNKNOWN, SWF_MPEG,		"MPEG files"			},
{ UNKNOWN, SWF_RGB,		"SGI rgb files"			},
{ UNKNOWN, SWF_QUICKTIME,	"Quicktime files"		},
{ UNKNOWN, SWF_MATIO,		"MATLAB i/o"			},
{ UNKNOWN, SWF_OPENCV,		"OpenCV machine vision library"	},

{ UNKNOWN, SWF_CURL,		"www i/o (w/ libcurl)"		},
{ UNKNOWN, SWF_SOUND,		"sound (portaudio/ALSA)"	},
{ UNKNOWN, SWF_MPLAYER,		"mplayer"			},
{ UNKNOWN, SWF_XINE,		"Xine"				},
{ UNKNOWN, SWF_SSE,		"SSE processor acceleration"	},
{ UNKNOWN, SWF_NUMREC,		"Numerical Recipes library"	},
{ UNKNOWN, SWF_CUDA,		"nVidia CUDA"			},
{ UNKNOWN, SWF_OPENCL,		"OpenCL acceleration"		},
{ UNKNOWN, SWF_LIBAVCODEC,	"AVI files (w/ libavcodec)"	},
{ UNKNOWN, SWF_X11,		"X11 window system"	},
{ UNKNOWN, SWF_X11_EXT,		"shared memory display w/ libXext"	},
{ UNKNOWN, SWF_OPENGL,		"OpenGL graphics"		},
{ UNKNOWN, SWF_MOTIF,		"Motif GUI widgets with libXm"	},
{ UNKNOWN, SWF_GSL,		"GNU Scientific Library"	},
{ UNKNOWN, SWF_GCRYPT,		"encryption w/ libgcrypt"	},
{ UNKNOWN, SWF_FLYCAP,		"firewire cameras w/ libflycap"	},
{ UNKNOWN, SWF_DC1394,		"firewire cameras w/ libdc1394"	},
{ UNKNOWN, SWF_V4L2,		"video-for-Linux II"	}
};

#ifdef NOW_DONE_WITH_ASSERTION
#ifdef CAUTIOUS
#define CHECKIT(code)						\
	if( swf_tbl[code].swf_code != code )			\
		ERROR1("CAUTIOUS:  Software feature table corruption!?");
#else
#define CHECKIT(code)
#endif
#endif // NOW_DONE_WITH_ASSERTION

#define FEATURE_ABSENT(code)					\
	/*CHECKIT(code)*/					\
	assert( swf_tbl[code].swf_code == code );		\
	swf_tbl[code].swf_state = ABSENT;
	
#define FEATURE_PRESENT(code)					\
	/*CHECKIT(code)*/					\
	assert( swf_tbl[code].swf_code == code );		\
	swf_tbl[code].swf_state = PRESENT;

static void get_feature_states(SINGLE_QSP_ARG_DECL)
{

#ifdef DEBUG
	FEATURE_PRESENT(SWF_DEBUG);
#else
	FEATURE_ABSENT(SWF_DEBUG);
#endif

#ifdef CAUTIOUS
	FEATURE_PRESENT(SWF_CAUTIOUS);
#else
	FEATURE_ABSENT(SWF_CAUTIOUS);
#endif


#ifdef HAVE_HISTORY
	FEATURE_PRESENT(SWF_HISTORY);
#else
	FEATURE_ABSENT(SWF_HISTORY);
#endif


#ifdef HAVE_CUDA
	FEATURE_PRESENT(SWF_CUDA);
#else
	FEATURE_ABSENT(SWF_CUDA);
#endif

#ifdef HAVE_OPENCL
	FEATURE_PRESENT(SWF_OPENCL);
#else
	FEATURE_ABSENT(SWF_OPENCL);
#endif


#ifdef HAVE_LIBAVCODEC
	FEATURE_PRESENT(SWF_LIBAVCODEC);
#else
	FEATURE_ABSENT(SWF_LIBAVCODEC);
#endif


#ifdef HAVE_MATIO
	FEATURE_PRESENT(SWF_MATIO);
#else
	FEATURE_ABSENT(SWF_MATIO);
#endif

#ifdef HAVE_OPENCV
	FEATURE_PRESENT(SWF_OPENCV);
#else
	FEATURE_ABSENT(SWF_OPENCV);
#endif


#ifdef HAVE_MPEG
	FEATURE_PRESENT(SWF_MPEG);
#else
	FEATURE_ABSENT(SWF_MPEG);
#endif


#ifdef HAVE_MPLAYER
	FEATURE_PRESENT(SWF_MPLAYER);
#else
	FEATURE_ABSENT(SWF_MPLAYER);
#endif

// HAVE_NUMREC can be set by autoconf based on whether numerical recipes is present
// on the system.  But USE_NUMREC is set by the user to indicate whether it should be used...

#ifdef HAVE_NUMREC
#ifdef USE_NUMREC
	FEATURE_PRESENT(SWF_NUMREC);
#else // ! USE_NUMREC
	FEATURE_ABSENT(SWF_NUMREC);
#endif // ! USE_NUMREC
#else // ! HAVE_NUMREC
	FEATURE_ABSENT(SWF_NUMREC);
#endif


#ifdef HAVE_PNG
	FEATURE_PRESENT(SWF_PNG);
#else
	FEATURE_ABSENT(SWF_PNG);
#endif


#ifdef HAVE_QUICKTIME
	FEATURE_PRESENT(SWF_QUICKTIME);
#else
	FEATURE_ABSENT(SWF_QUICKTIME);
#endif


#ifdef HAVE_LIBCURL
	FEATURE_PRESENT(SWF_CURL);
#else
	FEATURE_ABSENT(SWF_CURL);
#endif


#ifdef HAVE_SOUND
	FEATURE_PRESENT(SWF_SOUND);
#else
	FEATURE_ABSENT(SWF_SOUND);
#endif


#ifdef HAVE_TIFF
	FEATURE_PRESENT(SWF_TIFF);
#else
	FEATURE_ABSENT(SWF_TIFF);
#endif


#ifdef HAVE_JPEG_SUPPORT
	FEATURE_PRESENT(SWF_JPEG);
#else
	FEATURE_ABSENT(SWF_JPEG);
#endif


#ifdef HAVE_RGB
	FEATURE_PRESENT(SWF_RGB);
#else
	FEATURE_ABSENT(SWF_RGB);
#endif


#ifdef HAVE_XINE
	FEATURE_PRESENT(SWF_XINE);
#else
	FEATURE_ABSENT(SWF_XINE);
#endif


#ifdef HAVE_X11
	FEATURE_PRESENT(SWF_X11);
#else
	FEATURE_ABSENT(SWF_X11);
#endif


#ifdef HAVE_X11_EXT
	FEATURE_PRESENT(SWF_X11_EXT);
#else
	FEATURE_ABSENT(SWF_X11_EXT);
#endif


#ifdef HAVE_OPENGL
	FEATURE_PRESENT(SWF_OPENGL);
#else
	FEATURE_ABSENT(SWF_OPENGL);
#endif


#ifdef HAVE_MOTIF
	FEATURE_PRESENT(SWF_MOTIF);
#else
	FEATURE_ABSENT(SWF_MOTIF);
#endif

#ifdef HAVE_GSL
	FEATURE_PRESENT(SWF_GSL);
#else
	FEATURE_ABSENT(SWF_GSL);
#endif

#ifdef HAVE_LIBGCRYPT
	FEATURE_PRESENT(SWF_GCRYPT);
#else
	FEATURE_ABSENT(SWF_GCRYPT);
#endif

#ifdef HAVE_LIBFLYCAP
	FEATURE_PRESENT(SWF_FLYCAP);
#else
	FEATURE_ABSENT(SWF_FLYCAP);
#endif

#ifdef HAVE_LIBDC1394
	FEATURE_PRESENT(SWF_DC1394);
#else
	FEATURE_ABSENT(SWF_DC1394);
#endif


#ifdef HAVE_V4L2
	FEATURE_PRESENT(SWF_V4L2);
#else
	FEATURE_ABSENT(SWF_V4L2);
#endif


#ifdef HELPFUL
	FEATURE_PRESENT(SWF_HELPFUL);
#else
	FEATURE_ABSENT(SWF_HELPFUL);
#endif


#ifdef MONITOR_COLLISIONS
	FEATURE_PRESENT(SWF_MONITOR_COLLISIONS);
#else
	FEATURE_ABSENT(SWF_MONITOR_COLLISIONS);
#endif


#ifdef USE_SSE
	FEATURE_PRESENT(SWF_SSE);
#else
	FEATURE_ABSENT(SWF_SSE);
#endif


#ifdef SUBMENU
	FEATURE_PRESENT(SWF_SUBMENU);
#else
	FEATURE_ABSENT(SWF_SUBMENU);
#endif


#ifdef THREAD_SAFE_QUERY
	FEATURE_PRESENT(SWF_THREAD_SAFE_QUERY);
#else
	FEATURE_ABSENT(SWF_THREAD_SAFE_QUERY);
#endif


#ifdef TTY_CTL
	FEATURE_PRESENT(SWF_TTY_CTL);
#else
	FEATURE_ABSENT(SWF_TTY_CTL);
#endif

}

COMMAND_FUNC( do_list_features )
{
	const char *pn;
	int i;

	pn=tell_progname();

	get_feature_states(SINGLE_QSP_ARG);

	prt_msg("");
	sprintf(msg_str,"Features present in this build of %s:",pn);
	prt_msg(msg_str);
	prt_msg("");
	for(i=0;i<N_SW_FEATURES;i++){
		if( swf_tbl[i].swf_state == PRESENT ){
			sprintf(msg_str,"\tSupport for %s.",
				swf_tbl[i].swf_desc);
			prt_msg(msg_str);
		}
	}

	prt_msg("");
	sprintf(msg_str,"Features absent in this build of %s:",pn);
	prt_msg(msg_str);
	prt_msg("");
	for(i=0;i<N_SW_FEATURES;i++){
		if( swf_tbl[i].swf_state == ABSENT ){
			sprintf(msg_str,"\tNO support for %s.",
				swf_tbl[i].swf_desc);
			prt_msg(msg_str);
		}
	}

#ifdef CAUTIOUS
	for(i=0;i<N_SW_FEATURES;i++){
//		if( swf_tbl[i].swf_state != ABSENT &&
//			swf_tbl[i].swf_state != PRESENT ){
//
//			sprintf(ERROR_STRING,
//		"CAUTIOUS:  need to test state of feature %d (%s)",i,
//				swf_tbl[i].swf_desc);
//			ERROR1(ERROR_STRING);
//		}
		assert( swf_tbl[i].swf_state == ABSENT || swf_tbl[i].swf_state == PRESENT );
	}
#endif /* CAUTIOUS */
}

