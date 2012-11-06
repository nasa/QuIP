/* Jeff's interface to the 1394 subsystem, to talk to the PGR camera.  */

#include "quip_config.h"

char VersionId_ptgrey_pgr[] = QUIP_VERSION_STRING;

#include "data_obj.h"
#include "query.h"
#include "debug.h"		/* verbose */

#include <stdio.h>

#ifdef HAVE_DC1394_CONTROL_H
#include <dc1394/control.h>
#endif

#ifdef HAVE_DC1394_UTILS_H
#include <dc1394/utils.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>		/* usleep */
#endif

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

#include "pgr.h"

#ifdef HAVE_LIBDC1394
static PGR_Cam pgc1;
static List *in_use_lp=NO_LIST;
static int ring_buffer_size=64;
static int capturing=0;
static int transmitting=0;
#endif


ITEM_INTERFACE_DECLARATIONS(PGR_Frame,pfrm)



//dc1394error_t dc1394_get_camera_feature_set(dc1394camera_t *camera, dc1394featureset_t *features);

/* Stores the bounds and options associated with the feature described by feature->feature_id */
//dc1394error_t dc1394_get_camera_feature(dc1394camera_t *camera, dc1394feature_info_t *feature);

/* Displays the bounds and options of the given feature or of the entire feature set */
//dc1394error_t dc1394_print_feature(dc1394feature_info_t *feature);
//dc1394error_t dc1394_print_feature_set(dc1394featureset_t *features);

void show_camera_features(PGR_Cam *pgcp)
{
#ifdef HAVE_LIBDC1394
	dc1394featureset_t features;

	if(
		/* dc1394_get_camera_feature_set */
		dc1394_feature_get_all
		(pgcp->pc_cam_p, &features) != DC1394_SUCCESS ){
		NWARN("error getting camera feature set");
		return;
	}
	if( /* dc1394_print_feature_set(&features) */
		dc1394_feature_print_all(&features,stdout) != DC1394_SUCCESS ){
		NWARN("error printing feature set");
	}
#endif
}

#ifdef HAVE_LIBDC1394

void
cleanup1394( dc1394camera_t *cam_p )
{
	if( capturing ) dc1394_capture_stop( cam_p );
	if( transmitting ) dc1394_video_set_transmission( cam_p, DC1394_OFF );
	/* dc1394_free_camera */
	dc1394_camera_free( cam_p );
}

static Named_Video_Mode all_video_modes[]={
{ "yuv444_160x120",	DC1394_VIDEO_MODE_160x120_YUV444	},

{ "yuv411_640x480",	DC1394_VIDEO_MODE_640x480_YUV411	},
{ "yuv422_640x480",	DC1394_VIDEO_MODE_640x480_YUV422	},
{ "rgb8_640x480",	DC1394_VIDEO_MODE_640x480_RGB8		},
{ "mono8_640x480",	DC1394_VIDEO_MODE_640x480_MONO8		},
{ "mono16_640x480",	DC1394_VIDEO_MODE_640x480_MONO16	},

{ "yuv422_800x600",	DC1394_VIDEO_MODE_800x600_YUV422	},
{ "rgb8_800x600",	DC1394_VIDEO_MODE_800x600_RGB8		},
{ "mono8_800x600",	DC1394_VIDEO_MODE_800x600_MONO8		},
{ "mono16_800x600",	DC1394_VIDEO_MODE_800x600_MONO16	},

{ "yuv422_1024x768",	DC1394_VIDEO_MODE_1024x768_YUV422	},
{ "rgb8_1024x768",	DC1394_VIDEO_MODE_1024x768_RGB8		},
{ "mono8_1024x768",	DC1394_VIDEO_MODE_1024x768_MONO8	},
{ "mono16_1024x768",	DC1394_VIDEO_MODE_1024x768_MONO16	},

{ "yuv422_1280x960",	DC1394_VIDEO_MODE_1280x960_YUV422	},
{ "rgb8_1280x960",	DC1394_VIDEO_MODE_1280x960_RGB8		},
{ "mono8_1280x960",	DC1394_VIDEO_MODE_1280x960_MONO8	},

{ "yuv422_1600x1200",	DC1394_VIDEO_MODE_1600x1200_YUV422	},
{ "rgb8_1600x1200",	DC1394_VIDEO_MODE_1600x1200_RGB8	},
{ "mono8_1600x1200",	DC1394_VIDEO_MODE_1600x1200_MONO8	},
{ "mono16_1600x1200",	DC1394_VIDEO_MODE_1600x1200_MONO16	},

{ "exif",		DC1394_VIDEO_MODE_EXIF			},
{ "format7_0",		DC1394_VIDEO_MODE_FORMAT7_0		},
{ "format7_1",		DC1394_VIDEO_MODE_FORMAT7_1		},
{ "format7_2",		DC1394_VIDEO_MODE_FORMAT7_2		},
{ "format7_3",		DC1394_VIDEO_MODE_FORMAT7_3		},
{ "format7_4",		DC1394_VIDEO_MODE_FORMAT7_4		},
{ "format7_5",		DC1394_VIDEO_MODE_FORMAT7_5		},
{ "format7_6",		DC1394_VIDEO_MODE_FORMAT7_6		},
{ "format7_7",		DC1394_VIDEO_MODE_FORMAT7_7		}
};

#define N_NAMED_VIDEO_MODES	(sizeof(all_video_modes)/sizeof(Named_Video_Mode))


/*
static Named_Color_Coding all_color_codes[]={
{ "mono8",	DC1394_COLOR_CODING_MONO8	},
{ "yuv411",	DC1394_COLOR_CODING_YUV411	},
{ "yuv422",	DC1394_COLOR_CODING_YUV422	},
{ "yuv444",	DC1394_COLOR_CODING_YUV444	},
{ "rgb8",	DC1394_COLOR_CODING_RGB8	},
{ "mono16",	DC1394_COLOR_CODING_MONO16	},
{ "rgb16",	DC1394_COLOR_CODING_RGB16	},
{ "mono16s",	DC1394_COLOR_CODING_MONO16S	},
{ "rgb16s",	DC1394_COLOR_CODING_RGB16S	},
{ "raw8",	DC1394_COLOR_CODING_RAW8	},
{ "raw16",	DC1394_COLOR_CODING_RAW16	}
};

#define N_NAMED_COLOR_CODES	(sizeof(all_color_codes)/sizeof(Named_Color_Coding))
*/

static Named_Frame_Rate all_framerates[]={
{	"1.875",	DC1394_FRAMERATE_1_875	},
{	"3.75",		DC1394_FRAMERATE_3_75	},
{	"7.5",		DC1394_FRAMERATE_7_5	},
{	"15",		DC1394_FRAMERATE_15	},
{	"30",		DC1394_FRAMERATE_30	},
{	"60",		DC1394_FRAMERATE_60	},
{	"120",		DC1394_FRAMERATE_120	},
{	"240",		DC1394_FRAMERATE_240	}
};

#define N_NAMED_FRAMERATES	(sizeof(all_framerates)/sizeof(Named_Frame_Rate))

static Named_Feature all_features[]={
{ "brightness",		DC1394_FEATURE_BRIGHTNESS	},
{ "exposure",		DC1394_FEATURE_EXPOSURE		},
{ "sharpness",		DC1394_FEATURE_SHARPNESS	},
{ "white_balance",	DC1394_FEATURE_WHITE_BALANCE	},
{ "hue",		DC1394_FEATURE_HUE		},
{ "saturation",		DC1394_FEATURE_SATURATION	},
{ "gamma",		DC1394_FEATURE_GAMMA		},
{ "shutter",		DC1394_FEATURE_SHUTTER		},
{ "gain",		DC1394_FEATURE_GAIN		},
{ "iris",		DC1394_FEATURE_IRIS		},
{ "focus",		DC1394_FEATURE_FOCUS		},
{ "temperature",	DC1394_FEATURE_TEMPERATURE	},
{ "trigger",		DC1394_FEATURE_TRIGGER		},
{ "trigger_delay",	DC1394_FEATURE_TRIGGER_DELAY	},
{ "white_shading",	DC1394_FEATURE_WHITE_SHADING	},
{ "frame_rate",		DC1394_FEATURE_FRAME_RATE	},
{ "zoom",		DC1394_FEATURE_ZOOM		},
{ "pan",		DC1394_FEATURE_PAN		},
{ "tilt",		DC1394_FEATURE_TILT		},
{ "optical_filter",	DC1394_FEATURE_OPTICAL_FILTER	},
{ "capture_size",	DC1394_FEATURE_CAPTURE_SIZE	},
{ "capture_quality",	DC1394_FEATURE_CAPTURE_QUALITY	}
};

#define N_NAMED_FEATURES	(sizeof(all_features)/sizeof(Named_Feature))

static Named_Trigger_Mode all_trigger_modes[]={
{ "mode0",		DC1394_TRIGGER_MODE_0		},
{ "mode1",		DC1394_TRIGGER_MODE_1		},
{ "mode2",		DC1394_TRIGGER_MODE_2		},
{ "mode3",		DC1394_TRIGGER_MODE_3		},
{ "mode4",		DC1394_TRIGGER_MODE_4		},
{ "mode5",		DC1394_TRIGGER_MODE_5		},
{ "mode14",		DC1394_TRIGGER_MODE_14		},
{ "mode15",		DC1394_TRIGGER_MODE_15		}
};

#define N_NAMED_TRIGGER_MODES	(sizeof(all_trigger_modes)/sizeof(Named_Trigger_Mode))

#endif /* HAVE_LIBDC1394 */

/* called once at camera initialization... */

void get_camera_features( PGR_Cam *pgcp )
{
#ifdef HAVE_LIBDC1394
	Node *np;
	int i;

	if ( /*dc1394_get_camera_feature_set*/
		dc1394_feature_get_all( pgcp->pc_cam_p, &pgcp->pc_features ) != DC1394_SUCCESS ) {
		NERROR1("get_camera_features:  unable to get camera feature set");
	}

	/* Now can the table and build the linked list */
#ifdef FOOBAR
#ifdef CAUTIOUS
	if( pgcp->pc_feat_lp != NO_LIST ) NERROR1("CAUTIOUS:  get_camera_features:  bad list ptr!?");
#endif /* CAUTIOUS */
#endif /* FOOBAR */
	/* We may call this again after we have diddled the controls... */
	/* releasing and rebuilding the list is wasteful, but should work... */
	if( pgcp->pc_feat_lp != NO_LIST ){
		while( (np=remHead(pgcp->pc_feat_lp)) != NO_NODE )
			rls_node(np);
	} else {
		pgcp->pc_feat_lp = new_list();
	}


	for(i=0;i<DC1394_FEATURE_NUM;i++){
		dc1394feature_info_t * f;

		f = &pgcp->pc_features.feature[i];

#ifdef CAUTIOUS
		if( f->id < DC1394_FEATURE_MIN || f->id > DC1394_FEATURE_MAX )
			NERROR1("CAUTIOUS:  bad feature id code");
#endif /* CAUTIOUS */

		if(f->available){
			np = mk_node(f);
			addTail(pgcp->pc_feat_lp,np);
		}
	}
#endif
}

/* report_camera_features:  this used the library's print routine (for comparison) */

#ifdef HAVE_LIBDC1394
int report_camera_features( PGR_Cam *pgcp )
{
	// report camera's features
	dc1394featureset_t	features;

	if ( /*dc1394_get_camera_feature_set*/
		dc1394_feature_get_all( pgcp->pc_cam_p, &features ) != DC1394_SUCCESS ) {
		NWARN("report_camera_features:  unable to get camera feature set");
		return -1;
	}
	/*dc1394_print_feature_set( &features ); */
	dc1394_feature_print_all( &features,stdout );
	return 0;
}

void list_camera_feature( dc1394feature_info_t *feat_p )
{
	sprintf(msg_str,"%s", /*dc1394_feature_desc[feat_p->id - DC1394_FEATURE_MIN]*/
			dc1394_feature_get_string(feat_p->id) );
	prt_msg(msg_str);
}

int list_camera_features( PGR_Cam *pgcp )
{
	Node *np;

	if( pgcp->pc_feat_lp == NO_LIST ) NERROR1("CAUTIOUS:  list_camera_features:  bad list");

	np = pgcp->pc_feat_lp->l_head;
	while(np!=NO_NODE){
		dc1394feature_info_t * f;
		f= (dc1394feature_info_t *) np->n_data;
		list_camera_feature(f);
		np=np->n_next;
	}
	return(0);
}

int get_feature_choices( PGR_Cam *pgcp, const char ***chp )
{
	int n;
	const char * /* const */ * sptr;
	Node *np;

	n=eltcount(pgcp->pc_feat_lp);
	if( n <= 0 ) return(0);

	sptr = (const char **) getbuf( n * sizeof(char *) );
	*chp = sptr;

	np=pgcp->pc_feat_lp->l_head;
	while(np!=NO_NODE){
		dc1394feature_info_t *f;
		f= (dc1394feature_info_t *) np->n_data;
		*sptr = /*(char *)dc1394_feature_desc[f->id - DC1394_FEATURE_MIN]*/
			dc1394_feature_get_string(f->id) ;
		sptr++;
		np=np->n_next;
	}
	return n;
}

void report_feature_info( PGR_Cam *pgcp, dc1394feature_t id )
{
	Node *np;
	dc1394feature_info_t *f;
	unsigned int i;
	const char *name;
	char nbuf[32];

	np = pgcp->pc_feat_lp->l_head;
	f=NULL;
	while( np != NO_NODE ){
		f= (dc1394feature_info_t *) np->n_data;

		if( f->id == id )
			np=NO_NODE;
		else
			f=NULL;

		if( np != NO_NODE )
			np = np->n_next;
	}

#ifdef CAUTIOUS
	if( f == NULL ){
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  report_feature_info:  couldn't find %s",
			/*dc1394_feature_desc[id - DC1394_FEATURE_MIN]*/
			dc1394_feature_get_string(id) );
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
#endif /* CAUTIOUS */

	name=/*dc1394_feature_desc[f->id - DC1394_FEATURE_MIN]*/
		dc1394_feature_get_string(f->id);
	sprintf(nbuf,"%s:",name);
	sprintf(msg_str,"%-16s",nbuf);
	prt_msg_frag(msg_str);

	if (f->on_off_capable) {
		if (f->is_on) 
			prt_msg_frag("ON\t");
		else
			prt_msg_frag("OFF\t");
	} else {
		prt_msg_frag("\t");
	}

	/*
	if (f->one_push){
		if (f->one_push_active)
			prt_msg_frag("  one push: ACTIVE");
		else
			prt_msg_frag("  one push: INACTIVE");
	}
	prt_msg("");
	*/
	/* BUG need to use (new?) feature_get_modes... */
	 /* FIXME */
	/*
	if( f->auto_capable ){
		if (f->auto_active) 
			prt_msg_frag("AUTO\t");
		else
			prt_msg_frag("MANUAL\t");
	} else {
		prt_msg_frag("\t");
	}
	*/

	/*
	prt_msg("");
	*/

	/*
	if( f->id != DC1394_FEATURE_TRIGGER ){
		sprintf(msg_str,"\tmin: %d max %d", f->min, f->max);
		prt_msg(msg_str);
	}
	if( f->absolute_capable){
		sprintf(msg_str,"\tabsolute settings:  value: %f  min: %f  max: %f",
			f->abs_value,f->abs_min,f->abs_max);
		prt_msg(msg_str);
	}
	*/

	switch(f->id){
		case DC1394_FEATURE_TRIGGER:
			switch(f->trigger_modes.num){
				case 0:
					prt_msg("no trigger modes available");
					break;
				case 1:
					sprintf(msg_str,"one trigger mode (%s)",
						name_for_trigger_mode(f->trigger_modes.modes[0]));
					prt_msg(msg_str);
					break;
				default:
					sprintf(msg_str,"%d trigger modes (",f->trigger_modes.num);
					prt_msg_frag(msg_str);
					for(i=0;i<f->trigger_modes.num-1;i++){
						sprintf(msg_str,"%s, ",
					name_for_trigger_mode(f->trigger_modes.modes[i]));
						prt_msg_frag(msg_str);
					}
					sprintf(msg_str,"%s)",
						name_for_trigger_mode(f->trigger_modes.modes[i]));
					prt_msg(msg_str);

					break;
			}
			break;
			/*
    printf("\n\tAvailableTriggerModes: ");
    if (f->trigger_modes.num==0) {
      printf("none");
    }
    else {
      int i;
      for (i=0;i<f->trigger_modes.num;i++) {
	printf("%d ",f->trigger_modes.modes[i]);
      }
    }
    printf("\n\tAvailableTriggerSources: ");
    if (f->trigger_sources.num==0) {
      printf("none");
    }
    else {
      int i;
      for (i=0;i<f->trigger_sources.num;i++) {
	printf("%d ",f->trigger_sources.sources[i]);
      }
    }
    printf("\n\tPolarity Change Capable: ");
    
    if (f->polarity_capable) 
      printf("True");
    else 
      printf("False");
    
    printf("\n\tCurrent Polarity: ");
    
    if (f->trigger_polarity) 
      printf("POS");
    else 
      printf("NEG");
    
    printf("\n\tcurrent mode: %d\n", f->trigger_mode);
    if (f->trigger_sources.num>0) {
      printf("\n\tcurrent source: %d\n", f->trigger_source);
    }
    */
		case DC1394_FEATURE_WHITE_BALANCE: 
		case DC1394_FEATURE_TEMPERATURE:
		case DC1394_FEATURE_WHITE_SHADING: 
			NWARN("unhandled case in feature type switch");
			break;
		default:
			sprintf(msg_str,"value: %-8d  range: %d-%d",f->value,f->min,f->max);
			prt_msg(msg_str);
			break;
	}
}
#endif /* HAVE_LIBDC1394 */

#ifdef HAVE_LIBDC1394

#define INDEX_SEARCH( name, type, count, array, member )		\
									\
int name( type x )							\
{									\
	unsigned int i;							\
									\
	for(i=0;i<count;i++){						\
		if( array[i].member == x )				\
			return(i);					\
	}								\
	return -1;							\
}

INDEX_SEARCH(index_of_feature,dc1394feature_t,N_NAMED_FEATURES,all_features,nft_feature)
INDEX_SEARCH(index_of_video_mode,dc1394video_mode_t,N_NAMED_VIDEO_MODES,all_video_modes,nvm_mode)
INDEX_SEARCH(index_of_framerate,dc1394framerate_t,N_NAMED_FRAMERATES,all_framerates,nfr_framerate)
INDEX_SEARCH(index_of_trigger_mode,dc1394trigger_mode_t,N_NAMED_TRIGGER_MODES,all_trigger_modes,ntm_mode)


const char *name_for_video_mode(dc1394video_mode_t mode)
{
	int i;

	i=index_of_video_mode(mode);
	if( i >= 0 )
		return(all_video_modes[i].nvm_name);
	return(NULL);
}

const char *name_for_framerate(dc1394framerate_t rate)
{
	int i;

	i=index_of_framerate(rate);
	if( i >= 0 )
		return(all_framerates[i].nfr_name);
	return(NULL);
}

const char *name_for_trigger_mode(dc1394trigger_mode_t mode)
{
	int i;

	i=index_of_trigger_mode(mode);
	if( i >= 0 )
		return(all_trigger_modes[i].ntm_name);
	return(NULL);
}

int list_video_modes(PGR_Cam *pgcp)
{
	dc1394video_modes_t	video_modes;
	const char *s;
	unsigned int i;

	if ( dc1394_video_get_supported_modes( pgcp->pc_cam_p, &video_modes ) != DC1394_SUCCESS )
		return -1;

	for( i = 0; i< video_modes.num; i++ ){
		s=name_for_video_mode(video_modes.modes[i]);
#ifdef CAUTIOUS
		if( s == NULL ){
			sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  No name for video mode %d!?",video_modes.modes[i]);
			s = DEFAULT_ERROR_STRING;
		}
#endif /* CAUTIOUS */
		prt_msg_frag("\t");
		prt_msg(s);
	}
	return 0;
}

#define BAD_VIDEO_MODE ((dc1394video_mode_t)-1)

dc1394video_mode_t pick_video_mode(QSP_ARG_DECL  PGR_Cam *pgcp,const char *pmpt)
{
	unsigned int i;
	int j,k;
	dc1394video_modes_t	video_modes;
	const char **choices;
	dc1394video_mode_t *	modelist;
	dc1394video_mode_t	m;

	if( pgcp == NULL ) return BAD_VIDEO_MODE;

	if ( dc1394_video_get_supported_modes( pgcp->pc_cam_p, &video_modes ) != DC1394_SUCCESS )
		return BAD_VIDEO_MODE;

	choices = (const char **)getbuf( video_modes.num * sizeof(char *) );
	modelist = (dc1394video_mode_t *)
		getbuf( video_modes.num * sizeof(dc1394video_mode_t) );
	j=0;
	for(i=0;i<video_modes.num;i++){
		k=index_of_video_mode(video_modes.modes[i]);	/* get index into our table... */
								/* could probably get this by subtraction if
								 * our table is in the correct order...
								 */
		if( k >= 0 ){
			choices[j] = all_video_modes[k].nvm_name;
			modelist[j] = all_video_modes[k].nvm_mode;
			j++;
		}
	}
	i=WHICH_ONE(pmpt,j,choices);
	givbuf(choices);
	if( i >= 0 )
		m=modelist[i];
	else
		m=BAD_VIDEO_MODE;
	givbuf(modelist);

	return(m);
}

dc1394video_mode_t pick_fmt7_mode(QSP_ARG_DECL  PGR_Cam *pgcp,const char *pmpt)
{
	unsigned int i;
	int j,k;
	dc1394video_modes_t	video_modes;
	const char **choices;
	dc1394video_mode_t *	modelist;
	dc1394video_mode_t	m;

	if( pgcp == NULL ) return BAD_VIDEO_MODE;

	if ( dc1394_video_get_supported_modes( pgcp->pc_cam_p, &video_modes ) != DC1394_SUCCESS )
		return BAD_VIDEO_MODE;

	choices = (const char **)getbuf( video_modes.num * sizeof(char *) );
	modelist = (dc1394video_mode_t *) getbuf( video_modes.num * sizeof(dc1394video_mode_t) );
	j=0;
	for(i=0;i<video_modes.num;i++){
		k=index_of_video_mode(video_modes.modes[i]);	/* get index into our table... */
								/* could probably get this by subtraction if
								 * our table is in the correct order...
								 */
		if( k >= 0 && video_modes.modes[i] >= DC1394_VIDEO_MODE_FORMAT7_0
			   && video_modes.modes[i] <= DC1394_VIDEO_MODE_FORMAT7_7 ){
			choices[j] = all_video_modes[k].nvm_name;
			modelist[j] = all_video_modes[k].nvm_mode;
			j++;
		}
	}
	i=WHICH_ONE(pmpt,j,choices);
	givbuf(choices);
	if( i >= 0 )
		m=modelist[i];
	else
		m=BAD_VIDEO_MODE;
	givbuf(modelist);

	return(m);
}



int pick_framerate(QSP_ARG_DECL  PGR_Cam *pgcp,const char *pmpt)
{
	unsigned int i;
	int j;
	dc1394framerates_t	framerates;
	const char **choices;
	const char *s;

	if( pgcp == NULL ) return -1;

	if ( dc1394_video_get_supported_framerates( pgcp->pc_cam_p, pgcp->pc_current_video_mode, &framerates )
			!= DC1394_SUCCESS )
		return -1;

	choices = (const char **) getbuf( framerates.num * sizeof(char *) );
	j=0;
	for(i=0;i<framerates.num;i++){
		s=name_for_framerate(framerates.framerates[i]);
		if( s != NULL ){
			choices[j] = s;
			j++;
		}
	}
	i=WHICH_ONE(pmpt,j,choices);
	givbuf(choices);
	return(i);
}

int set_video_mode(PGR_Cam *pgcp, dc1394video_mode_t mode)
{
	if( pgcp == NULL ) return -1;

	if( dc1394_video_set_mode( pgcp->pc_cam_p, mode) != DC1394_SUCCESS ){
		NWARN("unable to set video mode");
		return -1;
	}
	pgcp->pc_current_video_mode = mode;
	return 0;
}

int set_framerate(PGR_Cam *pgcp, int framerate_index)
{
	dc1394framerate_t rate;

	if( pgcp == NULL ) return -1;

	rate = all_framerates[framerate_index].nfr_framerate;
	if( dc1394_video_set_framerate( pgcp->pc_cam_p, rate) != DC1394_SUCCESS ){
		NWARN("unable to set framerate");
		return -1;
	}
	pgcp->pc_current_framerate = rate;
	return 0;
}

int list_framerates(PGR_Cam *pgcp)
{
	unsigned int i;
	dc1394framerates_t	framerates;
	const char *s;

	if ( dc1394_video_get_supported_framerates( pgcp->pc_cam_p, pgcp->pc_current_video_mode, &framerates )
			!= DC1394_SUCCESS ){
		NWARN("error fetching framerates");
		return -1;
	}

	if( framerates.num <= 0 ){
		NWARN("no framerates for this video mode!?");
		return -1;
	}

	for(i=0;i<framerates.num; i++){
		s=name_for_framerate(framerates.framerates[i]);
#ifdef CAUTIOUS
		if( s == NULL ){
			sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  No name for framerate%d!?",framerates.framerates[i]);
			s = DEFAULT_ERROR_STRING;
		}
#endif /* CAUTIOUS */
		prt_msg_frag("\t");
		prt_msg(s);
	}
	return 0;
}

int get_video_mode(PGR_Cam *pgcp)
{
	dc1394camera_t*	cam_p;
	dc1394video_modes_t	video_modes;
	dc1394video_mode_t	video_mode;
	int i;

	cam_p = pgcp->pc_cam_p;

	//  get the best video mode and highest framerate. This can be skipped
	//  if you already know which mode/framerate you want...
	// get video modes:
	if ( dc1394_video_get_supported_modes( cam_p, &video_modes ) != DC1394_SUCCESS )
		return -1;

	// select highest res mode that is greyscale (MONO8)
	/*
	printf( "Searching for the highest resolution MONO8 mode available (of %d modes)...\n",
		video_modes.num);
		*/

	dc1394color_coding_t coding;

	// assign an invalid value to video_mode to quiet compiler,
	// then check below to make sure a mode we want was found...

	video_mode = BAD_VIDEO_MODE;

	for ( i = video_modes.num-1; i >= 0; i-- ) {
		// don't consider FORMAT 7 modes (i.e. "scalable")
		if ( !dc1394_is_video_mode_scalable( video_modes.modes[i] ) ) {
			dc1394_get_color_coding_from_video_mode( cam_p, video_modes.modes[i],
									&coding );
			if ( coding == DC1394_COLOR_CODING_MONO8 ) {
				video_mode = video_modes.modes[i];
				break;
			}
		}
	}
#ifdef CAUTIOUS
	if( video_mode == BAD_VIDEO_MODE )
		NERROR1("CAUTIOUS:  get_video_mode:  unable to find a video mode!?");
#endif /* CAUTIOUS */

	// double check that we found a video mode  that is MONO8
	dc1394_get_color_coding_from_video_mode( cam_p, video_mode, &coding );
	if ( ( dc1394_is_video_mode_scalable( video_mode ) ) ||
			( coding != DC1394_COLOR_CODING_MONO8 ) ) {
		NWARN("Could not get a valid MONO8 mode" );
		return -1;
	}

	dc1394_video_set_mode( pgcp->pc_cam_p, video_mode );
	pgcp->pc_current_video_mode = video_mode;
	return 0;
}

int get_framerate(PGR_Cam *pgcp)
{
	// get highest framerate
	dc1394framerates_t	framerates;
	dc1394framerate_t	framerate;

	if ( dc1394_video_get_supported_framerates( pgcp->pc_cam_p, pgcp->pc_current_video_mode, &framerates )
			!= DC1394_SUCCESS ) {
		NWARN("Can't get framerates");
		return -1;
	}
	//printf("%d frame rates to choose from...\n",framerates.num);

	framerate = framerates.framerates[ framerates.num-1];

	dc1394_video_set_framerate( pgcp->pc_cam_p, framerate );
//printf("video mode and framerate set...\n");

	pgcp->pc_current_framerate = framerate;

	return 0;
}

PGR_Cam * init_firewire_system()
{
	dc1394camera_t*	cam_p;
	int			err;
	PGR_Cam *pgcp;

	// Find cameras on the 1394 buses
	/*
	uint32_t		nCameras;
	dc1394camera_t**	cameras=NULL;
	*/
	dc1394_t *		firewire_context;
	dc1394camera_list_t *	camera_list_p;

	/*
	err = dc1394_find_cameras( &cameras, &nCameras );
	*/
	firewire_context = dc1394_new();
	/* BUG check for error */
	err=dc1394_camera_enumerate(firewire_context,&camera_list_p);

	if ( err != DC1394_SUCCESS ) {
		sprintf( DEFAULT_ERROR_STRING, "Unable to look for cameras\n\n"
			"Please check \n"
			"  - if the kernel modules `ieee1394',`raw1394' and `ohci1394' are loaded \n"
			"  - if you have read/write access to /dev/raw1394\n\n");
		NWARN(DEFAULT_ERROR_STRING);
		return(NULL);
	}

	//  get the camera nodes and describe them as we find them
	if( camera_list_p == NULL ){
		NWARN("dc1394_camera_enumerate returned a null list pointer...");
		return(NULL);
	}
	if ( camera_list_p->num < 1 ) {
		sprintf( DEFAULT_ERROR_STRING, "No cameras found!\n");
		NWARN(DEFAULT_ERROR_STRING);
		return(NULL);
	}
	if( camera_list_p->num > 1 ){
		sprintf(DEFAULT_ERROR_STRING,
			"%d cameras found, using first...\n", camera_list_p->num );
		NWARN(DEFAULT_ERROR_STRING);
#ifdef FOOBAR
		// free the other cameras
		for( i = 1; i < camera_list_p->num; i++ )
			/*dc1394_free_camera */
			dc1394_camera_free( camera_list_p->ids[i] );
#endif /* FOOBAR */
	}

	cam_p = dc1394_camera_new( firewire_context, camera_list_p->ids[0].guid );

	//free(cameras);

	pgcp = &pgc1;
	pgcp->pc_cam_p = cam_p;
	pgcp->pc_feat_lp=NO_LIST;
	pgcp->pc_flags = 0;		/* assume no B-mode unless we are told otherwise... */

	if( get_video_mode(pgcp) < 0 ){
		NWARN("error getting video mode");
		cleanup1394( cam_p );
		return(NULL);
	}

	/* used to set B-mode stuff here... */
	dc1394_video_set_iso_speed( cam_p, DC1394_ISO_SPEED_400 );

	if( get_framerate(pgcp) < 0 ){
		NWARN("error getting framerate");
		cleanup1394( cam_p );
		return(NULL);
	}

	dc1394_get_image_size_from_video_mode( pgcp->pc_cam_p,
					pgcp->pc_current_video_mode,
					&pgcp->pc_nCols,
					&pgcp->pc_nRows );
	get_camera_features(pgcp);

	return(pgcp);
}

int start_firewire_transmission(QSP_ARG_DECL  PGR_Cam * pgcp, int _ring_buffer_size )
{
	int i;

	ring_buffer_size = _ring_buffer_size;

	/* older version had third flags arg... */
advise("calling dc1394_capture_setup");
	if( dc1394_capture_setup(pgcp->pc_cam_p,ring_buffer_size ,DC1394_CAPTURE_FLAGS_DEFAULT )
		!= DC1394_SUCCESS ) {

		fprintf( stderr,"unable to setup camera-\n"
			"check line %d of %s to make sure\n"
			"that the video mode and framerate are\n"
			"supported by your camera\n",
			__LINE__,__FILE__ );
		fprintf( stderr,
			"video_mode = %d, framerate = %d\n"
			"Check dc1394_control.h for the meanings of these values\n",
			pgcp->pc_current_video_mode, pgcp->pc_current_framerate );

		NERROR1("error starting capture");

		return(-1);
	}
	capturing=1;

	sprintf(msg_str,"%d",ring_buffer_size);		/* tell the scripting language */
	ASSIGN_VAR("ring_buffer_size",msg_str);

	// have the camera start sending us data
advise("calling dc1394_video_set_transmission");
	if ( dc1394_video_set_transmission( pgcp->pc_cam_p, DC1394_ON ) != DC1394_SUCCESS ) {
		sprintf(DEFAULT_ERROR_STRING,"Unable to start camera iso transmission" );
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	transmitting=1;

	//  Sleep untill the camera has a transmission
	dc1394switch_t status = DC1394_OFF;

	for ( i = 0; i <= 5; i++ ) {
		usleep(50000);
advise("calling dc1394_video_get_transmission");
		if ( dc1394_video_get_transmission( pgcp->pc_cam_p, &status ) != DC1394_SUCCESS ) {
			fprintf( stderr, "Unable to get transmision status\n" );
			return(-1);
		}
		if ( status != DC1394_OFF )
			break;

		if( i == 5 ) {
			fprintf(stderr,"Camera doesn't seem to want to turn on!\n");
			return(-1);
		}
	}
	return(0);
}

PGR_Frame *make_1394frame_obj(QSP_ARG_DECL  dc1394video_frame_t *framep)
{
	Dimension_Set dimset;
	Data_Obj *dp;
	char fname[32];
	PGR_Frame *pfp;

	sprintf(fname,"_frame%d",framep->id);
	pfp = new_pfrm(QSP_ARG  fname);
	if( pfp == NO_PGR_FRAME ){
		NWARN("make_1394frame_obj:  unable to create frame object");
		return(NO_PGR_FRAME);
	}
	pfp->pf_framep = framep;


	dimset.ds_dimension[0] = 1;	/* 1 or two depending on video mode (8 or 16) */
	dimset.ds_dimension[1] = framep->size[0];
	dimset.ds_dimension[2] = framep->size[1];
	dimset.ds_dimension[3] = 1;
	dimset.ds_dimension[4] = 1;

	dp = _make_dp(QSP_ARG  fname,&dimset,PREC_UBY);

	/* Do we need to test for a good return value??? */
	/* Only one buffer?  where do we specify the index? BUG */
	dp->dt_data = framep->image;

	pfp->pf_dp = dp;

	if( framep->total_bytes != framep->image_bytes ){
		sprintf(DEFAULT_ERROR_STRING,"image may be padded...");
		NWARN(DEFAULT_ERROR_STRING);
	}

	return(pfp);
}

static int serial=1;

PGR_Frame * grab_newest_firewire_frame( QSP_ARG_DECL  PGR_Cam * pgcp )
{
	PGR_Frame *pfp;
	int i=0;

	do {
		if( (pfp=grab_firewire_frame(QSP_ARG  pgcp)) == NULL ){
			NWARN("grab_newest_firewire_frame:  error grabbing single frame");
			return(NULL);
		}
		/* now release up to newest... */
		if( pfp->pf_framep->frames_behind > 0 ){
			release_oldest_frame(pgcp);
			/* will that change frames_behind??? only "grabbing" should... */
			i++;
		}
	} while( pfp->pf_framep->frames_behind > 0 );

	if( verbose && i > 0 ){
		sprintf(msg_str,"F %d   newest = %d, %d frames dropped",serial++,pfp->pf_framep->id,i);
		prt_msg(msg_str);
	}
	return(pfp);
}


PGR_Frame * grab_firewire_frame(QSP_ARG_DECL  PGR_Cam * pgcp )
{
	dc1394video_frame_t *framep;
	dc1394capture_policy_t policy=DC1394_CAPTURE_POLICY_WAIT;
	char fname[32];
	PGR_Frame *pfp;
	Node *np;

	if ( dc1394_capture_dequeue( pgcp->pc_cam_p, policy, &framep ) != DC1394_SUCCESS) {
		fprintf( stderr, "Unable to capture a frame\n" );
		return(NULL);
	}

	/* Now, we might want to determine if this is the newest available frame,
	 * and what policy we want to enforce if it is not...
	 */

	sprintf(fname,"_frame%d",framep->id);
	pfp = pfrm_of(QSP_ARG  fname);
	if( pfp == NO_PGR_FRAME )
		pfp = make_1394frame_obj(QSP_ARG  framep);

	if( pfp == NO_PGR_FRAME ){
		NWARN("unable to create frame object");
		return(NULL);
	}

	pfp->pf_framep = framep;	/* redundant only if make_1394frame_obj was called */
					/* in the other case, the pointer is likely to be unchanged,
					 * but we don't assume...
					 * We *do* assume that the old size is still ok.
					 */

	np=mk_node(pfp);
	if( in_use_lp == NO_LIST )
		in_use_lp = new_list();

	addHead(in_use_lp,np);

	return(pfp);
}

void release_oldest_frame(PGR_Cam *pgcp)
{
	Node *np;
	PGR_Frame *pfrm_p;

	if( in_use_lp == NO_LIST ){
		NWARN("release_oldest_frame:  no frames have been grabbed");
		return;
	}

	np = remTail(in_use_lp);
	if( np == NO_NODE ){
		NWARN("release_oldest_frame:  no frames are currently dequeued");
		return;
	}
	pfrm_p = (PGR_Frame *) np->n_data;

	if( dc1394_capture_enqueue(pgcp->pc_cam_p,pfrm_p->pf_framep)
		!= DC1394_SUCCESS ){
		NWARN("error enqueueing frame");
	}
	/* BUG free object too?? */
}

int stop_firewire_capture( PGR_Cam * pgcp )
{
	/* Transmission has to be stopped before capture is stopped,
	 * or the iso bandwidth will not be freed - jbm.
	 */

	if( ! transmitting ){
		/* CAUTIOUS? */
		NWARN("stop_firewire_capture:  not transmitting!?");
		return(-1);
	} else {
		//  Stop data transmission
		if( dc1394_video_set_transmission( pgcp->pc_cam_p, DC1394_OFF ) != DC1394_SUCCESS ){
			NWARN("stop_firewire_capture:  Couldn't stop transmission!?");
			return(-1);
		}
		transmitting=0;
	}

	if( ! capturing ){
		NWARN("stop_firewire_capture:  not capturing!?");
		return(-1);
	} else {
		dc1394_capture_stop( pgcp->pc_cam_p );
		capturing = 0;
	}

	/* why free the camera here? */
	/*
	dc1394_free_camera( pgcp->pc_cam_p );
	*/


	pgcp->pc_flags &= ~PGR_CAM_IS_RUNNING;

	return(0);
}

int reset_camera(PGR_Cam *pgcp)
{
	if( /* dc1394_reset_camera */
		dc1394_camera_reset(pgcp->pc_cam_p) != DC1394_SUCCESS ){
		NWARN("Could not initilize camera");
		return -1;
	}
	return 0;
}


void list_trig(PGR_Cam *pgcp)
{
	dc1394bool_t has_p;
	dc1394trigger_mode_t tm;

	if( dc1394_external_trigger_has_polarity(pgcp->pc_cam_p,&has_p) !=
		DC1394_SUCCESS ){
		NWARN("error querying for trigger polarity capability");
		return;
	}
	if( has_p ){
		prt_msg("external trigger has polarity");
	} else {
		prt_msg("external trigger does not have polarity");
	}

	if( dc1394_external_trigger_get_mode(pgcp->pc_cam_p,&tm) !=
		DC1394_SUCCESS ){
		NWARN("error querying trigger mode");
		return;
	}
	switch(tm){
		case DC1394_TRIGGER_MODE_0:
			prt_msg("Mode 0:  falling edge + shutter"); break;
		case DC1394_TRIGGER_MODE_1:
			prt_msg("Mode 1:  falling edge to rising edge"); break;
		case DC1394_TRIGGER_MODE_2:
			prt_msg("Mode 2:  falling edge to nth falling edge"); break;
		case DC1394_TRIGGER_MODE_3:
			prt_msg("Mode 3:  internal trigger"); break;
		case DC1394_TRIGGER_MODE_4:
			prt_msg("Mode 4:  multiple exposure (shuttered)"); break;
		case DC1394_TRIGGER_MODE_5:
			prt_msg("Mode 5:  multiple exposure (triggered)"); break;
		case DC1394_TRIGGER_MODE_14:
			prt_msg("Mode 14:  vendor-specific"); break;
		case DC1394_TRIGGER_MODE_15:
			prt_msg("Mode 15:  vendor-specific"); break;
		default:
			NWARN("unrecognized trigger mode");
			break;
	}
}

void report_bandwidth( PGR_Cam *pgcp )
{
	unsigned int bw;

	if( dc1394_video_get_bandwidth_usage(pgcp->pc_cam_p,&bw) !=
		DC1394_SUCCESS ){
		NWARN("error querying bandwidth usage");
		return;
	}
	/* What are the units of bandwidth??? */
	sprintf(msg_str,"Bandwidth:  %d",bw);
	prt_msg(msg_str);
}

void print_camera_info(PGR_Cam *pgcp)
{
	int i;

	prt_msg("\nCurrent camera:");

	i=index_of_video_mode(pgcp->pc_current_video_mode);
	sprintf(msg_str,"\tmode:  %s",all_video_modes[i].nvm_name);
	prt_msg(msg_str);

	i=index_of_framerate(pgcp->pc_current_framerate);
	sprintf(msg_str,"\trate:  %s",all_framerates[i].nfr_name);
	prt_msg(msg_str);

	/* report_camera_features(pgcp); */
}

#endif /* HAVE_LIBDC1394 */
