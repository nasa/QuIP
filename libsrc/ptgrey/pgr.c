/* Jeff's interface to the 1394 subsystem, to talk to the PGR camera.  */

#include "quip_config.h"

#ifdef HAVE_LIBDC1394

#include "quip_prot.h"
#include "data_obj.h"

#include <stdio.h>
#include <string.h>

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

#define TMPSIZE	32	// for temporary object names, e.g. _frame55

ITEM_INTERFACE_DECLARATIONS(PGR_Cam,pgc,0)

//dc1394error_t dc1394_get_camera_feature_set(dc1394camera_t *camera, dc1394featureset_t *features);

/* Stores the bounds and options associated with the feature described by feature->feature_id */
//dc1394error_t dc1394_get_camera_feature(dc1394camera_t *camera, dc1394feature_info_t *feature);

/* Displays the bounds and options of the given feature or of the entire feature set */
//dc1394error_t dc1394_print_feature(dc1394feature_info_t *feature);
//dc1394error_t dc1394_print_feature_set(dc1394featureset_t *features);

#ifdef NOT_USED
void show_camera_features(PGR_Cam *pgcp)
{
	dc1394featureset_t features;

	if(
		/* dc1394_get_camera_feature_set */
		dc1394_feature_get_all
		(pgcp->pc_cam_p, &features) != DC1394_SUCCESS ){
		warn("error getting camera feature set");
		return;
	}
	if( /* dc1394_print_feature_set(&features) */
		dc1394_feature_print_all(&features,stdout) != DC1394_SUCCESS ){
		warn("error printing feature set");
	}
}
#endif /* NOT_USED */

void
cleanup_cam( PGR_Cam *pgcp )
{
	if( IS_CAPTURING(pgcp) ) dc1394_capture_stop( pgcp->pc_cam_p );
	if( IS_TRANSMITTING(pgcp) )
		dc1394_video_set_transmission( pgcp->pc_cam_p, DC1394_OFF );
	/* dc1394_free_camera */
	dc1394_camera_free( pgcp->pc_cam_p );
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
{ "mono16_1280x960",	DC1394_VIDEO_MODE_1280x960_MONO16	},

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

#ifdef NOT_USED
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
#endif /* NOT_USED */

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

/* called once at camera initialization... */

void get_camera_features( PGR_Cam *pgcp )
{
	Node *np;
	int i;

	if ( /*dc1394_get_camera_feature_set*/
		dc1394_feature_get_all( pgcp->pc_cam_p, &pgcp->pc_features ) != DC1394_SUCCESS ) {
		NERROR1("get_camera_features:  unable to get camera feature set");
	}

	/* Now can the table and build the linked list */
	/* We may call this again after we have diddled the controls... */
	/* releasing and rebuilding the list is wasteful, but should work... */
	if( pgcp->pc_feat_lp != NULL ){
		while( (np=remHead(pgcp->pc_feat_lp)) != NULL )
			rls_node(np);
	} else {
		pgcp->pc_feat_lp = new_list();
	}


	for(i=0;i<DC1394_FEATURE_NUM;i++){
		dc1394feature_info_t * f;

		f = &pgcp->pc_features.feature[i];

		assert( f->id >= DC1394_FEATURE_MIN && f->id <= DC1394_FEATURE_MAX );

		if(f->available){
			np = mk_node(f);
			addTail(pgcp->pc_feat_lp,np);
		}
	}
}


/* report_camera_features:  this used the library's print routine (for comparison) */

#define report_camera_features( pgcp ) _report_camera_features( QSP_ARG  pgcp )


static int _report_camera_features( QSP_ARG_DECL  PGR_Cam *pgcp )
{
	// report camera's features
	dc1394featureset_t	features;

	if ( /*dc1394_get_camera_feature_set*/
		dc1394_feature_get_all( pgcp->pc_cam_p, &features ) != DC1394_SUCCESS ) {
		warn("report_camera_features:  unable to get camera feature set");
		return -1;
	}
	/*dc1394_print_feature_set( &features ); */
	dc1394_feature_print_all( &features,stdout );
	return 0;
}

static void list_camera_feature(QSP_ARG_DECL  dc1394feature_info_t *feat_p )
{
	sprintf(msg_str,"%s", /*dc1394_feature_desc[feat_p->id - DC1394_FEATURE_MIN]*/
			dc1394_feature_get_string(feat_p->id) );
	prt_msg(msg_str);
}

int list_camera_features(QSP_ARG_DECL  PGR_Cam *pgcp )
{
	Node *np;

	assert( pgcp->pc_feat_lp != NULL );

	np = QLIST_HEAD(pgcp->pc_feat_lp);
	while(np!=NULL){
		dc1394feature_info_t * f;
		f= (dc1394feature_info_t *) np->n_data;
		list_camera_feature(QSP_ARG  f);
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

	np=QLIST_HEAD(pgcp->pc_feat_lp);
	while(np!=NULL){
		dc1394feature_info_t *f;
		f= (dc1394feature_info_t *) np->n_data;
		*sptr = /*(char *)dc1394_feature_desc[f->id - DC1394_FEATURE_MIN]*/
			dc1394_feature_get_string(f->id) ;
		sptr++;
		np=np->n_next;
	}
	return n;
}

void report_feature_info(QSP_ARG_DECL  PGR_Cam *pgcp, dc1394feature_t id )
{
	Node *np;
	dc1394feature_info_t *f;
	unsigned int i;
	const char *name;
	char nbuf[32];

	np = QLIST_HEAD(pgcp->pc_feat_lp);
	f=NULL;
	while( np != NULL ){
		f= (dc1394feature_info_t *) np->n_data;

		if( f->id == id )
			np=NULL;
		else
			f=NULL;

		if( np != NULL )
			np = np->n_next;
	}

	assert( f != NULL );

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
			warn("unhandled case in feature type switch");
			break;
		default:
			sprintf(msg_str,"value: %-8d  range: %d-%d",f->value,f->min,f->max);
			prt_msg(msg_str);
			break;
	}
}

#define INDEX_SEARCH( name, type, count, array, member )		\
									\
static int name( type x )						\
{									\
	unsigned int i;							\
									\
	for(i=0;i<count;i++){						\
		if( array[i].member == x )				\
			return(i);					\
	}								\
	return -1;							\
}

#ifdef NOT_USED
INDEX_SEARCH(index_of_feature,dc1394feature_t,N_NAMED_FEATURES,all_features,nft_feature)
#endif /* NOT_USED */
INDEX_SEARCH(index_of_video_mode,dc1394video_mode_t,N_NAMED_VIDEO_MODES,all_video_modes,nvm_mode)
INDEX_SEARCH(index_of_framerate,dc1394framerate_t,N_NAMED_FRAMERATES,all_framerates,nfr_framerate)
INDEX_SEARCH(index_of_trigger_mode,dc1394trigger_mode_t,N_NAMED_TRIGGER_MODES,all_trigger_modes,ntm_mode)


static const char *name_for_video_mode(dc1394video_mode_t mode)
{
	int i;

	i=index_of_video_mode(mode);
	if( i >= 0 )
		return(all_video_modes[i].nvm_name);
	return(NULL);
}

static const char *name_for_framerate(dc1394framerate_t rate)
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

int list_video_modes(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	dc1394video_modes_t	video_modes;
	const char *s;
	unsigned int i;

	if ( dc1394_video_get_supported_modes( pgcp->pc_cam_p, &video_modes ) != DC1394_SUCCESS )
		return -1;

	for( i = 0; i< video_modes.num; i++ ){
		s=name_for_video_mode(video_modes.modes[i]);
		assert( s != NULL );
		prt_msg_frag("\t");
		prt_msg(s);
	}
	return 0;
}

dc1394video_mode_t pick_video_mode(QSP_ARG_DECL  PGR_Cam *pgcp,const char *pmpt)
{
	int i;
	int j,k;
	const char **choices;
	dc1394video_mode_t *	modelist;
	dc1394video_mode_t	m;

	if( pgcp == NULL ) return BAD_VIDEO_MODE;

	choices = (const char **)getbuf( pgcp->pc_video_modes.num * sizeof(char *) );
	modelist = (dc1394video_mode_t *)
		getbuf( pgcp->pc_video_modes.num * sizeof(dc1394video_mode_t) );
	j=0;
	for(i=0;i<pgcp->pc_video_modes.num;i++){
		/* get index into our table... */
		/* could probably get this by subtraction if
		 * our table is in the correct order...
		 */
		k=index_of_video_mode(pgcp->pc_video_modes.modes[i]);
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

int get_camera_names( QSP_ARG_DECL  Data_Obj *str_dp )
{
	// Could check format of object here...
	// Should be string table with enough entries to hold the modes
	// Should the strings be rows or multidim pixels?
	List *lp;
	Node *np;
	PGR_Cam *pgcp;
	int i, n;

	lp = pgc_list();
	if( lp == NULL ){
		WARN("No cameras!?");
		return 0;
	}

	n=eltcount(lp);
	if( OBJ_COLS(str_dp) < n ){
		sprintf(ERROR_STRING,"String object %s has too few columns (%ld) to hold %d camera names",
			OBJ_NAME(str_dp),(long)OBJ_COLS(str_dp),n);
		WARN(ERROR_STRING);
		n = OBJ_COLS(str_dp);
	}
		
	np=QLIST_HEAD(lp);
	i=0;
	while(np!=NULL){
		char *dst;
		pgcp = (PGR_Cam *) NODE_DATA(np);
		dst = OBJ_DATA_PTR(str_dp);
		dst += i * OBJ_PXL_INC(str_dp);
		if( strlen(pgcp->pc_name)+1 > OBJ_COMPS(str_dp) ){
			sprintf(ERROR_STRING,"String object %s has too few components (%ld) to hold camera name \"%s\"",
				OBJ_NAME(str_dp),(long)OBJ_COMPS(str_dp),pgcp->pc_name);
			WARN(ERROR_STRING);
		} else {
			strcpy(dst,pgcp->pc_name);
		}
		i++;
		if( i>=n )
			np=NULL;
		else
			np = NODE_NEXT(np);
	}

	return i;
}

int get_video_mode_strings( QSP_ARG_DECL  Data_Obj *str_dp, PGR_Cam *pgcp )
{
	// Could check format of object here...
	// Should be string table with enough entries to hold the modes
	// Should the strings be rows or multidim pixels?

	int i, n;

	if( OBJ_COLS(str_dp) < pgcp->pc_video_modes.num ){
		sprintf(ERROR_STRING,"String object %s has too few columns (%ld) to hold %d modes",
			OBJ_NAME(str_dp),(long)OBJ_COLS(str_dp),pgcp->pc_video_modes.num);
		WARN(ERROR_STRING);
		n = OBJ_COLS(str_dp);
	} else {
		n=pgcp->pc_video_modes.num;
	}
		
	for(i=0;i<n;i++){
		int k;
		const char *src;
		char *dst;

		k=index_of_video_mode(pgcp->pc_video_modes.modes[i]);
		src = all_video_modes[k].nvm_name;
		dst = OBJ_DATA_PTR(str_dp);
		dst += i * OBJ_PXL_INC(str_dp);
		if( strlen(src)+1 > OBJ_COMPS(str_dp) ){
			sprintf(ERROR_STRING,"String object %s has too few components (%ld) to hold mode string \"%s\"",
				OBJ_NAME(str_dp),(long)OBJ_COMPS(str_dp),src);
			WARN(ERROR_STRING);
		} else {
			strcpy(dst,src);
		}
	}
	set_script_var_from_int(QSP_ARG  "n_video_modes",n);
	return n;
}

int get_framerate_strings( QSP_ARG_DECL  Data_Obj *str_dp, PGR_Cam *pgcp )
{
	// Could check format of object here...
	// Should be string table with enough entries to hold the modes
	// Should the strings be rows or multidim pixels?

	int i, n;

	if( OBJ_COLS(str_dp) < pgcp->pc_framerates.num ){
		sprintf(ERROR_STRING,"String object %s has too few columns (%ld) to hold %d framerates",
			OBJ_NAME(str_dp),(long)OBJ_COLS(str_dp),pgcp->pc_framerates.num);
		WARN(ERROR_STRING);
		n = OBJ_COLS(str_dp);
	} else {
		n=pgcp->pc_framerates.num;
	}
		
	for(i=0;i<n;i++){
		const char *src;
		char *dst;

		src = name_for_framerate(pgcp->pc_framerates.framerates[i]);
		dst = OBJ_DATA_PTR(str_dp);
		dst += i * OBJ_PXL_INC(str_dp);
		if( strlen(src)+1 > OBJ_COMPS(str_dp) ){
			sprintf(ERROR_STRING,"String object %s has too few components (%ld) to hold framerate string \"%s\"",
				OBJ_NAME(str_dp),(long)OBJ_COMPS(str_dp),src);
			WARN(ERROR_STRING);
		} else {
			strcpy(dst,src);
		}
	}
	return n;
}

dc1394video_mode_t pick_fmt7_mode(QSP_ARG_DECL  PGR_Cam *pgcp,const char *pmpt)
{
	int i;
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

static int mode_is_format7( PGR_Cam *pgcp )
{
	if( pgcp->pc_video_mode >= DC1394_VIDEO_MODE_FORMAT7_MIN &&
			pgcp->pc_video_mode <= DC1394_VIDEO_MODE_FORMAT7_MAX )
		return 1;
	return 0;
}

int pick_framerate(QSP_ARG_DECL  PGR_Cam *pgcp,const char *pmpt)
{
	int i;
	int j;
	dc1394framerates_t	framerates;
	const char **choices;
	const char *s;

	if( pgcp == NULL ) return -1;

	// format7 doesn't have a framerate!?
	if( mode_is_format7(pgcp) ){
		WARN("Can't specify framerate for format7 video mode...");
		// eat the argument
		s = NAMEOF("dummy argument");
		return -1;
	}

	if ( dc1394_video_get_supported_framerates( pgcp->pc_cam_p, pgcp->pc_video_mode, &framerates )
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

static int set_default_framerate(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	int r;

	// get highest framerate

	// Hmm...  the avaialable framerates appear to depend on the video mode!
	// Therefore, we should re-call this when we change the video mode!

	// format7 doesn't have a framerate!?

	// format7 doesn't have a framerate!?
	if( mode_is_format7(pgcp) ){
		//warn("set_default_framerate:  No framerate associated with format7 video mode!?");
		pgcp->pc_framerate = -1;
		set_script_var_from_int(QSP_ARG "n_framerates",0);
		return 0;
	}

	if ( dc1394_video_get_supported_framerates( pgcp->pc_cam_p, pgcp->pc_video_mode, &(pgcp->pc_framerates) )
			!= DC1394_SUCCESS ) {
		warn("Can't get framerates");
		return -1;
	}

	r = pgcp->pc_framerates.framerates[ pgcp->pc_framerates.num-1];
	pgcp->pc_framerate = r;

sprintf(ERROR_STRING,"set_default_framerate:  setting to %s", name_for_framerate(r));
advise(ERROR_STRING);

	dc1394_video_set_framerate( pgcp->pc_cam_p, pgcp->pc_framerate );
	// BUG if this fails, then pc_framerate is wrong!?

	// stash the number of framerates in a script variable
	// in case the user wants to fetch the strings...
	set_script_var_from_int(QSP_ARG
			"n_framerates",pgcp->pc_framerates.num);

	return 0;
}

int set_video_mode(QSP_ARG_DECL  PGR_Cam *pgcp, dc1394video_mode_t mode)
{
	dc1394framerate_t fr;

	if( pgcp == NULL ) return -1;

	if( dc1394_video_set_mode( pgcp->pc_cam_p, mode) != DC1394_SUCCESS ){
		WARN("unable to set video mode");
		return -1;
	}
	pgcp->pc_video_mode = mode;

	fr=pgcp->pc_framerate;
	if( set_default_framerate(QSP_ARG  pgcp) < 0 ){
		advise("Unable to set default framerate for new video mode");
	} else {
		if( fr != pgcp->pc_framerate ){
			sprintf(ERROR_STRING,
		"set_video_mode:  framerate changed from %s to %s",
				name_for_framerate(fr),
				name_for_framerate(pgcp->pc_framerate) );
			advise(DEFAULT_ERROR_STRING);
		}
	}

	return 0;
}

int _set_framerate(QSP_ARG_DECL  PGR_Cam *pgcp, int framerate_index)
{
	dc1394framerate_t rate;

	if( pgcp == NULL ) return -1;

	rate = all_framerates[framerate_index].nfr_framerate;
	if( dc1394_video_set_framerate( pgcp->pc_cam_p, rate) != DC1394_SUCCESS ){
		warn("unable to set framerate");
		return -1;
	}
	pgcp->pc_framerate = rate;
	return 0;
}

void show_framerate(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	sprintf(MSG_STR,"%s framerate:  %s",
		pgcp->pc_name,name_for_framerate(pgcp->pc_framerate));
	advise(MSG_STR);
}

void show_video_mode(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	sprintf(MSG_STR,"%s video mode:  %s",
		pgcp->pc_name,name_for_video_mode(pgcp->pc_video_mode));
	advise(MSG_STR);
}

int list_framerates(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	unsigned int i;
	dc1394framerates_t	framerates;
	const char *s;

	// format7 doesn't have a framerate!?
	if( mode_is_format7(pgcp) ){
		WARN("list_framerates:  No framerates associated with format7 video mode!?");
		return -1;
	}

	if ( dc1394_video_get_supported_framerates( pgcp->pc_cam_p, pgcp->pc_video_mode, &framerates )
			!= DC1394_SUCCESS ){
		warn("error fetching framerates");
		return -1;
	}

	if( framerates.num <= 0 ){
		warn("no framerates for this video mode!?");
		return -1;
	}

	for(i=0;i<framerates.num; i++){
		s=name_for_framerate(framerates.framerates[i]);
		assert( s != NULL );
		prt_msg_frag("\t");
		prt_msg(s);
	}
	return 0;
}

static int set_default_video_mode(PGR_Cam *pgcp)
{
	dc1394camera_t*	cam_p;
	dc1394video_mode_t	video_mode;
	int i;

	cam_p = pgcp->pc_cam_p;

	//  get the best video mode and highest framerate. This can be skipped
	//  if you already know which mode/framerate you want...
	// get video modes:

	if( dc1394_video_get_supported_modes( cam_p, &pgcp->pc_video_modes )
						!= DC1394_SUCCESS ){
		return -1;
	}

	// select highest res mode that is greyscale (MONO8)
	/*
	printf("Searching for the highest resolution MONO8 mode available (of %d modes)...\n",
		video_modes.num);
		*/

	dc1394color_coding_t coding;

	// assign an invalid value to video_mode to quiet compiler,
	// then check below to make sure a mode we want was found...

	video_mode = BAD_VIDEO_MODE;

//fprintf(stderr,"Checking %d video modes...\n",pgcp->pc_video_modes.num);
	for ( i = pgcp->pc_video_modes.num-1; i >= 0; i-- ) {
		// don't consider FORMAT 7 modes (i.e. "scalable")
		if ( !dc1394_is_video_mode_scalable( pgcp->pc_video_modes.modes[i] ) ) {
			dc1394_get_color_coding_from_video_mode( cam_p,
				pgcp->pc_video_modes.modes[i], &coding );
//fprintf(stderr,"Checking non-scalable mode %d\n",pgcp->pc_video_modes.modes[i]);
			if ( coding == DC1394_COLOR_CODING_MONO8 ) {
				video_mode = pgcp->pc_video_modes.modes[i];
				break;
			}
		} else {
//fprintf(stderr,"Not checking scalable mode %d\n",pgcp->pc_video_modes.modes[i]);
		}
	}
	if( video_mode == BAD_VIDEO_MODE ){	// only scalable modes?
		for ( i = pgcp->pc_video_modes.num-1; i >= 0; i-- ) {
			dc1394_get_color_coding_from_video_mode( cam_p, 
				pgcp->pc_video_modes.modes[i], &coding );
			if( coding == DC1394_COLOR_CODING_MONO8  ||
					coding == DC1394_COLOR_CODING_RAW8 ) {
				video_mode = pgcp->pc_video_modes.modes[i];
				break;
			}
		}
	}

	assert( video_mode != BAD_VIDEO_MODE );

#ifdef FOOBAR
	// double check that we found a video mode  that is MONO8
	dc1394_get_color_coding_from_video_mode( cam_p, video_mode, &coding );
	if ( ( dc1394_is_video_mode_scalable( video_mode ) ) ||
			( coding != DC1394_COLOR_CODING_MONO8 &&
			  coding != DC1394_COLOR_CODING_RAW8 ) ) {
		warn("Could not get a valid MONO8 mode" );
		return -1;
	}
#endif // FOOBAR

	dc1394_video_set_mode( pgcp->pc_cam_p, video_mode );
	pgcp->pc_video_mode = video_mode;

	return 0;
}

static void fix_string( char *s )
{
	while( *s ){
		if( *s == ' ' ) *s='_';
		// other chars to map also?
		s++;
	}
}

static PGR_Cam *unique_camera_instance( QSP_ARG_DECL  dc1394camera_t *cam_p )
{
	int i;
	char cname[80];	// How many chars is enough?
	PGR_Cam *pgcp;

	i=1;
	pgcp=NULL;
	while(pgcp==NULL){
		sprintf(cname,"%s_%d",cam_p->model,i);
		fix_string(cname);	// change spaces to underscores
		pgcp = pgc_of( cname );
		if( pgcp == NULL ){	// This index is free
			pgcp = new_pgc( cname );
			if( pgcp == NULL ){
				sprintf(ERROR_STRING,
			"Failed to create camera %s!?",cname);
				error1(ERROR_STRING);
			}
		} else {
			pgcp = NULL;
		}
		i++;
		if( i>=5 ){
			error1("Too many cameras!?"); 
		}
	}
	return pgcp;
}

static PGR_Cam *setup_my_camera( QSP_ARG_DECL  dc1394camera_t * cam_p )
{
	PGR_Cam *pgcp;

	// We could have multiple instances of the same model...
	pgcp = unique_camera_instance(QSP_ARG  cam_p);

	pgcp->pc_cam_p = cam_p;
	pgcp->pc_feat_lp=NULL;
	pgcp->pc_in_use_lp=NULL;
	pgcp->pc_flags = 0;		/* assume no B-mode unless we are told otherwise... */

	if( set_default_video_mode(pgcp) < 0 ){
		WARN("error setting default video mode");
		cleanup_cam( pgcp );
		return(NULL);
	}

	/* used to set B-mode stuff here... */
	// What if the camera is a usb cam???
	dc1394_video_set_iso_speed( cam_p, DC1394_ISO_SPEED_400 );

	if( set_default_framerate(QSP_ARG  pgcp) < 0 ){
		// This happens for format7...
		//warn("error setting default framerate");
		//cleanup_cam( pgcp );
		//return(NULL);
	}

	dc1394_get_image_size_from_video_mode( pgcp->pc_cam_p,
					pgcp->pc_video_mode,
					&pgcp->pc_nCols,
					&pgcp->pc_nRows );
	get_camera_features(pgcp);

	/* We need to poll if we have multiple cameras with different frame rates */
	pgcp->pc_policy = DC1394_CAPTURE_POLICY_POLL;	// or WAIT ...
	//pgcp->pc_policy = DC1394_CAPTURE_POLICY_WAIT;	// or POLL ...

	// Make a data_obj context for the frames...
	pgcp->pc_do_icp = create_dobj_context( pgcp->pc_name );

	return(pgcp);
}

void pop_camera_context(SINGLE_QSP_ARG_DECL)
{
	// pop old context...
	Item_Context *icp;
	icp=pop_dobj_context();
	assert( icp != NULL );
}

void push_camera_context(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	push_dobj_context(pgcp->pc_do_icp);
}

int init_firewire_system(SINGLE_QSP_ARG_DECL)
{
	dc1394camera_t*	cam_p;
	int err;
	PGR_Cam *pgcp;
	int i,n_good_cameras;
	static int firewire_system_inited=0;

	if( firewire_system_inited ){
		WARN("Firewire system has already been initialized!?");
		return -1;
	}
	firewire_system_inited=1;

	// Find cameras on the 1394 buses
	dc1394_t *		firewire_context;
	dc1394camera_list_t *	camera_list_p;

	firewire_context = dc1394_new();
	/* BUG check for error */
	err=dc1394_camera_enumerate(firewire_context,&camera_list_p);

	if ( err != DC1394_SUCCESS ) {
		sprintf( ERROR_STRING, "Unable to look for cameras\n\n"
			"Please check \n"
			"  - if the kernel modules `ieee1394',`raw1394' and `ohci1394' are loaded \n"
			"  - if you have read/write access to /dev/raw1394\n\n");
		WARN(ERROR_STRING);
		return -1;
	}

	//  get the camera nodes and describe them as we find them
	if( camera_list_p == NULL ){
		warn("dc1394_camera_enumerate returned a null list pointer...");
		return -1;
	}
	if ( camera_list_p->num < 1 ) {
		sprintf( ERROR_STRING, "No cameras found!");
		WARN(ERROR_STRING);
		advise("Check permissions on /dev/fw? ...");
		return -1;
	}

	sprintf(ERROR_STRING,
		"%d camera%s found.", camera_list_p->num, camera_list_p->num==1?"":"s" );
	advise(ERROR_STRING);

	// Why not initialize all the cameras???

	n_good_cameras=0;
	for(i=0;i<camera_list_p->num;i++){
		cam_p = dc1394_camera_new( firewire_context, camera_list_p->ids[i].guid );
		if( cam_p == NULL ){
			sprintf(ERROR_STRING,"dc1394_camera_new failed...");
			WARN(ERROR_STRING);
		} else {
			pgcp = setup_my_camera(QSP_ARG  cam_p);
			n_good_cameras ++;
			sprintf(ERROR_STRING,
				"%s set up...", pgcp->pc_cam_p->model );
			advise(ERROR_STRING);
		}
	}

	set_script_var_from_int(QSP_ARG  "n_cameras",
				/*camera_list_p->num*/ n_good_cameras);

	// BUG make this a reserved var

	// Set camera name in a script variable here...
	//assign_var("camera_model",pgcp->pc_cam_p->model);
	//return pgcp;

	return 0;
}

static Data_Obj *make_1394frame_obj(QSP_ARG_DECL  dc1394video_frame_t *framep)
{
	Dimension_Set dimset;
	Data_Obj *dp;
	char fname[32];

	sprintf(fname,"_frame%d",framep->id);

	dimset.ds_dimension[0] = 1;	/* 1 or two depending on video mode (8 or 16) */
	dimset.ds_dimension[1] = framep->size[0];
	dimset.ds_dimension[2] = framep->size[1];
	dimset.ds_dimension[3] = 1;
	dimset.ds_dimension[4] = 1;

	dp = _make_dp(QSP_ARG  fname,&dimset,PREC_FOR_CODE(PREC_UBY));

	/* Do we need to test for a good return value??? */
	/* Only one buffer?  where do we specify the index? BUG */
	SET_OBJ_DATA_PTR(dp, framep->image);
if( verbose )
fprintf(stderr,"Object %s, data ptr set to 0x%lx\n",OBJ_NAME(dp),(long)OBJ_DATA_PTR(dp));

	if( framep->total_bytes != framep->image_bytes ){
		sprintf(DEFAULT_ERROR_STRING,"image may be padded...");
		warn(DEFAULT_ERROR_STRING);
	}

	return(dp);
}

static void init_buffer_objects(QSP_ARG_DECL  PGR_Cam * pgcp )
{
	int i;
	dc1394video_frame_t *framep;
	char fname[TMPSIZE];
	Data_Obj *dp;

sprintf(ERROR_STRING,"Initializing %d buffer objects...",
pgcp->pc_ring_buffer_size);
advise(ERROR_STRING);

	// Cycle once through the ring buffer,
	// making a data object for each frame
	for(i=0;i<pgcp->pc_ring_buffer_size;i++){
		if ( dc1394_capture_dequeue( pgcp->pc_cam_p, 
			DC1394_CAPTURE_POLICY_WAIT, &framep )
			!= DC1394_SUCCESS) {
	error1("init_buffer_objects:  error in dc1394_capture_dequeue!?" );
		}
		snprintf(fname,TMPSIZE,"_frame%d",framep->id);
		assert( i == framep->id );
		dp = make_1394frame_obj(QSP_ARG  framep);
		if( dc1394_capture_enqueue(pgcp->pc_cam_p,framep)
				!= DC1394_SUCCESS ){
			error1("init_buffer_objects:  error enqueueing frame!?");
		}
		// Here we might store dp in a table...
	}
advise("Done setting up buffer objects.");
}

int start_firewire_transmission(QSP_ARG_DECL  PGR_Cam * pgcp, int _ring_buffer_size )
{
	int i;
	dc1394error_t err;
	Data_Obj *dp;

//advise("start_firewire_transmission BEGIN");
	/* older version had third flags arg... */
//advise("calling dc1394_capture_setup");
	if( (err=dc1394_capture_setup(pgcp->pc_cam_p,_ring_buffer_size ,DC1394_CAPTURE_FLAGS_DEFAULT ))
		!= DC1394_SUCCESS ) {

		WARN("dc1394_capture_setup failed!?");
		describe_dc1394_error( QSP_ARG  err );

		if( err == DC1394_IOCTL_FAILURE ){
			advise("Try decreasing the number of ring buffer frames requested?");
			return -1;
		}

		fprintf( stderr,"unable to setup camera-\n"
			"check line %d of %s to make sure\n"
			"that the video mode and framerate are\n"
			"supported by your camera\n",
			__LINE__,__FILE__ );

		/*
		fprintf( stderr,
			"video_mode = %d, framerate = %d\n"
			"Check dc1394_control.h for the meanings of these values\n",
			pgcp->pc_video_mode, pgcp->pc_framerate );
			*/
		fprintf( stderr,
			"video_mode = %s (%d), framerate = %s (%d)\n",
			name_for_video_mode(pgcp->pc_video_mode),
			pgcp->pc_video_mode,
			name_for_framerate(pgcp->pc_framerate),
			pgcp->pc_framerate );

		NERROR1("error starting capture");

		return(-1);
	}

	pgcp->pc_ring_buffer_size = _ring_buffer_size;
	pgcp->pc_n_avail = _ring_buffer_size;

	pgcp->pc_flags |= PGR_CAM_IS_CAPTURING;

#ifdef FOOBAR	// we set this in the script already!?
	sprintf(msg_str,"%d",ring_buffer_size);		/* tell the scripting language */
	assign_var("ring_buffer_size",msg_str);
#endif // FOOBAR

	// have the camera start sending us data
//advise("calling dc1394_video_set_transmission");
	if( (err=dc1394_video_set_transmission( pgcp->pc_cam_p, DC1394_ON ))
			!= DC1394_SUCCESS ) {
		WARN("Unable to start camera iso transmission");
		describe_dc1394_error( QSP_ARG  err );

		// do we need to undo capture_setup?
		dc1394_capture_stop( pgcp->pc_cam_p );
		pgcp->pc_flags &= ~PGR_CAM_IS_CAPTURING;

		return(-1);
	}
	pgcp->pc_flags |= PGR_CAM_IS_TRANSMITTING;

	//  Sleep untill the camera has a transmission
	dc1394switch_t status = DC1394_OFF;

	for ( i = 0; i <= 5; i++ ) {
		usleep(50000);
//advise("calling dc1394_video_get_transmission");
		if ( dc1394_video_get_transmission( pgcp->pc_cam_p, &status )
				!= DC1394_SUCCESS ) {
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
//advise("start_firewire_transmission DONE");

	// Now make sure that we have the frame objects...
	dp = dobj_of("_frame1");
	if( dp == NULL ) init_buffer_objects(QSP_ARG  pgcp);

	return(0);
}

static int ready_to_grab( QSP_ARG_DECL  PGR_Cam *pgcp )
{
	if( pgcp->pc_n_avail <= 0 ){
		WARN("grab_firewire_frame:  no available frames");
		advise("Need to release frames before grabbing.");
		return 0;
	}
	return 1;
}

static Data_Obj *dobj_for_frame(QSP_ARG_DECL  dc1394video_frame_t *framep)
{
	Data_Obj *dp;
	char fname[32];

	// For speed, we could keep a table of the dobj's associated with the camera.  BUG
	sprintf(fname,"_frame%d",framep->id);
	dp = get_obj(fname);
	return dp;
}

static void note_frame_usage(PGR_Cam *pgcp, dc1394video_frame_t *framep)
{
	Node *np;

	np=mk_node(framep);
	if( pgcp->pc_in_use_lp == NULL )
		pgcp->pc_in_use_lp = new_list();

	addHead(pgcp->pc_in_use_lp,np);
}

Data_Obj * grab_newest_firewire_frame( QSP_ARG_DECL  PGR_Cam * pgcp )
{
	dc1394video_frame_t *framep, *prev_framep=NULL;
	//int i=0;
	int n_dequeued=0;

	if( ! ready_to_grab( QSP_ARG  pgcp ) )
		return NULL;

	// We might want to release all of the frames we have now, in case we
	// need to automatically release any that we grab in the meantime,
	// so that we release in order...

	// We get the newest by dequeueing in POLL mode, until we come up empty.
	// If we have at least one frame at that time, then that's the frame.
	// If we don't have any, then we WAIT.  If at any time we have
	// more than 1, then we release the older.

	while( 1 ){
		if ( dc1394_capture_dequeue( pgcp->pc_cam_p, DC1394_CAPTURE_POLICY_POLL,
				&framep ) != DC1394_SUCCESS) {
			fprintf( stderr, "Unable to capture a frame\n" );
			return(NULL);
		}
		if( framep == NULL ){	// No frame to fetch?
			if( n_dequeued > 0 ){	// already have something?
				// The last one is the newest!
				sprintf(msg_str,"%d",prev_framep->id);
				assign_var("newest",msg_str);
				note_frame_usage(pgcp,prev_framep);
				return dobj_for_frame(QSP_ARG  prev_framep);
			} else {		// No frames yet...
				// We don't want to call the WAIT version here, because
				// we might have multiple cameras...
				return NULL;
			}
		} else {	// We have a new frame
			if( prev_framep != NULL ){	// already have one?
				if( dc1394_capture_enqueue(pgcp->pc_cam_p,prev_framep)
						!= DC1394_SUCCESS ){
					WARN("error enqueueing frame");
				}
			} else {
				// This counts the frame we dequeued.
				// We don't bother if we just enqueued
				// the previous one.
				pgcp->pc_n_avail--;
			}
			prev_framep = framep;
			n_dequeued++;
		}
	}
	// NOTREACHED
}

Data_Obj * grab_firewire_frame(QSP_ARG_DECL  PGR_Cam * pgcp )
{
	dc1394video_frame_t *framep;
	//dc1394capture_policy_t policy=DC1394_CAPTURE_POLICY_WAIT;
	Data_Obj *dp;
	char fname[TMPSIZE];

	// Before attempting to dequeue, make sure that we have at least one
	// available...  The library seems to hang if we keep
	// grabbing without releasing.

	if( ! ready_to_grab(QSP_ARG  pgcp) )
		return NULL;

	/* POLICY_WAIT waits for the next frame...
	 * POLICY_POLL returns right away if there is no frame available.
	 */
	if ( dc1394_capture_dequeue( pgcp->pc_cam_p, 
		pgcp->pc_policy, &framep ) != DC1394_SUCCESS) {
		fprintf( stderr, "Unable to capture a frame\n" );
		return(NULL);
	}
	if( framep == NULL ){
		if( pgcp->pc_policy != DC1394_CAPTURE_POLICY_POLL )
			WARN("dc1394_capture_dequeue returned a null frame.");
		return NULL;
	}

	pgcp->pc_n_avail--;

	//sprintf(fname,"_frame%d",framep->id);
	snprintf(fname,TMPSIZE,"_frame%d",framep->id);
	dp = get_obj(fname);
	if( dp == NULL ){
		warn("grab_firewire_frame:  unable to create frame object");
		return(NULL);
	}

	assert( OBJ_DATA_PTR(dp) == framep->image );

	/* in the other case, the pointer is likely to be unchanged,
	 * but we don't assume...
	 * We *do* assume that the old size is still ok.
	 */

	note_frame_usage(pgcp,framep);
	return(dp);
} // end grab_firewire_frame

void _release_oldest_frame(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	Node *np;
	dc1394video_frame_t *framep;

	// Only can do this if the camera is running...
	if( (pgcp->pc_flags & PGR_CAM_IS_RUNNING) == 0 ){
		warn("release_oldest_frame:  camera is not running!?");
		return;
	}

	if( pgcp->pc_n_avail == pgcp->pc_ring_buffer_size ){
		warn("release_oldest_frame:  no frames are currently dequeued");
		return;
	}

	if( pgcp->pc_in_use_lp == NULL ){
		warn("release_oldest_frame:  no frames have been grabbed");
		return;
	}

	np = remTail(pgcp->pc_in_use_lp);

	// This is CAUTIOUS because of the pc_n_avail test above...
	assert( np != NULL );

	framep = (dc1394video_frame_t *) np->n_data;

	if( dc1394_capture_enqueue(pgcp->pc_cam_p,framep)
		!= DC1394_SUCCESS ){
		warn("error enqueueing frame");
	}
	pgcp->pc_n_avail++;
}

int _stop_firewire_capture(QSP_ARG_DECL  PGR_Cam * pgcp )
{
	/* Transmission has to be stopped before capture is stopped,
	 * or the iso bandwidth will not be freed - jbm.
	 */

	if( ! IS_TRANSMITTING(pgcp) ){
		/* CAUTIOUS? */
		warn("stop_firewire_capture:  not transmitting!?");
		return(-1);
	} else {
		//  Stop data transmission
		if( dc1394_video_set_transmission( pgcp->pc_cam_p, DC1394_OFF ) != DC1394_SUCCESS ){
			warn("stop_firewire_capture:  Couldn't stop transmission!?");
			return(-1);
		}
		pgcp->pc_flags &= ~PGR_CAM_IS_TRANSMITTING;
	}

	if( ! IS_CAPTURING(pgcp) ){
		warn("stop_firewire_capture:  not capturing!?");
		return(-1);
	} else {
		dc1394_capture_stop( pgcp->pc_cam_p );
		pgcp->pc_flags &= ~PGR_CAM_IS_CAPTURING;
	}

	/* why free the camera here? */
	/*
	dc1394_free_camera( pgcp->pc_cam_p );
	*/


	pgcp->pc_flags &= ~PGR_CAM_IS_RUNNING;

	return(0);
}

int _reset_camera(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	if( /* dc1394_reset_camera */
		dc1394_camera_reset(pgcp->pc_cam_p) != DC1394_SUCCESS ){
		warn("Could not initilize camera");
		return -1;
	}
	return 0;
}


void list_trig(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	dc1394bool_t has_p;
	dc1394trigger_mode_t tm;

	if( dc1394_external_trigger_has_polarity(pgcp->pc_cam_p,&has_p) !=
		DC1394_SUCCESS ){
		warn("error querying for trigger polarity capability");
		return;
	}
	if( has_p ){
		prt_msg("external trigger has polarity");
	} else {
		prt_msg("external trigger does not have polarity");
	}

	if( dc1394_external_trigger_get_mode(pgcp->pc_cam_p,&tm) !=
		DC1394_SUCCESS ){
		warn("error querying trigger mode");
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
			warn("unrecognized trigger mode");
			break;
	}
}

void report_bandwidth(QSP_ARG_DECL  PGR_Cam *pgcp )
{
	unsigned int bw;

	if( dc1394_video_get_bandwidth_usage(pgcp->pc_cam_p,&bw) !=
		DC1394_SUCCESS ){
		warn("error querying bandwidth usage");
		return;
	}
	/* What are the units of bandwidth??? */
	sprintf(msg_str,"%s bandwidth:  %d",pgcp->pc_name,bw);
	prt_msg(msg_str);
}

void print_camera_info(QSP_ARG_DECL  PGR_Cam *pgcp)
{
	int i;

	prt_msg("\nCurrent camera:");

	i=index_of_video_mode(pgcp->pc_video_mode);
	sprintf(msg_str,"\tmode:  %s",all_video_modes[i].nvm_name);
	prt_msg(msg_str);

	i=index_of_framerate(pgcp->pc_framerate);
	sprintf(msg_str,"\trate:  %s",all_framerates[i].nfr_name);
	prt_msg(msg_str);

	report_camera_features(pgcp);
}

void describe_dc1394_error( QSP_ARG_DECL  dc1394error_t e )
{
	switch( e ){
		case DC1394_SUCCESS:
			advise("Success!"); break;
		case DC1394_FAILURE:
			advise("Failure!?"); break;
		case DC1394_NOT_A_CAMERA:
			advise("Not a camera."); break;
		case DC1394_FUNCTION_NOT_SUPPORTED:
			advise("Function not supported."); break;
		case DC1394_CAMERA_NOT_INITIALIZED:
			advise("Camera not initialized."); break;
		case DC1394_MEMORY_ALLOCATION_FAILURE:
			advise("Memory allocation failure."); break;
		case DC1394_TAGGED_REGISTER_NOT_FOUND:
			advise("Tagged register not found."); break;
		case DC1394_NO_ISO_CHANNEL:
			advise("No ISO channel."); break;
		case DC1394_NO_BANDWIDTH:
			advise("No bandwidth."); break;
		case DC1394_IOCTL_FAILURE:
			advise("Ioctl failure."); break;
		case DC1394_CAPTURE_IS_NOT_SET:
			advise("Capture is not set."); break;
		case DC1394_CAPTURE_IS_RUNNING:
			advise("Capture is running."); break;
		case DC1394_RAW1394_FAILURE:
			advise("Raw 1394 failure."); break;
		case DC1394_FORMAT7_ERROR_FLAG_1:
			advise("Format 7 error flag 1."); break;
		case DC1394_FORMAT7_ERROR_FLAG_2:
			advise("Format 7 error flag 2."); break;
		case DC1394_INVALID_ARGUMENT_VALUE:
			advise("Invalid argument value."); break;
		case DC1394_REQ_VALUE_OUTSIDE_RANGE:
			advise("Req. value outside range."); break;
		case DC1394_INVALID_FEATURE:
			advise("Invalid feature."); break;
		case DC1394_INVALID_VIDEO_FORMAT:
			advise("Invalid video format."); break;
		case DC1394_INVALID_VIDEO_MODE:
			advise("Invalid video mode."); break;
		case DC1394_INVALID_FRAMERATE:
			advise("Invalid frame rate."); break;
		case DC1394_INVALID_TRIGGER_MODE:
			advise("Invalid trigger mode."); break;
		case DC1394_INVALID_TRIGGER_SOURCE:
			advise("Invalid trigger source."); break;
		case DC1394_INVALID_ISO_SPEED:
			advise("Invalid ISO speed."); break;
		case DC1394_INVALID_IIDC_VERSION:
			advise("Invalid IIDC version."); break;
		case DC1394_INVALID_COLOR_CODING:
			advise("Invalid color coding."); break;
		case DC1394_INVALID_COLOR_FILTER:
			advise("Invalid color filter."); break;
		case DC1394_INVALID_CAPTURE_POLICY:
			advise("Invalid capture policy."); break;
		case DC1394_INVALID_ERROR_CODE:
			advise("Invalid error code."); break;
		case DC1394_INVALID_BAYER_METHOD:
			advise("Invalid Bayer method."); break;
		case DC1394_INVALID_VIDEO1394_DEVICE:
			advise("Invalid video1394 device."); break;
		case DC1394_INVALID_OPERATION_MODE:
			advise("Invalid operation mode."); break;
		case DC1394_INVALID_TRIGGER_POLARITY:
			advise("Invalid trigger polarity."); break;
		case DC1394_INVALID_FEATURE_MODE:
			advise("Invalid feature mode."); break;
		case DC1394_INVALID_LOG_TYPE:
			advise("Invalid log type."); break;
		case DC1394_INVALID_BYTE_ORDER:
			advise("Invalid byte order."); break;
		case DC1394_INVALID_STEREO_METHOD:
			advise("Invalid stereo method."); break;
		case DC1394_BASLER_NO_MORE_SFF_CHUNKS:
			advise("No more SFF chunks (Baseler)."); break;
		case DC1394_BASLER_CORRUPTED_SFF_CHUNK:
			advise("Corrupted SFF chunk (Baseler)."); break;
		case DC1394_BASLER_UNKNOWN_SFF_CHUNK:
			advise("Unknown SFF chunk (Baseler)."); break;
		default:
			sprintf(ERROR_STRING,"describe_dc1394_error:  unhandled error code %d!?",e);
			advise(ERROR_STRING);
			break;
	}
}

#endif /* HAVE_LIBDC1394 */

