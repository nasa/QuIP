#include "quip_config.h"

/* This code must have been cribbed from coriander? */

#include "quip_prot.h"
#include "pgr.h"

#ifdef HAVE_LIBDC1394
int set_camera_framerate(PGR_Cam *pgcp, dc1394framerate_t framerate )
{
	/*
	IsoFlowCheck(&state);
	*/
	if( dc1394_video_set_framerate(pgcp->pc_cam_p, framerate ) != DC1394_SUCCESS ){
		NWARN("Could not set framerate");
		return -1;
	}
	/*
	IsoFlowResume(&state);
	UpdateFeatureWindow();	// because several controls may change,
				// especially exposure, gamma, shutter,
				// ... since the framerate changes.
	*/
	return 0;
}
#endif


int power_on_camera(PGR_Cam *pgcp)
{
#ifdef HAVE_LIBDC1394
	if( dc1394_camera_set_power(pgcp->pc_cam_p, DC1394_ON) != DC1394_SUCCESS ){
		NWARN("Could not set camera 'on'");
		return(-1);
	}
#endif
	return 0;
}


int power_off_camera(PGR_Cam *pgcp)
{
#ifdef HAVE_LIBDC1394
	if( dc1394_camera_set_power(pgcp->pc_cam_p, DC1394_OFF) != DC1394_SUCCESS ){
		NWARN("Could not set camera 'off'");
		return -1;
	}
#endif
	return 0;
}




#ifdef HAVE_LIBDC1394
int set_camera_trigger_polarity(PGR_Cam *pgcp,
	dc1394trigger_polarity_t polarity)
{
	if( dc1394_external_trigger_set_polarity(pgcp->pc_cam_p,polarity) != DC1394_SUCCESS ){
		NWARN("Cannot set trigger polarity");
		return -1;
	}
	return 0;
		/*
		camera->feature_set.feature[DC1394_FEATURE_TRIGGER-DC1394_FEATURE_MIN].trigger_polarity=(int)togglebutton->active;
		*/
}



int set_camera_trigger_mode(PGR_Cam *pgcp, dc1394trigger_mode_t mode)
{
	if( dc1394_external_trigger_set_mode(pgcp->pc_cam_p, mode) != DC1394_SUCCESS ){
		NWARN("Could not set trigger mode");
		return -1;
	}
	return 0;
	/*
		camera->feature_set.feature[DC1394_FEATURE_TRIGGER-DC1394_FEATURE_MIN].trigger_mode=(int)user_data;
	UpdateTriggerFrame();
*/

}


int set_camera_trigger_source(PGR_Cam *pgcp, dc1394trigger_source_t source)
{
	if( dc1394_external_trigger_set_source(pgcp->pc_cam_p, source) != DC1394_SUCCESS ){
		NWARN("Could not set trigger source");
		return -1;
	}
	return 0;
	/*
	camera->feature_set.feature[DC1394_FEATURE_TRIGGER-DC1394_FEATURE_MIN].trigger_source=(int)user_data;
	UpdateTriggerFrame();
	*/
}
#endif /* HAVE_LIBDC1394 */

int set_camera_temperature(PGR_Cam *pgcp, int temp)
{
#ifdef HAVE_LIBDC1394
	if( dc1394_feature_temperature_set_value(pgcp->pc_cam_p,temp) != DC1394_SUCCESS ){
		//NWARN("set_camera_temperature not implemented");
		return -1;
	}
#endif
	return 0;
}

int set_camera_white_balance(PGR_Cam *pgcp, int wb)
{
#ifdef HAVE_LIBDC1394
	if( dc1394_feature_whitebalance_set_value(pgcp->pc_cam_p,wb, 0 /* old value? */ ) != DC1394_SUCCESS ){
		NWARN("Could not set B/U white balance");
		return -1;
	}
#endif
	return 0;
}

int set_camera_white_shading(PGR_Cam *pgcp, int val)
{
#ifdef HAVE_LIBDC1394
	if( dc1394_feature_whiteshading_set_value(pgcp->pc_cam_p,val,0,0/*BUG what are these args*/)!=DC1394_SUCCESS){
		NWARN("Could not set white shading / blue channel");
		return -1;
	}
#endif
	return 0;
}


#ifdef HAVE_LIBDC1394
int set_iso_speed(PGR_Cam *pgcp, dc1394speed_t speed)
{
fprintf(stderr,"set_iso_speed:  speed = 0x%lx\n",(long)speed);
	if( dc1394_video_set_iso_speed(pgcp->pc_cam_p, speed) != DC1394_SUCCESS ){
		//NWARN("error setting ISO speed");
		return -1;
	}
	return 0;
}
#endif


int set_camera_bmode(PGR_Cam *pgcp, int bmode_true)
{
#ifdef HAVE_LIBDC1394
	if( bmode_true ){
		//fprintf(stderr,"setting 1394b\n");
		if( dc1394_video_set_operation_mode(pgcp->pc_cam_p, DC1394_OPERATION_MODE_1394B) != DC1394_SUCCESS ){
			NWARN("unable to set 1394b mode");
			return -1;
		}
		return 0;
	} else {
		dc1394speed_t iso_speed;
		if( dc1394_video_get_iso_speed(pgcp->pc_cam_p, &iso_speed ) != DC1394_SUCCESS ){
			NWARN("error getting iso speed");
			return -1;
		}
		if (iso_speed>DC1394_ISO_SPEED_400){
			if( dc1394_video_set_iso_speed(pgcp->pc_cam_p,DC1394_ISO_SPEED_400) != DC1394_SUCCESS ){
				NWARN("unable to set 1394 400 mode");
				return -1;
			}
			return 0;
		}
		/* old camera */
		if( dc1394_video_set_operation_mode(pgcp->pc_cam_p, DC1394_OPERATION_MODE_LEGACY) != DC1394_SUCCESS ){
			NWARN("unable to set 1394 legacy mode");
			return -1;
		}
	}
#endif
	return 0;
}

#ifdef HAVE_LIBDC1394
int is_auto_capable( dc1394feature_info_t *feat_p )
{
	dc1394feature_modes_t *mp;
	int i;

#ifdef CAUTIOUS
	if( feat_p == NULL ){
		NWARN("CAUTIOUS:  is_auto_capable:  null feature pointer!?");
		return 0;
	}
#endif // CAUTIOUS

	mp = &(feat_p->modes);

	for( i = 0; i < mp->num ; i++ ){
		if( mp->modes[i] == DC1394_FEATURE_MODE_AUTO ) return 1;
	}
	return 0;
}
#endif // HAVE_LIBDC1394

