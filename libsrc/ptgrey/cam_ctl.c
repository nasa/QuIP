#include "quip_config.h"

char VersionId_ptgrey_cam_ctl[] = QUIP_VERSION_STRING;

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

/*
void on_trigger_external_toggled (GtkToggleButton *togglebutton,
					gpointer user_data)
{
#ifdef HAVE_LIBDC1394
	if (dc1394_feature_set_power(camera->camera_info, DC1394_FEATURE_TRIGGER, togglebutton->active)!=DC1394_SUCCESS)
		Error("Could not set external trigger source");
	else
		camera->feature_set.feature[DC1394_FEATURE_TRIGGER-DC1394_FEATURE_MIN].is_on=togglebutton->active;
	UpdateTriggerFrame();
#endif
}
*/


/*
void on_load_mem_clicked(GtkButton *button, gpointer user_data)
{
#ifdef HAVE_LIBDC1394
	if (dc1394_memory_load(camera->camera_info, camera->memory_channel)!=DC1394_SUCCESS)
		Error("Cannot load memory channel");
	UpdateAllWindows();

#endif
}
*/

/*
void on_save_mem_clicked(GtkButton *button, gpointer user_data)
{
#ifdef HAVE_LIBDC1394
	unsigned long int timeout_bin=0;
	unsigned long int step;
	dc1394bool_t value=TRUE;
	step=(unsigned long int)(1000000.0/preferences.auto_update_frequency);

	if (dc1394_memory_save(camera->camera_info, camera->memory_channel)!=DC1394_SUCCESS)
		Error("Could not save setup to memory channel");
	else {
		while ((value==DC1394_TRUE) &&(timeout_bin<(unsigned long int)(preferences.op_timeout*1000000.0)) ) {
			usleep(step);
			if (dc1394_memory_is_save_in_operation(camera->camera_info, &value)!=DC1394_SUCCESS)
	Error("Could not query if memory save is in operation");
			timeout_bin+=step;
		}
		if (timeout_bin>=(unsigned long int)(preferences.op_timeout*1000000.0))
			Warning("Save operation function timed-out!");
	}
#endif
}
*/

/*
void on_iso_start_clicked(GtkButton *button, gpointer user_data)
{
#ifdef HAVE_LIBDC1394
	dc1394switch_t status;

	if (dc1394_video_set_transmission(camera->camera_info,DC1394_ON)!=DC1394_SUCCESS)
		Error("Could not start ISO transmission");
	else {
		usleep(DELAY);
		if (dc1394_video_get_transmission(camera->camera_info, &status)!=DC1394_SUCCESS)
			Error("Could get ISO status");
		else {
			if (status==DC1394_FALSE) {
	Error("ISO transmission refuses to start");
			}
			camera->camera_info->is_iso_on=status;
			UpdateIsoFrame();
		}
	}
	UpdateTransferStatusFrame();
#endif
}
*/


/*
void on_iso_stop_clicked(GtkButton *button, gpointer user_data)
{
#ifdef HAVE_LIBDC1394
	dc1394switch_t status;

	if (dc1394_video_set_transmission(camera->camera_info,DC1394_OFF)!=DC1394_SUCCESS)
		Error("Could not stop ISO transmission");
	else {
		usleep(DELAY);
		if (dc1394_video_get_transmission(camera->camera_info, &status)!=DC1394_SUCCESS)
			Error("Could get ISO status");
		else {
			if (status==DC1394_TRUE) {
	Error("ISO transmission refuses to stop");
			}
			camera->camera_info->is_iso_on=status;
			UpdateIsoFrame();
		}
	}
	UpdateTransferStatusFrame();
#endif
}
*/


int set_camera_temperature(PGR_Cam *pgcp, int temp)
{
#ifdef HAVE_LIBDC1394
	if( dc1394_feature_temperature_set_value(pgcp->pc_cam_p,temp) != DC1394_SUCCESS ){
		NWARN("set_camera_temperature not implemented");
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
/*
	case DC1394_FEATURE_WHITE_BALANCE+RV: // why oh why is there a *4?
		if (dc1394_feature_whitebalance_set_value(camera->camera_info, camera->feature_set.feature[DC1394_FEATURE_WHITE_BALANCE-DC1394_FEATURE_MIN].BU_value, adj->value)!=DC1394_SUCCESS)
			Error("Could not set R/V white balance");
		else {
			camera->feature_set.feature[DC1394_FEATURE_WHITE_BALANCE-DC1394_FEATURE_MIN].RV_value=adj->value;
			if (camera->feature_set.feature[DC1394_FEATURE_WHITE_BALANCE-DC1394_FEATURE_MIN].absolute_capable!=0) {
	GetAbsValue(DC1394_FEATURE_WHITE_BALANCE);
			}
		}
		break;
		*/

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
	if( dc1394_video_set_iso_speed(pgcp->pc_cam_p, speed) != DC1394_SUCCESS ){
		NWARN("error setting ISO speed");
		return -1;
	}
	return 0;
}
#endif


/*
void on_range_menu_activate(GtkMenuItem *menuitem, gpointer user_data)
{
#ifdef HAVE_LIBDC1394
	int feature;
	int action;

	// single auto variables:
	unsigned long int timeout_bin=0;
	unsigned long int step;
	dc1394feature_mode_t value=DC1394_FEATURE_MODE_ONE_PUSH_AUTO;

	action=((int)(unsigned long)user_data)%1000;
	feature=(((int)(unsigned long)user_data)-action)/1000;

	switch (action) {
	case RANGE_MENU_OFF : // ============================== OFF ==============================
		if (dc1394_feature_set_power(camera->camera_info, feature, DC1394_OFF)!=DC1394_SUCCESS)
			Error("Could not set feature on/off");
		else {
			camera->feature_set.feature[feature-DC1394_FEATURE_MIN].is_on=FALSE;
			UpdateRange(feature);
		}
		break;
	case RANGE_MENU_MAN : // ============================== MAN ==============================
			if (camera->feature_set.feature[feature-DC1394_FEATURE_MIN].on_off_capable) {
	if (dc1394_feature_set_power(camera->camera_info, feature, DC1394_ON)!=DC1394_SUCCESS) {
		Error("Could not set feature on");
		break;
	}
	else
		camera->feature_set.feature[feature-DC1394_FEATURE_MIN].is_on=TRUE;
			}
			if (dc1394_feature_set_mode(camera->camera_info, feature, DC1394_FEATURE_MODE_MANUAL)!=DC1394_SUCCESS)
	Error("Could not set manual mode");
			else {
	camera->feature_set.feature[feature-DC1394_FEATURE_MIN].auto_active=FALSE;
	if (camera->feature_set.feature[feature-DC1394_FEATURE_MIN].absolute_capable)
		SetAbsoluteControl(feature, FALSE);
	UpdateRange(feature);
			}
			break;
	case RANGE_MENU_AUTO : // ============================== AUTO ==============================
		if (camera->feature_set.feature[feature-DC1394_FEATURE_MIN].on_off_capable) {
			if (dc1394_feature_set_power(camera->camera_info, feature, DC1394_ON)!=DC1394_SUCCESS) {
	Error("Could not set feature on");
	break;
			}
			else
	camera->feature_set.feature[feature-DC1394_FEATURE_MIN].is_on=TRUE;
		}
		if (dc1394_feature_set_mode(camera->camera_info, feature, DC1394_FEATURE_MODE_AUTO)!=DC1394_SUCCESS)
			Error("Could not set auto mode");
		else {
			camera->feature_set.feature[feature-DC1394_FEATURE_MIN].auto_active=TRUE;
			if (camera->feature_set.feature[feature-DC1394_FEATURE_MIN].absolute_capable)
	SetAbsoluteControl(feature, FALSE);
			UpdateRange(feature);
		}
		break;
		case RANGE_MENU_SINGLE : // ============================== SINGLE ==============================
			if (camera->feature_set.feature[feature-DC1394_FEATURE_MIN].on_off_capable) {
	if (dc1394_feature_set_power(camera->camera_info, feature, DC1394_ON)!=DC1394_SUCCESS) {
		Error("Could not set feature on");
		break;
	}
	else
		camera->feature_set.feature[feature-DC1394_FEATURE_MIN].is_on=TRUE;
			}
			step=(unsigned long int)(1000000.0/preferences.auto_update_frequency);
			if (dc1394_feature_set_mode(camera->camera_info, feature, DC1394_FEATURE_MODE_ONE_PUSH_AUTO)!=DC1394_SUCCESS)
	Error("Could not start one-push operation");
			else {
	SetScaleSensitivity(GTK_WIDGET(menuitem),feature,FALSE);
	while ((value==DC1394_FEATURE_MODE_ONE_PUSH_AUTO) && (timeout_bin<(unsigned long int)(preferences.op_timeout*1000000.0)) ) {
		usleep(step);
		if (dc1394_feature_get_mode(camera->camera_info, feature, &value)!=DC1394_SUCCESS)
			Error("Could not query one-push operation");
		timeout_bin+=step;
		UpdateRange(feature);
	}
	if (timeout_bin>=(unsigned long int)(preferences.op_timeout*1000000.0))
		Warning("One-Push function timed-out!");

	if (camera->feature_set.feature[feature-DC1394_FEATURE_MIN].absolute_capable)
		SetAbsoluteControl(feature, FALSE);
	UpdateRange(feature);
	// should switch back to manual mode here. Maybe a recursive call??
	// >> Not necessary because UpdateRange reloads the status which folds
	// back to 'man' in the camera
			}
			break;
	case RANGE_MENU_ABSOLUTE : // ============================== ABSOLUTE ==============================
		if (camera->feature_set.feature[feature-DC1394_FEATURE_MIN].on_off_capable) {
			if (dc1394_feature_set_power(camera->camera_info, feature, TRUE)!=DC1394_SUCCESS) {
	Error("Could not set feature on");
	break;
			}
			else
	camera->feature_set.feature[feature-DC1394_FEATURE_MIN].is_on=TRUE;
		}
		SetAbsoluteControl(feature, TRUE);
		UpdateRange(feature);
		break;
	}
#endif
}
*/

/*
void on_global_iso_stop_clicked(GtkButton *button, gpointer user_data)
{
#ifdef HAVE_LIBDC1394
	dc1394switch_t status;
	camera_t* camera_ptr;
	camera_ptr=cameras;

	while (camera_ptr!=NULL) {
		if (dc1394_video_get_transmission(camera_ptr->camera_info, &camera_ptr->camera_info->is_iso_on)!=DC1394_SUCCESS) {
			Error("Could not get ISO status");
		}
		if (camera_ptr->camera_info->is_iso_on==DC1394_TRUE) {
			if (dc1394_video_set_transmission(camera_ptr->camera_info, DC1394_OFF)!=DC1394_SUCCESS) {
	Error("Could not stop ISO transmission");
			}
			else {
	if (dc1394_video_get_transmission(camera_ptr->camera_info, &status)!=DC1394_SUCCESS)
		Error("Could not get ISO status");
	else {
		if (status==DC1394_TRUE)
			Error("Broacast ISO stop failed for a camera");
		else
			camera_ptr->camera_info->is_iso_on=DC1394_FALSE;
	}
			}
		}
		camera_ptr=camera_ptr->next;
	}

	UpdateIsoFrame();
	UpdateTransferStatusFrame();
#endif
}

void on_global_iso_start_clicked(GtkButton *button, gpointer user_data)
{
#ifdef HAVE_LIBDC1394
	dc1394switch_t status;
	camera_t* camera_ptr;
	camera_ptr=cameras;

	while (camera_ptr!=NULL) {
		if (dc1394_video_get_transmission(camera_ptr->camera_info, &camera_ptr->camera_info->is_iso_on)!=DC1394_SUCCESS) {
			Error("Could not get ISO status");
		}
		if (camera_ptr->camera_info->is_iso_on==DC1394_FALSE) {
			if (dc1394_video_set_transmission(camera_ptr->camera_info, DC1394_ON)!=DC1394_SUCCESS) {
				Error("Could not start ISO transmission");
			} else {
				if( dc1394_video_get_transmission(
			camera_ptr->camera_info, &status)!=DC1394_SUCCESS)
					Error("Could not get ISO status");
				else {
					if (status==DC1394_FALSE)
				Error("Broacast ISO start failed for a camera");
					else
						camera_ptr->camera_info->is_iso_on=DC1394_TRUE;
				}
			}
		}
		camera_ptr=camera_ptr->next;
	}

	UpdateIsoFrame();
	UpdateTransferStatusFrame();
#endif
}
*/

/*
void on_broadcast_button_toggled(GtkToggleButton *togglebutton, gpointer user_data)
{
#ifdef HAVE_LIBDC1394
	if(togglebutton->active>0) {
		dc1394_camera_set_broadcast(camera->camera_info, DC1394_ON);
	}
	else {
		dc1394_camera_set_broadcast(camera->camera_info, DC1394_OFF);
	}
	camera->prefs.broadcast=togglebutton->active;
	gnome_config_set_int("coriander/global/broadcast",camera->prefs.broadcast);
	gnome_config_sync();
#endif
}
*/


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

