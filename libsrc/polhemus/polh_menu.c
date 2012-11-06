#include "quip_config.h"

char VersionId_polhemus_polh_menu[] = QUIP_VERSION_STRING;
#include <stdio.h>
#include <string.h>

#include "dataprot.h"
#include "debug.h"
#include "version.h"

#include "polh_dev.h"
#include "ioctl_polhemus.h"
#include "polh_menu.h"

#ifdef DEBUG
debug_flag_t debug_polhemus;
#endif /* DEBUG */

typedef enum { CM, INCHES } Units;

/* set to centimeters intially because the system is in centimeters upon startup */
static Units unit = CM;		

static Data_Obj *tmp_pt_dp=NO_OBJ;


/* This defines the number of numbers in a reading...  The polhemus can report more or fewer
 * data, but for now we hard code this to the default, xyz + euler angles.
 */
#define POLHEMUS_READING_COUNT	6


/* Make a temporary dp to hold the current point... */
#define INSURE_TMP_PT										\
												\
	if( tmp_pt_dp == NO_OBJ ){								\
		/* BUG?  should we get the precision based on the current format? */		\
		tmp_pt_dp = mk_vec("tmp_polhemus_pt",1,POLHEMUS_READING_COUNT,PREC_SP);		\
		if( tmp_pt_dp == NO_OBJ ) error1("error creating temporary polhemus point");	\
	}

Output_Datum od_tbl[N_OUTPUT_TYPES]={
/*	name		our code	code	nbytes	nstrings	*/
{	"station",	STATION,	-1,	0,	1	},
{	"date",		DATE,		-1,	2,	1	},
{	"seconds",	SECONDS,	-1,	2,	1	},
{	"milliseconds",	MSECS,		-1,	2,	1	},
{	"xyz_int",	XYZ_INT,	2,	3,	3	},
{	"euler_int",	EULER_INT,	4,	3,	3	},
{	"x_dir_int",	X_DIR_INT,	5,	3,	3	},
{	"y_dir_int",	Y_DIR_INT,	6,	3,	3	},
{	"z_dir_int",	Z_DIR_INT,	7,	3,	3	},
{	"quat_int",	QUAT_INT,	11,	4,	4	},
{	"xyz_flt",	XYZ_FLT,	52,	6,	3	},
{	"euler_flt",	EULER_FLT,	54,	6,	3	},
{	"x_dir_flt",	X_DIR_FLT,	55,	6,	3	},
{	"y_dir_flt",	Y_DIR_FLT,	56,	6,	3	},
{	"z_dir_flt",	Z_DIR_FLT,	57,	6,	3	},
{	"quat_flt",	QUAT_FLT,	61,	8,	4	}
};

static char *od_names[N_OUTPUT_TYPES];

static COMMAND_FUNC( do_reset_align )
{
	if( send_polh_cmd(PH_RESET_ALIGNMENT,NULL) < 0 ) 
		warn("Unable to reset polhemus alignment!");
}

static COMMAND_FUNC( do_set_curr_align )
{
	char align[LLEN];
	//short pdp[2*N_OUTPUT_TYPES];	
	//Fmt_Pt fdp;

	do_reset_align(SINGLE_QSP_ARG);

	INSURE_TMP_PT
	
	/* read the current data point */
	if( read_single_polh_dp(tmp_pt_dp) < 0 ) {
		warn("do_set_curr_align: error reading single polhemus data point");
		return;
	}

	/* BUG need to make sure that xyz is measured */

/*
	format_data(&fdp,tmp_pt_dp, &station_info[curr_station].sd_single_prf);
	*/

	/* what units are these supposed to be in??? */

	/*
	sprintf(align, "%3.2f,%3.2f,%3.2f,%3.2f,0,0,0,%3.2f,0,0,0,%3.2f",
		pdp[ixyz], pdp[ixyz+1], pdp[ixyz+2], pdp[ixyz], pdp[ixyz+1], pdp[ixyz+2]); 
		*/
	error1("need to fix alignment code");


	if(send_polh_cmd(PH_ALIGNMENT, align) < 0) warn("Unable to set polhemus alignment!");	
}

static COMMAND_FUNC( do_set_align )
{
	char align[LLEN];
	float Ox, Oy, Oz, Xx, Xy, Xz, Yx, Yy, Yz;

	/* We have to reset the alignment before setting it
	 * to a new value (Polhemus manual A-12).
	 */
	do_reset_align(SINGLE_QSP_ARG);

	/* FIXME - we use how_many() but we would like to let
	 * the user only enter values for the alignment 
	 * points they would like to change.
	 */

	Ox = (float)HOW_MUCH("Ox - origin");
	Oy = (float)HOW_MUCH("Oy - origin");
	Oz = (float)HOW_MUCH("Oz - origin");
	Xx = (float)HOW_MUCH("Xx - positive direction of X-axis");
	Xy = (float)HOW_MUCH("Xy - positive direction of X-axis");
	Xz = (float)HOW_MUCH("Xz - positive direction of X-axis");
	Yx = (float)HOW_MUCH("Yx - positive Y direction from X-axis");
	Yy = (float)HOW_MUCH("Yy - positive Y direction from X-axis");
	Yz = (float)HOW_MUCH("Yz - positive Y direction from X-axis");

	/* FIXME - We should check these values to ensure that they are reasonable ... */
	sprintf(align, "%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f", 
		Ox, Oy, Oz, Xx, Xy, Xz, Yx, Yy, Yz);
	
	if(send_polh_cmd(PH_ALIGNMENT, align) < 0) warn("Unable to set polhemus alignment!");	
}

static COMMAND_FUNC( do_get_align )
{
	if( get_polh_info(PH_ALIGNMENT,NULL) < 0 ) 
		warn("Unable to get current polhemus alignment!");
}

/* Polhemus manual (pg. A-15) - The manual says that if the alignment
 * command had been previously invoked then the results of the
 * boresight are unpredictable ... 
 */

static COMMAND_FUNC( do_set_bore )
{
	if( send_polh_cmd(PH_BORESIGHT,NULL) < 0 ) 
		warn("Unable to set polhemus boresight to station line sight values!");
}

static COMMAND_FUNC( do_set_ref_bore )
{
	if( SET_POLH_ANGLES(PH_REF_BORESIGHT) < 0 ) 
		warn("Unable to set boresight reference angles!");
}

static COMMAND_FUNC( do_set_curr_ref_bore )
{
	char bore[LLEN];
	//short pdp[2*N_OUTPUT_TYPES];	/* conservative overestimate */
	Fmt_Pt fp1;
	
	INSURE_TMP_PT

	/* read the current data point */
	if( read_single_polh_dp(tmp_pt_dp) < 0 ) {
		warn("do_set_curr_align: error reading single polhemus data point");
		return;
	}

	/* BUG make sure we have euler angles */

	/*
	format_data(&fp1,tmp_pt_dp, &station_info[curr_station].sd_single_prf );
	*/
	sprintf(bore, "%3.2f %3.2f %3.2f", fp1.fp_azim, fp1.fp_elev, fp1.fp_roll);

	if( send_polh_cmd(PH_REF_BORESIGHT, bore) < 0 ) 
		warn("Unable to set boresight reference angles!");
}

static COMMAND_FUNC( do_get_ref_bore )
{
	if( get_polh_info(PH_REF_BORESIGHT,NULL) < 0 ) 
		warn("Unable to get boresight reference angles!");
}

static COMMAND_FUNC( do_set_trans )
{
	if( SET_POLH_ANGLES(PH_XMTR_ANGLES) < 0 ) 
		warn("Unable to set transmitter mount frame angles!");
}

static COMMAND_FUNC( do_get_trans )
{
	if( get_polh_info(PH_XMTR_ANGLES,NULL) < 0 ) 
		warn("Unable to get transmitter mount frame angles!");
}

static COMMAND_FUNC( do_set_recv_bore )
{
	/* Polhemus manual (pg. A21) - The manual
	 * says that we can choose which receiver
	 * but since we only have one we don't
	 * need to choose.
	 */

	if( SET_POLH_ANGLES(PH_RECV_ANGLES) < 0 ) 
		warn("Unable to set receiver boresight angles!");
}

static COMMAND_FUNC( do_get_recv_bore )
{
	if( get_polh_info(PH_RECV_ANGLES,NULL) < 0 ) 
		warn("Unable to get receiver boresight angles!");
}

static COMMAND_FUNC( do_reset_bore )
{
	if( send_polh_cmd(PH_RESET_BORESIGHT,NULL) < 0 ) 
		warn("Unable to reset system boresight!");
}
static COMMAND_FUNC( do_get_angl )
{
	if( get_polh_info(PH_ANGULAR_ENV,NULL) < 0 ) 
		warn("Unable to get angular operational envelope!");
}

/* These values are from Polhmemus manual A-41. */
#define MAX_IN_COORD	(78.74)
#define MIN_IN_COORD	(-78.74)
#define MAX_CM_COORD	(200.0)
#define MIN_CM_COORD	(-200.0)

static COMMAND_FUNC( do_set_post )
{
	char post[LLEN];
	float max, min;
	float xmax, ymax, zmax, xmin, ymin, zmin;

	/* We have different max and min values depending
	 * on what units we are using
	 */

	if( unit == CM ) {
		max = MAX_CM_COORD;
		min = MIN_CM_COORD;
	} else if( unit == INCHES ) {
		max = MAX_IN_COORD;
		min = MIN_IN_COORD;
	} else {
		warn("unknown unit conversion for system");
		return;
	}
	
	if( (ASK_ENV("x-coordinate","positional operational envelope", &xmax, &xmin, max, min) < 0)
	    || (ASK_ENV("y-coordinates","positional operational envelope", &ymax, &ymin, max, min) < 0)
	    || (ASK_ENV("z-coordinates","positional operational envelope", &zmax, &zmin, max, min) < 0)
	  ){
		return;
	}

	sprintf(post, "%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f", xmax, ymax, zmax, xmin, ymin, zmin);

	if( send_polh_cmd(PH_POSITIONAL_ENV, post) < 0 ) 
		warn("Unable to set positional operational envelope");
}

static COMMAND_FUNC( do_get_post )
{
	/* FIXME - doesn't seem to be giving right coordinates */
	if( get_polh_info(PH_POSITIONAL_ENV,NULL) < 0 ) 
		warn("Unable to get position operational envelope!");
}

/* Polhemus manual A-42 */
#define MAX_HEMI_VALUE		1
#define MIN_HEMI_VALUE		-1
#define N_HEMI_COMPS		3

static COMMAND_FUNC( do_set_hemi )
{
	char hemi[LLEN];
	float vecs[N_HEMI_COMPS];
	char *ask_strs[N_HEMI_COMPS] = { "x-component", "y-component", "z-component" };
	int i=0;

	for(i=0;i<N_HEMI_COMPS;i++){
		sprintf(msg_str,"%s of vector in direction of hemisphere", ask_strs[i]);
		vecs[i]=(float)HOW_MUCH(msg_str);
		if( vecs[i] < MIN_HEMI_VALUE || vecs[i] > MAX_HEMI_VALUE ){
			sprintf(error_string,
			"bad %s %f specified, value must be between %d and %d",
			ask_strs[i], vecs[i], MIN_HEMI_VALUE, MAX_HEMI_VALUE);
			warn(error_string);
			return;
		}		
	}

	sprintf(hemi, "%1.2f,%1.2f,%1.2f", vecs[0], vecs[1], vecs[2]);

	if( send_polh_cmd(PH_HEMISPHERE, hemi) < 0 ) 
		warn("Unable to set polhemus operational hemisphere!");
}

static COMMAND_FUNC( do_read_raw_vector )
{
	Data_Obj *dp;

	dp = PICK_OBJ("");
	if(dp == NULL) return;

	/* BUG? where do we check that the vector is of proper type and shape? */

	read_polh_vector(dp);
}

static COMMAND_FUNC( do_next_read )
{
	Data_Obj *dp;

	dp = PICK_OBJ("data object for single polhemus record");
	if( dp == NO_OBJ ) return;

	if( read_next_polh_dp(dp) < 0 ) {
		warn("do_single_read: error reading single polhemus data point");
		return;
	}
}

static COMMAND_FUNC( do_cont_read )
{
	Data_Obj *dp;

	dp = PICK_OBJ("data object for continuous polhemus data acquisition");
	if( dp == NO_OBJ ) return;

	if( read_cont_polh_dp(dp) < 0 ) {
		warn("do_single_read: error reading polhemus continuously");
		return;
	}
}

static COMMAND_FUNC( do_single_read )
{
	Data_Obj *dp;

	dp = PICK_OBJ("data object for single polhemus record");

	if( read_single_polh_dp(dp) < 0 ) {
		warn("do_single_read: error reading single polhemus data point");
		return;
	}


#ifdef FOOBAR
	for(station=0;station<2;station++){
		if( STATION_IS_ACTIVE(station) ){
			/* BUG for multiple stations, need to index dp... */
			format_data(&fp1,dp,&station_info[station].sd_single_prf);
			display_formatted_point(&fp1,&station_info[station].sd_single_prf);
		}
	}
#endif /* FOOBAR */
}

static COMMAND_FUNC( do_fmt_raw_vector )
{
	Data_Obj *dp;

	dp = PICK_OBJ("polhemus data vector");

	if( ! good_polh_vector(dp) ) return;

	/*
	format_polh_vector(dp);
	*/
}

static COMMAND_FUNC( do_cvt_raw_vector )
{
	Data_Obj *fdp, *pdp;

	fdp = PICK_OBJ("float data vector");
	pdp = PICK_OBJ("polhemus data vector");

	if( ! good_polh_vector(pdp) ) return;

	/*
	convert_polh_vector(fdp,pdp);
	*/
}

static Polh_Output_Type get_record_type(SINGLE_QSP_ARG_DECL)
{
	int i;

	/* BUG this only needs to be done once... */
	for(i=0;i<N_OUTPUT_TYPES;i++){
		od_names[i] = od_tbl[i].od_name;
#ifdef CAUTIOUS
		if( od_tbl[i].od_type != i ){
			sprintf(error_string,
				"CAUTIOUS:  Output data table entry %d has type code %d!?",
				i,od_tbl[i].od_type);
			error1(error_string);
		}
#endif /* CAUTIOUS */
	}

	/* this only works because the type codes are the same as the
	 * indices - we should sort the table to be sure!
	 */

	return( WHICH_ONE("type of datum",N_OUTPUT_TYPES,od_names) );
}

static COMMAND_FUNC( do_set_record )
{
	int n,i;
	Polh_Record_Format prf;

	n=HOW_MANY("number of measurements to transfer");
	if( n <= 0 || n > N_OUTPUT_TYPES ){
		sprintf(error_string,"measurements:  number of measurements must be > 0 and <= %d",
			N_OUTPUT_TYPES);
		warn(error_string);
		return;
	}
	for(i=0;i<n;i++){
		prf.rf_output[i] = get_record_type(SINGLE_QSP_ARG);

		/* BUG we should do some error checking here,
		 * check for duplications...
		 */
	}
	prf.rf_n_data = n;

	if( n_active_stations == 2 ){	/* If both are active, do both */
		prf.rf_station = 0;
		polhemus_output_data_format(&prf);
		prf.rf_station = 1;
		polhemus_output_data_format(&prf);
	} else {
		prf.rf_station = curr_station;
		polhemus_output_data_format(&prf);
		/* show_output_data_format(curr_station); */
	}
}

static COMMAND_FUNC( do_mk_vector )
{
	Data_Obj *dp;
	char name[LLEN];
	Dimension_Set ds1;
	uint32_t n;

	strcpy(name, NAMEOF("name for new polhemus data vector") );
	n = HOW_MANY("number of records");

	dp = dobj_of(name);
	if( dp != NO_OBJ ){
		sprintf(error_string,"Can't create new polhemus data vector %s, name is in use already",
			name);
		warn(error_string);
		return;
	}

	if( n <= 0 ){
		sprintf(error_string,"number of records (%d) must be positive",n);
		warn(error_string);
		return;
	}

	ds1.ds_seqs = 1;
	ds1.ds_frames = 1;
	ds1.ds_rows = 1;
	ds1.ds_cols = n;
	/* BUG need to handle more stations...  this code is for insidetrak only! */
	if( n_active_stations == 2 )
		ds1.ds_tdim = station_info[0].sd_multi_prf.rf_n_words
			+ station_info[1].sd_multi_prf.rf_n_words;
	else if( n_active_stations < 1 ){
		warn("At least one station must be active to create a polhemus data vector");
		return;
	} else
		ds1.ds_tdim = station_info[curr_station].sd_multi_prf.rf_n_words;

	/* WHY SHORT??? good for binary data, but... */
	/* BUG need to handle other precisions... */
	dp = make_dobj(name,&ds1,PREC_IN);
	//dp = make_dobj(name,&ds1,PREC_SP);

	if( dp == NO_OBJ )
		warn("unable to create polhemus data vector");
}

#ifdef FOOBAR
static COMMAND_FUNC( do_assign_var )
{
	char *varname;
	int i_type, index;
	Data_Obj *dp;

	varname=NAMEOF("variable name");
	dp = PICK_OBJ("");
	i_type = get_record_type(SINGLE_QSP_ARG);
	index = HOW_MANY("index");

	if(varname == NULL) return;
	if(dp == NULL) return;
	if( i_type < 0 ) return;
	if( index < 0 || index >= od_tbl[i_type].od_strings ){
		if( od_tbl[i_type].od_strings==1){
			sprintf(error_string,
		"For %s records, index must be 0",od_tbl[i_type].od_name);
		} else {
			sprintf(error_string,
		"For %s records, index must be between 0 and %d",
				od_tbl[i_type].od_name,
				od_tbl[i_type].od_strings-1);
		}
		warn(error_string);
		return;
	}

	assign_polh_var_data(varname,dp,i_type,index);
}
#endif /* FOOBAR */


static COMMAND_FUNC( do_set_async )
{
	polh_read_async( ASKIF("asynchronous reads from polhemus device") );
}

static COMMAND_FUNC( do_polh_flush )
{
	flush_polh_buffer();
}


static Command ph_acq_ctbl[] = {
{ "measurements",	do_set_record,		"specify the data that compose a record"		},
{ "single",		do_single_read,		"read single polhemus point"				},
{ "next",		do_next_read,		"read next streaming polhemus point"			},
{ "cont",		do_cont_read,		"read polhemus continuously"			},
{ "create_vec",		do_mk_vector,		"create a vector for polhemus data"			},
{ "readvec",		do_read_raw_vector,	"read raw polhemus data point into a vector"		},
{ "async",		do_set_async,		"set/clear asynchronous read mode"			},
{ "wait",		polhemus_wait,		"wait for async read to finish"				},
{ "halt",		polhemus_halt,		"terminate asynchronous read"				},
{ "fmtvec",		do_fmt_raw_vector,	"print recorded polhemus data"				},
{ "cvtvec",		do_cvt_raw_vector,	"convert recorded polhemus data to float"		},
#ifdef FOOBAR
{ "fmtvar",		do_assign_var,		"assign formatted polhemus data to a variable"		},
#endif /* FOOBAR */
{ "start",		do_start_continuous_mode,	"start continuous output mode"				},
{ "stop",		do_stop_continuous_mode,	"stop continuous output mode"				},
{ "flush",		do_polh_flush,		"flush polhemus buffer"					},
{ "quit",		popcmd,			"exit submenu"						},
{ NULL_COMMAND												}
};

static COMMAND_FUNC( do_reinit )
{ 
	if( send_polh_cmd(PH_REINIT_SYS,NULL) < 0 ) 
		warn("Unable to reinitialize polhemus system!");
}

static COMMAND_FUNC( do_units )
{ 
        Ph_Cmd_Code cmd;
	char *units[] = { "inches", "cm" };
	
	int n = WHICH_ONE("system units (inches/cm)", 2, units);

	if(n < 0) return;

	/* FIXME - we should only set the units after we know 
	 * that the conversion happened.
	 */

	switch(n) {
		case 0 : cmd = PH_INCHES_FMT; unit = INCHES; break;
		case 1 : cmd = PH_CM_FMT; unit = CM; break;
#ifdef CAUTIOUS
		default : warn("CAUTIOUS: unexpected system unit!?"); break;
#endif
	}
	
	set_polh_units(cmd);
}

static COMMAND_FUNC( do_send_string )
{
	const char *s;

	s=NAMEOF("command string");
	send_string(s);
}

static COMMAND_FUNC( do_chk_resp )
{
	int n;

	n=polhemus_word_count();
	if( n== 0 ) {
		advise("No data available");
		return;
	}

	read_response(1);
}

static COMMAND_FUNC( do_get_hemi )
{
	if( get_polh_info(PH_HEMISPHERE,NULL) < 0 ) 
		warn("Unable to get polhemus operational hemisphere!");
}

static COMMAND_FUNC( do_clr )
{
	clear_polh_dev();
}

/* The defaults are the maximum allowable
 * values for the envelope commands.
 */

#define MAX_AZ	180
#define MAX_EL	90
#define MAX_RL	180
#define MIN_AZ	-180
#define MIN_EL	-90
#define MIN_RL	-180

	
static COMMAND_FUNC( do_set_angl )
{
	char angl[LLEN];
	float amax, emax, rmax, amin, emin, rmin;

	if( (ASK_ENV("azimuth angle", "angular operational envelop", &amax, &amin, MAX_AZ, MIN_AZ) < 0) 
	    || (ASK_ENV("elevation angle", "angular operational envelop", &emax, &emin, MAX_EL, MIN_EL) < 0)
	    || (ASK_ENV("roll angle", "angular operational envelop", &rmax, &rmin, MAX_RL, MIN_RL) < 0)
	  ){
		return;
	}
		
	sprintf(angl, "%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f", amax, emax, rmax, amin, emin, rmin);

	if( send_polh_cmd(PH_ANGULAR_ENV, angl ) < 0) 
		warn("Unable to set angular operational envelope");
}

static COMMAND_FUNC( do_reopen )
{
	if( reopen_polh_dev() < 0 ) 
		warn("unable to reopen polhemus device");
}

static char * stat_choices[] = { "1", "2" };

static int get_station(SINGLE_QSP_ARG_DECL)
{
	int n;

	n = WHICH_ONE("station number", 2, stat_choices);
	if( n < 0 ) {
		warn("bad station number specified, must be 1 or 2");
		return(-1);
	}	
	return(n);
}

static COMMAND_FUNC( do_activate_station )
{
	int n;
	
	n=get_station(SINGLE_QSP_ARG);
	if( n < 0 ) return;

	activate_station(n,1);
}

static COMMAND_FUNC( do_deactivate_station )
{
	int n;
	
	n=get_station(SINGLE_QSP_ARG);
	if( n < 0 ) return;

	activate_station(n,0);
}

static COMMAND_FUNC( do_get_status )
{ 
	if( get_polh_info(PH_STATUS,NULL) < 0 ) 
		warn("Unable to get polhemus system status!");
}

static void inform_activation_state(int station)
{
	if( STATION_IS_ACTIVE(station) ){
		sprintf(msg_str,"Station %d is activated",station+1);
		prt_msg(msg_str);
	} else {
		sprintf(msg_str,"Station %d is not activated",station+1);
		prt_msg(msg_str);
	}
}


static COMMAND_FUNC( do_get_active_stations )
{
	get_active_stations();
	inform_activation_state(0);
	inform_activation_state(1);
}

static Command ph_dev_ctbl[] = {
{ "reinit",		do_reinit,		"reinitialize system"			},
{ "units",		do_units,		"set system distance unit"		},
{ "send",		do_send_string,		"send a command string"			},
{ "activate",		do_activate_station,	"activate a station"			},
{ "deactivate",		do_deactivate_station,	"deactivate a station"			},
{ "check_active",	do_get_active_stations,	"get current active stations"		},
{ "response",		do_chk_resp,		"check for command response"		},
{ "clear",		do_clr,			"clear polhemus device"			},
{ "reopen",		do_reopen,		"close polhemus device and reopen"	},
{ "status",		do_get_status,		"get system status record"		},
{ "quit",		popcmd,			"exit submenu"				},
{ NULL_COMMAND										}
};

static COMMAND_FUNC( do_get_sync )
{
	if( get_polh_info(PH_SYNC_MODE,NULL) < 0 ) 
		warn("Unable to get synchronization mode!");

	/*
	if( !strncmp((char *)(&resp_buf[1]),polh_cmds[PH_INTERNAL_SYNC].pc_cmdstr,2) )
		prt_msg("Current sync mode is internal");
	else if( !strncmp((char *)(&resp_buf[1]),polh_cmds[PH_EXTERNAL_SYNC].pc_cmdstr,2) )
		prt_msg("Current sync mode is external");
	else if( !strncmp((char *)(&resp_buf[1]),polh_cmds[PH_SOFTWARE_SYNC].pc_cmdstr,2) )
		prt_msg("Current sync mode is software");
	else {
		sprintf(error_string,"Unrecognized sync mode string:  \"%s\"",(char *)(&resp_buf[1]) );
		warn(error_string);
	}
	*/
}


static COMMAND_FUNC( do_set_sync )
{
	char *sync_choices[] = { "internal", "external", "software" };
	int stat=0;

	int n = WHICH_ONE("synchronization type (internal/external/software)", 3, sync_choices);

	if(n < 0) return;

	switch(n){
		case 0:  stat=set_polh_sync_mode(0); break;
		case 1:  stat=set_polh_sync_mode(1); break;
		case 2:  warn("Sorry, software sync not supported for polhemus"); break;
	}
	if( stat < 0 ) warn("Error setting polhemus sync");
}

static COMMAND_FUNC( do_set_att )
{
	if( SET_POLH_FILTER(PH_ATT_FILTER) < 0 ) 
		warn("Unable to set attitude filter parameters!");
}

static COMMAND_FUNC( do_get_att )
{
	if( get_polh_info(PH_ATT_FILTER,NULL) < 0 ) 
		warn("Unable to get attitude filter parameters!");
}


static Command ph_bore_ctbl[] = {
{ "sight",		do_set_bore,		"set boresight (zero) angles to current sight"	},
{ "refer",		do_set_ref_bore,	"set boresight (zero) reference angles" 	},
{ "current_refer",	do_set_curr_ref_bore,	"set boresight (zero) reference angles to current position angles" },
{ "get_refer",		do_get_ref_bore,	"get current boresight (zero) reference angles"	},
{ "xmitr",		do_set_trans,		"set transmitter mount frame angles"		},
{ "get_xmitr",		do_get_trans,		"get current transmitter mount frame angles"	},
{ "recv",		do_set_recv_bore,	"set boresight receiver angles"			},
{ "get_recv",		do_get_recv_bore,	"get current boresight receiver angles"		},
{ "reset",		do_reset_bore,		"reset boresight to factory defaults"		},
{ "quit",		popcmd,			"exit submenu"					},
{ NULL_COMMAND											}
};

static COMMAND_FUNC( ph_bore_menu )
{
	PUSHCMD(ph_bore_ctbl, "boresight");
}

static COMMAND_FUNC( do_set_pos )
{
	if( SET_POLH_FILTER(PH_POS_FILTER) < 0 ) 
		warn("Unable to set position filter parameters!");
}

static COMMAND_FUNC( do_get_pos )
{
	if( get_polh_info(PH_POS_FILTER,NULL) < 0 ) 
		warn("Unable to get position filter parameters!");
}

static COMMAND_FUNC( ph_pdata_menu )
{ PUSHCMD(ph_acq_ctbl, "acquire"); }

static COMMAND_FUNC( ph_dev_menu )
{ PUSHCMD(ph_dev_ctbl, "device"); }




static Command ph_comp_ctbl[] = {
{ "attitude",		do_set_att,		"set adaptive filter attitude controls"		},
{ "get_attitude",	do_get_att,		"get current adaptive filter attitude controls"	},
{ "position",		do_set_pos,		"set adaptive filter position controls"		},
{ "get_position",	do_get_pos,		"get current adaptive filter position controls"	},
{ "quit",		popcmd,			"exit submenu"					},
{ NULL_COMMAND											}
};

static COMMAND_FUNC( ph_comp_menu )
{ PUSHCMD(ph_comp_ctbl, "compensate"); }

static Command ph_env_ctbl[] = {
{ "angular",		do_set_angl,		"set software angular limits"		},	
{ "get_angular",	do_get_angl,		"get software angular limits"		},	
{ "positional",		do_set_post,		"set positional angular limits"		},
{ "get_positional",	do_get_post,		"get positional angular limits"		},
{ "quit",		popcmd,			"exit submenu"				},
{ NULL_COMMAND										}
};

static COMMAND_FUNC( ph_env_menu )
{ PUSHCMD(ph_env_ctbl, "envelope"); }

static Command ph_hemi_ctbl[] = {
{ "set",		do_set_hemi,		"set operational hemisphere"			},
{ "get",		do_get_hemi,		"get current operational hemisphere"		},
{ "quit",		popcmd,			"exit submenu"					},
{ NULL_COMMAND											}
};

static COMMAND_FUNC( ph_hemi_menu )
{ PUSHCMD(ph_hemi_ctbl, "hemisphere"); }

static Command ph_sys_ctbl[] = {
{ "reinit",		do_reinit,		"reinitialize system"		},
{ "units",		do_units,		"set system distance unit"	},
{ "status",		do_get_status,		"get system status record"	},
{ "send",		do_send_string,		"send a command string"		},
{ "response",		do_chk_resp,		"check for command response"	},
{ "quit",		popcmd,			"exit submenu"			},
{ NULL_COMMAND									}
};

static COMMAND_FUNC( ph_sys_menu )
{ PUSHCMD(ph_sys_ctbl, "system"); }

static Command ph_align_ctbl[] = {
{ "set",		do_set_align,		"set alignment points"				},
{ "current",		do_set_curr_align,	"set alignment to current coordinate position"	},
{ "get",		do_get_align,		"get current alignment points"			},
{ "reset",		do_reset_align,		"reset alignment to factory default"		},
{ "quit",		popcmd,			"exit submenu"					},
{ NULL_COMMAND											}
};

static COMMAND_FUNC( ph_align_menu )
{ PUSHCMD(ph_align_ctbl, "alignment"); }

/* BUG these commands should be organized better... */

static Command ph_misc_ctbl[]={
{ "system",		ph_sys_menu,		"system submenu"		},
{ "device",		ph_dev_menu,		"device submenu"		},
{ "alignment",		ph_align_menu,		"alignment submenu"		},
{ "boresight",		ph_bore_menu,		"boresight submenu"		},
{ "compensate",		ph_comp_menu,		"compensation submenu"		},
{ "envelope",		ph_env_menu,		"envelope submenu"		},
{ "hemisphere",		ph_hemi_menu,		"hemisphere submenu"		},
{ "quit",		popcmd,			"exit submenu"			},
{ NULL_COMMAND									}
};

static COMMAND_FUNC( ph_misc_menu )
{
	PUSHCMD(ph_misc_ctbl, "misc");
}

static Command ph_ctbl[] = {
{ "acquire",		ph_pdata_menu,		"data acquisition submenu"		},
{ "get_sync",		do_get_sync,		"get current synchronization mode"	},	
{ "set_sync",		do_set_sync,		"set synchronization mode"		},
{ "misc",		ph_misc_menu,		"miscellaneous command submenu"		},
{ "quit",		popcmd,			"exit submenu"				},
{ NULL_COMMAND										}
};

COMMAND_FUNC( ph_menu )
{
	static int polh_init = 0;

	if( !polh_init ) {

#ifdef DEBUG
		debug_polhemus = add_debug_module("polhemus");
		//debug |= debug_polhemus;	/* turn it on for testing */
#endif /* DEBUG */

		auto_version("POLHEMUS","VersionId_polhemus");
		sort_table();
		if( init_polh_dev() < 0 ) {
			warn("ph_menu: Unable to initialize polhemus device");
			return;
		}
		polh_init = 1;
	}

	PUSHCMD(ph_ctbl, "polhemus");
}

