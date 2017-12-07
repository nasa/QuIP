#include "quip_config.h"

#include <string.h>
#include <ctype.h>

#include "quip_prot.h"
//#include "myerror.h"
#include "debug.h"

#include "polh_dev.h"

#define PH_RECORD_TYPE		'2'

static char * ph_err_msgs[] = {
	"check transmitter connections/possible distortion environment(proximity of large metal objects too great)",
	"check receiver connections",
	"movement of receiver towards sensor too rapid for AGC",
	"operating outside of mapped bounds",
	"receiver outside of user-defined motion box (V command)",
	"orientation angles outside of allowed user-defined angular envelope (Q command)",
	"transmitter characterization matrix - consult factory",
	"receiver characterization matrix - consult factory",
	"compensation structure - consult factory",
	NULL
};

/* We initially set the msg strings to null,
 * and fill them in later based on the code value.
 */

#define N_PH_ERRS	18

Ph_Err polh_errs[N_PH_ERRS] = {
	{	"A",	PH_ERR_XMTR_ENV_CONN,	NULL	},
	{	"B",	PH_ERR_XMTR_ENV_CONN,	NULL	},
	{	"C",	PH_ERR_XMTR_ENV_CONN,	NULL	},
	{	"G",	PH_ERR_XMTR_ENV_CONN,	NULL	},
	{	"H",	PH_ERR_XMTR_ENV_CONN,	NULL	},
	{	"I",	PH_ERR_XMTR_ENV_CONN,	NULL	},
	{	"a",	PH_ERR_XMTR_ENV_CONN,	NULL	},
	{	"b",	PH_ERR_XMTR_ENV_CONN,	NULL	},
	{	"c",	PH_ERR_XMTR_ENV_CONN,	NULL	},
	{	"k",	PH_ERR_RCVR_CONN,	NULL	},
	{	"l",	PH_ERR_RCVR_CONN,	NULL	},
	{ 	"j",	PH_ERR_MOV_RAPID,	NULL	},
	{	"u",	PH_ERR_OUT_BOUNDS,	NULL	},
	{	"x",	PH_ERR_V_BOUNDS,	NULL	},
	{	"y",	PH_ERR_Q_BOUNDS,	NULL	},
	{	"X",	PH_ERR_XMTR_MAT,	NULL	},
	{	"Y",	PH_ERR_RCVR_MAT,	NULL	},
	{	"t",	PH_ERR_COMP,		NULL	}
};

#define check_polh_err(err) _check_polh_err(QSP_ARG  err)

static inline int _check_polh_err(QSP_ARG_DECL  char err)
{
	int i;

	for(i = 0; i < N_PH_ERRS; i++) {

		/* check if error code matches */
		if(strchr(polh_errs[i].pe_str, err)) {
			/* set the error message */		
			polh_errs[i].pe_msg = ph_err_msgs[ polh_errs[i].pe_code ]; 

			sprintf(ERROR_STRING, "check_polh_err: err. code: %s err msg.: %s", 
				polh_errs[i].pe_str, polh_errs[i].pe_msg);
			warn(ERROR_STRING);

			return(-1);
		}		
	}

	return(0);
}	

int _check_polh_data(QSP_ARG_DECL  char *rawdata, Polh_Record_Format *prfp)
{
	/* Polhemus manual (pg A64) - The manual says
	 * that the first byte is either an error code
	 * or blank space. The second byte is the station
	 * number.
	 */
	
#ifdef QUIP_DEBUG
if( debug & debug_polhemus ){
	sprintf(ERROR_STRING,"check_polh_data:  station = %d, nwords = %d",prfp->rf_station,prfp->rf_n_words);
	_advise(DEFAULT_QSP_ARG  ERROR_STRING);
	display_buffer((short *)rawdata,prfp->rf_n_words);
}
#endif /* QUIP_DEBUG */

	/* check for error code */
	if(check_polh_err(rawdata[0]) < 0) return(-1);
	
	/* check for blank space */
	if(!isspace(rawdata[0])) {
		warn("check_polh_data: unexpected polhemus data format (no initial blank space found)");
sprintf(ERROR_STRING,"rawdata[0] = 0x%x",rawdata[0]);
_advise(DEFAULT_QSP_ARG  ERROR_STRING);
		return(-1);
	}
	
	/* check for the correct station number */
	if(rawdata[1] != STATION_CHAR(prfp->rf_station)) {
		sprintf(ERROR_STRING, "check_polh_data: unexpected station number %c, expected station number %c",
			rawdata[1], STATION_CHAR(prfp->rf_station));
		warn(ERROR_STRING);
		return(-1);
	}

	return(0);		
}

int _check_polh_output(QSP_ARG_DECL  char *out, int expected_station, Ph_Cmd_Code cmd)
{
	/* All output records from initiating command begin with 
	 * the record type "2" followed be either a blank or the
	 * station number, and then the the sub-record type 
	 * after that.
	 *
	 * byte 1 - Record type "2"
	 * byte 2 - blank or station number
	 * byte 3 - sub-record type (corresponds to the initiating 
	 *          command used to request the output)
	 *
	 * Commands that have a blank instead of station number: v, x, y.
	 */

	/* check for the record type */
#ifdef QUIP_DEBUG
if( debug & debug_polhemus ){
sprintf(ERROR_STRING,"check_polh_output:  cmd=%d, expect station %d, checking \"%s\"",
cmd,expected_station,show_printable(DEFAULT_QSP_ARG  out));
_advise(DEFAULT_QSP_ARG  ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	if(*out != PH_RECORD_TYPE) {
		sprintf(ERROR_STRING, "check_polh_output: Unexpected record type %c in output", out[0]);
		warn(ERROR_STRING);
		return(-1);	
	}
	
	/* check for either space or station number */
	if( IS_FILTER_CMD(cmd) || IS_SYNC_CMD(cmd) ) {
		/* check for blank space */
		if(!isspace(out[1])) {
			warn("check_polh_output: Unexpected polhemus record output format (no blank space found)");
			return(-1);	
		}
	} else {
		/* how do we know what the current station is?  We should probably ignore this if the command
		 * doesn't need a station...
		 */
		if( expected_station != -1 ){
			if(out[1] != STATION_CHAR(expected_station)) {
				sprintf(ERROR_STRING, "check_polh_output: unexpected station number %c, expected station %d",
					out[1], expected_station);
				warn(ERROR_STRING);
				return(-1);
			}
		}
	}

	/* check for sub-record type */
	if( *polh_cmds[cmd].pc_cmdstr != out[2]) {
		sprintf(ERROR_STRING, "check_polh_output: unexpected sub-record type %c, expected sub-record type %c",
			out[2], *polh_cmds[cmd].pc_cmdstr);
		warn(ERROR_STRING);
		return(-1);	
	}

	return(0);
}
