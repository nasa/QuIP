#include "quip_config.h"

#include <sys/ioctl.h>
//#include "ioctl_polhemus.h"
#include <unistd.h>		/* usleep */
#include <pthread.h>
#include <time.h>		/* ctime */
#include <signal.h>
#include <ctype.h>
#include <string.h>

#include "quip_prot.h"
//#include "sigpush.h"
//#include "debug.h"
//#include "vars.h"
//#include "tryhard.h"		/* try_open */

#include "polh_dev.h"
#include "data_obj.h"

#define NUMERIC_POLH_PRINT_FORMAT "%-8ld\t%-8ld\t%-8g\t%-8g\t%-8g\t%-8g\t%-8g\t%-8g"
#define READABLE_POLH_PRINT_FORMAT "%s\t%3ld.%03ld\t%-8g\t%-8g\t%-8g\t%-8g\t%-8g\t%-8g" 

static FILE * polh_output_file = NULL;	/* output file */
static char *separator="";

static int _read_async=0;
static int halting=0;

/* data processing threads */
static pthread_t polh_thread;
static pthread_attr_t thread_attr;
static int thread_running=0;

typedef struct polh_read_info {
	int	pri_n_requested;		/* number to read */
	int	pri_n_obtained;			/* number actually read */
	void *	pri_addr;
	prec_t	pri_prec;
	int	pri_station;
} Polh_Read_Info;

#define INT_TO_COS(ptr)		( ((float) (*(ptr))) / 0x7fff )

#ifdef NOT_USED

static void format_chunk(Fmt_Pt *fpp, short *pdp, Polh_Output_Type type )
{
	char *s;

	switch(type){
		case STATION:
			fpp->fp_station = *pdp;
			break;

		case DATE:
			s = ctime((time_t*)pdp);
			if( s[ strlen(s) - 1 ] == '\n' )
				s[ strlen(s) - 1 ] = 0;
			strcpy(fpp->fp_date_str,s);
			break;

		case SECONDS:
			fpp->fp_seconds = *((int32_t *)pdp);
			break;

		case MSECS:
			fpp->fp_usecs = *((int32_t *)pdp);
			break;

		case XYZ_INT:
#ifdef QUIP_DEBUG
if( debug & debug_polhemus ){
NADVISE("raw buffer for XYZ_INT:");
display_buffer(pdp,3);
}
#endif /* QUIP_DEBUG */
			/* coordinates */
			if( polh_units == PH_CM_FMT ) {
				fpp->fp_x = RAW_TO_CM(pdp); pdp ++;
				fpp->fp_y = RAW_TO_CM(pdp); pdp ++;
				fpp->fp_z = RAW_TO_CM(pdp); pdp ++;
			} else if( polh_units == PH_INCHES_FMT ) {
				fpp->fp_x = RAW_TO_IN(pdp); pdp ++;
				fpp->fp_y = RAW_TO_IN(pdp); pdp ++;
				fpp->fp_z = RAW_TO_IN(pdp); pdp ++;
			}
#ifdef CAUTIOUS
			else {
				NWARN("CAUTIOUS:  format_chunk: not cm or inches!?");
				return;
			}
#endif /* CAUTIOUS */
			break;

		case XYZ_FLT:
			fpp->fp_x = *((float *)pdp); pdp+=2;
			fpp->fp_y = *((float *)pdp); pdp+=2;
			fpp->fp_z = *((float *)pdp); pdp+=2;
			break;

		case EULER_INT:
			fpp->fp_azim = RAW_TO_DEG(pdp); pdp ++;
			fpp->fp_elev = RAW_TO_DEG(pdp); pdp ++;
			fpp->fp_roll = RAW_TO_DEG(pdp); pdp ++;
			break;

		case X_DIR_INT:
			fpp->fp_xdc_x = INT_TO_COS(pdp); pdp++;
			fpp->fp_xdc_y = INT_TO_COS(pdp); pdp++;
			fpp->fp_xdc_z = INT_TO_COS(pdp); pdp++;
			break;

		case Y_DIR_INT:
			fpp->fp_ydc_x = INT_TO_COS(pdp); pdp++;
			fpp->fp_ydc_y = INT_TO_COS(pdp); pdp++;
			fpp->fp_ydc_z = INT_TO_COS(pdp); pdp++;
			break;

		case Z_DIR_INT:
			fpp->fp_zdc_x = INT_TO_COS(pdp); pdp++;
			fpp->fp_zdc_y = INT_TO_COS(pdp); pdp++;
			fpp->fp_zdc_z = INT_TO_COS(pdp); pdp++;
			break;

		case QUAT_INT:
			fpp->fp_q1 = INT_TO_COS(pdp); pdp++;
			fpp->fp_q2 = INT_TO_COS(pdp); pdp++;
			fpp->fp_q3 = INT_TO_COS(pdp); pdp++;
			fpp->fp_q4 = INT_TO_COS(pdp); pdp++;
			break;

		case QUAT_FLT:
			fpp->fp_q1 = *((float *)pdp); pdp+=2;
			fpp->fp_q2 = *((float *)pdp); pdp+=2;
			fpp->fp_q3 = *((float *)pdp); pdp+=2;
			fpp->fp_q4 = *((float *)pdp); pdp+=2;
			break;

		case X_DIR_FLT:
			fpp->fp_xdc_x = *((float *)pdp); pdp+=2;
			fpp->fp_xdc_y = *((float *)pdp); pdp+=2;
			fpp->fp_xdc_z = *((float *)pdp); pdp+=2;
			break;

		case Y_DIR_FLT:
			fpp->fp_ydc_x = *((float *)pdp); pdp+=2;
			fpp->fp_ydc_y = *((float *)pdp); pdp+=2;
			fpp->fp_ydc_z = *((float *)pdp); pdp+=2;
			break;

		case Z_DIR_FLT:
			fpp->fp_zdc_x = *((float *)pdp); pdp+=2;
			fpp->fp_zdc_y = *((float *)pdp); pdp+=2;
			fpp->fp_zdc_z = *((float *)pdp); pdp+=2;
			break;

#ifdef CAUTIOUS
		default:
			sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  format_chunk:  Oops, not programmed to format %s",
					od_tbl[type].od_name);
			NWARN(DEFAULT_ERROR_STRING);
			break;
#endif /* CAUTIOUS */
	}
}
#endif // NOT_USED

#ifdef FOOBAR
/* What does this do?
 * This is old inside-track code that assumes (short) binary data...
 */

int format_polh_data(Fmt_Pt *fpp, short *pdp, Polh_Record_Format *prfp)
{
	short *orig_pdp;
	int i;

	/* The polhemus data is as follows:
	 *
	 *  ____   ____   ____   ____   ____   ____   ____   ____ 
	 * |    | |    | |    | |    | |    | |    | |    | |    |
	 * |    | |    | |    | |    | |    | |    | |    | |    |
	 * |____| |____| |____| |____| |____| |____| |____| |____| 
	 *
	 *  E  S  x-pts  y-pts  z-pts    az     el     rl    cr/nl 
	 *  
	 *
	 * Each box up above represents a short. The first expected
	 * character is a space or an error code(so we know our data 
	 * is aligned properly). The second character is the station 
	 * number. The first short is thrown away, and the next 
	 * six shorts are the data we want. The last short is 
	 * a carriage return/line feed pair.
	 */

#ifdef QUIP_DEBUG
	/*
if( debug & debug_polhemus ){
NADVISE("raw buffer for single record:");
display_buffer(pdp,station_info[ curr_station_idx ].sd_single_prf.rf_n_words);
}
*/
#endif /* QUIP_DEBUG */

	if( check_polh_data((char *)pdp, prfp) < 0 ){
		NWARN("problem found in polhemus data");
		return(-1);
	}

	pdp ++;			/* bypass first short chunk */

	orig_pdp = pdp;

	for(i=0;i<prfp->rf_n_data;i++){
		Polh_Output_Type type;

		type = prfp->rf_output[i];
		if( type == STATION )
			format_chunk(fpp,&prfp->rf_station,type);
		else
			format_chunk(fpp,pdp,type);

		pdp += od_tbl[type].od_words;
	}
	return(0);
}
#endif /* FOOBAR */

static int parse_polh_reading( QSP_ARG_DECL  Data_Obj *dp, char * s )
{
	char str[32];
	float *f_p;

#ifdef QUIP_DEBUG
if( debug & debug_polhemus ){
sprintf(DEFAULT_ERROR_STRING,"parse_polh_reading \"%s\"",show_printable(DEFAULT_QSP_ARG  s));
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	if( OBJ_PREC(dp) != PREC_SP ){
		sprintf(DEFAULT_ERROR_STRING,"Object %s has %s precision, should be %s",
			OBJ_NAME(dp),
			PREC_NAME(OBJ_PREC_PTR(dp)),
			PREC_NAME(prec_for_code(PREC_SP)) );
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(DEFAULT_ERROR_STRING,"Object %s should be contiguous",OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( OBJ_N_MACH_ELTS(dp) < 6 ){
		sprintf(DEFAULT_ERROR_STRING,"Object %s should have at least 6 elements",OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

	f_p = OBJ_DATA_PTR(dp);

	if( sscanf(s,"%s %f %f %f %f %f %f",str,
		f_p+0,
		f_p+1,
		f_p+2,
		f_p+3,
		f_p+4,
		f_p+5
		) != 7 ){
		sprintf(DEFAULT_ERROR_STRING,"Error scanning polhemus data string");
		NWARN(DEFAULT_ERROR_STRING);
		sprintf(DEFAULT_ERROR_STRING,"String:  \"%s\"",show_printable(DEFAULT_QSP_ARG  s));
		NADVISE(DEFAULT_ERROR_STRING);
		return(-1);
	}
	return(0);
}

/* call this routine to read the next reading when we are streaming in continuous mode.
 * We trust that we are reading fast enough to keep up...
 */

int read_next_polh_dp(QSP_ARG_DECL  Data_Obj *dp)
{
	char *s;

	if(dp == NO_OBJ) return(-1);

	if( !polh_continuous )
		start_continuous_mode();

	s=read_polh_line();
	parse_polh_reading(QSP_ARG  dp,s);

	return(0);
}

static void *cont_reader(void *vp)
{
	Data_Obj *dp;
	dp=vp;

	while(!halting){
		read_next_polh_dp(DEFAULT_QSP_ARG  dp);
	}
	return NULL;
}

/* call this routine to read continuously when we are streaming in continuous mode.
 * This forks a separate thread - we do this to avoid having to externally trigger the polhemus
 * in a case where the program is clocked by another event (e.g. video refresh).
 */

int read_cont_polh_dp(Data_Obj *dp)
{
	halting=0;

	/* create data-processing threads */
	pthread_attr_init(&thread_attr);

	/* Why do we set it to inherit the parent thread scheduling? */
	pthread_attr_setinheritsched(&thread_attr, PTHREAD_INHERIT_SCHED);

	if( pthread_create(&polh_thread, &thread_attr, cont_reader, dp) != 0 ){
		perror("pthread_create");
		NWARN("error creating polhemus cont_reader thread");
		return -1;
	}

	thread_running=1;
	return 0;
}

int read_single_polh_dp(QSP_ARG_DECL  Data_Obj *dp)
{
	//int n_want_bytes;
	char *s;

	if(dp == NO_OBJ) return(-1);

#ifdef FOOBAR
	/* Set the polhemus device to non-continuous mode. */
	if( ioctl(polh_fd, POLHEMUS_SET_NONCONTINUOUS_MODE, NULL) < 0 ) {
		NWARN("read_single_polh_dp: Unable to set polhemus device to non-continuous mode!");
		return(-1);
	}
#endif
	if( polh_continuous )
		stop_continuous_mode();

	/* request for a single record to be sent to the host */
	send_polh_cmd(PH_SINGLE_RECORD, NULL); 

	/* Originally, we slept for 1 millisec, but that
	 * caused problems so we increased the time to
	 * sleep.
	 * What problems???
	 */
	//usleep(100000);	/* sleep 1/10 sec. */
	usleep(1000);	/* sleep 1 msec. */

	if( n_active_stations < 1 ){
		sprintf(DEFAULT_ERROR_STRING,"read_single_polh_dp:  at least one station must be active (%d)",
				n_active_stations);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

#ifdef INSIDE_TRACK
	/* Do we get both?? */
	n_want_bytes = n_active_stations * station_info[ curr_station_idx ].sd_single_prf.rf_n_words * sizeof(short);

	if( read_polh_dev(pdp, n_want_bytes ) < 0 ){
		NWARN("error reading polhemus data");
		return(-1);
	}
#else

	s=read_polh_line();
	parse_polh_reading(QSP_ARG  dp,s);

#endif

	return(0);
} /* end read_single_polh_dp */

/* Return 1 if appropriate for polhemus, 0 otherwise */

int good_polh_vector(QSP_ARG_DECL  Data_Obj *dp)
{
	int n_expected_words;

	/* The polhemus can output floats, but we'll assume it's all short here... */

	if( dp == NO_OBJ ) return(0);

	if( OBJ_PREC(dp) != PREC_IN && OBJ_PREC(dp) != PREC_UIN ) {
		sprintf(DEFAULT_ERROR_STRING, "Object %s has %s precision, should be %s or %s for polhemus data",
			OBJ_NAME(dp), OBJ_PREC_NAME(dp), NAME_FOR_PREC_CODE(PREC_IN),
			NAME_FOR_PREC_CODE(PREC_UIN) );
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}

	if( n_active_stations == 2 )
		n_expected_words = station_info[0].sd_multi_prf.rf_n_words +
			station_info[1].sd_multi_prf.rf_n_words;
	else
		n_expected_words = station_info[curr_station_idx].sd_multi_prf.rf_n_words ;

	if( OBJ_COMPS(dp) != n_expected_words ) {
		sprintf(DEFAULT_ERROR_STRING, "Object %s has bad type dimensions (%d), should be %d",
			OBJ_NAME(dp), OBJ_COMPS(dp), n_expected_words);
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}

	if( !IS_CONTIGUOUS(dp) ) {
		sprintf(DEFAULT_ERROR_STRING, "good_polh_vector: Object %s must be contiguous", OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}
	return(1);
}

/* This code reads an ascii record from the fastrak */ 

static int good_polh_char(void)
{
	int c;

	while(1){ 
		c=polh_getc();
		if( c>=0 ) return(c);
		fill_polh_buffer();
	}
	/* NOTREACHED */
}

static void read_data_string(char *s)
{
	int c;

	/* read any leading spaces */
	do {
		c=good_polh_char();
	} while( isspace(c) );

	/* c is the first char of the string */
	/* Sometimes we have number strings butted together, e.g. 2.31-4.65, instead
	 * of having a space before the minus sign...
	 */

	do {
		*s++ = (char) c;
		c=good_polh_char();
	} while( (!isspace(c)) && c!='-' );

	if( c == '-' ) polh_ungetc(c);	/* BUG check error status */

	/* c is white space or CR */
	*s = 0;
}

/* Serial port stuff */

/* BUG - this does not put the data into the binary buffer??? */

static int read_next_record(QSP_ARG_DECL  void * raw_pdp, int station )
{
	char data_string[6][32];
	int i;
	int c, expect_c;

fprintf(stderr,"read_next_record 0x%lx\n",(long)raw_pdp);
	/* read any leading spaces */
	do {
		c=good_polh_char();
	} while( isspace(c) );

	while( c != '0' ){
		sprintf(DEFAULT_ERROR_STRING,
	"read 0x%x, expected '0' in first header position",c);
		NWARN(DEFAULT_ERROR_STRING);
		c=good_polh_char();
	}
	c=good_polh_char();
	expect_c = '1' + curr_station_idx;
	while( c != expect_c ){
		sprintf(DEFAULT_ERROR_STRING,
	"read '%c' (0x%x), expected '%c' in second header position",
			c,c,expect_c);
		NWARN(DEFAULT_ERROR_STRING);
		c=good_polh_char();
	}
	c=good_polh_char();
	if( c != 'D' && c != 'Y' ){
		sprintf(DEFAULT_ERROR_STRING,
	"read '%c' (0x%x), expected 'D' or 'Y' in third header position",c,c);
		//NWARN(DEFAULT_ERROR_STRING);
		NADVISE(DEFAULT_ERROR_STRING);

		c=good_polh_char();

		sprintf(DEFAULT_ERROR_STRING,
	"read '%c' (0x%x) after unexpected character",c,c);
		//NWARN(DEFAULT_ERROR_STRING);
		NADVISE(DEFAULT_ERROR_STRING);
	}

	/* BUG don't assume 6 data - use format? */
	for(i=0;i<6;i++){
		read_data_string(data_string[i]);
	//	NADVISE(data_string[i]);
		/* raw_pdp doesn't indicate the format? */
	}

	// BUG how do we know that these are the data that were requested???
	assign_var(QSP_ARG  "polh_x_pos",data_string[0]);
	assign_var(QSP_ARG  "polh_y_pos",data_string[1]);
	assign_var(QSP_ARG  "polh_z_pos",data_string[2]);
	assign_var(QSP_ARG  "polh_x_rot",data_string[3]);
	assign_var(QSP_ARG  "polh_y_rot",data_string[4]);
	assign_var(QSP_ARG  "polh_z_rot",data_string[5]);

	return(0);
}

/* transfer_polh_data is a new thread entry point */

static void *transfer_polh_data(void *prip)
{
	int records_to_read;
	char *data_p;
	int station;

	records_to_read	= ((Polh_Read_Info *)prip)->pri_n_requested;
	data_p		= ((Polh_Read_Info *)prip)->pri_addr;
	station		= ((Polh_Read_Info *)prip)->pri_station;

#ifdef INSIDE_TRACK
	/* This code, with its fixed read size, seems to assume binary transfer mode... */
	while( records_to_read-- && !halting ) {
		if( read_polh_data(data_p, station_info[ station ].sd_multi_prf.rf_n_words*sizeof(short) ) < 0 ) {
			sprintf(DEFAULT_ERROR_STRING,
		"transfer_polh_data: Error reading raw polhemus data (%d points left to read)",
				records_to_read);
			NWARN(DEFAULT_ERROR_STRING);
			halting=1;
		} else {
			((Polh_Read_Info *)prip)->pri_n_obtained ++ ;
		}
		/* BUG should have size in struct */
		data_p += station_info[ station ].sd_multi_prf.rf_n_words;
		if( n_active_stations == 2 )
			station ^= 1;	/* xor 1->0, 0->1 */
	}
#else
	while( records_to_read-- && !halting ) {
		if( read_next_record(DEFAULT_QSP_ARG  data_p,station) < 0 ){
			NWARN("error reading record");
			halting=1;
		} else {
			((Polh_Read_Info *)prip)->pri_n_obtained ++ ;
		}
		/* BUG raw_pdp is void * ??? */
		data_p += sizeof(short) * station_info[ station ].sd_multi_prf.rf_n_words;
		/* No multiple station support yet... */
		//if( n_active_stations == 2 )
		//	station ^= 1;	/* xor 1->0, 0->1 */
	}
#endif

	if( polh_continuous )
		stop_continuous_mode();

	sprintf(DEFAULT_MSG_STR,"%d", ((Polh_Read_Info *)prip)->pri_n_obtained  );
	assign_var(DEFAULT_QSP_ARG  "n_polh_records_obtained",DEFAULT_MSG_STR);

	return(NULL);
}

static void start_data_thread(Polh_Read_Info *prip)
{
	/* create data-processing threads */
	pthread_attr_init(&thread_attr);

	/* Why do we set it to inherit the parent thread scheduling? */
	pthread_attr_setinheritsched(&thread_attr, PTHREAD_INHERIT_SCHED);

	if( pthread_create(&polh_thread, &thread_attr, transfer_polh_data, prip) != 0 ){
		perror("pthread_create");
		NWARN("error creating polhemus transfer thread");
		return;
	}

	thread_running=1;
}

COMMAND_FUNC( polhemus_wait )
{
	if( ! thread_running ){
		NWARN("polhemus_wait:  no thread running!?");
		return;
	}

	if( pthread_join( polh_thread, NULL ) != 0 ){
		perror("pthread_join");
		NWARN("error waiting for data collection thread to exit");
	}
	thread_running=0;
}

COMMAND_FUNC( polhemus_halt )
{
	halting=1;

	polhemus_wait(SINGLE_QSP_ARG);
}

/* read_polh_vector	-	read a bunch of polhemus data.
 * This function is not really what we need for interactive work!?
 * Unless the vector consists of just one reading...
 */

void read_polh_vector(QSP_ARG_DECL  Data_Obj *dp)
{
	void *raw_pdp;
	int records_to_read;
	int station;
	static Polh_Read_Info pri1;

	if( polh_continuous ){
		WARN("start_async_read:  polhemus is already in continuous mode!?");
	} else	{
//NADVISE("read_polh_vector:  starting continuous mode");
		start_continuous_mode();
//WARN("read_polh_vector:  exiting before reading for debugging purposes!");
		//return;

	}

	if( ! good_polh_vector(QSP_ARG  dp) ) return;

	raw_pdp = OBJ_DATA_PTR(dp);
	records_to_read = OBJ_N_MACH_ELTS(dp) / station_info[ curr_station_idx ].sd_multi_prf.rf_n_words;
//fprintf(stderr,"Will attempt to read %d records...\n",records_to_read);

	/* BUG station login assumes max 2 stations as in insidetrak... */
	if( n_active_stations == 2 ){
		station=0;
	} else if( n_active_stations == 1 )
		station = curr_station_idx;
	else {
		sprintf(DEFAULT_ERROR_STRING,"read_polh_vector:  number of active stations (%d) should be 1 or 2",
				n_active_stations);
		WARN(DEFAULT_ERROR_STRING);
		return;
	}

	pri1.pri_n_requested = records_to_read;
	pri1.pri_n_obtained = 0;
	pri1.pri_addr = raw_pdp;
	pri1.pri_station = station;

	halting=0;

//sprintf(DEFAULT_ERROR_STRING,"read_polh_vector:  _read_async = %d",_read_async);
//NADVISE(DEFAULT_ERROR_STRING);

	if( _read_async ){
ADVISE("read_polh_vector:  calling start_data_thread");
		start_data_thread(&pri1);
	} else {
ADVISE("read_polh_vector:  calling transfer_polh_data");
		transfer_polh_data(&pri1);
	}
}

static int fmt_one(QSP_ARG_DECL  char *str, Fmt_Pt *fpp, Polh_Output_Type type )
{
	switch(type){
		case DATE:
			sprintf(msg_str,"%s%s",separator,fpp->fp_date_str);
			break;

		case SECONDS:
			sprintf(msg_str,"%s%d",separator,fpp->fp_seconds);
			break;

		case MSECS:
			sprintf(msg_str,"%s%3d.%03d",separator,fpp->fp_usecs/1000,fpp->fp_usecs%1000);
			break;

		case STATION:
			sprintf(msg_str,"%s%d",separator,fpp->fp_station);
			break;

		case XYZ_INT:
		case XYZ_FLT:
			sprintf(msg_str,"%s%g\t%g\t%g",separator,fpp->fp_x,fpp->fp_y,fpp->fp_z);
			break;

		case QUAT_INT:
		case QUAT_FLT:
			sprintf(msg_str,"%s%g\t%g\t%g\t%g",separator,fpp->fp_q1,fpp->fp_q2,fpp->fp_q3,
					fpp->fp_q4);
			break;

		case EULER_INT:
		case EULER_FLT:
			sprintf(msg_str,"%s%g\t%g\t%g",separator,fpp->fp_azim,fpp->fp_elev,fpp->fp_roll);
			break;
		case X_DIR_INT:
		case X_DIR_FLT:
			sprintf(msg_str,"%s%g\t%g\t%g",separator,fpp->fp_xdc_x,fpp->fp_xdc_y,fpp->fp_xdc_z);
			break;
		case Y_DIR_INT:
		case Y_DIR_FLT:
			sprintf(msg_str,"%s%g\t%g\t%g",separator,fpp->fp_ydc_x,fpp->fp_ydc_y,fpp->fp_ydc_z);
			break;
		case Z_DIR_INT:
		case Z_DIR_FLT:
			sprintf(msg_str,"%s%g\t%g\t%g",separator,fpp->fp_zdc_x,fpp->fp_zdc_y,fpp->fp_zdc_z);
			break;
		default:
			sprintf(DEFAULT_ERROR_STRING,"disp_one:  Oops, no display routine for %s",od_tbl[type].od_name);
			NWARN(DEFAULT_ERROR_STRING);
			return(-1);
			break;
	}
	return(0);
}

static void disp_one( QSP_ARG_DECL  Fmt_Pt *fpp, Polh_Output_Type type)
{
	if( fmt_one(QSP_ARG  msg_str,fpp,type) < 0 ) return;
	prt_msg_frag(msg_str);
}

void display_formatted_point(QSP_ARG_DECL  Fmt_Pt * fpp, Polh_Record_Format *prfp )
{
	int i;

	if( fpp == NULL ) return;

	separator="";	/* no leading tab before the first record */
	for(i=0;i<prfp->rf_n_data;i++){
		Polh_Output_Type type;

		type = prfp->rf_output[i];

		disp_one(QSP_ARG  fpp,type);

		separator="\t";	/* for succeeding records, prepend a tab */
	}
	prt_msg("");
} 

int set_continuous_output_file(QSP_ARG_DECL  const char *fname)
{
	polh_output_file = try_open(QSP_ARG  fname, "w");

	if( !polh_output_file ) {
		sprintf(DEFAULT_ERROR_STRING, "Unable to open file for continuous output: %s", fname);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

	return(0);
}

#ifdef FOOBAR
void assign_polh_var_data(char *varname, Data_Obj *dp, Polh_Output_Type type, int index )
{
	short *rpdp;
	Fmt_Pt fp1;
	Polh_Record_Format *prfp;
	char str[64];
	char *s,*s2;
	int i;

	if( OBJ_COLS(dp) != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"assign_polh_var_data:  object %s has %d columns, should be 1",
				OBJ_NAME(dp),OBJ_COLS(dp));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	rpdp = (short *) OBJ_DATA_PTR(dp);
	prfp = &station_info[curr_station_idx].sd_multi_prf;
	format_polh_data(&fp1,rpdp,prfp);
	separator="";	/* no leading tab before the first record */
	if( fmt_one(msg_str,&fp1,type) < 0 )
		return;
	s=msg_str;
	i=index;
	while(i--){
		while( *s!=0 && !isspace(*s) ) s++;	/* skip data */
		while( *s!=0 && isspace(*s) ) s++;	/* skip spaces */
	}
#ifdef CAUTIOUS
	if( *s == 0 ){
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  assign_polh_var_data:  not enough words for index %d",
				index);
		NWARN(DEFAULT_ERROR_STRING);
		sprintf(DEFAULT_ERROR_STRING,"formatted string:  %s",msg_str);
		NADVISE(DEFAULT_ERROR_STRING);
		return;
	}
#endif /* CAUTIOUS */

	/* we have to do this check in case index is zero, so the check for leading spaces hasn't been
	 * performed above.  This is an issue with  milliseconds, where a leading space may appear
	 * in the string for values less than 100.
	 */
	while( *s!=0 && isspace(*s) ) s++;	/* skip any leading spaces */

	s2 = str;
	while( *s != 0 && !isspace(*s) )
		*s2++ = *s++;
	*s2=0;				/* BUG should check for overflow */

	assign_var(varname,str);

} /* end assign_polh_var_data */


void format_polh_vector(Data_Obj *dp)
{
	uint32_t i;
	short *rpdp;
	Fmt_Pt fp1;
	int station;

	/* assume that the object has already been checked for proper dim, type... */

	if( n_active_stations == 2 )
		station=0;
	else if(n_active_stations < 1 ){
		NWARN("format_polh_vector:  no active stations!?");
		return;
	} else
		station=curr_station_idx;

	rpdp = (short *) OBJ_DATA_PTR(dp);
	for(i=0;i<OBJ_COLS(dp);i++){
		Polh_Record_Format *prfp;

		prfp = &station_info[station].sd_multi_prf;
		format_polh_data(&fp1,rpdp,prfp);
		display_formatted_point(QSP_ARG  &fp1,prfp);
		rpdp += prfp->rf_n_words;
		if( n_active_stations == 2 )
			station ^= 1;
	}
}
#endif /* FOOBAR */

#ifdef NOT_USED

static void convert_chunk(float *fltp,Fmt_Pt *fpp, Polh_Output_Type type )
{
	switch(type){
		case DATE:
			NWARN("convert_chunk:  can't convert DATE");
			break;

		case SECONDS:
			*fltp = fpp->fp_seconds;
			break;

		case MSECS:
			*fltp = fpp->fp_usecs / 1000.0;
			break;

		default:
			sprintf(DEFAULT_ERROR_STRING,"convert_chunk:  unhandled type code %d!?",type);
			NWARN(DEFAULT_ERROR_STRING);
			break;
	}
}

void convert_polh_vector(Data_Obj *flt_dp,Data_Obj *polh_dp)
{
	uint32_t i,j;
	short *rpdp;
	float *cvt_p;
	Fmt_Pt fp1;
	int station;

	if( OBJ_PREC(flt_dp) != PREC_SP ){
		sprintf(DEFAULT_ERROR_STRING,"convert_polh_data:  object %s has precision %s, should be %s",
				OBJ_NAME(flt_dp),PREC_NAME(OBJ_PREC_PTR(flt_dp)),PREC_NAME(prec_for_code(PREC_SP)));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( OBJ_COLS(flt_dp) != OBJ_COLS(polh_dp) ){
		sprintf(DEFAULT_ERROR_STRING,"convert_polh_data:  vectors %s (%d) and %s (%d) do not have the same number of columns",
				OBJ_NAME(flt_dp),OBJ_COLS(flt_dp),OBJ_NAME(polh_dp),OBJ_COLS(polh_dp));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	/* BUG should make sure that tdim of flt_dp is correct! */

	/* assume that the object has already been checked for proper dim, type... */

	if( n_active_stations == 2 )
		station=0;
	else if(n_active_stations < 1 ){
		NWARN("format_polh_vector:  no active stations!?");
		return;
	} else
		station=curr_station_idx;

	rpdp = (short *) OBJ_DATA_PTR(polh_dp);

	for(i=0;i<OBJ_COLS(polh_dp);i++){
		Polh_Record_Format *prfp;

		cvt_p = ((float *) OBJ_DATA_PTR(flt_dp)) + i * OBJ_PXL_INC(flt_dp);

		prfp = &station_info[station].sd_multi_prf;
		format_polh_data(&fp1,rpdp,prfp);
		for(j=0;j<prfp->rf_n_data;j++){
			Polh_Output_Type type;
			type = prfp->rf_output[j];
			convert_chunk(cvt_p,&fp1,type);
			cvt_p += od_tbl[type].od_strings;	/* BUG not the right variable?... */
		}
		rpdp += prfp->rf_n_words;
		if( n_active_stations == 2 )
			station ^= 1;
	}
}
#endif // NOT_USED


/* Put the polhemus into continuous data mode.
 * Does this mean that it will buffer readings?
 * How many readings will be buffered by the driver???
 */

COMMAND_FUNC( do_start_continuous_mode )
{
	start_continuous_mode();
}

void start_continuous_mode(void)
{
#ifdef INSIDE_TRACK
	if( ioctl( polh_fd, POLHEMUS_SET_CONTINUOUS_MODE, NULL )  < 0 ){
		perror("ioctl");
		NWARN("Unable to set polhemus device to continuous output");
		return;
	}
	usleep(100000);	/* 1/10 sec */
#else
	if(send_polh_cmd(PH_CONTINUOUS, NULL) < 0 ) {
		NWARN("start_continuous_mode:  unable to start continuous mode");
		return;
	}
#endif

	/* should we call clear_polh_dev() here???
	 *
	 * How do we know for sure that the buffer has been emptied?
	 */

	polh_continuous = 1;
//NADVISE("start_continuous_mode:  done.");
}

COMMAND_FUNC( do_stop_continuous_mode )
{
	stop_continuous_mode();
}

void stop_continuous_mode(void)
{
	if( !polh_continuous ){
		NWARN("stop_continuous_mode:  device is already stopped!?");
	}

#ifdef INSIDE_TRACK
	/* Set the polhemus device back to non-continuous mode. */
	if( ioctl(polh_fd, POLHEMUS_SET_NONCONTINUOUS_MODE, NULL) < 0 ) 
		NWARN("stop_continuous_output_file: Unable to set polhemus device to non-continuous mode!");
	else
		polh_continuous = 0;
#else
	if(send_polh_cmd(PH_NONCONTINUOUS, NULL) < 0 ) {
		NWARN("stop_continuous_mode:  unable to stop continuous mode");
		return;
	}
	flush_input_data();
#endif

	polh_continuous = 0;
}

void polh_read_async( int flag )
{
//sprintf(DEFAULT_ERROR_STRING,"polh_read_async:  setting flag to %d",flag);
//NADVISE(DEFAULT_ERROR_STRING);
	_read_async=flag;
}

