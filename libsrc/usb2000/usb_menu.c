/* usb2000.c */
#include "quip_config.h"

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	/* malloc(), strtoul() */
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>	/* usleep() */
#endif

#include "quip_prot.h"

#include "usb2000.h"


static COMMAND_FUNC( pixel_inq)
{
	int pb_width;

	pb_width = get_pb_width(SINGLE_QSP_ARG);

	if ( pb_width < 0 )
		return;

	sprintf(msg_str,"pixel boxcar width: %d", pb_width);
	advise(msg_str);
}

static COMMAND_FUNC( scan_inq )
{
	int n_scans;

	n_scans = get_n_scans(SINGLE_QSP_ARG);

	if ( n_scans < 0 )
		return;

	sprintf(msg_str, "number of discrete spectra being summed together: %d",n_scans);
	advise(msg_str);
}

static COMMAND_FUNC( it_inq )
{
	int integ_time;

	integ_time = get_integ_time(SINGLE_QSP_ARG);

	if ( integ_time < 0)
		return;

	sprintf(msg_str, "integration time: %d",integ_time);
	advise(msg_str);
}


static COMMAND_FUNC( baud_inq )
{
	int baud_rate;

	baud_rate = get_baud_rate(SINGLE_QSP_ARG);

	if ( baud_rate < 0 )
		return;

	switch(baud_rate) {
		case 0: advise("baud rate: 2400"); break;
		case 1: advise("baud rate: 4800"); break;
		case 2: advise("baud rate: 9600"); break;
		case 3: advise("baud rate: 19200"); break;
		case 4: advise("baud rate: 38400"); break;
		case 5: advise("baud rate: 57600"); break;
#ifdef CAUTIOUS
		default:
			sprintf(ERROR_STRING, "CAUTIOUS: unknown baud rate: %d", baud_rate);
			WARN(ERROR_STRING);
#endif /* CAUTIOUS */
	}
}

static COMMAND_FUNC( trig_mode_inq )
{
	int trig_mode;

	trig_mode = get_trig_mode(SINGLE_QSP_ARG);

	if ( trig_mode < 0 )
		return;

	switch(trig_mode) {
		case 0: advise("trigger mode: normal"); break;
		case 1: advise("trigger mode: software trigger"); break;
		case 3: advise("trigger mode: external hardware trigger"); break;
#ifdef CAUTIOUS
		default: WARN("CAUTIOUS: unknown trigger mode");
#endif /* CAUTIOUS */
	}
}

static COMMAND_FUNC( lamp_inq )
{
	int status;

	status = get_lamp_status(SINGLE_QSP_ARG);

	if (status < 0)
		return;

	switch(status){
		case LAMP_HIGH:	prt_msg("lamp enable: high"); break;
		case LAMP_LOW:	prt_msg("lamp enable: low"); break;
#ifdef CAUTIOUS
		default:
			WARN("CAUTIOUS:  unrecognized status returned by get_lamp_status()!?");
			break;
#endif /* CAUTIOUS */
	}
}

static COMMAND_FUNC( calib_const_inq )
{
	int calib_index;
	char calib_const[MAX_SIZEOF_CALIB_CONST];
	char prompt[LLEN];

	calib_index = HOW_MANY("calib constant index");

	if( calib_index<MIN_CALIB_INDEX || calib_index>MAX_CALIB_INDEX ) {
	       sprintf(ERROR_STRING, " (%d) should be in range %d - %d",
			calib_index, MIN_CALIB_INDEX, MAX_CALIB_INDEX);

	       WARN(ERROR_STRING);
	       return;
	}

	if ( do_calib_inq(QSP_ARG  CALIB_CONSTS, calib_index, calib_const) < 0 )
		return;

	if( calib_index == 15 ) {
		char grating[3];
		char filter_wav_len[4];
		char slit_size[4];

		sprintf( grating,"%c%c", calib_const[0], calib_const[1] );
		sprintf( filter_wav_len,"%c%c%c", calib_const[3], calib_const[4], calib_const[5] );
		sprintf( slit_size,"%c%c%c", calib_const[7], calib_const[8], calib_const[9] );

		sprintf(prompt, "Grating: %s\nFilter Wavelength: %s\nSlit Size: %s", grating, filter_wav_len, slit_size );

	} else if( calib_index == 16 ) {
		sprintf(prompt, "USB2000 Configuration: %s", calib_const);

	} else {
		sprintf(prompt, "Calibration Constant: %s", calib_const);
	}

	advise(prompt);
}

static COMMAND_FUNC( timer_inq )
{
	int timer;

	timer = get_timer_type(SINGLE_QSP_ARG);

	if ( timer < 0 )
		return;

	switch(timer) {
		case 0: advise("Integration clock: 16 bit\nBaud rate generator: 8 bit"); break;
		case 1: advise("Integration clock: 8 bit\nBaud rate generator: 16 bit"); break;

#ifdef CAUTIOUS
		default:
			WARN("CAUTIOUS:  unrecognized value returned by get_timer_type()!?");
#endif /* CAUTIOUS */
	}

}

static COMMAND_FUNC( accessories_inq )	{ advise("sorry: this inquiry has not been implemented"); }

static COMMAND_FUNC( do_n_of_scans )
{
	int n_of_scans;
	char prompt[LLEN];

	n_of_scans = get_n_of_scans(SINGLE_QSP_ARG);

	if ( n_of_scans < 0 )
		return;

	sprintf(prompt,"Number of scans in spectral data memory: %d", n_of_scans);
	advise(prompt);
}


#define ADD_CMD(s,f,h)	ADD_COMMAND(usb2k_inquiry_menu,s,f,h)
MENU_BEGIN(usb2k_inquiry)
ADD_CMD( integ_time,	it_inq,			integration time		)
ADD_CMD( baud,		baud_inq,		baud rate			)
ADD_CMD( timer,		timer_inq,		timer operation			)
ADD_CMD( pb_width,	pixel_inq,		pixel boxcar width		)
ADD_CMD( scan,		scan_inq,		number of discrete spectra being summed together )
ADD_CMD( trig_mode,	trig_mode_inq,		trigger mode			)
ADD_CMD( lamp,		lamp_inq,		lamp enable			)
ADD_CMD( calib_const,	calib_const_inq,	calibration constant		)
ADD_CMD( accessories,	accessories_inq,	read plugged-in ocean optics compatible accessories )
ADD_CMD( n_of_scans,		do_n_of_scans,		return number of scans in spectral data memory )
MENU_END(usb2k_inquiry)

static short mk_spec_vector(QSP_ARG_DECL  short size, Spectral_Data *sdp)
{
	Data_Obj *dp;
	char name[LLEN];
	u_short n_of_bytes;
	u_short i,j;
	float tmp_data[MAX_N_OF_PIXELS*2];

	strcpy(	name, NAMEOF("name for new spectra data vector") );

	dp = dobj_of(QSP_ARG  name);

	if( dp != NULL ){
		sprintf(ERROR_STRING,"Can't create new data vector %s, name is in use already",
			name);
		WARN(ERROR_STRING);
		/* We could delete the existing object here, or recycle it if the dimension is correct... */
		return -1;
	}

	dp = mk_vec(QSP_ARG  name,size,2,PREC_FOR_CODE(PREC_SP));

	if( dp == NULL ) {
		WARN("unable to create spectra data vector");
		return -1;
	}

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING, "mk_spec_vector: object %s must be contiguous",OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return -1;
	}

	n_of_bytes = size*sizeof(float);

	/* we now format the data */

/* BUG: instead of i=0, we should start from the first pixel number, this is for cases in which the user supplies the starting and ending pixel numbers, or should the values read be returned from the start of the array? need to think more on this. */

	j=0;

	for( i=0; i<size; i++) {
		tmp_data[j++] = sdp->sd_wavlen[i];
		tmp_data[j++] = sdp->sd_spec_data[i];

	}

	SET_OBJ_DATA_PTR(dp, (float *)memcpy(OBJ_DATA_PTR(dp), tmp_data, n_of_bytes*2) );

	return 0;
}

static COMMAND_FUNC( do_spec_acq )
{
	short size;
	Spectral_Data sdp;

	if ( ( size = spec_acq(QSP_ARG  &sdp) ) < 0 )
		return;

	if( data_strg_mode == DISABLE ) {
		if( mk_spec_vector(QSP_ARG  size, &sdp) < 0 )
			return;
	}

}

static COMMAND_FUNC( do_add_scans )
{
	int data_word;
	char prompt[LLEN];

#define MIN_N_SCANS		1
#define MAX_N_SCANS		15

	sprintf(prompt, "number of discrete spectra to be summed together (%d - %d)", MIN_N_SCANS, MAX_N_SCANS);
	data_word= HOW_MANY(prompt);

	if( data_word<MIN_N_SCANS || data_word>MAX_N_SCANS ) {
		sprintf(ERROR_STRING, "number (%d) should be in range %d - %d",
			data_word, MIN_N_SCANS, MAX_N_SCANS);

		WARN(ERROR_STRING);
		return;
	}

	if ( add_scans(QSP_ARG  data_word) < 0 )
		return;
}

static COMMAND_FUNC( do_clr_spectra )
{
	int data_word;

	data_word = 1;		/* 0 or 1 doesn't matter (pg.10) */

	if ( clr_spectra(QSP_ARG  data_word) < 0 )
		return;
}

#define N_PIXEL_MODES	4

/* TODO: Replace all references to pixels by wavelengths */

static COMMAND_FUNC( do_sampling_mode )
{
	int pxl_mode;
	float n;
	float x,y;
	float wavelengths[10];
					/*
					 * we can specify a max of
					 * 10 pixel locations for
					 * case 4 on pg.11
					 */

	static const char *pxl_mode_strs[N_PIXEL_MODES]={"all_wavlens", "every_nth", "nth_x_to_y", "choose"};

	pxl_mode = WHICH_ONE("sampling mode", N_PIXEL_MODES, pxl_mode_strs);
	if( pxl_mode < 0 ) return;

	switch(pxl_mode) {
		case 0:
			/* there are no args for case 'P0': manual pg.11 */

			/* set these vars just to silence compiler warnings */
			x=y=n=0;
			break;

		case 1:
			n = HOW_MUCH("number of wavelengths to skip");

			if( n<(1/BIN_WIDTH) || n>=WAVELENGTH_RANGE ) {
				sprintf(ERROR_STRING, "number of wavelengths(%f) has to be in range(%f-%f)",
						n, (1/BIN_WIDTH), WAVELENGTH_RANGE );
				WARN(ERROR_STRING);

				return;
			}
			/* set these vars just to silence compiler warnings */
			x=y=0;

			break;

		case 2:
			x = HOW_MUCH("starting wavelength");

			if( x<MIN_WAVELENGTH || x>MAX_WAVELENGTH ) {
				sprintf(ERROR_STRING, "wavelength(%f) has to be in range(%f-%f)",
						x, MIN_WAVELENGTH, MAX_WAVELENGTH);
				WARN(ERROR_STRING);

				return;
			}

			y = HOW_MUCH("ending wavelength");

			if( y<MIN_WAVELENGTH || y>MAX_WAVELENGTH ) {
				sprintf(ERROR_STRING, "wavelength(%f) has to be in range(%f-%f)",
						y, MIN_WAVELENGTH, MAX_WAVELENGTH);
				WARN(ERROR_STRING);

				return;
			}

			n = HOW_MUCH("number of wavelengths to skip");

			if( n<(1/BIN_WIDTH) || n>=WAVELENGTH_RANGE ) {
				sprintf(ERROR_STRING, "number of wavelength(%f) has to be in range(%f-%f)",
						n, (1/BIN_WIDTH), WAVELENGTH_RANGE);
				WARN(ERROR_STRING);

				return;
			}

			break;

		case 3:
		{
			int i;

			n = HOW_MANY("number of sampling wavelengths");

			/* see manual pg.11 for these limits */
			if( n<0 || n>10 ) {
				sprintf(ERROR_STRING, "n (%f) has to be in range(0-10)",n);
				WARN(ERROR_STRING);
				return;
			}

			for(i=0; i<n; i++) {
				char prompt[LLEN];

				sprintf(prompt, "wavelength (%d)", i+1);
				wavelengths[i] = HOW_MUCH(prompt);

				if( wavelengths[i]<MIN_WAVELENGTH || wavelengths[i]>MAX_WAVELENGTH ) {
					sprintf(ERROR_STRING, "wavelength(%f) has to be in range(%f-%f)",
						wavelengths[i], MIN_WAVELENGTH, MAX_WAVELENGTH);
					WARN(ERROR_STRING);

					return;
				}
			}
			/* set these vars just to silence compiler warnings */
			x=y=0;

			break;

		} /* case 3 */
		default:
			WARN("do_sampling_mode:  bad case!?");
			return;
			break;

	} /* switch */

	if ( set_pxl_mode(QSP_ARG  pxl_mode, x, y, n, wavelengths) < 0 )
		return;
}

static COMMAND_FUNC( do_data_strg )
{
	int data_word;

	/* manual pg.10: 1 for storage */
	data_word = ASKIF("store scans in spectral memory");

	if ( data_strg(QSP_ARG  data_word) < 0 )
		return;


}

static COMMAND_FUNC( do_get_scan )
{
	short size;
	Spectral_Data sdp;

	if ( ( size = get_scan(QSP_ARG  &sdp) ) < 0 )
		return;

	if( mk_spec_vector(QSP_ARG  size, &sdp) < 0 )
		return;

}

static COMMAND_FUNC( do_save_spec )
{
	Data_Obj *dp;
	char name[LLEN];
	FILE *fp;
	/* BUG what if pathname is longer than 256??? */
	const char *filename;

	strcpy(name, NAMEOF("name of spec vector to save") );

	dp = dobj_of(QSP_ARG  name);
	if( dp == NULL ){
		sprintf(ERROR_STRING,"%s does not exist", name);
		WARN(ERROR_STRING);
		return;
	}

	filename = NAMEOF("output file");

	if( strcmp(filename,"-") && strcmp(filename,"stdout") ){
		fp=TRYNICE( filename, "w" );
		if( !fp ) return;

	} else {
		fp = stdout;
	}

	pntvec(QSP_ARG  dp,fp);

	if( fp != stdout ) {
		fclose(fp);
	}

	return;
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(spectra_menu,s,f,h)
MENU_BEGIN(spectra)
ADD_CMD( sampling_mode, do_sampling_mode,	set sampling mode			)
ADD_CMD( acquire,	do_spec_acq,		acquire spectra with current set of operating params )
ADD_CMD( save,		do_save_spec,		write acquired spectra to ascii file	)
ADD_CMD( add_scans,	do_add_scans,		number of discrete spectra to sum together )
ADD_CMD( clr_spectra,	do_clr_spectra,		clear memory				)
ADD_CMD( storage,	do_data_strg,		enable/disable data storage mode	)
ADD_CMD( scan,		do_get_scan,		read out one scan from specified spectral memory )
MENU_END(spectra)

static COMMAND_FUNC( do_data_comp )
{
	int data_word;

	data_word = ASKIF("enable data compression");

	if ( data_comp(QSP_ARG  data_word) < 0 )
		return;
}


static COMMAND_FUNC( do_data_mode )
{
	Cmd_Index data_mode;
	int i;

#define N_DATA_MODES	2

	static const char *data_mode_strs[N_DATA_MODES]={"binary", "ascii"};

	i = WHICH_ONE("data mode", N_DATA_MODES, data_mode_strs);
	if( i < 0 ) return;

	if(i)
		data_mode = ASCII;
	else
		data_mode = BINARY;

	if ( set_data_mode(QSP_ARG  data_mode) < 0 )
		return;
}

static COMMAND_FUNC( do_checksum )
{
	int data_word;

	/* manual pg.13 */
	data_word = ASKIF("transmit checksum value at end of scan");

	if ( set_checksum(QSP_ARG  data_word) < 0 )
		return;
}

static COMMAND_FUNC( do_pb_width )
{
	int data_word;
	char prompt[LLEN];

#define MIN_PB_WIDTH		0
#define MAX_PB_WIDTH		15

	/*
	 * BUG: Should this feature should be asking for pb width in terms of wavelengths?
	 */

	sprintf(prompt, "number of pixels to be averaged together (%d - %d)", MIN_PB_WIDTH, MAX_PB_WIDTH);
	data_word = HOW_MANY(prompt);

	if( data_word<MIN_PB_WIDTH || data_word>MAX_PB_WIDTH ) {
		sprintf(ERROR_STRING, "number (%d) should be in range %d - %d",
			data_word, MIN_PB_WIDTH, MAX_PB_WIDTH);

		WARN(ERROR_STRING);
		return;
	}

	if ( set_pb_width(QSP_ARG  data_word) < 0 )
		return;

}

static COMMAND_FUNC( do_integ_time )
{
	int data_word;
	char prompt[LLEN];

#define MIN_INTEG_TIME		3		/* the windows version (OOBase32) goes down to 3 */
#define MAX_INTEG_TIME		65535

	sprintf(prompt, "integration time (milli secs) (%d - %d)", MIN_INTEG_TIME, MAX_INTEG_TIME);
	data_word= HOW_MANY(prompt);

	if( data_word<MIN_INTEG_TIME || data_word>MAX_INTEG_TIME ) {
		sprintf(ERROR_STRING, "integration time(%d) should be in range %d - %d",
			data_word, MIN_INTEG_TIME, MAX_INTEG_TIME);

		WARN(ERROR_STRING);
		return;
	}

	if ( set_integ_time(QSP_ARG  data_word) < 0 )
		return;
}

static COMMAND_FUNC( do_lamp )
{
	int data_word;

	data_word = ASKIF("enable lamp");

	if ( set_lamp(QSP_ARG  data_word) < 0 )
		return;
}

#ifdef USE_SERIAL_LINE
static COMMAND_FUNC( do_baud )
{
	int data_word;

#define N_BAUD_MODES	6

	static const char *baud_mode_strs[N_BAUD_MODES]={"2400", "4800", "9600", "19200", "38400", "57600"};

	data_word = WHICH_ONE("baud rate", N_BAUD_MODES, baud_mode_strs);
	if( data_word < 0 ) return;

	if ( set_baud_rate(QSP_ARG  data_word) < 0 )
		return;
}
#endif /* USE_SERIAL_LINE */

static COMMAND_FUNC( do_trig_mode )
{
	int data_word;

#define N_TRIG_MODES	4

	static const char *trig_mode_strs[N_TRIG_MODES]={"normal", "sw_trig", "N/A", "ext_hw_trig"};

	data_word = WHICH_ONE("trig mode", N_TRIG_MODES, trig_mode_strs);
	if( data_word < 0 ) return;

	if ( set_trig_mode(QSP_ARG  data_word) < 0 )
		return;
}

static COMMAND_FUNC( do_calib_const )
{
	char prompt[LLEN];
	int calib_index;
	const char *calib_const;

	/* need to make sure that mode is ascii (manual pg.13)
	 * or it would be better to just check the mode and if it
	 * isn't ascii then change the mode to ascii, do the cmd
	 * and then change the mode back to binary.
	 */

	if(!ascii_mode) {
		WARN("This command requires ASCII mode");
		return;
	}

	sprintf(prompt, "calib constant index (%d - %d)", MIN_CALIB_INDEX, MAX_CALIB_INDEX);
	calib_index = HOW_MANY(prompt);

	if( calib_index<MIN_CALIB_INDEX || calib_index>MAX_CALIB_INDEX ) {
		sprintf(ERROR_STRING, " (%d) should be in range %d - %d",
			calib_index, MIN_CALIB_INDEX, MAX_CALIB_INDEX);

		WARN(ERROR_STRING);
		return;
	}

	/* BUG: need to make sure that the const values are given in the specified format.
	 * specified format can be seen in the file calib_consts or by reading the calib consts.
	 */

	calib_const = NAMEOF("calib constant value");

	if ( set_calib_const(QSP_ARG  CALIB_CONSTS, calib_index, calib_const) < 0 )
		return;
}

static COMMAND_FUNC( do_timer )
{
	int data_word;

	/* Baud rate generator has 16-bit timer by default.
	 * This command will re-initialize baud rate, trig mode
	 * and lamp enable (manual pg.14).
	 */

	if( ASKIF("enable 8-bit timer for baud rate generator") )
		data_word = 0;
	else
		data_word = 1;

	if ( set_timer(QSP_ARG  data_word) < 0 )
		return;
}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(usb2k_control_menu,s,f,h)
MENU_BEGIN(usb2k_control)
ADD_CMD( integ_time,		do_integ_time,		set integtration time			)
ADD_CMD( data_mode,		do_data_mode,		select mode for data sent and received (default mode: binary) )
ADD_CMD( compress,		do_data_comp,		enable/disable data compression	)
ADD_CMD( checksum,		do_checksum,		enable/disable checksum mode		)
ADD_CMD( pb_width,		do_pb_width,		set pixel boxcar width		)
ADD_CMD( lamp,			do_lamp,		set lamp enable to low/high		)
#ifdef USE_SERIAL_LINE
ADD_CMD( baud,			do_baud,		change baud rate			)
#endif /* USE_SERIAL_LINE */
ADD_CMD( trig_mode,		do_trig_mode,		set trigger mode			)
ADD_CMD( timer,			do_timer,		set timer operation			)
ADD_CMD( set_calib_const,	do_calib_const,		write a calibration constant to EEPROM )
MENU_END(usb2k_control)

static COMMAND_FUNC( analog_op_inq )
{
	int analog_op;
	char prompt[LLEN];

	analog_op = get_analog_op(SINGLE_QSP_ARG);

	if ( analog_op < 0 )
		return;

	sprintf(prompt,"analog output: %d", analog_op);
	advise(prompt);

}

static COMMAND_FUNC( led_mode_inq )
{
	int led_mode;

	led_mode = get_led_mode(SINGLE_QSP_ARG);

	if (led_mode< 0)
		return;

	switch(led_mode) {
		case 0: advise("LED mode: CW"); break;
		case 1: advise("LED mode: pulsed"); break;
#ifdef CAUTIOUS
		default: WARN("CAUTIOUS: unknown led mode");
#endif /* CAUTIOUS */
	}

}

static COMMAND_FUNC( do_ls450_const_inq )
{
	int calib_index;
	char calib_const[MAX_SIZEOF_CALIB_CONST];
	char prompt[LLEN];

	calib_index = HOW_MANY("calib constant index");

	if( calib_index<MIN_LS450_CALIB_INDEX || calib_index>MAX_LS450_CALIB_INDEX ) {
	       sprintf(ERROR_STRING, " (%d) should be in range %d - %d",
			calib_index, MIN_LS450_CALIB_INDEX, MAX_LS450_CALIB_INDEX);

	       WARN(ERROR_STRING);
	       return;
	}

	if ( do_calib_inq(QSP_ARG  LS_450_CALIB_CONST, calib_index, calib_const) < 0 )
		return;

	sprintf(prompt, "Calibration Constant: %s", calib_const);
	advise(prompt);
}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(ls450_inquiry_menu,s,f,h)
MENU_BEGIN(ls450_inquiry)
ADD_CMD( analog_op,	analog_op_inq,		analog output				)
ADD_CMD( led_mode,	led_mode_inq,		LED operational mode			)
ADD_CMD( calib_const,	do_ls450_const_inq,	USB-LS450 calibration coefficient	)
MENU_END(ls450_inquiry)


static COMMAND_FUNC( do_analog_op )
{
	int analog_op;
	char prompt[LLEN];

#define MIN_ANALOG_OP		0
#define MAX_ANALOG_OP		65535

	sprintf(prompt, "analog output %d(0) - %d(20mA)", MIN_ANALOG_OP, MAX_ANALOG_OP);
	analog_op= HOW_MANY(prompt);

	if( analog_op<MIN_ANALOG_OP|| analog_op>MAX_ANALOG_OP) {
		sprintf(ERROR_STRING, "analog output (%d) should be in range %d - %d",
			analog_op, MIN_ANALOG_OP, MAX_ANALOG_OP);

		WARN(ERROR_STRING);
		return;
	}

	if ( set_analog_op(QSP_ARG  analog_op) < 0 )
		return;
}

static COMMAND_FUNC( do_led_mode )
{
	int led_mode;

#define N_LED_MODES	2

	static const char *led_mode_strs[N_LED_MODES]={"cw", "pulsed"};

	led_mode = WHICH_ONE("LED mode", N_LED_MODES, led_mode_strs);
	if( led_mode < 0 ) return;

	if ( set_led_mode(QSP_ARG  led_mode) < 0 )
		return;
}

static COMMAND_FUNC( do_temp )
{
	char *temp;
	char prompt[LLEN];

#define SIZEOF_TEMP_STRING	5	/* wild guess, I think it should be no more than 5
					 * since the returned temp is of the form ***0
					 */

	if(!ascii_mode) {
		WARN("This command requires ASCII mode");
		return;
	}

	temp = (char *)malloc(SIZEOF_TEMP_STRING);

	if( get_temp(QSP_ARG  temp) < 0 )
		return;

	sprintf(prompt, "Temperature: %s (Centigrade)", temp);
	advise(prompt);
}

static COMMAND_FUNC( do_ls450_calib_const )
{
	char prompt[LLEN];
	int calib_index;
	const char *calib_const;

	/* make sure that mode is ascii (manual pg.16) */

	if(!ascii_mode) {
		WARN("This command requires ASCII mode");
		return;
	}

	sprintf(prompt, "calib constant index (%d - %d)", MIN_LS450_CALIB_INDEX, MAX_LS450_CALIB_INDEX);
	calib_index = HOW_MANY(prompt);

	if(calib_index<MIN_LS450_CALIB_INDEX || calib_index>MAX_LS450_CALIB_INDEX ) {
		sprintf(ERROR_STRING, " (%d) should be in range %d - %d",
			calib_index, MIN_LS450_CALIB_INDEX, MAX_LS450_CALIB_INDEX);

		WARN(ERROR_STRING);
		return;
	}

	/* BUG: need to make sure that the const values are given in the specified format.
	 * specified format can be seen in the file calib_consts or by reading the calib consts.
	 */

	calib_const = NAMEOF("calib constant value");

	if ( set_calib_const(QSP_ARG LS_450_CALIB_CONST, calib_index, calib_const) < 0 )
		return;

}

static COMMAND_FUNC( inq_ls450 )
{
	PUSH_MENU(ls450_inquiry);
}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(ls450_menu,s,f,h)
MENU_BEGIN(ls450)
ADD_CMD( analog_op,		do_analog_op,		set analog output (0-20mA)		)
ADD_CMD( led_mode,		do_led_mode,		set LED operational mode		)
ADD_CMD( temp,			do_temp,		read temperature			)
ADD_CMD( set_calib_const,	do_ls450_calib_const,	set a USB-LS450 calibration coefficient )
ADD_CMD( inquire,		inq_ls450,		inquire the settings			)
MENU_END(ls450)



static COMMAND_FUNC( do_spectra_menu )
{
	PUSH_MENU(spectra);
}

static COMMAND_FUNC( do_cntrl_menu )
{
	PUSH_MENU(usb2k_control);
}

static COMMAND_FUNC( do_inq_menu )
{
	PUSH_MENU(usb2k_inquiry);
}

static COMMAND_FUNC( do_ver )
{
	char *ver;
	char prompt[LLEN];

#define SIZEOF_VER_STRING	7
	ver = (char *)malloc(SIZEOF_VER_STRING);

	if( get_ver(QSP_ARG  ver) < 0 )
		return;

	sprintf(prompt, "Version: %s", ver);
	advise(prompt);

}

#ifdef NOT_USED
static void poll_accessories()
{
	/*BUG: this should be a combination of + followed by ?+ */

	WARN("sorry: This cmd not yet implemented");

}
#endif /* NOT_USED */

static COMMAND_FUNC( do_ls450_menu )
{
	PUSH_MENU(ls450);
}

static COMMAND_FUNC(do_clear_input_buf)
{
	clear_input_buf(SINGLE_QSP_ARG);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(usb2k_menu,s,f,h)
MENU_BEGIN(usb2k)
ADD_CMD( spectra,	do_spectra_menu,	spectra sub-menu			)
ADD_CMD( cntrl,		do_cntrl_menu,		control sub-menu			)
ADD_CMD( inquire,	do_inq_menu,		inquire the settings			)
ADD_CMD( version,	do_ver,			microcode version number		)
/*
ADD_CMD( accessories,	poll_accessories,	poll plugged-in ocean optics compatible accessories)
*/
ADD_CMD( usb_ls450,	do_ls450_menu,		USB-LS450 sub-menu			)
ADD_CMD( flush,		do_clear_input_buf,	flush the contents of input buffer	)
MENU_END(usb2k)


COMMAND_FUNC( do_usb2000_menu )
{
	static int inited=0;

	if( !inited ){
		init_usb2000(SINGLE_QSP_ARG);
		inited=1;
	}

	PUSH_MENU(usb2k);

}



