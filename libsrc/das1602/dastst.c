#include "quip_config.h"

#ifdef HAVE_DAS1602

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

#include "data_obj.h"
#include "submenus.h"
#include "query.h"
#include "debug.h"	/* verbose */

#include "ioctl_das1602.h"
#include "pacer.h"

#define VALID_ADC_PREC(dp)					\
								\
	((MACHINE_PREC(dp)==PREC_UIN)||(MACHINE_PREC(dp)==PREC_IN))



#define VALID_DIO_PREC(dp)					\
								\
	((MACHINE_PREC(dp)==PREC_UBY)||(MACHINE_PREC(dp)==PREC_BY))

#define MAX_BYTE_VALUE	255
#define NVRAM_SIZE	256
static int first_adc_channel=(-1),last_adc_channel=(-1);
static int adc_trigger_channel=(-1);
static u_short adc_trig_level=0x8000;
static int use_sw_trigger=0;

static int das_fd = (-1);

static COMMAND_FUNC( set_reg )
{
	Reg_Data reg_setting;

	reg_setting.reg_blocki = HOW_MANY("block_index (1-4)");
	reg_setting.reg_offset = HOW_MANY("register offset");

	reg_setting.reg_data.u_s = HOW_MANY("register data");
	/* BUG?  does this work for char data?? */

	if( das_fd < 0 ){
		NWARN("device not open");
		return;
	}

	if( ioctl(das_fd,DAS_SET_REG,&reg_setting) < 0 ){
		perror("ioctl");
		NWARN("error setting das1602 register");
	}
}


static COMMAND_FUNC( get_reg )
{
	Reg_Data reg_setting;

	reg_setting.reg_blocki = HOW_MANY("block_index (1-4)");
	reg_setting.reg_offset = HOW_MANY("register offset");

	if( das_fd < 0 ){
		NWARN("device not open");
		return;
	}

	reg_setting.reg_data.u_s = 0;	/* clear all bits */
	if( ioctl(das_fd,DAS_GET_REG,&reg_setting) < 0 ){
		perror("ioctl");
		NWARN("error getting das1602 register");
	}

	sprintf(error_string,"Reg %d %d:  0x%x (0x%x)",
		reg_setting.reg_blocki,reg_setting.reg_offset,
		reg_setting.reg_data.u_s,reg_setting.reg_data.u_c);
	advise(error_string);

	sprintf(error_string,"0x%x",reg_setting.reg_data.u_s);
	ASSIGN_VAR("reg_data",error_string);
}

static COMMAND_FUNC( open_device )
{
	das_fd = open("/dev/das1602",O_RDWR);
	if( das_fd < 0 ){
		perror("open");
		NWARN("error opening /dev/das1602");
	}
}

static Command reg_ctbl[]={
{ "open",	open_device,	"open device register file"	},
{ "getreg",	get_reg,	"read device register"		},
{ "setreg",	set_reg,	"write device register"		},
{ "quit",	popcmd,		"exit submenu"			},
{ NULL_COMMAND							}
};

static COMMAND_FUNC( reg_menu )
{
	PUSHCMD(reg_ctbl,"registers");
}

static int adc_fd=(-1);
static int dac_fd=(-1);

#define CONFIRM_ADC						\
				if( adc_fd < 0 ){		\
					NWARN("adc not open");	\
					return;			\
				}
#define CONFIRM_DAC						\
				if( dac_fd < 0 ){		\
					NWARN("dac not open");	\
					return;			\
				}

#define CONFIRM_DIO(channel)					\
								\
	if( dio_fd[ channel ] < 0 ){				\
		sprintf(error_string,				\
			"dio channel %d not open",channel);	\
		NWARN(error_string);				\
		return;						\
	}

static int get_adc_channel(QSP_ARG_DECL  const char *pmpt,
						int chmin, int chmax)
{
	int chno;

	sprintf(msg_str,"%s (%d-%d)",pmpt,chmin,chmax);
	chno = HOW_MANY(msg_str);

	if( chno < chmin || chno > chmax ){
		sprintf(error_string,
	"channel number (%d) must be between %d and %d",chno,chmin,chmax);
		NWARN(error_string);
		return(-1);
	}
	return(chno);
}

static int get_dac_channel(SINGLE_QSP_ARG_DECL)
{
	int chno;

	chno = HOW_MANY("channel number (0-1, or 2 for both)");

	if( chno < 0 || chno > 2 ){
		sprintf(error_string,
	"channel number (%d) must be between 0 and 2",chno);
		NWARN(error_string);
		return(-1);
	}
	return(chno);
}

static COMMAND_FUNC( open_adc )
{
	char fn[16];

	first_adc_channel = get_adc_channel(QSP_ARG  "first a/d channel",0,7);
	last_adc_channel = get_adc_channel(QSP_ARG  "last a/d channel",first_adc_channel,7);

	if( first_adc_channel < 0 || last_adc_channel < 0 ) return;

	if( adc_fd >= 0 ){
		NWARN("An adc is already open!?");
		return;
	}

	if( last_adc_channel == first_adc_channel )
		sprintf(fn,"/dev/adc%d",first_adc_channel);
	else
		sprintf(fn,"/dev/adc%d%d",first_adc_channel,last_adc_channel);

	adc_fd = open(fn,O_RDONLY);
	if( adc_fd < 0 ){
		perror("open");
		sprintf(error_string,"Error opening adc file %s",fn);
		NWARN(error_string);
		return;
	}
}

static COMMAND_FUNC( close_adc )
{
	CONFIRM_ADC

	close(adc_fd);
	adc_fd = -1;
}

#define GET_ADC_DATA( addr , n )									\
													\
		if( (nr=read(adc_fd,(addr),(n))) != (n) ){						\
			if( nr < 0 ){									\
				sprintf(error_string,"read (errno=%d)",errno);				\
				perror(error_string);							\
			} else {									\
				sprintf(error_string,							\
		"%d bytes requested, %d actually read,nw=%d", n,nr,nw);				\
				advise(error_string);							\
			}										\
			NWARN("error reading adc data");							\
		}

#define N_TRIG	2

static COMMAND_FUNC( read_adc )
{
	Data_Obj *dp;
	int nb;
	int nr;
	u_short tmp_data[16*N_TRIG];	/* this is the maximum number of channels? */
uint32_t nw=0;

	dp = PICK_OBJ("data vector");

	if( dp == NO_OBJ ) return;

	if( ! VALID_ADC_PREC(dp) ){
		sprintf(error_string,
	"Object %s has precision %s, should be %s or %s for ADC data",
			dp->dt_name,prec_name[MACHINE_PREC(dp)],
			prec_name[PREC_IN],prec_name[PREC_UIN]);
		NWARN(error_string);
		return;
	}
	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(error_string,
	"read_adc:  object %s must be contiguous",dp->dt_name);
		NWARN(error_string);
		return;
	}

	CONFIRM_ADC


	nb = N_TRIG*sizeof(u_short)*(1+last_adc_channel-first_adc_channel);
	if( use_sw_trigger ){
		/* first wait for the signal to drop below the threshold */
		do {
			GET_ADC_DATA(&tmp_data[0],nb)
			nw++;
if( verbose ){
sprintf(error_string,"polling %d for tmp_data[%d] = 0x%x to go below 0x%x",
nw,adc_trigger_channel,tmp_data[adc_trigger_channel-first_adc_channel],
adc_trig_level);
advise(error_string);
}
		} while( tmp_data[adc_trigger_channel-first_adc_channel] > adc_trig_level );

sprintf(error_string,"after %d polls, signal 0x%x below threshold 0x%x",
nw, tmp_data[adc_trigger_channel-first_adc_channel], adc_trig_level );
advise(error_string);
sprintf(error_string,"tmp_data = 0x%x   0x%x   0x%x   0x%x   0x%x   0x%x",
tmp_data[0],tmp_data[1],tmp_data[2],tmp_data[3],tmp_data[4],tmp_data[5]);
advise(error_string);

		/* Now wait for it to go above the threshold */
		nw=0;
		do {
			GET_ADC_DATA(&tmp_data[0],nb)
			nw++;
if( verbose ){
sprintf(error_string,"polling %d for tmp_data[%d] = 0x%x to go above 0x%x",
nw,adc_trigger_channel,tmp_data[adc_trigger_channel-first_adc_channel],
adc_trig_level);
advise(error_string);
}
if( tmp_data[ adc_trigger_channel-first_adc_channel] == 0xffff ){
sprintf(error_string,"tmp_data = 0x%x   0x%x   0x%x   0x%x   0x%x   0x%x",
tmp_data[0],tmp_data[1],tmp_data[2],tmp_data[3],tmp_data[4],tmp_data[5]);
advise(error_string);
}
		} while( tmp_data[adc_trigger_channel-first_adc_channel] < adc_trig_level );

sprintf(error_string,"after %d polls, signal 0x%x above threshold 0x%x",
nw, tmp_data[adc_trigger_channel-first_adc_channel], adc_trig_level );
advise(error_string);

	}
	nb = dp->dt_n_mach_elts*siztbl[PREC_IN];

	GET_ADC_DATA(dp->dt_data,nb)

}

static const char *pol_strs[2]={"unipolar","bipolar"};
static const char *volt_strs[4]={"10","5","2.5","1.25"};

static COMMAND_FUNC( adc_range )
{
	int i_pol,i_v;
	int range;

	i_pol = WHICH_ONE("polarity",2,pol_strs);
	i_v = WHICH_ONE("voltage",4,volt_strs);

	if( i_pol < 0 || i_v < 0 ) return;

	CONFIRM_ADC

	if( i_pol == 0 ){	/* unipolar */
		switch( i_v ){
			case 0:	range = RANGE_10V_UNI; break;
			case 1:	range = RANGE_5V_UNI; break;
			case 2:	range = RANGE_2_5V_UNI; break;
			case 3:	range = RANGE_1_25V_UNI; break;
#ifdef CAUTIOUS
			default:
				ERROR1("CAUTIOUS:  adc_range:  impossible unipolar value!?");
				range=(-1);	/* quiet compiler */
				break;
#endif /* CAUTIOUS */
		}
	} else {		/* bipolar */
		switch( i_v ){
			case 0: range = RANGE_10V_BI; break;
			case 1:	range = RANGE_5V_BI; break;
			case 2:	range = RANGE_2_5V_BI; break;
			case 3:	range = RANGE_1_25V_BI; break;
#ifdef CAUTIOUS
			default:
				ERROR1("CAUTIOUS:  adc_range:  impossible bipolar value!?");
				range=(-1);	/* quiet compiler */
				break;
#endif /* CAUTIOUS */
		}
	}


	if( ioctl(adc_fd,ADC_SET_RANGE,range) < 0 ){
		perror("ioctl");
		NWARN("error setting adc range");
		return;
	}

}



static const char *cfg_strs[2]={"single_ended","differential"};

static COMMAND_FUNC( adc_config )
{
	int i_c;
	int code;

	i_c = WHICH_ONE("adc input configuration (single_ended/differential)",
		2,cfg_strs);
	if( i_c < 0 ) return;

	CONFIRM_ADC

	if( i_c == 0 ){		/* SE */
		code = ADC_CFG_SE;
	} else {
		code = ADC_CFG_DIFF;
	}

	if( ioctl(adc_fd,ADC_SET_CFG,code) < 0 ){
		perror("ioctl");
		NWARN("error setting adc configuration");
	}
}

static const char *pacer_strs[2]={"software","clock"};

static COMMAND_FUNC( adc_pacer )
{
	int i_p;
	int code;

	i_p = WHICH_ONE("adc pacer source (software/clock)",
		2,pacer_strs);
	if( i_p < 0 ) return;

	CONFIRM_ADC

	if( i_p == 0 ){		/* SE */
		code = PACER_SW;
	} else {
		code = PACER_CTR;
	}

	if( ioctl(adc_fd,ADC_SET_PACER,code) < 0 ){
		perror("ioctl ADC_SET_PACER");
		NWARN("error setting adc pacer mode");
	}
}

static const char *adc_mode_strs[]={"polled","intr","paced"};

static COMMAND_FUNC( adc_mode )
{
	int i;
	int arg;

	i=WHICH_ONE("adc mode",3,adc_mode_strs);
	if( i < 0 ) return;

	switch(i){
		case 0: arg = ADC_MODE_POLLED; break;
		case 1: arg = ADC_MODE_INTR; break;
		case 2: arg = ADC_MODE_PACED; break;
#ifdef CAUTIOUS
		default:
			ERROR1("CAUTIOUS:  adc_mode:  impossible mode!?");
			arg = (-1);	/* quiet compiler */
			break;
#endif /* CAUTIOUS */
	}
	if( ioctl(adc_fd,ADC_SET_MODE,arg) < 0 ){
		perror("ioctl");
		NWARN("error setting adc mode");
	}
}

static COMMAND_FUNC( pacer_freq )
{
	double f;
	u_short dividers[2];

	f=HOW_MUCH("pacer frequency");
	if( f <= 0 ){
		NWARN("pacer frequency must be positive");
		return;
	}
	f=SetPacerFreq(f,&dividers[0],&dividers[1]);

	if( verbose ){
		sprintf(error_string,
	"SetPacerFreq: ctr1 %d   ctr2 %d   freq %g", dividers[0], dividers[1], f);
		advise(msg_str);
	}

	CONFIRM_ADC

	if( ioctl(adc_fd,ADC_PACER_FREQ,dividers) < 0 ){
		perror("ioctl ADC_PACER_FREQ");
		NWARN("error setting adc pacer mode");
	}
}

/* Out software trigger works by starting a conversion after a specified channel exceeds a certain value */

static COMMAND_FUNC( set_sw_trig )
{
	use_sw_trigger = ASKIF("use software trigger to initiate paced conversions?");
}

static COMMAND_FUNC( set_trig_chan )
{
	int n;

	n=HOW_MANY("trigger channel");
	if( first_adc_channel < 0 ){
		NWARN("select adc channels (with open) before selecting trigger channel");
		return;
	}
	if( n < first_adc_channel || n > last_adc_channel ){
		sprintf(error_string,"requested trigger channel (%d) is not in range of active adc channels (%d-%d)",
			n,first_adc_channel,last_adc_channel);
		NWARN(error_string);
		return;
	}
	adc_trigger_channel = n;
}

static COMMAND_FUNC( set_trig_level )
{
	adc_trig_level = HOW_MANY("adc trigger level");
}

static COMMAND_FUNC( do_ld_dac08 )
{
	int value;

	value = HOW_MANY("dac08 value");

	if( value < 0 || value > MAX_BYTE_VALUE ){
		sprintf(error_string,"ld_dac08:  value %d must be in the range 0-255",value);
		NWARN(error_string);
		return;
	}

	CONFIRM_ADC

	if( ioctl(adc_fd,ADC_LOAD_DAC08,value) < 0 ){
		perror("ioctl ADC_LOAD_DAC08");
		NWARN("error load dac08 value");
	}
}

static COMMAND_FUNC( do_ld_8402 )
{
	int addr,value;

	addr = HOW_MANY("8402 addr (0/1)");
	value = HOW_MANY("8402 value");

	if( addr < 0 || addr > 1 ) {
		sprintf(error_string,"ld_8402:  address %d must be 0 or 1",addr);
		NWARN(error_string);
		return;
	}

	if( value < 0 || value > MAX_BYTE_VALUE ){
		sprintf(error_string,"ld_8402:  value %d must be in the range 0-255",value);
		NWARN(error_string);
		return;
	}

	CONFIRM_ADC

	/* put the address bit into the second byte of the word... */
	value |= (addr<<8);

	if( ioctl(adc_fd,ADC_LOAD_8402,value) < 0 ){
		perror("ioctl ADC_LOAD_8402");
		NWARN("error load 8402 value");
	}
}

static COMMAND_FUNC( do_calib )
{
	int yesno;

	yesno=ASKIF("enable calibration mode");

	CONFIRM_ADC

	if( yesno ){
		if( ioctl(adc_fd,ADC_CALIB_ENABLE,NULL) < 0 ){
			perror("ioctl ADC_CALIB_ENABLE");
			NWARN("error setting calibration mode");
		}
	} else {
		if( ioctl(adc_fd,ADC_CALIB_DISABLE,NULL) < 0 ){
			perror("ioctl ADC_CALIB_DISABLE");
			NWARN("error clearing calibration mode");
		}
	}
}

#define N_CALIB_SOURCES	8

static const char *src_names[N_CALIB_SOURCES]={
	"analog_ground",
	"7.0V",
	"3.5V",
	"1.75V",
	"0.875V",
	"-10.0V",
	"dac0",
	"dac1"
};

static COMMAND_FUNC( do_setsrc )
{
	int i;

	i=WHICH_ONE("calibration source",N_CALIB_SOURCES,src_names);

	if( i < 0 ) return;

	CONFIRM_ADC

	if( ioctl(adc_fd,ADC_CALIB_SRC,i) < 0 ){
		perror("ioctl ADC_CALIB_SRC");
		NWARN("error setting calibration source");
	}
}

static Command calib_ctbl[]={
{ "calibrate",	do_calib,	"enable/disable calibration mode"	},
{ "source",	do_setsrc,	"select calibration source"	},
{ "ld_dac08",	do_ld_dac08,	"load data to the dac08"	},
{ "ld_8402",	do_ld_8402,	"load data to the 8402"		},
{ "quit",	popcmd,		"exit submenu"			},
{ NULL_COMMAND							}
};

static COMMAND_FUNC( adc_calib )
{
	PUSHCMD(calib_ctbl,"adc_calib");
}

static Command adc_ctbl[]={
{ "open",	open_adc,	"open adc file"			},
{ "close",	close_adc,	"close adc file"		},
{ "read",	read_adc,	"read adc data"			},
{ "range",	adc_range,	"set adc input voltage range"	},
{ "pacer",	adc_pacer,	"set adc pacer source"		},
{ "freq",	pacer_freq,	"set adc pacer frequency"	},
{ "config",	adc_config,	"select input configuration"	},
{ "mode",	adc_mode,	"select adc mode"		},
{ "calib",	adc_calib,	"calibrate adc"			},
{ "sw_trig",	set_sw_trig,	"enable/disable software trigger"	},
{ "trig_channel",set_trig_chan,	"set trigger channel"		},
{ "trig_level",	set_trig_level,	"set trigger threshold"		},
{ "quit",	popcmd,		"exit submenu"			},
{ NULL_COMMAND							}
};

static COMMAND_FUNC( adc_menu )
{
	PUSHCMD(adc_ctbl,"adc");
}

static int dac_chno;

static COMMAND_FUNC( open_dac )
{
	char fn[16];

	dac_chno = get_dac_channel(SINGLE_QSP_ARG);
	if( dac_chno < 0 ) return;

	if( dac_fd >= 0 ){
		NWARN("A dac is already open!?");
		return;
	}

	if( dac_chno == 2 ){
		strcpy(fn,"/dev/dac01");
	} else {
		sprintf(fn,"/dev/dac%d",dac_chno);
	}

	dac_fd = open(fn,O_WRONLY);
	if( dac_fd < 0 ){
		perror("open");
		sprintf(error_string,"Error opening dac file %s",fn);
		NWARN(error_string);
		return;
	}
}

static COMMAND_FUNC( close_dac )
{
	CONFIRM_DAC

	close(dac_fd);
	dac_fd = -1;
}

static COMMAND_FUNC( write_dac )
{
	Data_Obj *dp;
	int32_t nwant;
	int n;

	dp=PICK_OBJ("data vector");

	CONFIRM_DAC

	if( dp == NO_OBJ ) return;
	if( ! VALID_ADC_PREC(dp) ){
		sprintf(error_string,
		"Object %s has precision %s, should be %s or %s",
			dp->dt_name,prec_name[MACHINE_PREC(dp)],
			prec_name[PREC_IN],prec_name[PREC_UIN]);
		NWARN(error_string);
		return;
	}

	nwant = sizeof(short)*dp->dt_n_mach_elts;
	if( (n=write(dac_fd,dp->dt_data,nwant)) != nwant ){
		sprintf(error_string,
	"Error writing object %s (%d bytes requested, %d actual)",
			dp->dt_name,nwant,n);
		NWARN(error_string);
	}
}

#define N_DAC_PACER_SOURCES	4
static const char *dac_mode_strings[N_DAC_PACER_SOURCES]=
	{"software","clock","external_rising","external_falling"};

#define INVALID_DAC_MODE	((DAC_Mode) -1 )

static COMMAND_FUNC( set_dac_mode )
{
	int i;
	DAC_Mode arg;

	i=WHICH_ONE("dac pacer source",N_DAC_PACER_SOURCES,dac_mode_strings);
	if( i < 0 ) return;

	switch(i){
		case 0: arg=DAC_MODE_POLLED; break;
		case 1: arg=DAC_MODE_PACED; break;
		case 2: arg=DAC_MODE_EXT_RISING; break;
		case 3: arg=DAC_MODE_EXT_FALLING; break;
#ifdef CAUTIOUS
		default:
			ERROR1("CAUTIOUS:  set_dac_mode:  impossible DAC mode!?");
			arg=(INVALID_DAC_MODE);	/* quiet compiler */
			break;
#endif /* CAUTIOUS */
	}

	CONFIRM_DAC

	if( ioctl(dac_fd,DAC_SET_MODE,arg) < 0 ){
		perror("ioctl");
		NWARN("error setting dac pacer mode");
	}
}

static COMMAND_FUNC( dac_pacer_freq )
{
	double f;
	u_short dividers[2];

	f=HOW_MUCH("pacer frequency");
	if( f <= 0 ){
		NWARN("pacer frequency must be positive");
		return;
	}
	f=SetPacerFreq(f,&dividers[0],&dividers[1]);

	if( verbose ){
		sprintf(msg_str,
	"SetPacerFreq: ctr1 %d   ctr2 %d   freq %g", dividers[0], dividers[1], f);
		advise(msg_str);
	}

	CONFIRM_DAC

	if( ioctl(dac_fd,DAC_PACER_FREQ,dividers) < 0 ){
		perror("ioctl DAC_PACER_FREQ");
		NWARN("error setting dac pacer freq");
	}
}

#define N_DAC_RANGES	4

static const char *dac_range[N_DAC_RANGES]={
	"bipolar_10V",
	"bipolar_5V",
	"unipolar_10V",
	"unipolar_5V"
};

/* BUG?  it looks as if the two dac's can have their ranges set independently,
 * but this utility forces them to be the same!?
 */

static COMMAND_FUNC( set_dac_range )
{
	int range;
	int arg;

	range = WHICH_ONE("DAC range",N_DAC_RANGES,dac_range);

	if( range < 0 ) return;

	CONFIRM_DAC

	switch( range ){
		case 0: arg=RANGE_10V_BI; break;
		case 1: arg=RANGE_5V_BI; break;
		case 2: arg=RANGE_10V_UNI; break;
		case 3: arg=RANGE_5V_UNI; break;
#ifdef CAUTIOUS
		default:
			ERROR1("CAUTIOUS:  set_dac_range:  impossible range index!?");
			arg=(-1);	/* quiet compiler */
			break;
#endif
	}

	if( ioctl(dac_fd,DAC1_SET_RANGE,arg) < 0 ){
		perror("ioctl");
		NWARN("error setting dac1 range");
	}

	if( ioctl(dac_fd,DAC0_SET_RANGE,arg) < 0 ){
		perror("ioctl");
		NWARN("error setting dac0 range");
	}
}

static COMMAND_FUNC( do_ld_8800 )
{
	int addr,value;

	addr = HOW_MANY("8800 addr (0-7)");
	value = HOW_MANY("8800 value");

	if( addr < 0 || addr > 7 ) {
		sprintf(error_string,"ld_8800:  address %d must between 0 and 7",addr);
		NWARN(error_string);
		return;
	}

	if( value < 0 || value > MAX_BYTE_VALUE ){
		sprintf(error_string,"ld_8800:  value %d must be in the range 0-255",value);
		NWARN(error_string);
		return;
	}

	CONFIRM_DAC

	/* put the address bits into the second byte of the word... */
	value |= (addr<<8);

	if( ioctl(dac_fd,DAC_LOAD_8800,value) < 0 ){
		perror("ioctl DAC_LOAD_8800");
		NWARN("error load 8800 value");
	}
}

static Command dac_ctbl[]={
{ "open",	open_dac,	"open dac file"			},
{ "close",	close_dac,	"close dac file"		},
{ "write",	write_dac,	"write dac data"		},
{ "pacer",	set_dac_mode,	"select dac pacer source"	},
{ "freq",	dac_pacer_freq,	"set dac pacer frequency"	},
{ "range",	set_dac_range,	"select dac voltage output range" },
{ "ld_8800",	do_ld_8800,	"load data to the 8800"		},
{ "quit",	popcmd,		"exit submenu"			},
{ NULL_COMMAND							}
};

static COMMAND_FUNC( dac_menu )
{
	PUSHCMD(dac_ctbl,"dac");
}

#define N_DIO_PORTS	4
static int dio_fd[N_DIO_PORTS]={-1,-1,-1,-1};

static int get_dio_channel(SINGLE_QSP_ARG_DECL)
{
	int i;

	i = HOW_MANY("dio port index (0-3)");
	if( i < 0 || i > 3 ) {
		sprintf(error_string,
	"Invalid dio port index %d, must be 0-3",i);
		NWARN(error_string);
		return(-1);
	}
	return(i);
}

static const char *dio_mode_strs[2]={"read","write"};

static COMMAND_FUNC( open_dio )
{
	char dio_name[32];
	int i;
	int mode;

	i=get_dio_channel(SINGLE_QSP_ARG);

	mode = WHICH_ONE("read/write mode",2,dio_mode_strs);

	if( i < 0 || mode < 0 ) return;

	sprintf(dio_name,"/dev/dio%d",i);

	if( mode == 0 ){
		dio_fd[i] = open(dio_name,O_RDONLY);
	} else {
		dio_fd[i] = open(dio_name,O_WRONLY);
	}

	if( dio_fd[i] < 0 ){
		perror("open");
		sprintf(error_string,"error opening dio port %d in %s mode",
			i,dio_mode_strs[mode]);
		NWARN(error_string);
	}
}

static COMMAND_FUNC( close_dio )
{
	int i;

	i=get_dio_channel(SINGLE_QSP_ARG);
	CONFIRM_DIO(i)
	if( close(dio_fd[i]) < 0 ) {
		perror("close");
		sprintf(error_string,"error closing dio port %d",i);
		NWARN(error_string);
	}
	dio_fd[i]=(-1);
}

static COMMAND_FUNC( write_dio )
{
	int i;
	int n;
	Data_Obj *dp;

	i=get_dio_channel(SINGLE_QSP_ARG);
	dp = PICK_OBJ("");

	if( i < 0 ) return;
	CONFIRM_DIO(i);

	if(dp==NO_OBJ) return;

	if( ! VALID_DIO_PREC(dp) ){
		sprintf(error_string,
		"Object %s has %s precision, should be %s or %s for dio",
			dp->dt_name,prec_name[MACHINE_PREC(dp)],
			prec_name[PREC_BY],prec_name[PREC_UBY]);
		NWARN(error_string);
		return;
	}

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(error_string,"Object %s must be contiguous for dio",
			dp->dt_name);
		NWARN(error_string);
		return;
	}

	if( (n=write(dio_fd[i],dp->dt_data,dp->dt_n_mach_elts)) != (int)dp->dt_n_mach_elts ){
		if( n < 0 ) perror("write");
		sprintf(error_string,
	"dio error writing object %s, %d bytes requested, %d actual",
			dp->dt_name,dp->dt_n_mach_elts,n);
		NWARN(error_string);
	}
}

static COMMAND_FUNC( read_dio )
{
	int i;
	int n;
	Data_Obj *dp;

	i=get_dio_channel(SINGLE_QSP_ARG);
	dp = PICK_OBJ("");

	if( i < 0 ) return;
	CONFIRM_DIO(i);

	if(dp==NO_OBJ) return;

	if( ! VALID_DIO_PREC(dp) ){
		sprintf(error_string,
		"Object %s has %s precision, should be %s or %s for dio",
			dp->dt_name,prec_name[MACHINE_PREC(dp)],
			prec_name[PREC_BY],prec_name[PREC_UBY]);
		NWARN(error_string);
		return;
	}

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(error_string,"Object %s must be contiguous for dio",
			dp->dt_name);
		NWARN(error_string);
		return;
	}

	if( (n=read(dio_fd[i],dp->dt_data,dp->dt_n_mach_elts)) != (int)dp->dt_n_mach_elts ){
		if( n < 0 ) perror("read");
		sprintf(error_string,
	"dio error reading object %s, %d bytes requested, %d actual",
			dp->dt_name,dp->dt_n_mach_elts,n);
		NWARN(error_string);
	}
}




static Command dio_ctbl[]={
{ "open",	open_dio,	"open dio device"		},
{ "close",	close_dio,	"close dio device"		},
{ "write",	write_dio,	"write dio data"		},
{ "read",	read_dio,	"read dio data"			},
{ "quit",	popcmd,		"exit submenu"			},
{ NULL_COMMAND							}
};

static COMMAND_FUNC( dio_menu )
{
	PUSHCMD(dio_ctbl,"dio");
}


#define VALID_NVRAM_PREC(dp)					\
								\
	((MACHINE_PREC(dp)==PREC_UBY)||(MACHINE_PREC(dp)==PREC_BY))


static int nvram_fd=(-1);

COMMAND_FUNC( open_nvram )
{

	nvram_fd = open("/dev/nvram0",O_RDWR);
	if( nvram_fd < 0 ){
		perror("open /dev/nvram0");
		return;
	}
}

COMMAND_FUNC( close_nvram )
{
	if( nvram_fd < 0 ){
		NWARN("nvram not open");
		return;
	}

	close(nvram_fd);
	nvram_fd = -1;

}

static COMMAND_FUNC( read_nvram )
{
	int n;
	Data_Obj *dp;

	dp = PICK_OBJ("");

	if(dp==NO_OBJ) return;

	if( nvram_fd < 0 ) ERROR1("unable to open nvram");

	if( ! VALID_NVRAM_PREC(dp) ){
		sprintf(error_string,
		"Object %s has %s precision, should be %s or %s for nvram",
			dp->dt_name,prec_name[MACHINE_PREC(dp)],
			prec_name[PREC_BY],prec_name[PREC_UBY]);
		NWARN(error_string);
		return;
	}

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(error_string,"Object %s must be contiguous for nvram",
			dp->dt_name);
		NWARN(error_string);
		return;
	}

	if( (n=read(nvram_fd,dp->dt_data,dp->dt_n_mach_elts)) != (int)dp->dt_n_mach_elts ){
		if( n < 0 ) perror("read nvram");
		sprintf(error_string,
	"nvram error reading object %s, %d bytes requested, %d actual",
			dp->dt_name,dp->dt_n_mach_elts,n);
		NWARN(error_string);
	}
}

static COMMAND_FUNC( write_nvram )		/* sq */
{
	int n;
	Data_Obj *dp;

	dp = PICK_OBJ("");

	if(dp==NO_OBJ) return;

	if( nvram_fd < 0 ) ERROR1("unable to open nvram");

	if( ! VALID_NVRAM_PREC(dp) ){
		sprintf(error_string,
		"Object %s has %s precision, should be %s or %s for nvram",
			dp->dt_name,prec_name[MACHINE_PREC(dp)],
			prec_name[PREC_BY],prec_name[PREC_UBY]);
		NWARN(error_string);
		return;
	}

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(error_string,"Object %s must be contiguous for nvram",
			dp->dt_name);
		NWARN(error_string);
		return;
	}

	if( (n=write(nvram_fd,dp->dt_data,dp->dt_n_mach_elts)) != (int)dp->dt_n_mach_elts ){
		if( n < 0 ) perror("write nvram");
		sprintf(error_string,
	"nvram error writing object %s, %d bytes requested, %d actual",
			dp->dt_name,dp->dt_n_mach_elts,n);
		NWARN(error_string);
	}
}

static COMMAND_FUNC( ld_nvram )		/*sq */
{
	int n;
	int addr;
	int original_fd_position;

	Data_Obj *dp;

	addr = HOW_MANY("NVRAM starting addr (0-255)");

	if( nvram_fd < 0 ) ERROR1("nvram not open");

	original_fd_position = lseek(nvram_fd,0,SEEK_CUR);

	if(lseek(nvram_fd,addr,SEEK_SET)!=addr) {
		perror("lseek NVRAM");
		NWARN("error setting NVRAM offset");
		return;
	}

	dp = PICK_OBJ("");
	if(dp==NO_OBJ) return;

	if( addr+dp->dt_n_mach_elts > NVRAM_SIZE ){
		sprintf(error_string,"NVRAM: Can't write requested number of bytes. Check address and size of object %s",dp->dt_name);
		NWARN(error_string);
		return;
	}


	if( ! VALID_NVRAM_PREC(dp) ){
		sprintf(error_string,"Object %s has %s precision, should be %s or %s for nvram",
			dp->dt_name,prec_name[MACHINE_PREC(dp)],prec_name[PREC_BY],prec_name[PREC_UBY]);
		NWARN(error_string);
		return;
	}

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(error_string,"Object %s must be contiguous for nvram",dp->dt_name);
		NWARN(error_string);
		return;
	}

	if( (n=write(nvram_fd,dp->dt_data,dp->dt_n_mach_elts)) != (int)dp->dt_n_mach_elts ){
		if( n < 0 ) perror("write nvram");
		sprintf(error_string, "nvram error writing object %s, %d bytes requested, %d actual",
					dp->dt_name,dp->dt_n_mach_elts,n);
		NWARN(error_string);
	}

	lseek(nvram_fd,original_fd_position,SEEK_SET);

}

static COMMAND_FUNC( rd_nvram )		/*sq */
{
	int n;
	int addr;
	int original_fd_position;

	Data_Obj *dp;

	addr = HOW_MANY("NVRAM starting addr (0-255)");

	if( nvram_fd < 0 ) ERROR1("nvram not open");

	original_fd_position = lseek(nvram_fd,0,SEEK_CUR);

	if(lseek(nvram_fd,addr,SEEK_SET)!=addr) {
		perror("lseek NVRAM");
		NWARN("error setting NVRAM offset");
		return;
	}

	dp = PICK_OBJ("");
	if(dp==NO_OBJ) return;

	if( addr+dp->dt_n_mach_elts > NVRAM_SIZE ){
		sprintf(error_string,"NVRAM: Can't read requested number of bytes. Check address and size of object %s",dp->dt_name);
		NWARN(error_string);
		return;
	}


	if( ! VALID_NVRAM_PREC(dp) ){
		sprintf(error_string,
		"Object %s has %s precision, should be %s or %s for nvram",
			dp->dt_name,prec_name[MACHINE_PREC(dp)],
			prec_name[PREC_BY],prec_name[PREC_UBY]);
		NWARN(error_string);
		return;
	}

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(error_string,"Object %s must be contiguous for nvram",dp->dt_name);
		NWARN(error_string);
		return;
	}

	if( (n=read(nvram_fd,dp->dt_data,dp->dt_n_mach_elts)) != (int)dp->dt_n_mach_elts ){
		if( n < 0 ) perror("read nvram");
		sprintf(error_string, "nvram error reading object %s, %d bytes requested, %d actual",
					dp->dt_name,dp->dt_n_mach_elts,n);
		NWARN(error_string);
	}

	lseek(nvram_fd,original_fd_position,SEEK_SET);

}

static Command nvram_ctbl[]={
{ "open",	open_nvram,	"open nvram device"			},
{ "close",	close_nvram,	"close nvram device"			},
{ "read",	read_nvram,	"read nvram data"			},
{ "write",	write_nvram,	"write nvram data"			},
{ "ld_nvram",	ld_nvram,	"load/write data to an address"		},
{ "rd_nvram",	rd_nvram,	"read nvram data from an address"	},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL_COMMAND								}
};

static COMMAND_FUNC( nvram_menu )
{
	PUSHCMD(nvram_ctbl,"nvram");
}

 static Command das_ctbl[]={
{ "registers",	reg_menu,	"manipulate device registers"	},
{ "adc",	adc_menu,	"access ADC"			},
{ "dac",	dac_menu,	"access DAC"			},
{ "dio",	dio_menu,	"digital I/O"			},
{ "nvram",	nvram_menu,	"access NVRAM"			},
{ "quit",	popcmd,		"exit submenu"			},
{ NULL_COMMAND							}
};

COMMAND_FUNC( aio_menu )
{
	PUSHCMD(das_ctbl,"das1602");
}

#endif /* HAVE_DAS1602 */
