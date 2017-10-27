#include "quip_config.h"


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

#include "quip_prot.h"
#include "data_obj.h"
#include "ioctl_das1602.h"
#include "pacer.h"

#define VALID_ADC_PREC(dp)					\
								\
	((OBJ_MACH_PREC(dp)==PREC_UIN)||(OBJ_MACH_PREC(dp)==PREC_IN))



#define VALID_DIO_PREC(dp)					\
								\
	((OBJ_MACH_PREC(dp)==PREC_UBY)||(OBJ_MACH_PREC(dp)==PREC_BY))

#define MAX_BYTE_VALUE	255
#define NVRAM_SIZE	256

// BUG?  globals are not thread safe...

static int first_adc_channel=(-1),last_adc_channel=(-1);
static int use_sw_trigger=0;
static u_short adc_trig_level=0x8000;

#ifdef HAVE_DAS1602

static int adc_trigger_channel=(-1);

static int das_fd = (-1);

static int adc_fd=(-1);
static int dac_fd=(-1);
static int nvram_fd=(-1);


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
		sprintf(ERROR_STRING,				\
			"dio channel %d not open",channel);	\
		NWARN(ERROR_STRING);				\
		return;						\
	}

#else /* ! HAVE_DAS1602 */

static int no_aio_warned=0;

#define NO_AIO_ALERT					\
							\
	if( ! no_aio_warned ){				\
		WARN("No analog I/O capabilities!?");	\
		no_aio_warned=1;			\
	}

#endif /* ! HAVE_DAS1602 */

static COMMAND_FUNC( set_reg )
{
	Reg_Data reg_setting;

	reg_setting.reg_blocki = HOW_MANY("block_index (1-4)");
	reg_setting.reg_offset = HOW_MANY("register offset");

	reg_setting.reg_data.u_s = HOW_MANY("register data");
	/* BUG?  does this work for char data?? */

#ifdef HAVE_DAS1602
	if( das_fd < 0 ){
		NWARN("device not open");
		return;
	}

	if( ioctl(das_fd,DAS_SET_REG,&reg_setting) < 0 ){
		perror("ioctl");
		NWARN("error setting das1602 register");
	}
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}


static COMMAND_FUNC( get_reg )
{
	Reg_Data reg_setting;

	reg_setting.reg_blocki = HOW_MANY("block_index (1-4)");
	reg_setting.reg_offset = HOW_MANY("register offset");

#ifdef HAVE_DAS1602
	if( das_fd < 0 ){
		NWARN("device not open");
		return;
	}

	reg_setting.reg_data.u_s = 0;	/* clear all bits */
	if( ioctl(das_fd,DAS_GET_REG,&reg_setting) < 0 ){
		perror("ioctl");
		NWARN("error getting das1602 register");
	}

	sprintf(ERROR_STRING,"Reg %d %d:  0x%x (0x%x)",
		reg_setting.reg_blocki,reg_setting.reg_offset,
		reg_setting.reg_data.u_s,reg_setting.reg_data.u_c);
	advise(ERROR_STRING);

	sprintf(ERROR_STRING,"0x%x",reg_setting.reg_data.u_s);
	assign_var("reg_data",ERROR_STRING);
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
	assign_var("reg_data","0");
#endif /* ! HAVE_DAS1602 */
}

static COMMAND_FUNC( open_device )
{
#ifdef HAVE_DAS1602
	das_fd = open("/dev/das1602",O_RDWR);
	if( das_fd < 0 ){
		perror("open");
		NWARN("error opening /dev/das1602");
	}
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(registers_menu,s,f,h)

MENU_BEGIN(registers)
ADD_CMD( open,	open_device,	open device register file )
ADD_CMD( getreg,	get_reg,	read device register )
ADD_CMD( setreg,	set_reg,	write device register )
MENU_END(registers)

static COMMAND_FUNC( do_reg_menu )
{
	PUSH_MENU(registers);
}

static int get_adc_channel(QSP_ARG_DECL  const char *pmpt,
						int chmin, int chmax)
{
	int chno;

	sprintf(msg_str,"%s (%d-%d)",pmpt,chmin,chmax);
	chno = HOW_MANY(msg_str);

	if( chno < chmin || chno > chmax ){
		sprintf(ERROR_STRING,
	"channel number (%d) must be between %d and %d",chno,chmin,chmax);
		NWARN(ERROR_STRING);
		return(-1);
	}
	return(chno);
}

static int get_dac_channel(SINGLE_QSP_ARG_DECL)
{
	int chno;

	chno = HOW_MANY("channel number (0-1, or 2 for both)");

	if( chno < 0 || chno > 2 ){
		sprintf(ERROR_STRING,
	"channel number (%d) must be between 0 and 2",chno);
		NWARN(ERROR_STRING);
		return(-1);
	}
	return(chno);
}

#ifdef FOOBAR
#ifdef HAVE_DAS1602
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
#endif // FOOBAR

static COMMAND_FUNC( open_adc )
{
#ifdef HAVE_DAS1602
	char fn[16];
#endif /* ! HAVE_DAS1602 */

	first_adc_channel = get_adc_channel(QSP_ARG  "first a/d channel",0,7);
	last_adc_channel = get_adc_channel(QSP_ARG  "last a/d channel",first_adc_channel,7);

	if( first_adc_channel < 0 || last_adc_channel < 0 ) return;

#ifdef HAVE_DAS1602
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
		sprintf(ERROR_STRING,"Error opening adc file %s",fn);
		NWARN(ERROR_STRING);
		return;
	}
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

static COMMAND_FUNC( close_adc )
{
#ifdef HAVE_DAS1602
	CONFIRM_ADC

	close(adc_fd);
	adc_fd = -1;
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

#define GET_ADC_DATA( addr , n )									\
													\
		if( (nr=read(adc_fd,(addr),(n))) != (n) ){						\
			if( nr < 0 ){									\
				sprintf(ERROR_STRING,"read (errno=%d)",errno);				\
				perror(ERROR_STRING);							\
			} else {									\
				sprintf(ERROR_STRING,							\
		"%d bytes requested, %d actually read,nw=%d", n,nr,nw);				\
				advise(ERROR_STRING);							\
			}										\
			NWARN("error reading adc data");							\
		}

#define N_TRIG	2

static COMMAND_FUNC( read_adc )
{
	Data_Obj *dp;
#ifdef HAVE_DAS1602
	int nb, nr;
	u_short tmp_data[16*N_TRIG];	/* this is the maximum number of channels? */
	uint32_t nw=0;
#endif // HAVE_DAS1602

	dp = pick_obj("data vector");

	if( dp == NULL ) return;

#ifdef HAVE_DAS1602
	if( ! VALID_ADC_PREC(dp) ){
		sprintf(ERROR_STRING,
	"Object %s has precision %s, should be %s or %s for ADC data",
			OBJ_NAME(dp),PREC_NAME(OBJ_MACH_PREC_PTR(dp)),
			PREC_IN_NAME,PREC_UIN_NAME);
		NWARN(ERROR_STRING);
		return;
	}
	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,
	"read_adc:  object %s must be contiguous",OBJ_NAME(dp));
		NWARN(ERROR_STRING);
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
sprintf(ERROR_STRING,"polling %d for tmp_data[%d] = 0x%x to go below 0x%x",
nw,adc_trigger_channel,tmp_data[adc_trigger_channel-first_adc_channel],
adc_trig_level);
advise(ERROR_STRING);
}
		} while( tmp_data[adc_trigger_channel-first_adc_channel] > adc_trig_level );

sprintf(ERROR_STRING,"after %d polls, signal 0x%x below threshold 0x%x",
nw, tmp_data[adc_trigger_channel-first_adc_channel], adc_trig_level );
advise(ERROR_STRING);
sprintf(ERROR_STRING,"tmp_data = 0x%x   0x%x   0x%x   0x%x   0x%x   0x%x",
tmp_data[0],tmp_data[1],tmp_data[2],tmp_data[3],tmp_data[4],tmp_data[5]);
advise(ERROR_STRING);

		/* Now wait for it to go above the threshold */
		nw=0;
		do {
			GET_ADC_DATA(&tmp_data[0],nb)
			nw++;
if( verbose ){
sprintf(ERROR_STRING,"polling %d for tmp_data[%d] = 0x%x to go above 0x%x",
nw,adc_trigger_channel,tmp_data[adc_trigger_channel-first_adc_channel],
adc_trig_level);
advise(ERROR_STRING);
}
if( tmp_data[ adc_trigger_channel-first_adc_channel] == 0xffff ){
sprintf(ERROR_STRING,"tmp_data = 0x%x   0x%x   0x%x   0x%x   0x%x   0x%x",
tmp_data[0],tmp_data[1],tmp_data[2],tmp_data[3],tmp_data[4],tmp_data[5]);
advise(ERROR_STRING);
}
		} while( tmp_data[adc_trigger_channel-first_adc_channel] < adc_trig_level );

sprintf(ERROR_STRING,"after %d polls, signal 0x%x above threshold 0x%x",
nw, tmp_data[adc_trigger_channel-first_adc_channel], adc_trig_level );
advise(ERROR_STRING);

	}

	// We've already checked above that the object has
	// a valid (short) precision...
	nb = OBJ_N_MACH_ELTS(dp)*OBJ_PREC_MACH_SIZE(dp);

	GET_ADC_DATA(OBJ_DATA_PTR(dp),nb)

#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

static const char *pol_strs[2]={"unipolar","bipolar"};
static const char *volt_strs[4]={"10","5","2.5","1.25"};

static COMMAND_FUNC( adc_range )
{
	int i_pol,i_v;
#ifdef HAVE_DAS1602
	int range;
#endif // HAVE_DAS1602

	i_pol = WHICH_ONE("polarity",2,pol_strs);
	i_v = WHICH_ONE("voltage",4,volt_strs);

	if( i_pol < 0 || i_v < 0 ) return;

#ifdef HAVE_DAS1602
	CONFIRM_ADC

	if( i_pol == 0 ){	/* unipolar */
		switch( i_v ){
			case 0:	range = RANGE_10V_UNI; break;
			case 1:	range = RANGE_5V_UNI; break;
			case 2:	range = RANGE_2_5V_UNI; break;
			case 3:	range = RANGE_1_25V_UNI; break;
#ifdef CAUTIOUS
			default:
				error1("CAUTIOUS:  adc_range:  impossible unipolar value!?");
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
				error1("CAUTIOUS:  adc_range:  impossible bipolar value!?");
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

#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}



static const char *cfg_strs[2]={"single_ended","differential"};

static COMMAND_FUNC( adc_config )
{
	int i_c;
#ifdef HAVE_DAS1602
	int code;
#endif // HAVE_DAS1602

	i_c = WHICH_ONE("adc input configuration (single_ended/differential)",
		2,cfg_strs);
	if( i_c < 0 ) return;

#ifdef HAVE_DAS1602
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
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

static const char *pacer_strs[2]={"software","clock"};

static COMMAND_FUNC( adc_pacer )
{
	int i_p;
#ifdef HAVE_DAS1602
	int code;
#endif // HAVE_DAS1602

	i_p = WHICH_ONE("adc pacer source (software/clock)",
		2,pacer_strs);
	if( i_p < 0 ) return;

#ifdef HAVE_DAS1602
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
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

static const char *adc_mode_strs[]={"polled","intr","paced"};

static COMMAND_FUNC( adc_mode )
{
	int i;
#ifdef HAVE_DAS1602
	int arg;
#endif // HAVE_DAS1602

	i=WHICH_ONE("adc mode",3,adc_mode_strs);
	if( i < 0 ) return;

#ifdef HAVE_DAS1602
	switch(i){
		case 0: arg = ADC_MODE_POLLED; break;
		case 1: arg = ADC_MODE_INTR; break;
		case 2: arg = ADC_MODE_PACED; break;
#ifdef CAUTIOUS
		default:
			error1("CAUTIOUS:  adc_mode:  impossible mode!?");
			arg = (-1);	/* quiet compiler */
			break;
#endif /* CAUTIOUS */
	}
	if( ioctl(adc_fd,ADC_SET_MODE,arg) < 0 ){
		perror("ioctl");
		NWARN("error setting adc mode");
	}
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

static COMMAND_FUNC( pacer_freq )
{
	double f;
#ifdef HAVE_DAS1602
	u_short dividers[2];
#endif // HAVE_DAS1602

	f=HOW_MUCH("pacer frequency");
	if( f <= 0 ){
		NWARN("pacer frequency must be positive");
		return;
	}
#ifdef HAVE_DAS1602
	f=SetPacerFreq(f,&dividers[0],&dividers[1]);

	if( verbose ){
		sprintf(ERROR_STRING,
	"SetPacerFreq: ctr1 %d   ctr2 %d   freq %g", dividers[0], dividers[1], f);
		advise(msg_str);
	}

	CONFIRM_ADC

	if( ioctl(adc_fd,ADC_PACER_FREQ,dividers) < 0 ){
		perror("ioctl ADC_PACER_FREQ");
		NWARN("error setting adc pacer mode");
	}
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
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
#ifdef HAVE_DAS1602
	if( n < first_adc_channel || n > last_adc_channel ){
		sprintf(ERROR_STRING,"requested trigger channel (%d) is not in range of active adc channels (%d-%d)",
			n,first_adc_channel,last_adc_channel);
		NWARN(ERROR_STRING);
		return;
	}
	adc_trigger_channel = n;
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
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
		sprintf(ERROR_STRING,"ld_dac08:  value %d must be in the range 0-255",value);
		NWARN(ERROR_STRING);
		return;
	}

#ifdef HAVE_DAS1602
	CONFIRM_ADC

	if( ioctl(adc_fd,ADC_LOAD_DAC08,value) < 0 ){
		perror("ioctl ADC_LOAD_DAC08");
		NWARN("error load dac08 value");
	}
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

static COMMAND_FUNC( do_ld_8402 )
{
	int addr,value;

	addr = HOW_MANY("8402 addr (0/1)");
	value = HOW_MANY("8402 value");

	if( addr < 0 || addr > 1 ) {
		sprintf(ERROR_STRING,"ld_8402:  address %d must be 0 or 1",addr);
		NWARN(ERROR_STRING);
		return;
	}

	if( value < 0 || value > MAX_BYTE_VALUE ){
		sprintf(ERROR_STRING,"ld_8402:  value %d must be in the range 0-255",value);
		NWARN(ERROR_STRING);
		return;
	}
#ifdef HAVE_DAS1602

	CONFIRM_ADC

	/* put the address bit into the second byte of the word... */
	value |= (addr<<8);

	if( ioctl(adc_fd,ADC_LOAD_8402,value) < 0 ){
		perror("ioctl ADC_LOAD_8402");
		NWARN("error load 8402 value");
	}
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

static COMMAND_FUNC( do_calib )
{
	int yesno;

	yesno=ASKIF("enable calibration mode");

#ifdef HAVE_DAS1602
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
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
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
#ifdef HAVE_DAS1602

	CONFIRM_ADC

	if( ioctl(adc_fd,ADC_CALIB_SRC,i) < 0 ){
		perror("ioctl ADC_CALIB_SRC");
		NWARN("error setting calibration source");
	}
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(adc_calib_menu,s,f,h)

MENU_BEGIN(adc_calib)
ADD_CMD( calibrate,	do_calib,	enable/disable calibration mode )
ADD_CMD( source,	do_setsrc,	select calibration source )
ADD_CMD( ld_dac08,	do_ld_dac08,	load data to the dac08 )
ADD_CMD( ld_8402,	do_ld_8402,	load data to the 8402 )
MENU_END(adc_calib)

static COMMAND_FUNC( adc_calib )
{
	PUSH_MENU(adc_calib);
}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(adc_menu,s,f,h)

MENU_BEGIN(adc)
ADD_CMD( open,		open_adc,	open adc file )
ADD_CMD( close,		close_adc,	close adc file )
ADD_CMD( read,		read_adc,	read adc data )
ADD_CMD( range,		adc_range,	set adc input voltage range )
ADD_CMD( pacer,		adc_pacer,	set adc pacer source )
ADD_CMD( freq,		pacer_freq,	set adc pacer frequency )
ADD_CMD( config,	adc_config,	select input configuration )
ADD_CMD( mode,		adc_mode,	select adc mode )
ADD_CMD( calib,		adc_calib,	calibrate adc )
ADD_CMD( sw_trig,	set_sw_trig,	enable/disable software trigger )
ADD_CMD( trig_channel,	set_trig_chan,	set trigger channel )
ADD_CMD( trig_level,	set_trig_level,	set trigger threshold )
MENU_END(adc)

static COMMAND_FUNC( do_adc_menu )
{
	PUSH_MENU(adc);
}

static int dac_chno;

static COMMAND_FUNC( open_dac )
{
#ifdef HAVE_DAS1602
	char fn[16];
#endif // HAVE_DAS1602

	dac_chno = get_dac_channel(SINGLE_QSP_ARG);
	if( dac_chno < 0 ) return;

#ifdef HAVE_DAS1602
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
		sprintf(ERROR_STRING,"Error opening dac file %s",fn);
		NWARN(ERROR_STRING);
		return;
	}
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

static COMMAND_FUNC( close_dac )
{
#ifdef HAVE_DAS1602
	CONFIRM_DAC

	close(dac_fd);
	dac_fd = -1;
#endif /* HAVE_DAS1602 */
}

static COMMAND_FUNC( write_dac )
{
	Data_Obj *dp;
#ifdef HAVE_DAS1602
	int32_t nwant;
	int n;
#endif /* HAVE_DAS1602 */

	dp=pick_obj("data vector");

#ifdef HAVE_DAS1602

	CONFIRM_DAC

	if( dp == NULL ) return;
	if( ! VALID_ADC_PREC(dp) ){
		sprintf(ERROR_STRING,
		"Object %s has precision %s, should be %s or %s",
			OBJ_NAME(dp),PREC_NAME(OBJ_MACH_PREC_PTR(dp)),
			PREC_IN_NAME,PREC_UIN_NAME);
		NWARN(ERROR_STRING);
		return;
	}

	nwant = sizeof(short)*OBJ_N_MACH_ELTS(dp);
	if( (n=write(dac_fd,OBJ_DATA_PTR(dp),nwant)) != nwant ){
		sprintf(ERROR_STRING,
	"Error writing object %s (%d bytes requested, %d actual)",
			OBJ_NAME(dp),nwant,n);
		NWARN(ERROR_STRING);
	}
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

#define N_DAC_PACER_SOURCES	4
static const char *dac_mode_strings[N_DAC_PACER_SOURCES]=
	{"software","clock","external_rising","external_falling"};

#define INVALID_DAC_MODE	((DAC_Mode) -1 )

static COMMAND_FUNC( set_dac_mode )
{
	int i;
#ifdef HAVE_DAS1602
	DAC_Mode arg;
#endif // HAVE_DAS1602

	i=WHICH_ONE("dac pacer source",N_DAC_PACER_SOURCES,dac_mode_strings);
	if( i < 0 ) return;

#ifdef HAVE_DAS1602

	switch(i){
		case 0: arg=DAC_MODE_POLLED; break;
		case 1: arg=DAC_MODE_PACED; break;
		case 2: arg=DAC_MODE_EXT_RISING; break;
		case 3: arg=DAC_MODE_EXT_FALLING; break;
#ifdef CAUTIOUS
		default:
			error1("CAUTIOUS:  set_dac_mode:  impossible DAC mode!?");
			arg=(INVALID_DAC_MODE);	/* quiet compiler */
			break;
#endif /* CAUTIOUS */
	}

	CONFIRM_DAC

	if( ioctl(dac_fd,DAC_SET_MODE,arg) < 0 ){
		perror("ioctl");
		NWARN("error setting dac pacer mode");
	}
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

static COMMAND_FUNC( dac_pacer_freq )
{
	double f;
#ifdef HAVE_DAS1602
	u_short dividers[2];
#endif // HAVE_DAS1602

	f=HOW_MUCH("pacer frequency");
	if( f <= 0 ){
		NWARN("pacer frequency must be positive");
		return;
	}
#ifdef HAVE_DAS1602

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
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
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
#ifdef HAVE_DAS1602
	int arg;
#endif // HAVE_DAS1602

	range = WHICH_ONE("DAC range",N_DAC_RANGES,dac_range);

	if( range < 0 ) return;

#ifdef HAVE_DAS1602

	CONFIRM_DAC

	switch( range ){
		case 0: arg=RANGE_10V_BI; break;
		case 1: arg=RANGE_5V_BI; break;
		case 2: arg=RANGE_10V_UNI; break;
		case 3: arg=RANGE_5V_UNI; break;
#ifdef CAUTIOUS
		default:
			error1("CAUTIOUS:  set_dac_range:  impossible range index!?");
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
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

static COMMAND_FUNC( do_ld_8800 )
{
	int addr,value;

	addr = HOW_MANY("8800 addr (0-7)");
	value = HOW_MANY("8800 value");

	if( addr < 0 || addr > 7 ) {
		sprintf(ERROR_STRING,"ld_8800:  address %d must between 0 and 7",addr);
		NWARN(ERROR_STRING);
		return;
	}

	if( value < 0 || value > MAX_BYTE_VALUE ){
		sprintf(ERROR_STRING,"ld_8800:  value %d must be in the range 0-255",value);
		NWARN(ERROR_STRING);
		return;
	}

#ifdef HAVE_DAS1602

	CONFIRM_DAC

	/* put the address bits into the second byte of the word... */
	value |= (addr<<8);

	if( ioctl(dac_fd,DAC_LOAD_8800,value) < 0 ){
		perror("ioctl DAC_LOAD_8800");
		NWARN("error load 8800 value");
	}
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(dac_menu,s,f,h)

MENU_BEGIN(dac)
ADD_CMD( open,	open_dac,	open dac file )
ADD_CMD( close,	close_dac,	close dac file )
ADD_CMD( write,	write_dac,	write dac data )
ADD_CMD( pacer,	set_dac_mode,	select dac pacer source )
ADD_CMD( freq,	dac_pacer_freq,	set dac pacer frequency )
ADD_CMD( range,	set_dac_range,	select dac voltage output range )
ADD_CMD( ld_8800,	do_ld_8800,	load data to the 8800 )
MENU_END(dac)

static COMMAND_FUNC( do_dac_menu )
{
	PUSH_MENU(dac);
}

#ifdef HAVE_DAS1602
#define N_DIO_PORTS	4
static int dio_fd[N_DIO_PORTS]={-1,-1,-1,-1};
#endif // HAVE_DAS1602

static int get_dio_channel(SINGLE_QSP_ARG_DECL)
{
	int i;

	i = HOW_MANY("dio port index (0-3)");
	if( i < 0 || i > 3 ) {
		sprintf(ERROR_STRING,
	"Invalid dio port index %d, must be 0-3",i);
		NWARN(ERROR_STRING);
		return(-1);
	}
	return(i);
}

static const char *dio_mode_strs[2]={"read","write"};

static COMMAND_FUNC( open_dio )
{
#ifdef HAVE_DAS1602
	char dio_name[32];
#endif // HAVE_DAS1602
	int i;
	int mode;

	i=get_dio_channel(SINGLE_QSP_ARG);

	mode = WHICH_ONE("read/write mode",2,dio_mode_strs);

	if( i < 0 || mode < 0 ) return;

#ifdef HAVE_DAS1602

	sprintf(dio_name,"/dev/dio%d",i);

	if( mode == 0 ){
		dio_fd[i] = open(dio_name,O_RDONLY);
	} else {
		dio_fd[i] = open(dio_name,O_WRONLY);
	}

	if( dio_fd[i] < 0 ){
		perror("open");
		sprintf(ERROR_STRING,"error opening dio port %d in %s mode",
			i,dio_mode_strs[mode]);
		NWARN(ERROR_STRING);
	}
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

static COMMAND_FUNC( close_dio )
{
	int i;

	i=get_dio_channel(SINGLE_QSP_ARG);
#ifdef HAVE_DAS1602

	CONFIRM_DIO(i)
	if( close(dio_fd[i]) < 0 ) {
		perror("close");
		sprintf(ERROR_STRING,"error closing dio port %d",i);
		NWARN(ERROR_STRING);
	}
	dio_fd[i]=(-1);
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

static COMMAND_FUNC( write_dio )
{
	int i;
	Data_Obj *dp;
#ifdef HAVE_DAS1602
	int n;
#endif // HAVE_DAS1602

	i=get_dio_channel(SINGLE_QSP_ARG);
	dp = pick_obj("");

	if( i < 0 ) return;
	if(dp==NULL) return;

#ifdef HAVE_DAS1602

	CONFIRM_DIO(i);


	if( ! VALID_DIO_PREC(dp) ){
		sprintf(ERROR_STRING,
		"Object %s has %s precision, should be %s or %s for dio",
			OBJ_NAME(dp),PREC_NAME(OBJ_MACH_PREC_PTR(dp)),
			PREC_BY_NAME,PREC_UBY_NAME);
		NWARN(ERROR_STRING);
		return;
	}

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"Object %s must be contiguous for dio",
			OBJ_NAME(dp));
		NWARN(ERROR_STRING);
		return;
	}

	if( (n=write(dio_fd[i],OBJ_DATA_PTR(dp),OBJ_N_MACH_ELTS(dp))) != (int)OBJ_N_MACH_ELTS(dp) ){
		if( n < 0 ) perror("write");
		sprintf(ERROR_STRING,
	"dio error writing object %s, %d bytes requested, %d actual",
			OBJ_NAME(dp),OBJ_N_MACH_ELTS(dp),n);
		NWARN(ERROR_STRING);
	}
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

static COMMAND_FUNC( read_dio )
{
	int i;
	Data_Obj *dp;
#ifdef HAVE_DAS1602
	int n;
#endif // HAVE_DAS1602

	i=get_dio_channel(SINGLE_QSP_ARG);
	dp = pick_obj("");

	if(dp==NULL) return;
	if( i < 0 ) return;

#ifdef HAVE_DAS1602

	CONFIRM_DIO(i);


	if( ! VALID_DIO_PREC(dp) ){
		sprintf(ERROR_STRING,
		"Object %s has %s precision, should be %s or %s for dio",
			OBJ_NAME(dp),PREC_NAME(OBJ_MACH_PREC_PTR(dp)),
			PREC_BY_NAME,PREC_UBY_NAME);
		NWARN(ERROR_STRING);
		return;
	}

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"Object %s must be contiguous for dio",
			OBJ_NAME(dp));
		NWARN(ERROR_STRING);
		return;
	}

	if( (n=read(dio_fd[i],OBJ_DATA_PTR(dp),OBJ_N_MACH_ELTS(dp))) != (int)OBJ_N_MACH_ELTS(dp) ){
		if( n < 0 ) perror("read");
		sprintf(ERROR_STRING,
	"dio error reading object %s, %d bytes requested, %d actual",
			OBJ_NAME(dp),OBJ_N_MACH_ELTS(dp),n);
		NWARN(ERROR_STRING);
	}
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}





#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(dio_menu,s,f,h)

MENU_BEGIN(dio)
ADD_CMD( open,	open_dio,	open dio device )
ADD_CMD( close,	close_dio,	close dio device )
ADD_CMD( write,	write_dio,	write dio data )
ADD_CMD( read,	read_dio,	read dio data )
MENU_END(dio)

static COMMAND_FUNC( do_dio_menu )
{
	PUSH_MENU(dio);
}


#define VALID_NVRAM_PREC(dp)					\
								\
	((OBJ_MACH_PREC(dp)==PREC_UBY)||(OBJ_MACH_PREC(dp)==PREC_BY))


static COMMAND_FUNC( do_open_nvram )
{

#ifdef HAVE_DAS1602

	nvram_fd = open("/dev/nvram0",O_RDWR);
	if( nvram_fd < 0 ){
		perror("open /dev/nvram0");
		return;
	}
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

static COMMAND_FUNC( do_close_nvram )
{
#ifdef HAVE_DAS1602

	if( nvram_fd < 0 ){
		NWARN("nvram not open");
		return;
	}

	close(nvram_fd);
	nvram_fd = -1;

#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

static COMMAND_FUNC( read_nvram )
{
#ifdef HAVE_DAS1602
	int n;
#endif // HAVE_DAS1602
	Data_Obj *dp;

	dp = pick_obj("");

	if(dp==NULL) return;

#ifdef HAVE_DAS1602

	if( nvram_fd < 0 ) error1("unable to open nvram");

	if( ! VALID_NVRAM_PREC(dp) ){
		sprintf(ERROR_STRING,
		"Object %s has %s precision, should be %s or %s for nvram",
			OBJ_NAME(dp),PREC_NAME(OBJ_MACH_PREC_PTR(dp)),
			PREC_BY_NAME,PREC_UBY_NAME);
		NWARN(ERROR_STRING);
		return;
	}

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"Object %s must be contiguous for nvram",
			OBJ_NAME(dp));
		NWARN(ERROR_STRING);
		return;
	}

	if( (n=read(nvram_fd,OBJ_DATA_PTR(dp),OBJ_N_MACH_ELTS(dp))) != (int)OBJ_N_MACH_ELTS(dp) ){
		if( n < 0 ) perror("read nvram");
		sprintf(ERROR_STRING,
	"nvram error reading object %s, %d bytes requested, %d actual",
			OBJ_NAME(dp),OBJ_N_MACH_ELTS(dp),n);
		NWARN(ERROR_STRING);
	}
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

static COMMAND_FUNC( write_nvram )		/* sq */
{
#ifdef HAVE_DAS1602
	int n;
#endif // HAVE_DAS1602
	Data_Obj *dp;

	dp = pick_obj("");

	if(dp==NULL) return;

#ifdef HAVE_DAS1602

	if( nvram_fd < 0 ) error1("unable to open nvram");

	if( ! VALID_NVRAM_PREC(dp) ){
		sprintf(ERROR_STRING,
		"Object %s has %s precision, should be %s or %s for nvram",
			OBJ_NAME(dp),PREC_NAME(OBJ_MACH_PREC_PTR(dp)),
			PREC_BY_NAME,PREC_UBY_NAME);
		NWARN(ERROR_STRING);
		return;
	}

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"Object %s must be contiguous for nvram",
			OBJ_NAME(dp));
		NWARN(ERROR_STRING);
		return;
	}

	if( (n=write(nvram_fd,OBJ_DATA_PTR(dp),OBJ_N_MACH_ELTS(dp))) != (int)OBJ_N_MACH_ELTS(dp) ){
		if( n < 0 ) perror("write nvram");
		sprintf(ERROR_STRING,
	"nvram error writing object %s, %d bytes requested, %d actual",
			OBJ_NAME(dp),OBJ_N_MACH_ELTS(dp),n);
		NWARN(ERROR_STRING);
	}
#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

static COMMAND_FUNC( ld_nvram )		/*sq */
{
	int addr;
#ifdef HAVE_DAS1602
	int n;
	int original_fd_position;
#endif // HAVE_DAS1602

	Data_Obj *dp;

	addr = HOW_MANY("NVRAM starting addr (0-255)");

	dp = pick_obj("");
	if(dp==NULL) return;

#ifdef HAVE_DAS1602

	if( nvram_fd < 0 ) error1("nvram not open");

	original_fd_position = lseek(nvram_fd,0,SEEK_CUR);

	if(lseek(nvram_fd,addr,SEEK_SET)!=addr) {
		perror("lseek NVRAM");
		NWARN("error setting NVRAM offset");
		return;
	}

	if( addr+OBJ_N_MACH_ELTS(dp) > NVRAM_SIZE ){
		sprintf(ERROR_STRING,"NVRAM: Can't write requested number of bytes. Check address and size of object %s",OBJ_NAME(dp));
		NWARN(ERROR_STRING);
		return;
	}


	if( ! VALID_NVRAM_PREC(dp) ){
		sprintf(ERROR_STRING,"Object %s has %s precision, should be %s or %s for nvram",
			OBJ_NAME(dp),PREC_NAME(OBJ_MACH_PREC_PTR(dp)),PREC_BY_NAME,PREC_UBY_NAME);
		NWARN(ERROR_STRING);
		return;
	}

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"Object %s must be contiguous for nvram",OBJ_NAME(dp));
		NWARN(ERROR_STRING);
		return;
	}

	if( (n=write(nvram_fd,OBJ_DATA_PTR(dp),OBJ_N_MACH_ELTS(dp))) != (int)OBJ_N_MACH_ELTS(dp) ){
		if( n < 0 ) perror("write nvram");
		sprintf(ERROR_STRING, "nvram error writing object %s, %d bytes requested, %d actual",
					OBJ_NAME(dp),OBJ_N_MACH_ELTS(dp),n);
		NWARN(ERROR_STRING);
	}

	lseek(nvram_fd,original_fd_position,SEEK_SET);

#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}

static COMMAND_FUNC( rd_nvram )		/*sq */
{
	int addr;
#ifdef HAVE_DAS1602
	int original_fd_position;
	int n;
#endif // HAVE_DAS1602

	Data_Obj *dp;

	addr = HOW_MANY("NVRAM starting addr (0-255)");

	dp = pick_obj("");
	if(dp==NULL) return;

#ifdef HAVE_DAS1602

	if( nvram_fd < 0 ) error1("nvram not open");

	original_fd_position = lseek(nvram_fd,0,SEEK_CUR);

	if(lseek(nvram_fd,addr,SEEK_SET)!=addr) {
		perror("lseek NVRAM");
		NWARN("error setting NVRAM offset");
		return;
	}

	if( addr+OBJ_N_MACH_ELTS(dp) > NVRAM_SIZE ){
		sprintf(ERROR_STRING,"NVRAM: Can't read requested number of bytes. Check address and size of object %s",OBJ_NAME(dp));
		NWARN(ERROR_STRING);
		return;
	}


	if( ! VALID_NVRAM_PREC(dp) ){
		sprintf(ERROR_STRING,
		"Object %s has %s precision, should be %s or %s for nvram",
			OBJ_NAME(dp),PREC_NAME(OBJ_MACH_PREC_PTR(dp)),
			PREC_BY_NAME,PREC_UBY_NAME);
		NWARN(ERROR_STRING);
		return;
	}

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"Object %s must be contiguous for nvram",OBJ_NAME(dp));
		NWARN(ERROR_STRING);
		return;
	}

	if( (n=read(nvram_fd,OBJ_DATA_PTR(dp),OBJ_N_MACH_ELTS(dp))) != (int)OBJ_N_MACH_ELTS(dp) ){
		if( n < 0 ) perror("read nvram");
		sprintf(ERROR_STRING, "nvram error reading object %s, %d bytes requested, %d actual",
					OBJ_NAME(dp),OBJ_N_MACH_ELTS(dp),n);
		NWARN(ERROR_STRING);
	}

	lseek(nvram_fd,original_fd_position,SEEK_SET);

#else /* ! HAVE_DAS1602 */
	NO_AIO_ALERT
#endif /* ! HAVE_DAS1602 */
}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(nvram_menu,s,f,h)

MENU_BEGIN(nvram)
ADD_CMD( open,		do_open_nvram,	open nvram device )
ADD_CMD( close,		do_close_nvram,	close nvram device )
ADD_CMD( read,		read_nvram,	read nvram data )
ADD_CMD( write,		write_nvram,	write nvram data )
ADD_CMD( ld_nvram,	ld_nvram,	load/write data to an address )
ADD_CMD( rd_nvram,	rd_nvram,	read nvram data from an address )
MENU_END(nvram)

static COMMAND_FUNC( do_nvram_menu )
{
	PUSH_MENU(nvram);
}


#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(das1602_menu,s,f,h)

MENU_BEGIN(das1602)
ADD_CMD( registers,	do_reg_menu,	manipulate device registers )
ADD_CMD( adc,		do_adc_menu,	access ADC )
ADD_CMD( dac,		do_dac_menu,	access DAC )
ADD_CMD( dio,		do_dio_menu,	digital I/O )
ADD_CMD( nvram,		do_nvram_menu,	access NVRAM )
MENU_END(das1602)

COMMAND_FUNC( do_aio_menu )
{
	PUSH_MENU(das1602);
}

