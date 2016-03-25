
#ifdef HAVE_LINUX_IOCTL_H
#include<linux/ioctl.h>
#endif

#ifdef HAVE_SYS_IOCCOM_H
#include <sys/ioccom.h>		// Solaris
#endif

typedef struct {
	int	reg_blocki;	/* index of register block 1-4 */
	int	reg_offset;
	union {
		unsigned char	u_c;
		unsigned short	u_s;
		unsigned long	u_l;
	} reg_data;
} Reg_Data;

/* These ioctl's are useful for implementing user-space drivers... */

#define DAS_SET_REG 		_IOW('y', 1, Reg_Data)
#define DAS_GET_REG 		_IOR('y', 1, Reg_Data)

#define ADC_SET_RANGE		_IOW('y', 2, int)
#define ADC_SET_CFG		_IOW('y', 3, int)
#define ADC_SET_POL		_IOW('y', 4, int)
#define ADC_SET_PACER		_IOW('y', 5, int)
#define ADC_PACER_FREQ		_IOW('y', 6, unsigned short * )

#define DAC0_SET_RANGE		_IOW('y',7,int)
#define DAC1_SET_RANGE		_IOW('y',8,int)
#define ADC_SET_MODE		_IOW('y',9,int)
#define DAC_SET_MODE		_IOW('y',10,int)
#define DAC_PACER_FREQ		_IOW('y', 19, unsigned short * )

#define ADC_LOAD_8402		_IOW('y', 11, int )
#define ADC_LOAD_DAC08		_IOW('y', 12, int )
#define DAC_LOAD_8800		_IOW('y', 13, int )

#define ADC_CALIB_ENABLE	_IOW('y', 14, int )
#define ADC_CALIB_DISABLE	_IOW('y', 15, int )
#define ADC_CALIB_SRC		_IOW('y', 16, int )

#ifdef USE_NVRAM_DEV
#define NVRAM_LD_ADC_COEFFS	_IOW('y', 17, int )
#define NVRAM_LD_DAC_COEFFS	_IOW('y', 18, int )
#endif /* USE_NVRAM_DEV */

/* These are codes used to pass to the driver */

enum {
	ADC_CFG_SE,
	ADC_CFG_DIFF,
	PACER_SW,
	PACER_CTR,
	RANGE_10V_BI,
	RANGE_10V_UNI,
	RANGE_5V_BI,
	RANGE_5V_UNI,
	RANGE_2_5V_BI,
	RANGE_2_5V_UNI,
	RANGE_1_25V_BI,
	RANGE_1_25V_UNI,
	DAC_WAIT
};

typedef enum {
	ADC_MODE_POLLED,
	ADC_MODE_INTR,
	ADC_MODE_PACED
} ADC_Mode;

typedef enum {
	DAC_MODE_POLLED,
	DAC_MODE_PACED,
	DAC_MODE_EXT_RISING,
	DAC_MODE_EXT_FALLING
	/* N_DAC_MODES */
} DAC_Mode;

