

#ifndef NO_BDF_INFO

#include "data_obj.h"

/* After the "info" part of the header, comes stuff with the per-channel info...
 *
 * The information for each channel is not grouped, rather for each field in this structure
 * we get all the channels info, then we move on to the next field.  This structure is useful
 * for helping us to calculate the offset of a particular field.
 */

typedef struct per_channel_text {
	char	pct_channel_name[16];		/* A1 B2 etc */
	char	pct_channel_desc[80];		/* "Active Electrode" or "Status ..." */
	char	pct_channel_unit[8];		/* uV or Boolean */
	char	pct_min_val[8];
	char	pct_max_val[8];
	char	pct_adc_min[8];			/* -2^24 */
	char	pct_adc_max[8];			/* 2^24-1 */
	char	pct_filter_desc[80];
	char	pct_sample_rate[8];			/* 512 */
	char	pct_signal_type[32];		/* "EEG" or "TRI" */
} Per_Channel_Text;

/* We create this structure ourselves to keep the info together... */

typedef struct per_channel_info {
	const char *	pci_channel_name;
	const char *	pci_channel_desc;
	const char *	pci_channel_unit;
	long		pci_min_val;
	long		pci_max_val;
	long		pci_adc_min;
	long		pci_adc_max;
	const char *	pci_filter_desc;
	long		pci_sample_rate;
	const char *	pci_signal_type;
} Per_Channel_Info;

#define N_SIGNAL_TYPES	2
#define N_SIGNAL_UNITS	2

/* jbm:  this "info" is what I would usually call the header... */

typedef struct bdf_info_text {
	char bdft_magic[8];
	char bdft_subject[80];
	char bdft_recording[80];
	char bdft_startdate[8];
	char bdft_starttime[8];
	char bdft_headerSize[8];
	char bdft_formatVersion[44];
	char bdft_numDataRecord[8];
	char bdft_dataDuration[8];
	char bdft_numChan[4];
} BDF_info_text;

typedef struct bdf_info {
	const char *		bdf_magic;
	const char *		bdf_subject;
	const char *		bdf_recording;
	const char *		bdf_startdate;
	const char *		bdf_starttime;
	const char *		bdf_headerSize;
	const char *		bdf_formatVersion;
	const char *		bdf_numDataRecord;
	const char *		bdf_dataDuration;
	const char *		bdf_numChan;
	long			bdf_n_channels;
	Per_Channel_Info *	bdf_pci_tbl;
} BDF_info;

#define NO_BDF_INFO	((BDF_info *)NULL)

#endif /* undef NO_BDF_INFO */

