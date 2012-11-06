
#ifndef WAV_HDR

#define WAV_HDR

#include "typedefs.h"

typedef struct wav_hdr {
	char	wh_riff_label[4];
	u_long	wh_chunksize;		/* always 36+datasize, # remaining header bytes + data  */
	char	wh_wave_label[4];
	char	wh_fmt_label[4];
	u_long	wh_always_16;		/* always 16??? */
	short	wh_fmt_tag;		/* always 1 */
	short	wh_n_channels;
	u_long	wh_samp_rate;
	u_long	wh_bytes_per_sec;
	short	wh_blk_align;		/* bytes per time slice */
	short	wh_bits_per_sample;
	char	wh_data_label[4];
	u_long	wh_datasize;		/* total # of bytes */
} Wav_Header;

#endif /* ! WAV_HDR */

