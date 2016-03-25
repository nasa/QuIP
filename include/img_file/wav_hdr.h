
#ifndef WAV_HDR

#define WAV_HDR

#include "typedefs.h"

// 10-28-14
// originally written on 32 bit machine, now u_long must
// be changed to uint32_t...

typedef struct wav_hdr {
	char	wh_riff_label[4];
	//u_long	wh_chunksize;		/* always 36+datasize, # remaining header bytes + data  */
	uint32_t	wh_chunksize;		/* always 36+datasize, # remaining header bytes + data  */
	char	wh_wave_label[4];
	char	wh_fmt_label[4];
	//u_long	wh_always_16;		/* always 16??? */
	uint32_t	wh_always_16;		/* always 16??? */
	short	wh_fmt_tag;		/* always 1 */
	short	wh_n_channels;
	//u_long	wh_samp_rate;
	uint32_t	wh_samp_rate;
	//u_long	wh_bytes_per_sec;
	uint32_t	wh_bytes_per_sec;
	short	wh_blk_align;		/* bytes per time slice */
	short	wh_bits_per_sample;
	char	wh_data_label[4];
	//u_long	wh_datasize;		/* total # of bytes */
	uint32_t	wh_datasize;		/* total # of bytes */
} Wav_Header;

#endif /* ! WAV_HDR */

