
#ifndef WAV_HDR

#define WAV_HDR

#include "typedefs.h"

// 7-6-2018	Got a wav file with different chunk layout,
// must overhaul header reading software to detect different
// chunk types.

// 10-28-14
// originally written on 32 bit machine, now u_long must
// be changed to uint32_t...

typedef struct wav_chunk_hdr {
	char		wch_label[4];
	uint32_t	wch_size;	// number of bytes remaining in the  
					// chunk - but in RIFF chunk is for
					// whole file???
} Wav_Chunk_Hdr;

typedef struct wav_hdr_chunk {
	Wav_Chunk_Hdr	whc_wch;
	char		whc_wave_label[4];
} Wav_Hdr_Chunk;

typedef struct wav_fmt_data {
	short		wfd_compression_code;	/* usually 1? */
	short		wfd_n_channels;
	uint32_t	wfd_samp_rate;
	uint32_t	wfd_bytes_per_sec;
	short		wfd_blk_align;		/* bytes per time slice */
	short		wfd_bits_per_sample;
	// possible extra bytes here???
} Wav_Fmt_Data;

typedef struct fmt_hdr_chunk {
	Wav_Chunk_Hdr	fhc_wch;
	Wav_Fmt_Data	fhc_wfd;
} Fmt_Hdr_Chunk;

typedef struct data_hdr_chunk {
	Wav_Chunk_Hdr	dhc_wch;
} Data_Hdr_Chunk;

typedef struct wav_hdr {
	Wav_Hdr_Chunk	wh_whc;
	Fmt_Hdr_Chunk	wh_fhc;
	Data_Hdr_Chunk	wh_dhc;
} Wav_Header;

#define wh_riff_label		wh_whc.whc_wch.wch_label
#define wh_chunksize		wh_whc.whc_wch.wch_size
#define wh_wave_label		wh_whc.whc_wave_label

#define wh_fmt_label		wh_fhc.fhc_wch.wch_label
#define wh_fmt_size		wh_fhc.fhc_wch.wch_size

#define wh_fmt_tag		wh_fhc.fhc_wfd.wfd_compression_code
#define wh_bits_per_sample	wh_fhc.fhc_wfd.wfd_bits_per_sample
#define wh_n_channels		wh_fhc.fhc_wfd.wfd_n_channels
#define wh_samp_rate		wh_fhc.fhc_wfd.wfd_samp_rate
#define wh_bytes_per_sec	wh_fhc.fhc_wfd.wfd_bytes_per_sec
#define wh_blk_align		wh_fhc.fhc_wfd.wfd_blk_align

#define wh_data_label		wh_dhc.dhc_wch.wch_label
#define wh_datasize		wh_dhc.dhc_wch.wch_size

#endif /* ! WAV_HDR */

#ifdef OLD
typedef struct wav_hdr {
	char	wh_riff_label[4];
	uint32_t	wh_chunksize;		/* always 36+datasize, # remaining header bytes + data  */
	char	wh_wave_label[4];
	char	wh_fmt_label[4];
	uint32_t	wh_always_16;		/* always 16??? */
	short	wh_fmt_tag;		/* always 1 */
	short	wh_n_channels;
	uint32_t	wh_samp_rate;
	uint32_t	wh_bytes_per_sec;
	short	wh_blk_align;		/* bytes per time slice */
	short	wh_bits_per_sample;
	char	wh_data_label[4];	// Can also be LIST???
	uint32_t	wh_datasize;		/* total # of bytes */
} Wav_Header;
#endif // OLD
