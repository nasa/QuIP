#ifndef _FRAME_H_

#define _FRAME_H_ 1

#include "quip_config.h"

#ifdef HAVE_TIME_H
#include <time.h>
#endif

#ifdef HAVE_LIBDV_DV_H
#include <libdv/dv.h>
#endif

#ifdef HAVE_LIBDV_DV_TYPES_H
#include <libdv/dv_types.h>
#endif

#include "dv_globals.h"

#define FRAME_MAX_WIDTH 720
#define FRAME_MAX_HEIGHT 576


typedef struct Pack
{
	/// the five bytes of a packet
	unsigned char data[5];
}
Pack;

typedef struct TimeCode
{
	int hour;
	int min;
	int sec;
	int frame;
} TimeCode;


typedef struct AudioInfo
{
	int frames;
	int frequency;
	int samples;
	int channels;
	int quantization;
} AudioInfo;


typedef struct VideoInfo
{
	int vi_width;
	int vi_height;
	bool vi_isPAL;
	TimeCode vi_timeCode;
	struct tm	vi_recDate;
} VideoInfo;

#define MAX_DATA_BYTES 144000

typedef struct Frame
{
	/// enough space to hold a PAL frame
	unsigned char frm_data[MAX_DATA_BYTES];
	/// the number of bytes written to the frame
	int frm_bytesInFrame;
#ifdef HAVE_LIBDV
	dv_decoder_t *frm_decoder;
#endif
	short *frm_audio_buffers[4];

} Frame;

#endif /* undef _FRAME_H */

extern int GetFrameSize(Frame *);
extern void ExtractHeader(Frame *);
extern bool IsComplete(Frame *);
//extern bool DoneWithFrame(Frame *);
extern bool GetTimeCode(Frame *frmp, TimeCode *timeCode_p);
extern bool GetRecordingDateTime(Frame *frmp, struct tm *recDate_p);
extern Frame *new_frame(void);
extern int ExtractRGB(Frame *frmp, void *rgb);


