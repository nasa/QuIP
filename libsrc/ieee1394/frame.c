
#include "quip_config.h"

#ifdef HAVE_LIBDV

// C includes

#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>		/* memset() */
#endif

// local includes
#include "frame.h"
#include "getbuf.h"

Frame *new_frame()
{
	Frame *frmp;
	int n;

	frmp = (Frame *)getbuf(sizeof(*frmp));

	memset(frmp->frm_data, 0, MAX_DATA_BYTES);

	frmp->frm_decoder = dv_decoder_new(0,0,0);
	frmp->frm_decoder->quality = DV_QUALITY_COLOR | DV_QUALITY_AC_1;
	dv_set_audio_correction ( frmp->frm_decoder, DV_AUDIO_CORRECT_AVERAGE );
	for ( n = 0; n < 4; n++ )
		frmp->frm_audio_buffers[n] = (short *) malloc(2*DV_AUDIO_MAX_SAMPLES*sizeof(short));
	return(frmp);
}


#ifdef NOT_USED
static void release_frame(Frame *frmp)
{
	int n;
	dv_decoder_free(frmp->frm_decoder);
	for (n = 0; n < 4; n++)
		free(frmp->frm_audio_buffers[n]);
}
#endif /* NOT_USED */


/** checks whether the frame is in PAL or NTSC format
 
	\todo function can't handle "empty" frame
	\return TRUE for PAL frame, FALSE for a NTSC frame
*/

static bool IsPAL(Frame *frmp)
{
	unsigned char dsf = frmp->frm_data[3] & 0x80;
	bool pal = (dsf == 0) ? FALSE : TRUE;
	if (!pal) {
		pal = (dv_system_50_fields (frmp->frm_decoder) == 1) ? TRUE : pal;
	}
	return pal;
}


#ifdef NOT_USED

/** gets a subcode data packet
 
	This function returns a SSYB packet from the subcode data section.
 
	\param packNum the SSYB package id to return
	\param pack a reference to the variable where the result is stored
	\return TRUE for success, FALSE if no pack could be found */

static bool GetSSYBPack(Frame *frmp, int packNum, Pack *packp)
{
	packp->data[ 0 ] = packNum;
#ifdef HAVE_LIBDV_1_0
	dv_get_vaux_pack( frmp->frm_decoder, packNum, packp->data[ 1 ] );
#else
	int id;
	if ( ( id = frmp->frm_decoder->ssyb_pack[ packNum ] ) != 0xff )
	{
		packp->data[ 1 ] = frmp->frm_decoder->ssyb_data[ id ][ 0 ];
		packp->data[ 2 ] = frmp->frm_decoder->ssyb_data[ id ][ 1 ];
		packp->data[ 3 ] = frmp->frm_decoder->ssyb_data[ id ][ 2 ];
		packp->data[ 4 ] = frmp->frm_decoder->ssyb_data[ id ][ 3 ];
	}
#endif
	return TRUE;
	
}


/** gets a video auxiliary data packet
 
	Every DIF block in the video auxiliary data section contains 15
	video auxiliary data packets, for a total of 45 VAUX packets. As
	the position of a VAUX packet is fixed, we could directly look it
	up, but I choose to walk through all data as with the other
	GetXXXX routines.
 
	\param packNum the VAUX package id to return
	\param pack a reference to the variable where the result is stored
	\return TRUE for success, FALSE if no pack could be found */

static bool GetVAUXPack(Frame *frmp,int packNum, Pack *packp)
{
	packp->data[ 0 ] = packNum;
	dv_get_vaux_pack( frmp->frm_decoder, packNum, &packp->data[ 1 ] );
	//cerr << "VAUX: 0x"
	//<< setw(2) << setfill('0') << hex << (int) packp->data[0]
	//<< setw(2) << setfill('0') << hex << (int) packp->data[1]
	//<< setw(2) << setfill('0') << hex << (int) packp->data[2]
	//<< setw(2) << setfill('0') << hex << (int) packp->data[3]
	//<< setw(2) << setfill('0') << hex << (int) packp->data[4]
	//<< endl;
	return TRUE;
	
}


/** gets an audio auxiliary data packet
 
	Every DIF block in the audio section contains 5 bytes audio
	auxiliary data and 72 bytes of audio data.  The function searches
	through all DIF blocks although AAUX packets are only allowed in
	certain defined DIF blocks.
 
	\param packNum the AAUX package id to return
	\param pack a reference to the variable where the result is stored
	\return TRUE for success, FALSE if no pack could be found */

static bool GetAAUXPack(Frame *frmp, int packNum, Pack *packp)
{
	bool done = FALSE;
	int seqCount,i,j;

	switch ( packNum )
	{
		case 0x50:
			memcpy( packp->data, &frmp->frm_decoder->audio->aaux_as, 5 );
			done = TRUE;
			break;

		case 0x51:
			memcpy( packp->data, &frmp->frm_decoder->audio->aaux_asc, 5 );
			done = TRUE;
			break;

#ifdef HAVE_LIBDV_1_0
		case 0x52:
			memcpy( packp->data, frmp->frm_decoder->audio->aaux_as1, 5 );
			done = TRUE;
			break;

		case 0x53:
			memcpy( packp->data, frmp->frm_decoder->audio->aaux_asc1, 5 );
			done = TRUE:
			break;
#else
		default:
			break;
#endif
	}
	if (done)
		return TRUE;
   
	/* number of DIF sequences is different for PAL and NTSC */

	seqCount = IsPAL(frmp) ? 12 : 10;

	/* process all DIF sequences */

	for (i = 0; i < seqCount; ++i) {

		/* there are nine audio DIF blocks */

		for (j = 0; j < 9; ++j) {

			/* calculate address: 150 DIF blocks per sequence, 80 bytes
			   per DIF block, audio blocks start at every 16th beginning
			   with block 6, block has 3 bytes header, followed by one
			   packet. */

			const unsigned char *s;
		s = &frmp->frm_data[i * 150 * 80 + 6 * 80 + j * 16 * 80 + 3];
			if (s[0] == packNum) {
				// printf("aaux %d: %2.2x %2.2x %2.2x %2.2x %2.2x\n",
				// j, s[0], s[1], s[2], s[3], s[4]);
				packp->data[0] = s[0];
				packp->data[1] = s[1];
				packp->data[2] = s[2];
				packp->data[3] = s[3];
				packp->data[4] = s[4];
				return TRUE;
			}
		}
	}
	return FALSE;
}

static const char * GetRecordingDateString(Frame *frmp)
{
	const char * recDate;
	static char s[64];
	if (dv_get_recording_datetime( frmp->frm_decoder, s))
		recDate = s;
	else
		recDate = "0000-00-00 00:00:00";
	return recDate;
}

static bool GetVideoInfo(Frame *frmp,VideoInfo *vip)
{
	GetTimeCode(frmp,&vip->vi_timeCode);
	GetRecordingDateTime(frmp,&vip->vi_recDate);
	vip->vi_isPAL = IsPAL(frmp);
	return TRUE;
}


/** get the video frame rate

	\return frames per second
*/
static float GetFrameRate(Frame *frmp)
{
	return IsPAL(frmp) ? 25.0 : 30000.0/1001.0;
}

/** checks whether this frame is the first in a new recording
 
	To determine this, the function looks at the recStartPoint bit in
	AAUX pack 51.
 
	\return TRUE if this frame is the start of a new recording */

static bool IsNewRecording(Frame *frmp)
{
	return (frmp->frm_decoder->audio->aaux_asc.pc2.rec_st == 0);
}


/** retrieves the audio data from the frame
 
	The DV frame contains audio data mixed in the video data blocks, 
	which can be retrieved easily using this function.
 
	The audio data consists of 16 bit, two channel audio samples (a 16 bit word for channel 1, followed by a 16 bit word
	for channel 2 etc.)
 
	\param sound a pointer to a buffer that holds the audio data
	\return the number of bytes put into the buffer, or 0 if no audio data could be retrieved */

static int ExtractAudio(Frame *frmp)
{
	AudioInfo info;
	
	if (GetAudioInfo(frmp,&info) == TRUE) {

		dv_decode_full_audio( frmp->frm_decoder, frmp->frm_data, (short **)frmp->frm_audio_buffers);
	} else
		info.samples = 0;
	
	return info.samples * info.channels * 2;
}



/** gets the audio properties of this frame
 
	get the sampling frequency and the number of samples in this particular DV frame (which can vary)
 
	\param info the AudioInfo record
	\return TRUE, if audio properties could be determined */

static bool GetAudioInfo(Frame *frmp, AudioInfo *info_p)
{
	info_p->frequency = frmp->frm_decoder->audio->frequency;
	info_p->samples = frmp->frm_decoder->audio->samples_this_frame;
	info_p->frames = (frmp->frm_decoder->audio->aaux_as.pc3.system == 1) ? 50 : 60;
	info_p->channels = frmp->frm_decoder->audio->num_channels;
	info_p->quantization = (frmp->frm_decoder->audio->aaux_as.pc4.qu == 0) ? 16 : 12;
	return TRUE;
}



static void Deinterlace( Frame *frmp,  void *image, int bpp )
{
	int i;
	int width = GetWidth(frmp) * bpp;
	int height = GetHeight(frmp);
	for ( i = 0; i < height; i += 2 )
		memcpy( (uint8_t *)image + width * ( i + 1 ), (uint8_t *)image + width * i, width );
}


/** Get the frame image width.

	\return the width in pixels.
*/
static int GetWidth(Frame *frmp)
{
	return frmp->frm_decoder->width;
}

/** Get the frame image height.

	\return the height in pixels.
*/
static int GetHeight(Frame *frmp)
{
	return frmp->frm_decoder->height;
}

static int ExtractYUV(Frame *frmp, void *yuv)
{
	unsigned char *pixels[3];
	int pitches[3];

	pixels[0] = (unsigned char*)yuv;
	pitches[0] = frmp->frm_decoder->width * 2;

	dv_decode_full_frame(frmp->frm_decoder, frmp->frm_data, e_dv_color_yuv, pixels, pitches);
	return 0;
}


/** Set the RecordingDate of the frame.

	This updates the calendar date and time and the timecode.
	However, timecode is derived from the time in the datetime
	parameter and frame number. Use SetTimeCode for more control
	over timecode.

	\param datetime A simple time value containing the
		   RecordingDate and time information. The time in this
		   structure is automatically incremented by one second
		   depending on the frame parameter and updatded.
	\param frame A zero-based running frame sequence/serial number.
		   This is used both in the timecode as well as a timestamp on
		   dif block headers.
*/
static void SetRecordingDate( Frame *frmp,  time_t *datetime, int frame )
{
	dv_encode_metadata( frmp->frm_data, IsPAL(frmp), IsWide(frmp), datetime, frame );
}

/** Get the frame aspect ratio.

	Indicates whether frame aspect ration is normal (4:3) or wide (16:9).

	\return TRUE if the frame is wide (16:9), FALSE if unknown or normal.
*/
static bool IsWide( Frame *frmp )
{
	return dv_format_wide(frmp->frm_decoder) > 0;
}


/** Set the TimeCode of the frame.

	This function takes a zero-based frame counter and automatically
	derives the timecode.

	\param frame The frame counter.
*/
static void SetTimeCode( Frame *frmp, int frame )
{
	dv_encode_timecode( frmp->frm_data, IsPAL(frmp), frame );
}


#endif /* NOT_USED */

/** gets the date and time of recording of this frame
 
	Returns a struct tm with date and time of recording of this frame.
 
	This code courtesy of Andy (http://www.videox.net/) 
 
	\param recDate the time and date of recording of this frame
	\return TRUE for success, FALSE if no date or time information could be found */

bool GetRecordingDateTime(Frame *frmp, struct tm *recDate_p)
{
	return dv_get_recording_datetime_tm( frmp->frm_decoder, recDate_p);
}



/** gets the timecode information of this frame
 
	Returns a string with the timecode of this frame. The timecode is
	the relative location of this frame on the tape, and is defined
	by hour, minute, second and frame (within the last second).
	 
	\param timeCode the TimeCode struct
	\return TRUE for success, FALSE if no timecode information could be found */

bool GetTimeCode(Frame *frmp, TimeCode *timeCode_p)
{
	int timestamp[4];
	
	dv_get_timestamp_int( frmp->frm_decoder, timestamp);
	
	timeCode_p->hour = timestamp[0];
	timeCode_p->min = timestamp[1];
	timeCode_p->sec = timestamp[2];
	timeCode_p->frame = timestamp[3];
	return TRUE;
}

/** gets the size of the frame
 
	Depending on the type (PAL or NTSC) of the frame, the length of the frame is returned 
 
	\return the length of the frame in Bytes */

int GetFrameSize(Frame *frmp)
{
	return IsPAL(frmp) ? 144000 : 120000;
}


/** check whether we have received as many bytes as expected for this frame
 
	\return TRUE if this frames is completed, FALSE otherwise */

bool IsComplete(Frame *frmp)
{
//sprintf(error_string,"IsComplete:  bytesInFrame = %d, frame size = %d",
//frmp->frm_bytesInFrame,GetFrameSize(frmp));
//advise(error_string);
	return frmp->frm_bytesInFrame == GetFrameSize(frmp);
}

void ExtractHeader(Frame *frmp)
{
	dv_parse_header(frmp->frm_decoder, frmp->frm_data);
	dv_parse_packs(frmp->frm_decoder, frmp->frm_data);
}

int ExtractRGB(Frame *frmp, void *rgb)
{
	unsigned char *pixels[3];
	int pitches[3];

	pixels[0] = (unsigned char*)rgb;
	pixels[1] = NULL;
	pixels[2] = NULL;

	pitches[0] = 720 * 3;		// where does this number come from???
	pitches[1] = 0;
	pitches[2] = 0;

	dv_decode_full_frame(frmp->frm_decoder, frmp->frm_data, e_dv_color_rgb, pixels, pitches);
	return 0;
}

#endif /* HAVE_LIBDV */

