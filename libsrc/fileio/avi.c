
/* can we force this string to be loaded? */

#include "quip_config.h"

char VersionId_fio_avi[] = QUIP_VERSION_STRING;

int force_avi_load;		/* see comment in matio.c */

#ifdef HAVE_LIBAVCODEC


#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#include "fio_prot.h"
#include "filetype.h"
#include "getbuf.h"
#include "data_obj.h"
#include "debug.h"
#include "savestr.h"

#include "my_avi.h"

#include "query.h"		/* assign_var() */

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#include "libswscale/swscale.h"
}
#else
#include "libswscale/swscale.h"
#endif

static int lib_avcodec_inited=0;	/* our flag */

/* local prototypes */
static void scan_file(QSP_ARG_DECL  Image_File *ifp);
static void scan_seek(QSP_ARG_DECL  Image_File *ifp);
static long unskewed_frame_index(long index,Image_File *ifp);
static long skewed_frame_index(long index, Image_File *ifp);

//#define HDR_P		((Image_File_Hdr *)ifp->if_hd)->ifh_u.avc_hd_p
#define HDR_P		(&(((Image_File_Hdr *)ifp->if_hd)->ifh_u.avc_hd))

void avi_info(QSP_ARG_DECL  Image_File *ifp)
{
	sprintf(msg_str,"File %s:",ifp->if_name);
	prt_msg(msg_str);
}

#ifdef FOOBAR
void SaveFrame(AVFrame *pFrame, int width, int height, int iFrame) {
	FILE *pFile;
	char szFilename[32];
	int y;

	// Open file
	sprintf(szFilename, "frame%d.ppm", iFrame);
	pFile=fopen(szFilename, "wb");
	if(pFile==NULL)
		return;

	// Write header
	fprintf(pFile, "P6\n%d %d\n255\n", width, height);

	// Write pixel data
	for(y=0; y<height; y++)
		fwrite(pFrame->data[0]+y*pFrame->linesize[0], 1, width*3, pFile);

	// Close file
	fclose(pFile);
}
#endif

static void print_fps(double r, const char *s)
{
	sprintf(DEFAULT_ERROR_STRING,"\t%s:  %g",s,r);
	advise(DEFAULT_ERROR_STRING);
}

int avi_to_dp(Data_Obj *dp,AVCodec_Hdr *hd_p)
{
	//uint32_t fps;

	dp->dt_cols = hd_p->avch_codec_ctx_p->width;
	dp->dt_rows = hd_p->avch_codec_ctx_p->height;
	dp->dt_comps = 3;

	/* Getting the number of frames is a bit tricky...
	 * We have the duration (in microseconds)
	 * but there are three different ways of computing the
	 * frame rate:
	 */

#ifdef FOR_INFO_ONLY
	/* This code comes from dump_format... */
	/* FOR_INFO_ONLY means don't try to compile it! */
	if(st->r_frame_rate.den && st->r_frame_rate.num)
		print_fps(av_q2d(st->r_frame_rate), "tbr");
	if(st->time_base.den && st->time_base.num)
		print_fps(1/av_q2d(st->time_base), "tbn");
	if(st->codec->time_base.den && st->codec->time_base.num)
		print_fps(1/av_q2d(st->codec->time_base), "tbc");
#endif

	/* This is the frame rate - 29.97 */
	if(hd_p->avch_video_stream_p->r_frame_rate.den && hd_p->avch_video_stream_p->r_frame_rate.num)
		print_fps(av_q2d(hd_p->avch_video_stream_p->r_frame_rate), "tbr");

	/* This is 90000 - why? */
	if(hd_p->avch_video_stream_p->time_base.den && hd_p->avch_video_stream_p->time_base.num)
		print_fps(1/av_q2d(hd_p->avch_video_stream_p->time_base), "tbn");

	/* This is the field rate 59.94 */
	if(hd_p->avch_video_stream_p->codec->time_base.den && hd_p->avch_video_stream_p->codec->time_base.num)
		print_fps(1/av_q2d(hd_p->avch_video_stream_p->codec->time_base), "tbc");


sprintf(DEFAULT_ERROR_STRING,"duration = %ld, AV_TIME_BASE = %d, fps? = %g, time base = %g",
hd_p->avch_format_ctx_p->duration,
AV_TIME_BASE,
1/av_q2d(hd_p->avch_video_stream_p->time_base),
av_q2d(hd_p->avch_video_stream_p->time_base)
);
advise(DEFAULT_ERROR_STRING);

/* The frame count isn't quite right ...
   When we seek, we seem to get to 10x the seek N_frames in seconds...
   which suggests that the software thinks we have specified 10 fps.
   (For our dtdm movies, it's really 15fps!?!?)
   But our total frame cound is not off by a lot...
   Seeking to 94674 is bad, 94673 is good.
   According to xine, the movie len is 1:45:12...
   = 60*60 + 45*60 + 12 seconds =  6312 seconds.
   6312 seconds * 15 fps = 94680 frames, pretty close...
   Reported movie len (in usecs) is: 6315933333
   This is longer than where we crap out, but note that the frame
   only changes every 250 "frames" (25 seconds).
   So perhaps our strategy for seeking should be to round the seek
   target down to a multiple of 25 seconds, and then play forward...
   This is going to make a seek-per-frame movie player (like jpsee)
   a real dog!
 */

/* With the shuttle videos, the "time base" is 1/90000, so the following line gives up
 * a frame rate of 90000 fps...  But this is nuts.  However, in scan_file, the pts's
 * are separated by 33 msec, BEFORE any scaling, consistent with 30 fps.  If we use
 * 30 for fps there, then we get good frame serial numbers.
 */
	hd_p->avch_fps = round( 1/av_q2d(hd_p->avch_video_stream_p->time_base) );
//sprintf(DEFAULT_ERROR_STRING,"avi_to_dp:  fps = %ld",hd_p->avch_fps);
//advise(DEFAULT_ERROR_STRING);

	/* We compute fps above, to avoid a numerical difference on 32bit and 64bit machines. */

	/* The integer division below causes the frame count to be rounded down to an
	 * integral number of seconds.
	 */

#ifdef LONG_64_BIT
sprintf(DEFAULT_ERROR_STRING,"avi_to_dp:  unscaled duration is %ld",hd_p->avch_format_ctx_p->duration);
#else
sprintf(DEFAULT_ERROR_STRING,"avi_to_dp:  unscaled duration is %lld",hd_p->avch_format_ctx_p->duration);
#endif
advise(DEFAULT_ERROR_STRING);

	dp->dt_frames = (hd_p->avch_format_ctx_p->duration / AV_TIME_BASE)	/* seconds */
				* hd_p->avch_fps;
				//* (1/av_q2d(hd_p->avch_video_stream_p->time_base));	/* fps */

sprintf(DEFAULT_ERROR_STRING,"nf = %d",dp->dt_frames);
advise(DEFAULT_ERROR_STRING);

	dp->dt_seqs = 1;

	dp->dt_prec = PREC_UBY;	/* BUG get this from file!!! */

	return 0;
} /* end avi_to_dp() */

#ifndef INT64_C
#define INT64_C		(uint64_t)
#endif

uint64_t global_video_pkt_pts = AV_NOPTS_VALUE;

/* These are called whenever we allocate a frame
 * buffer. We use this to store the global_pts in
 * a frame at the time it is allocated.
 */
int our_get_buffer(struct AVCodecContext *c, AVFrame *pic)
{
	int ret = avcodec_default_get_buffer(c, pic);
	uint64_t *pts = (uint64_t *) av_malloc(sizeof(uint64_t));
	*pts = global_video_pkt_pts;
	pic->opaque = pts;

	return ret;
}

void our_release_buffer(struct AVCodecContext *c, AVFrame *pic)
{
	if(pic) av_freep(&pic->opaque);
	avcodec_default_release_buffer(c, pic);
}

FIO_OPEN_FUNC( avi_open )
{
	Image_File *ifp;
	int i;
	long numBytes;

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_AVI);
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	/* image_file_open updates if_pathname if a directory has been specified */

	ifp->if_hd = getbuf( sizeof(AVCodec_Hdr) );

	HDR_P->avch_img_convert_ctx_p = NULL;

	/* create file struct, but don't actually open the file - the lib does it... */
	if( ! lib_avcodec_inited ){
		/* init the library */
		av_register_all();
		lib_avcodec_inited=1;
	}

#if LIBAVFORMAT_VERSION_INT >=	AV_VERSION_INT(53,4,0)

	/* This code uses the new API but has not been tested */
/*
	int avformat_open_input(
		AVFormatContext **ps,
		const char *filename,
		AVInputFormat *fmt,
		AVDictionary **options);
*/

	if( avformat_open_input(&HDR_P->avch_format_ctx_p,
				ifp->if_pathname,
				NULL,
				NULL) != 0 ){
		sprintf(ERROR_STRING,"libavformat error opening file %s",ifp->if_pathname);
		WARN(ERROR_STRING);
		return(NO_IMAGE_FILE);
	}

#else /* OLD_VERSION */

	if( av_open_input_file(&HDR_P->avch_format_ctx_p,ifp->if_pathname,NULL,0,NULL) != 0 ){
		sprintf(ERROR_STRING,"libavcodec error opening file %s",ifp->if_pathname);
		WARN(ERROR_STRING);
		return(NO_IMAGE_FILE);
	}


#endif

	if( av_find_stream_info(HDR_P->avch_format_ctx_p)<0 ){
		sprintf(ERROR_STRING,"Couldn't find stream info for file %s",name);
		WARN(ERROR_STRING);
		return(NO_IMAGE_FILE);
	}
//dump_format(HDR_P->avch_format_ctx_p,0,name,0);

	HDR_P->avch_video_stream_index=(-1);
	for(i=0;i<(int)HDR_P->avch_format_ctx_p->nb_streams;i++){

#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(51,9,0)
#define EXPECTED_VIDEO_TYPE	AVMEDIA_TYPE_VIDEO
#else
#define EXPECTED_VIDEO_TYPE	CODEC_TYPE_VIDEO
#endif

		if(
	HDR_P->avch_format_ctx_p->streams[i]->codec->codec_type ==
			EXPECTED_VIDEO_TYPE ) {
//sprintf(ERROR_STRING,"Found video at stream #%d",i);
//advise(ERROR_STRING);
			HDR_P->avch_video_stream_index=i;
			HDR_P->avch_video_stream_p =
				HDR_P->avch_format_ctx_p->streams[i];
			break;
		}
	}
	if(HDR_P->avch_video_stream_index==-1){
		WARN("no video stream");
		return(NO_IMAGE_FILE);
	}
	// Get a pointer to the codec context for the video stream
	HDR_P->avch_codec_ctx_p = HDR_P->avch_video_stream_p->codec;

	/* use custom buffer functions to allow us to cache time stamps... */
	HDR_P->avch_codec_ctx_p->get_buffer = our_get_buffer;
	HDR_P->avch_codec_ctx_p->release_buffer = our_release_buffer;


	// Find the decoder for the video stream
	HDR_P->avch_codec_p=avcodec_find_decoder(HDR_P->avch_codec_ctx_p->codec_id);
	if(HDR_P->avch_codec_p==NULL) {
		WARN("Unsupported codec!?");
		return(NO_IMAGE_FILE);
	}
	// Open codec
	if(avcodec_open(HDR_P->avch_codec_ctx_p, HDR_P->avch_codec_p)<0){
		WARN("couldn't open codec");
		return(NO_IMAGE_FILE);
	}

	// Allocate video frame
	HDR_P->avch_frame_p=avcodec_alloc_frame();
	if(HDR_P->avch_frame_p==NULL){
		WARN("couldn't allocate first frame");
		return(NO_IMAGE_FILE);
	}

	// Allocate an AVFrame structure
	HDR_P->avch_rgb_frame_p=avcodec_alloc_frame();
	if(HDR_P->avch_rgb_frame_p==NULL){
		WARN("couldn't allocate another frame");
		return(NO_IMAGE_FILE);
	}

	// Determine required buffer size and allocate buffer
	numBytes=avpicture_get_size(PIX_FMT_RGB24, HDR_P->avch_codec_ctx_p->width,
			HDR_P->avch_codec_ctx_p->height);
	HDR_P->avch_buffer=(uint8_t *)av_malloc(numBytes*sizeof(uint8_t));

	// Assign appropriate parts of buffer to image planes in HDR_P->avch_rgb_frame_p
	// Note that HDR_P->avch_rgb_frame_p is an AVFrame, but AVFrame is a superset
	// of AVPicture
	avpicture_fill((AVPicture *)HDR_P->avch_rgb_frame_p, HDR_P->avch_buffer, PIX_FMT_RGB24,
				HDR_P->avch_codec_ctx_p->width, HDR_P->avch_codec_ctx_p->height);


	HDR_P->avch_duration = HDR_P->avch_format_ctx_p->duration / AV_TIME_BASE;	/* seconds */

	avi_to_dp(ifp->if_dp,ifp->if_hd);


	/* initialize our private flag */
	HDR_P->avch_have_frame_ready=0;

advise("checking avi info...");
	if( check_avi_info(ifp) < 0 ){
		/* info doesn't already exist */
		advise("No cached avi info, scanning file...");
		scan_file(QSP_ARG  ifp);

		/* With shuttle mpeg files, this generates a lot of errors... */
		scan_seek(QSP_ARG  ifp);

		/*--- Otherwise, we can do this:  ---*/
		//HDR_P->avch_n_seek_tbl_entries = 0;
		//HDR_P->avch_seek_tbl = NULL;
		/*----*/

		/* now write out the file... */
		save_avi_info(ifp);
	}

	return(ifp);
} /* end avi_open() */


FIO_CLOSE_FUNC( avi_close )
{
	/* This stuff cleans up after reading,
	 * at the moment we don't know how to write...
	 */

	// Free the RGB image
	av_free(HDR_P->avch_buffer);
	av_free(HDR_P->avch_rgb_frame_p);

	// Free the YUV frame
	av_free(HDR_P->avch_frame_p);

	// Close the codec
	avcodec_close(HDR_P->avch_codec_ctx_p);

	// Close the video file
	av_close_input_file(HDR_P->avch_format_ctx_p);

	/* TIFFClose(ifp->if_avi); */
	GENERIC_IMGFILE_CLOSE(ifp);
}

int dp_to_avi(AVCodec_Hdr *avip,Data_Obj *dp)
{
	NERROR1("Sorry, dp_to_avi not implemented");
	return(-1);
}

FIO_SETHDR_FUNC( set_avi_hdr )
{
	/*
	if( dp_to_avi(ifp->info_p,ifp->if_dp) < 0 ){
		avi_close(ifp);
		return(-1);
	}
	*/
	return(0);
}

static void copy_frame_data(Data_Obj *dp, AVFrame * frame_p )
{
	/* BUG here we ignore the offsets, and also the fact that the avcodec frame may not have
	 * contiguous lines...
	 */
	dimension_t r;
	u_char *cp_to, *cp_fr;

	cp_to = (u_char *)dp->dt_data;
	cp_fr = frame_p->data[0];
	for(r=0;r<dp->dt_rows;r++){
		memcpy( cp_to, cp_fr , dp->dt_comps * dp->dt_cols );
		cp_to += dp->dt_rowinc;
		cp_fr += frame_p->linesize[0];
	}
}

static void convert_video_frame(Image_File *ifp)
{
	int ret;

	// Did we get a video frame?
	if( ! HDR_P->avch_frame_finished) {
		NWARN("convert_video_frame:  frame is not finished!?");
		return;
	}

	// Convert the image from its native format to RGB
	/*
	img_convert((AVPicture *)HDR_P->avch_rgb_frame_p, PIX_FMT_RGB24,
		(AVPicture*)HDR_P->avch_frame_p, HDR_P->avch_codec_ctx_p->pix_fmt,
		HDR_P->avch_codec_ctx_p->width, HDR_P->avch_codec_ctx_p->height);
		*/

	// This comment from the sample program looks wrong!?
	// Convert the image into YUV format that SDL uses

	/* BUG don't do all this conversion if we're just skipping during a seek */
	if(HDR_P->avch_img_convert_ctx_p == NULL) {
		int w = HDR_P->avch_codec_ctx_p->width;
		int h = HDR_P->avch_codec_ctx_p->height;

//sprintf(error_string,"source format is %d (0x%x)",
//HDR_P->avch_codec_ctx_p->pix_fmt,
//HDR_P->avch_codec_ctx_p->pix_fmt );
//advise(error_string);
		HDR_P->avch_img_convert_ctx_p = sws_getContext(w, h,
				HDR_P->avch_codec_ctx_p->pix_fmt,
				w, h, /*PIX_FMT_RGB24*/ PIX_FMT_BGR24, SWS_BICUBIC,
				NULL, NULL, NULL);
		if(HDR_P->avch_img_convert_ctx_p == NULL) {
			NWARN("Cannot initialize the conversion context!");
			return;
		}
	}
	// We have a problem here due to using the latest version of
	// the library??

//#if LIBSWSCALE_VERSION_INT >= AV_VERSION_INT(2,0,0)
#if LIBSWSCALE_VERSION_INT >= AV_VERSION_INT(0,11,0)

	/* BUG this hasn't been fixed to work with the new version!? */
	ret = sws_scale(	HDR_P->avch_img_convert_ctx_p,
				(const uint8_t * const * )
				HDR_P->avch_frame_p->data,
				HDR_P->avch_frame_p->linesize,
				0,
				HDR_P->avch_codec_ctx_p->height,
				HDR_P->avch_rgb_frame_p->data,
				HDR_P->avch_rgb_frame_p->linesize);

#else /* OLD_VERSION */

	ret = sws_scale(	HDR_P->avch_img_convert_ctx_p,
				/* cast here? */ HDR_P->avch_frame_p->data,
				HDR_P->avch_frame_p->linesize,
				0,
				HDR_P->avch_codec_ctx_p->height,
				HDR_P->avch_rgb_frame_p->data,
				HDR_P->avch_rgb_frame_p->linesize);

#endif

	/* Now we've converted into our libavcodec rgb frame,
	 * but can we go straight to our data obj?
	 */
}

static int get_next_avi_frame(QSP_ARG_DECL  Image_File *ifp)
{
	HDR_P->avch_frame_finished = 0;
	while( ! HDR_P->avch_frame_finished ){
		if( av_read_frame(HDR_P->avch_format_ctx_p, &HDR_P->avch_packet)<0) {
			NWARN("av_read_frame:  no data!?");
			return(-1);
		}
		// Is this a packet from the video stream?
		if(HDR_P->avch_packet.stream_index==HDR_P->avch_video_stream_index) {

			// Decode video frame
			// at which version did avcodec_decode_video
			// get changed to avcodec_decode_video2 ???
			//
			// The newer version isn't guaranteed to decode
			// a whole frame, we need to check when a frame
			// has been completed???  BUG???

#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(51,9,0)

/*
int avcodec_decode_video2(	AVCodecContext *avctx,
				AVFrame *picture,
				int *got_picture_ptr,
				AVPacket *avpkt);
*/

	avcodec_decode_video2(	HDR_P->avch_codec_ctx_p,
				HDR_P->avch_frame_p,
				&HDR_P->avch_frame_finished,
				&HDR_P->avch_packet
				);

#else
			avcodec_decode_video(HDR_P->avch_codec_ctx_p,
						HDR_P->avch_frame_p,
						&HDR_P->avch_frame_finished,
						HDR_P->avch_packet.data,
						HDR_P->avch_packet.size);

#endif
			/* decoding time stamp (dts) should be equal to pts thanks
			 * to ffmpeg...
			 */
			if(HDR_P->avch_packet.dts == (int64_t)AV_NOPTS_VALUE 
		&& HDR_P->avch_frame_p->opaque
		&& *(uint64_t*)HDR_P->avch_frame_p->opaque != AV_NOPTS_VALUE) {

				HDR_P->avch_pts = *(uint64_t *)HDR_P->avch_frame_p->opaque;
//advise("pts set from opaque...");
			} else if(HDR_P->avch_packet.dts != (int64_t)AV_NOPTS_VALUE) {
//advise("get_next_avi_frame:  pts set from packet...");
				HDR_P->avch_pts = HDR_P->avch_packet.dts;
			} else {
				HDR_P->avch_pts = 0;
			}

			HDR_P->avch_pts *= av_q2d(HDR_P->avch_video_stream_p->time_base);
//sprintf(error_string,"after avcodec_decode_video, pts = %g",HDR_P->avch_pts);
//advise(error_string);
			/* Not sure how much this slows us down... */
			sprintf(error_string,"%g",HDR_P->avch_pts);
			ASSIGN_VAR("pts",error_string);
		}

		// Free the packet that was allocated by av_read_frame
		// COMMENTED OUT AFTER YUM INSTALL HAS NO FUNC
		//av_free_packet(&HDR_P->avch_packet);
	}
	/* frame is finished */
	ifp->if_nfrms++;
	return(0);
}

/* read the next frame */

FIO_RD_FUNC( avi_rd )
{
	if( HDR_P->avch_have_frame_ready ){
		HDR_P->avch_have_frame_ready = 0;
		ifp->if_nfrms++;
	} else {
		if( get_next_avi_frame(QSP_ARG  ifp) < 0 ) return;
		convert_video_frame(ifp);
	}
	/* Now copy to our data object */
	copy_frame_data(dp,HDR_P->avch_rgb_frame_p);
}


int avi_unconv(void *hdr_pp,Data_Obj *dp)
{
	NWARN("avi_unconv not implemented");
	return(-1);
}

int avi_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("avi_conv not implemented");
	return(-1);
}

#ifdef FOOBAR
static AVPacket flush_pkt;
static int flush_pkt_inited=0;

static void init_flush_pkt(void)
{
	av_init_packet(&flush_pkt);
	flush_pkt.data = "FLUSH";
}

static void packet_queue_flush(PacketQueue *q)
{
	AVPacketList *pkt, *pkt1;

	SDL_LockMutex(q->mutex);
	for(pkt = q->first_pkt; pkt != NULL; pkt = pkt1) {
		pkt1 = pkt->next;
		// COMMENTED OUT AFTER YUM INSTALL HAS NO FUNC
		//av_free_packet(&pkt->pkt);
		av_freep(&pkt);
	}
	q->last_pkt = NULL;
	q->first_pkt = NULL;
	q->nb_packets = 0;
	q->size = 0;
	SDL_UnlockMutex(q->mutex);
}

int packet_queue_put(PacketQueue *q, AVPacket *pkt)
{
	AVPacketList *pkt1;

	if(pkt != &flush_pkt && av_dup_packet(pkt) < 0) {
		return -1;
	}
	return(0);
}
#endif /* FOOBAR */

int avi_seek_frame( QSP_ARG_DECL  Image_File *ifp, uint32_t n )
{
	int64_t seek_target, seek_result;
	//double seek_target_seconds, seek_point_seconds;
	int n_skip_frames;
	int i;

	seek_result = (-1);	// quiet compiler

//sprintf(error_string,"avi_seek %ld, current posn = %ld",
//n,ifp->if_nfrms);
//advise(error_string);
	if( n == ifp->if_nfrms ) return(0);	/* we're already there */
	else if( n == ifp->if_nfrms-1 ){	/* holding in place */
		/* like an unget */
		HDR_P->avch_have_frame_ready=1;
		ifp->if_nfrms--;
		return(0);
	}

	/* To seek, first we get the "skewed" index of the frame we want.
	 * This may be greater than the serial index, because of dropped frames,
	 * but is what we need to compare to the seek table.
	 */
	long skewed_index;

	skewed_index = skewed_frame_index(n,ifp);

	/* Now look this up in the seek table.
	 * We observe that seeking to 0 gets us 0, and seeking to
	 * anything in the range 1-500 takes us to 500.  So we search the
	 * table until we find something larger, then back up one...
	 */

	i=0;
	seek_target=(-1);
	while( seek_target < 0 && i < HDR_P->avch_n_seek_tbl_entries ){
		if( skewed_index < (long)HDR_P->avch_seek_tbl[i].seek_result ){
			/* Found the first seek that goes past our goal */
			i--;
#ifdef CAUTIOUS
			if( i < 0 ) NERROR1("CAUTIOUS:  bad avi seek table");
#endif /* CAUTIOUS */
			seek_target = HDR_P->avch_seek_tbl[i].seek_target;
			seek_result = HDR_P->avch_seek_tbl[i].seek_result;
		}
		i++;
	}
	if( seek_target < 0 ){	/* not found yet */
		i--;
#ifdef CAUTIOUS
		if( i < 0 ) NERROR1("CAUTIOUS:  bad avi seek table");
#endif /* CAUTIOUS */
		seek_target = HDR_P->avch_seek_tbl[i].seek_target;
		seek_result = HDR_P->avch_seek_tbl[i].seek_result;

		if( seek_result == (-1) ){
			/* the last table entry can be an illegal seek */
			i--;
#ifdef CAUTIOUS
			if( i < 0 ) NERROR1("CAUTIOUS:  bad avi seek table");
#endif /* CAUTIOUS */
			seek_target = HDR_P->avch_seek_tbl[i].seek_target;
			seek_result = HDR_P->avch_seek_tbl[i].seek_result;
		}
	}

	/* seek target is specified in frames by the user... */

	/* Here is what happens:  when we seek, we get back a PTS that is one-tenth of the
	 * requested seek value, and seems to indicate seconds...  So even though
	 * the nominal frame rate is 15 fps, the count seems to be interpreted as if it
	 * were 10 fps... and then we get rounded (up) to a multiple of 25 seconds!
	 *
	 * 0 -> 0
	 * 1-250 -> 16.6 -> really frame 16.6*15 = 250
	 * 251-500 -> 33.3
	 * 501-750 -> 50
	 * ...
	 *
	 * What should we do?  We can convert our frame request to seconds, then multiply by
	 * 10 to get a seek offset...
	 *
	 * But wait - on another input this does the wrong thing:  seeking to 750 gets
	 * us a pts of 50 (seconds), which happens to be 750 frames at 15 fps!?
	 * This is a different file than the one originally used to test the function...
	 * Are some of the files 10 fps recordings?
	 *
	 * changing the 10's to fps's didn't quite work, rather than going to a multiple
	 * of 25 seconds we seem to go to a multiple of 250 frames!
	 */

	if(av_seek_frame(HDR_P->avch_format_ctx_p,
		HDR_P->avch_video_stream_index, seek_target, 0 /* or AVSEEK_FLAG_BACKWARD */) < 0) {
		sprintf(error_string, "%s: error while seeking\n", ifp->if_name);
		WARN(error_string);
		return(-1);
	}

	ifp->if_nfrms = unskewed_frame_index(seek_result,ifp);

	/* Now skip some frames...  The seek has taken us to seek_result, but this is "skewed",
	 * we have to un-skew it to compare to our original request...
	 */

	n_skip_frames = n - ifp->if_nfrms;

#ifdef CAUTIOUS
	if( n_skip_frames < 0 ){
		sprintf(error_string,"n = %d, n_frms = %d",
			n,ifp->if_nfrms);
		advise(error_string);
		WARN("CAUTIOUS:  avi_seek:  n_skip_frames < 0 !?");
		n_skip_frames=0;
	}
#endif /* CAUTIOUS */

	while( n_skip_frames > 0 ){
		/* read a frame and throw it away */
		if( get_next_avi_frame(QSP_ARG  ifp) < 0 )
			return(-1);
		n_skip_frames --;
	}

	return(0);
} /* end avi_seek() */

/* We seem to kind of have seeking figured out, but we get messed up because the presentation
 * time stamps (pts's) don't always march along linearly...  Perhaps this is because a frame
 * was dropped?  Or the timestamps got messed up in the encoding process?
 *
 * Our approach is to seek to a point, and then see if the time stamp matches what we expect.
 */

long check_seek_offset(QSP_ARG_DECL  Image_File *ifp, int64_t pos)
{
	int64_t result;

	if(av_seek_frame(HDR_P->avch_format_ctx_p,
		HDR_P->avch_video_stream_index, pos, 0 /* or AVSEEK_FLAG_BACKWARD */) < 0) {
		sprintf(error_string, "check_for_dropped_frames %s: error while seeking\n", ifp->if_name);
		WARN(error_string);
		return(0);
	}
	if( get_next_avi_frame(QSP_ARG  ifp) < 0 )
		return(-1);

	result = round( HDR_P->avch_pts * HDR_P->avch_fps );
	result -= pos;
	return( (long) result );
}

/* Build a table mapping between frame numbers (serial in file) and time stamps.
 * These are stored as offsets.  Dropped frames (time stamp is later than expected)
 * are represented as positive offsets, and we only store the frames where the offset
 * changes.  This generally increases sporadically by one here and there, but we have
 * tested a file where at the end of the file the same timestamp is repeated over and over,
 * producing a steady decline in the offset...
 */

static Seek_Info seek_tbl[MAX_AVI_SEEK_TBL_SIZE];	/* BUG should be per-file */
static Frame_Skew skew_tbl[MAX_STORED_OFFSETS]; /* BUG should be per-file */

static void scan_file(QSP_ARG_DECL  Image_File *ifp)
{
	uint32_t i;
	int32_t offset, previous=0;
	int n_stored=0;
	uint32_t n_to_check;
	int32_t fps;

	offset = (-1);		// quiet compiler
	n_to_check = ifp->if_dp->dt_frames;	/* if there are no dropped frames, this is correct */

sprintf(error_string,"scan_file %s, checking %d frames",ifp->if_name,n_to_check);
advise(error_string);

#define MAX_VALID_FPS		100
#define DEFAULT_VALID_FPS	30

	fps = HDR_P->avch_fps;
	/* don't know where this comes from, this is a total hack to deal w/ jsc videos... */
	if( fps > MAX_VALID_FPS ){
		sprintf(error_string,
	"frames-per-second (%d) exceeds threshold (%d), resetting to default (%d)",
			HDR_P->avch_fps,MAX_VALID_FPS,DEFAULT_VALID_FPS);
		advise(error_string);
		fps = DEFAULT_VALID_FPS;
	}

	for(i=0;i<n_to_check;i++){
		double serial;
		if( get_next_avi_frame(QSP_ARG  ifp) < 0 ){
			sprintf(error_string,"scan_file:  Error reading frame %d",i);
			WARN(error_string);
			sprintf(error_string,"offset = %d    n_to_check = %d",
				offset,n_to_check);
			advise(error_string);
			n_to_check = i;		/* terminate loop */
		} else {
			serial=round( HDR_P->avch_pts * fps);

			offset = serial - i;
			if( offset != previous ){
				if( n_stored >= MAX_STORED_OFFSETS ){
					sprintf(error_string,"scan_file:  need to increase MAX_STORED_OFFSETS");
					WARN(error_string);
				} else {
					skew_tbl[n_stored].frame_index = i;
					skew_tbl[n_stored].pts_offset = offset;
					n_stored++;
				}
				/* If we have an offset, that means there are dropped frames,
				 * so our estimate of the length of the file is wrong...
				 */
				n_to_check -= (offset-previous);
			}
			previous = offset;
		}
	}

sprintf(msg_str,"%d offsets stored",n_stored);
prt_msg(msg_str);

	HDR_P->avch_n_skew_tbl_entries = n_stored;
	HDR_P->avch_skew_tbl = (Frame_Skew *)getbuf( n_stored * sizeof(Frame_Skew) );
	memcpy( HDR_P->avch_skew_tbl, skew_tbl, n_stored * sizeof(Frame_Skew) );

	if( n_to_check != ifp->if_dp->dt_frames ){
		sprintf(error_string,"Resetting number of frames from %d to %d",
			ifp->if_dp->dt_frames, n_to_check);
		advise(error_string);
		ifp->if_dp->dt_frames = n_to_check;
	}
} /* end scan_file() */

/* Given a frame serial number (e.g., the 1000th frame in the file), return
 * the index in the original sequence, corresponding to the time stamp (which may
 * be larger due to dropped frames).
 */

static long skewed_frame_index(long index, Image_File *ifp)
{
	/* The table records where the offset increases (assuming it
	 * starts at 0).  We scan the table until we find an entry which
	 * is larger than our desired frame.
	 */
	int i=0;
	int skew=0;

	while(i<HDR_P->avch_n_skew_tbl_entries){
		if( HDR_P->avch_skew_tbl[i].frame_index > index )
			return(index+skew);
		skew = HDR_P->avch_skew_tbl[i].pts_offset;
		i++;
	}
	return(index+skew);
}

/* Given a time frame index, return the serial number of the actual frame in the file.
 * If the request corresponds to a dropped frame, print a warning and return the previous.
 */

static long unskewed_frame_index(long index,Image_File *ifp)
{
	int i, offset;
	Frame_Skew * fs_p;

	i=0;
	offset = 0;
	while(i<HDR_P->avch_n_skew_tbl_entries){
		fs_p = &HDR_P->avch_skew_tbl[i];
		if( fs_p->frame_index+fs_p->pts_offset > index )
			return( index - offset );
		offset = fs_p->pts_offset;
		i++;
	}
	return( index - offset );
}

static void scan_seek(QSP_ARG_DECL  Image_File *ifp)
{
	uint32_t i;
	long previous = (-1);
	long result;
	int64_t seek_target;
	int n_stored=0;

	for(i=0;i<ifp->if_dp->dt_frames;i++){
		seek_target = i;
		if(av_seek_frame(HDR_P->avch_format_ctx_p,
			HDR_P->avch_video_stream_index, seek_target, 0 /* or AVSEEK_FLAG_BACKWARD */) < 0) {
			/*
			sprintf(error_string, "scan_seek %s: error while seeking to offset %d\n", ifp->if_name,i);
			WARN(error_string);
			return;
			*/
			/* Sometimes a seek error occurs for targets that seem
			 * less than the end of the file... (although it may be
			 * that they are less than 250 from the end?)
			 */
			result = -1;
		} else {
			get_next_avi_frame(QSP_ARG  ifp);
			result = round( HDR_P->avch_pts * HDR_P->avch_fps );
		}

		if( result >= 0 )
			ifp->if_nfrms = unskewed_frame_index(result,ifp);

		if( result != previous ){
			if( n_stored >= MAX_AVI_SEEK_TBL_SIZE ){
				sprintf(error_string,"scan_seek:  Need to increase MAX_AVI_SEEK_TBL_SIZE");
				WARN(error_string);
			} else {
				seek_tbl[n_stored].seek_target = i;
				seek_tbl[n_stored].seek_result = result;
				n_stored++;
			}
		}
		previous = result;
	}

	HDR_P->avch_n_seek_tbl_entries = n_stored;
	HDR_P->avch_seek_tbl = (Seek_Info *)getbuf( n_stored * sizeof(Seek_Info) );
	memcpy( HDR_P->avch_seek_tbl, seek_tbl, n_stored * sizeof(Seek_Info) );

	avi_seek_frame(QSP_ARG  ifp,0);
}


#endif /* HAVE_LIBAVCODEC */
