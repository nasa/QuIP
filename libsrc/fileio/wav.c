
#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "fio_prot.h"
#include "img_file/wav.h"
#include "img_file/raw.h"
//#include "img_file/readhdr.h"

#define HDR_P(ifp)	((Wav_Header *)ifp->if_hdr_p)

static int valid_wav_header(Wav_Header *hd_p)
{
	if( strncmp(hd_p->wh_riff_label,"RIFF",4) ) return(0);
	if( strncmp(hd_p->wh_wave_label,"WAVE",4) ) return(0);
	if( strncmp(hd_p->wh_fmt_label,"fmt ",4) ) return(0);
	if( strncmp(hd_p->wh_data_label,"data",4) ) return(0);
	return(1);
}

int wav_to_dp(Data_Obj *dp,Wav_Header *hd_p)
{
	Precision * prec_p;
	dimension_t total_samples, samples_per_channel;

	switch( hd_p->wh_bits_per_sample ){
		case 8:  prec_p=PREC_FOR_CODE(PREC_UBY); break;
		case 16: prec_p=PREC_FOR_CODE(PREC_IN); break;
		default:
			sprintf(DEFAULT_ERROR_STRING,
		"wav_to_dp:  unexpected # of bits per sample %d",
				hd_p->wh_bits_per_sample);
			NWARN(DEFAULT_ERROR_STRING);
			return(-1);
	}
	SET_OBJ_PREC_PTR(dp, prec_p);

	SET_OBJ_COMPS(dp, hd_p->wh_n_channels );

	total_samples = (dimension_t) (hd_p->wh_datasize / PREC_SIZE( prec_p ));
	samples_per_channel = total_samples / OBJ_COMPS(dp);

	SET_OBJ_COLS(dp, samples_per_channel);
	SET_OBJ_ROWS(dp, 1);
	SET_OBJ_FRAMES(dp, 1);
	SET_OBJ_SEQS(dp, 1);

	SET_OBJ_COMP_INC(dp, 1);
	SET_OBJ_PXL_INC(dp, 1);
	SET_OBJ_ROW_INC(dp, 1);
	SET_OBJ_FRM_INC(dp, 1);
	SET_OBJ_SEQ_INC(dp, 1);

	SET_OBJ_PARENT(dp, NO_OBJ);
	SET_OBJ_CHILDREN(dp, NULL);

	SET_OBJ_AREA(dp, ram_area_p);		/* the default */

	/* dp->dt_data = hd_p->image; */		/* where do we allocate data??? */

	SET_OBJ_N_TYPE_ELTS(dp, OBJ_COMPS(dp) * OBJ_COLS(dp) * OBJ_ROWS(dp)
			* OBJ_FRAMES(dp) * OBJ_SEQS(dp) );

	auto_shape_flags(OBJ_SHAPE(dp),dp);

	return(0);
}

#define DEFAULT_SAMPLE_RATE	16000

static uint32_t default_samp_rate_func(void)
{
	return( DEFAULT_SAMPLE_RATE );
}

static uint32_t (*samp_rate_func)(void)=default_samp_rate_func;

#ifdef NOT_USED

void set_samp_rate_func( uint32_t (*func)(void) )
{
	samp_rate_func = func;
}

#endif /* NOT_USED */

/* initialize the fixed header fields */

static void init_wav_hdr(Wav_Header *hd_p)
{
	strncpy(hd_p->wh_riff_label,"RIFF",4);
	strncpy(hd_p->wh_wave_label,"WAVE",4);
	strncpy(hd_p->wh_fmt_label,"fmt ",4);
	strncpy(hd_p->wh_data_label,"data",4);

	hd_p->wh_always_16 = 16;
	hd_p->wh_fmt_tag = 1;
}

FIO_OPEN_FUNC( wav )
{
	Image_File *ifp;

	ifp = IMG_FILE_CREAT(name,rw,FILETYPE_FOR_CODE(IFT_WAV));
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	ifp->if_hdr_p = getbuf( sizeof(Wav_Header) );

	if( IS_READABLE(ifp) ){
		/* BUG: should check for error here */

		/* An error can occur here if the file exists,
		 * but is empty...  we therefore initialize
		 * the header strings (above) with nulls, so
		 * that the program won't try to free nonsense
		 * addresses
		 */

		if( fread(ifp->if_hdr_p,sizeof(Wav_Header),1,ifp->if_fp) != 1 ){
			sprintf(ERROR_STRING,"Error reading wav header, file %s",
				ifp->if_name);
			WARN(ERROR_STRING);
			wav_close(QSP_ARG  ifp);
			return(NO_IMAGE_FILE);
		}
		if( ! valid_wav_header(ifp->if_hdr_p) ){
			sprintf(ERROR_STRING,"File %s does not appear to be a wav file",
				ifp->if_name);
			WARN(ERROR_STRING);
			wav_close(QSP_ARG  ifp);
			return(NO_IMAGE_FILE);
		}
		wav_to_dp(ifp->if_dp,ifp->if_hdr_p);
	} else {	/* writable */
		init_wav_hdr(ifp->if_hdr_p);		/* initialize the fixed fields */
	}
	return(ifp);
}


FIO_CLOSE_FUNC( wav )
{
	/* see if we need to edit the header */

	if( ifp->if_hdr_p != NULL )
		givbuf(ifp->if_hdr_p);

	GENERIC_IMGFILE_CLOSE(ifp);
}

FIO_DP_TO_FT_FUNC(wav,Wav_Header)
//int dp_to_wav(Wav_Header *hd_p,Data_Obj *dp)
{
	/* num_frame set when when write request given */

	/* BUG questionable cast */
	hd_p->wh_n_channels = (short) OBJ_COMPS(dp);
	switch( OBJ_MACH_PREC(dp) ){
		case PREC_UBY:
			hd_p->wh_bits_per_sample = 8;
			break;
		case PREC_IN:
			hd_p->wh_bits_per_sample = 16;
			break;
		default:
			sprintf(DEFAULT_ERROR_STRING,
		"dp_to_wav:  vector %s has unsupported source precision %s",
				OBJ_NAME(dp),PREC_NAME(OBJ_MACH_PREC_PTR(dp)));
			NWARN(DEFAULT_ERROR_STRING);
			return(-1);
			break;
	}
			
	hd_p->wh_datasize = OBJ_N_TYPE_ELTS(dp) * PREC_SIZE( OBJ_MACH_PREC_PTR(dp) );
	hd_p->wh_chunksize = 36 + hd_p->wh_datasize;

	/* BUG dp's don't have a way to carry around the sample rate with them??? */
	/* So we assume that the current sample rate is the one that corresponds
	 * to this object BUG
	 */
	hd_p->wh_samp_rate = (*samp_rate_func)();
	hd_p->wh_blk_align = hd_p->wh_n_channels * hd_p->wh_bits_per_sample / 8;
	hd_p->wh_bytes_per_sec = hd_p->wh_blk_align * hd_p->wh_samp_rate;

	return(0);
}

FIO_SETHDR_FUNC( wav )
{
	if( FIO_DP_TO_FT_FUNC_NAME(wav)(ifp->if_hdr_p,ifp->if_dp) < 0 ){
		wav_close(QSP_ARG  ifp);
		return(-1);
	}
	/* BUG we need to write the numbers out byte by byte
	 * to be independent of endian-ness
	 */
	if( fwrite(ifp->if_hdr_p,sizeof(Wav_Header),1,ifp->if_fp) != 1 ){
		sprintf(ERROR_STRING,"error writing wav header file %s",ifp->if_name);
		WARN(ERROR_STRING);
		return(-1);
	}
	return(0);
}

FIO_WT_FUNC( wav )
{
	/* We shouldn't need to do this, but it is easier to put these
	 * lines in than to modify wt_raw_data()...
	 */
	if( ifp->if_dp == NO_OBJ ){	/* first time? */
		/* set the rows & columns in our file struct */
		setup_dummy(ifp);
		copy_dimensions(ifp->if_dp, dp);
		if( set_wav_hdr(QSP_ARG  ifp) < 0 ) return(-1);
	}

	wt_raw_data(QSP_ARG  dp,ifp);
	return(0);
}

FIO_RD_FUNC( wav )
{
	raw_rd(QSP_ARG  dp,ifp,x_offset,y_offset,t_offset);
}

FIO_UNCONV_FUNC(wav)
{
	Wav_Header **hdr_pp;

	hdr_pp = (Wav_Header **) hd_pp;

	/* allocate space for new header */

	*hdr_pp = (Wav_Header *)getbuf( sizeof(Wav_Header) );
	if( *hdr_pp == NULL ) return(-1);

	FIO_DP_TO_FT_FUNC_NAME(wav)(*hdr_pp,dp);

	return(0);
}

FIO_CONV_FUNC(wav)
{
	NWARN("wav_conv not implemented");
	return(-1);
}

FIO_INFO_FUNC(wav)
{
	sprintf(msg_str,"\tsample rate %d",HDR_P(ifp)->wh_samp_rate);
	prt_msg(msg_str);
}

FIO_SEEK_FUNC( wav )
{
	sprintf(ERROR_STRING,"wav_seek_frame not implemented!?");
	WARN(ERROR_STRING);

	return -1;
}

