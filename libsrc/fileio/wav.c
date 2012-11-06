
#include "quip_config.h"

char VersionId_fio_wav[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "fio_prot.h"
#include "filetype.h"
#include "getbuf.h"
#include "data_obj.h"
#include "debug.h"
#include "savestr.h"
#include "wav.h"
#include "raw.h"
#include "readhdr.h"

//#define HDR_P(ifp)	((Image_File_Hdr *)ifp->if_hd)->ifh_u.wav_hd_p
#define HDR_P(ifp)	((Wav_Header *)&((Image_File_Hdr *)ifp->if_hd)->ifh_u.wav_hd)

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
	short prec;
	dimension_t total_samples, samples_per_channel;

	switch( hd_p->wh_bits_per_sample ){
		case 8:  prec=PREC_UBY; break;
		case 16: prec=PREC_IN; break;
		default:
			sprintf(DEFAULT_ERROR_STRING,
		"wav_to_dp:  unexpected # of bits per sample %d",
				hd_p->wh_bits_per_sample);
			NWARN(DEFAULT_ERROR_STRING);
			return(-1);
	}
	dp->dt_prec = prec;

	dp->dt_comps = hd_p->wh_n_channels;

	total_samples = hd_p->wh_datasize / siztbl[ prec ];
	samples_per_channel = total_samples / dp->dt_comps;

	dp->dt_cols = samples_per_channel;
	dp->dt_rows = 1;
	dp->dt_frames = 1;
	dp->dt_seqs = 1;

	dp->dt_cinc = 1;
	dp->dt_pinc = 1;
	dp->dt_rowinc = 1;
	dp->dt_finc = 1;
	dp->dt_sinc = 1;

	dp->dt_parent = NO_OBJ;
	dp->dt_children = NO_LIST;

	dp->dt_ap = ram_area;		/* the default */

	/* dp->dt_data = hd_p->image; */		/* where do we allocate data??? */

	dp->dt_n_type_elts = dp->dt_comps * dp->dt_cols * dp->dt_rows
			* dp->dt_frames * dp->dt_seqs;

	set_shape_flags(&dp->dt_shape,dp);

	return(0);
}

#define DEFAULT_SAMPLE_RATE	16000

static u_long default_samp_rate_func(void)
{
	return( DEFAULT_SAMPLE_RATE );
}

static u_long (*samp_rate_func)(VOID)=default_samp_rate_func;


void set_samp_rate_func( u_long (*func)(VOID) )
{
	samp_rate_func = func;
}

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

FIO_OPEN_FUNC( wav_open )
{
	Image_File *ifp;

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_WAV);
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	ifp->if_hd = getbuf( sizeof(Wav_Header) );

	if( IS_READABLE(ifp) ){
		/* BUG: should check for error here */

		/* An error can occur here if the file exists,
		 * but is empty...  we therefore initialize
		 * the header strings (above) with nulls, so
		 * that the program won't try to free nonsense
		 * addresses
		 */

		if( fread(ifp->if_hd,sizeof(Wav_Header),1,ifp->if_fp) != 1 ){
			sprintf(error_string,"Error reading wav header, file %s",
				ifp->if_name);
			NWARN(error_string);
			wav_close(QSP_ARG  ifp);
			return(NO_IMAGE_FILE);
		}
		if( ! valid_wav_header(ifp->if_hd) ){
			sprintf(error_string,"File %s does not appear to be a wav file",
				ifp->if_name);
			NWARN(error_string);
			wav_close(QSP_ARG  ifp);
			return(NO_IMAGE_FILE);
		}
		wav_to_dp(ifp->if_dp,ifp->if_hd);
	} else {	/* writable */
		init_wav_hdr(ifp->if_hd);		/* initialize the fixed fields */
	}
	return(ifp);
}


FIO_CLOSE_FUNC( wav_close )
{
	/* see if we need to edit the header */

	if( ifp->if_hd != NULL )
		givbuf(ifp->if_hd);

	GENERIC_IMGFILE_CLOSE(ifp);
}

int dp_to_wav(Wav_Header *hd_p,Data_Obj *dp)
{
	/* num_frame set when when write request given */

	/* BUG questionable cast */
	hd_p->wh_n_channels = dp->dt_comps;
	switch( MACHINE_PREC(dp) ){
		case PREC_UBY:
			hd_p->wh_bits_per_sample = 8;
			break;
		case PREC_IN:
			hd_p->wh_bits_per_sample = 16;
			break;
		default:
			sprintf(DEFAULT_ERROR_STRING,
		"dp_to_wav:  vector %s has unsupported source precision %s",
				dp->dt_name,prec_name[MACHINE_PREC(dp)]);
			NWARN(DEFAULT_ERROR_STRING);
			return(-1);
			break;
	}
			
	hd_p->wh_datasize = dp->dt_n_type_elts * siztbl[ MACHINE_PREC(dp) ];
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

FIO_SETHDR_FUNC( set_wav_hdr )
{
	if( dp_to_wav(ifp->if_hd,ifp->if_dp) < 0 ){
		wav_close(QSP_ARG  ifp);
		return(-1);
	}
	/* BUG we need to write the numbers out byte by byte
	 * to be independent of endian-ness
	 */
	if( fwrite(ifp->if_hd,sizeof(Wav_Header),1,ifp->if_fp) != 1 ){
		sprintf(error_string,"error writing wav header file %s",ifp->if_name);
		WARN(error_string);
		return(-1);
	}
	return(0);
}

FIO_WT_FUNC( wav_wt )
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

FIO_RD_FUNC( wav_rd )
{
	raw_rd(QSP_ARG  dp,ifp,x_offset,y_offset,t_offset);
}

int wav_unconv(void *hdr_pp,Data_Obj *dp)
{
	Wav_Header **hd_pp;

	hd_pp = (Wav_Header **) hdr_pp;

	/* allocate space for new header */

	*hd_pp = (Wav_Header *)getbuf( sizeof(Wav_Header) );
	if( *hd_pp == NULL ) return(-1);

	dp_to_wav(*hd_pp,dp);

	return(0);
}

int wav_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("wav_conv not implemented");
	return(-1);
}

void wav_info(QSP_ARG_DECL  Image_File *ifp)
{
	sprintf(msg_str,"\tsample rate %ld",HDR_P(ifp)->wh_samp_rate);
	prt_msg(msg_str);
}

