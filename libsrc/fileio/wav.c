
#include "quip_config.h"

#include <stdio.h>
#include <ctype.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "fio_prot.h"
#include "img_file/wav.h"
#include "img_file/raw.h"
//#include "img_file/readhdr.h"

#define HDR_P(ifp)	((Wav_Header *)ifp->if_hdr_p)

#define wav_error(msg, ifp) _wav_error(QSP_ARG  msg, ifp)

static void _wav_error(QSP_ARG_DECL  const char *msg, Image_File *ifp)
{
	sprintf(ERROR_STRING,"File %s:  %s!?",msg,ifp->if_name);
	warn(ERROR_STRING);
}

#define wav_fatal_error(msg, ifp) _wav_fatal_error(QSP_ARG  msg, ifp)

static void _wav_fatal_error(QSP_ARG_DECL  const char *msg, Image_File *ifp)
{
	wav_error(msg,ifp);
	wav_close(QSP_ARG  ifp);
}

#define valid_riff_hdr(hd_p) _valid_riff_hdr(QSP_ARG  hd_p)

static int _valid_riff_hdr(QSP_ARG_DECL  Wav_Header *hd_p)
{
	if( strncmp(hd_p->wh_riff_label,"RIFF",4) ){
		warn("bad riff label!?");
		return 0;
	}
	if( strncmp(hd_p->wh_wave_label,"WAVE",4) ){
		warn("bad wave lavel!?");
		return 0;
	}
	return 1;
}

//int _wav_to_dp(QSP_ARG_DECL  Data_Obj *dp,Wav_Header *hd_p)
FIO_FT_TO_DP_FUNC(wav,Wav_Header)
{
	Precision * prec_p;
	dimension_t total_samples, samples_per_channel;

	switch( hd_p->wh_bits_per_sample ){
		case 8:  prec_p=PREC_FOR_CODE(PREC_UBY); break;
		case 16: prec_p=PREC_FOR_CODE(PREC_IN); break;
		default:
			sprintf(ERROR_STRING,
		"wav_to_dp:  unexpected # of bits per sample %d",
				hd_p->wh_bits_per_sample);
			warn(ERROR_STRING);
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

	SET_OBJ_PARENT(dp, NULL);
	SET_OBJ_CHILDREN(dp, NULL);

	SET_OBJ_AREA(dp, ram_area_p);		/* the default */

	/* dp->dt_data = hd_p->image; */		/* where do we allocate data??? */

	SET_OBJ_N_TYPE_ELTS(dp, OBJ_COMPS(dp) * OBJ_COLS(dp) * OBJ_ROWS(dp)
			* OBJ_FRAMES(dp) * OBJ_SEQS(dp) );

	auto_shape_flags(OBJ_SHAPE(dp));

	return 0;
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

	hd_p->wh_fmt_size = 16;
	hd_p->wh_fmt_tag = 1;	// compression code - 1 == no compression?
}

#define read_wav_header_chunk(ifp) _read_wav_header_chunk(QSP_ARG  ifp)

static int _read_wav_header_chunk(QSP_ARG_DECL  Image_File *ifp)
{
	if( fread(&(HDR_P(ifp)->wh_whc),sizeof(Wav_Hdr_Chunk),1,ifp->if_fp) != 1 ){
		wav_fatal_error("Error reading WAV header chunk",ifp);
		return -1;
	}
	if( ! valid_riff_hdr(HDR_P(ifp)) ){
		wav_fatal_error("Not a WAV file",ifp);
		return -1;
	}
	return 0;
}

static int is_printable(const char *s)
{
	while( *s ){
		if( ! isprint(*s) ) return 0;
		s++;
	}
	return 1;
}

#define read_format_chunk(ifp, wch_p) _read_format_chunk(QSP_ARG  ifp, wch_p)

static int _read_format_chunk(QSP_ARG_DECL  Image_File *ifp, Wav_Chunk_Hdr *wch_p)
{
	int n_mandatory, n_extra;
	char *b;
	int status = 0;

	HDR_P(ifp)->wh_fhc.fhc_wch = *wch_p;	// copy chunk header

	n_mandatory = sizeof(Wav_Fmt_Data);
fprintf(stderr,"format chunk has %d mandatory bytes (should be 16?)\n",
n_mandatory);
	if( fread(&(HDR_P(ifp)->wh_fhc.fhc_wfd),sizeof(Wav_Fmt_Data),1,ifp->if_fp) != 1 ){
		wav_fatal_error("Error reading format data",ifp);
		return -1;
	}
	n_extra = wch_p->wch_size - n_mandatory;
	if( n_extra == 0 ) return 0;

	if( n_extra < 0 ){
		wav_error("Bad WAV format chunk size",ifp);
		return -1;
	}

fprintf(stderr,"format chunk has %d extra bytes???\n",n_extra);
	b = getbuf(n_extra);
	if( fread(b,1,n_extra,ifp->if_fp) != n_extra ){
		wav_error("Error reading extra format data",ifp);
		status = -1;
	}
	givbuf(b);	// just throw away for now...
	return status;
}

#define read_data_chunk(ifp, wch_p) _read_data_chunk(QSP_ARG  ifp, wch_p)

static int _read_data_chunk(QSP_ARG_DECL  Image_File *ifp, Wav_Chunk_Hdr *wch_p)
{
	// we don't read the data, we just copy the chunk header
	HDR_P(ifp)->wh_dhc.dhc_wch = *wch_p;	// copy chunk header
	return 0;
}

#define read_next_chunk_header(ifp) _read_next_chunk_header(QSP_ARG  ifp)

static Wav_Chunk_Hdr * _read_next_chunk_header(QSP_ARG_DECL  Image_File *ifp)
{
	static Wav_Chunk_Hdr wch1;

	if( fread(&wch1,sizeof(Wav_Chunk_Hdr),1,ifp->if_fp) != 1 ){
		wav_fatal_error("Error reading chunk header",ifp);
		return NULL;
	}
	return &wch1;  // BUG static object not thread-safe!
}

#define ignore_chunk(ifp, wch_p) _ignore_chunk(QSP_ARG  ifp, wch_p)

static int _ignore_chunk(QSP_ARG_DECL  Image_File *ifp, Wav_Chunk_Hdr *wch_p)
{
	int nb;
	char *buf;
	int status=0;

	nb = wch_p->wch_size;
fprintf(stderr,"ignore_chunk:  size is %d\n",nb);
	buf = getbuf(nb);
	if( fread(buf,1,nb,ifp->if_fp) != nb ){
		wav_fatal_error("Error reading ignored chunk data",ifp);
		status = -1;
	}
	givbuf(buf);
	return status;
}

#define process_chunk(ifp, wch_p) _process_chunk(QSP_ARG  ifp, wch_p)

static int _process_chunk(QSP_ARG_DECL  Image_File *ifp, Wav_Chunk_Hdr *wch_p)
{
	// See what kind of chunk it is...
	if( !strncmp(wch_p->wch_label,"fmt ",4) ){
fprintf(stderr,"format chunk seen!\n");
		return read_format_chunk(ifp,wch_p);
fprintf(stderr,"back from read_format_chunk\n");
	} else if( !strncmp(wch_p->wch_label,"data",4) ){
fprintf(stderr,"data chunk seen!\n");
		if( read_data_chunk(ifp,wch_p) != 0 )
			return -1;
		return 1;	// special return val for data chunk
	} else if( !strncmp(wch_p->wch_label,"LIST",4) ){
fprintf(stderr,"list chunk seen!\n");
		return ignore_chunk(ifp,wch_p);
	} else {
		char s[5];
		strncpy(s,wch_p->wch_label,4);
		s[4]=0;

		if( is_printable(s) ){
			sprintf(ERROR_STRING,
	"read_next_chunk_header (%s):  unrecognized chunk label \"%s\"!?\n",
				ifp->if_name,s);
			warn(ERROR_STRING);
		} else {
			wav_error("unprintable chunk label",ifp);
		}
		// should read the chunk data!?
		return 0;
	}
}

#define read_wav_header(ifp) _read_wav_header(QSP_ARG  ifp)

static int _read_wav_header(QSP_ARG_DECL  Image_File *ifp)
{
	Wav_Chunk_Hdr *ch_p;

	if( read_wav_header_chunk(ifp) < 0 )
		return -1;
fprintf(stderr,"read_wav_header:  size is %d (0x%x)\n",
HDR_P(ifp)->wh_whc.whc_wch.wch_size,
HDR_P(ifp)->wh_whc.whc_wch.wch_size);

	while( (ch_p=read_next_chunk_header(ifp)) != NULL ){
		int status;
		status = process_chunk(ifp,ch_p);
		if( status < 0 ) return -1;	// some error
		if( status == 1 ) return 0;	// data chunk
		// if status is 0, read next chunk...
	}
	return -1;	// error return, null chunk
}
		

FIO_OPEN_FUNC( wav )
{
	Image_File *ifp;

	ifp = img_file_creat(name,rw,FILETYPE_FOR_CODE(IFT_WAV));
	if( ifp==NULL ) return(ifp);

	ifp->if_hdr_p = getbuf( sizeof(Wav_Header) );

	if( IS_READABLE(ifp) ){
		/* BUG: should check for error here */

		/* An error can occur here if the file exists,
		 * but is empty...  we therefore initialize
		 * the header strings (above) with nulls, so
		 * that the program won't try to free nonsense
		 * addresses
		 */

		if( read_wav_header(ifp) < 0 )
			return NULL;
		wav_to_dp(ifp->if_dp,HDR_P(ifp));
	} else {	/* writable */
		init_wav_hdr(HDR_P(ifp));		/* initialize the fixed fields */
	}
	return(ifp);
}


FIO_CLOSE_FUNC( wav )
{
	/* see if we need to edit the header */

	if( ifp->if_hdr_p != NULL )
		givbuf(ifp->if_hdr_p);

	generic_imgfile_close(ifp);
}

FIO_DP_TO_FT_FUNC(wav,Wav_Header)
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
			sprintf(ERROR_STRING,
		"dp_to_wav:  vector %s has unsupported source precision %s",
				OBJ_NAME(dp),PREC_NAME(OBJ_MACH_PREC_PTR(dp)));
			warn(ERROR_STRING);
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

	return 0;
}

FIO_SETHDR_FUNC( wav )
{
	if( FIO_DP_TO_FT_FUNC_NAME(wav)(QSP_ARG  HDR_P(ifp),ifp->if_dp) < 0 ){
		wav_close(QSP_ARG  ifp);
		return(-1);
	}
	/* BUG we need to write the numbers out byte by byte
	 * to be independent of endian-ness
	 */
	if( fwrite(ifp->if_hdr_p,sizeof(Wav_Header),1,ifp->if_fp) != 1 ){
		sprintf(ERROR_STRING,"error writing wav header file %s",ifp->if_name);
		warn(ERROR_STRING);
		return(-1);
	}
	return 0;
}

FIO_WT_FUNC( wav )
{
	/* We shouldn't need to do this, but it is easier to put these
	 * lines in than to modify wt_raw_data()...
	 */
	if( ifp->if_dp == NULL ){	/* first time? */
		/* set the rows & columns in our file struct */
		setup_dummy(ifp);
		copy_dimensions(ifp->if_dp, dp);
		if( set_wav_hdr(ifp) < 0 ) return(-1);
	}

	wt_raw_data(dp,ifp);
	return 0;
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

	FIO_DP_TO_FT_FUNC_NAME(wav)(QSP_ARG  *hdr_pp,dp);

	return 0;
}

FIO_CONV_FUNC(wav)
{
	warn("wav_conv not implemented");
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
	warn(ERROR_STRING);

	return -1;
}

