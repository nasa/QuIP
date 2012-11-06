
#include "quip_config.h"

char VersionId_fio_bdf[] = QUIP_VERSION_STRING;

#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>		/* atoi */
#endif

#include "fio_prot.h"		/* define HAVE_TIFF if os has it... */
#include "filetype.h"
#include "getbuf.h"
#include "data_obj.h"
#include "debug.h"
#include "savestr.h"
#include "bdf_hdr.h"
#include "bdf.h"

//#define HDR_P	((Image_File_Hdr *)ifp->if_hd)->ifh_u.bdf_info_p
#define HDR_P	((BDF_info *)&(((Image_File_Hdr *)ifp->if_hd)->ifh_u.bdf_info))

/* static int num_color=1; */
/* static void rewrite_bdf_nf(TIFF* bdf,dimension_t n); */

void print_channel_info(BDF_info *bdfp,long index)
{
	if( index < 0 || index >= bdfp->bdf_n_channels ){
		sprintf(DEFAULT_ERROR_STRING,"print_channel_info:  channel index %ld out of range (%ld channels)",
			index,bdfp->bdf_n_channels);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

#ifdef CAUTIOUS
	if( bdfp->bdf_pci_tbl == NULL ){
		NWARN("CAUTIOUS:  print_channel_info:  channel_tbl is NULL!?");
		return;
	}
#endif /* CAUTIOUS */

	sprintf(msg_str,"Channel %s:",bdfp->bdf_pci_tbl[index].pci_channel_name);			prt_msg(msg_str);
	sprintf(msg_str,"\t%s",bdfp->bdf_pci_tbl[index].pci_channel_desc);				prt_msg(msg_str);
	sprintf(msg_str,"\tunit: %s",bdfp->bdf_pci_tbl[index].pci_channel_unit);			prt_msg(msg_str);
	sprintf(msg_str,"\tunit range: %ld - %ld",bdfp->bdf_pci_tbl[index].pci_min_val,
						bdfp->bdf_pci_tbl[index].pci_max_val);			prt_msg(msg_str);
	sprintf(msg_str,"\tadc range: %ld - %ld",bdfp->bdf_pci_tbl[index].pci_min_val,
						bdfp->bdf_pci_tbl[index].pci_max_val);			prt_msg(msg_str);
	sprintf(msg_str,"\tfiltering:  %s",bdfp->bdf_pci_tbl[index].pci_filter_desc);			prt_msg(msg_str);
	sprintf(msg_str,"\tsampling rate: %ld",bdfp->bdf_pci_tbl[index].pci_sample_rate);		prt_msg(msg_str);
	sprintf(msg_str,"\tsignal type:  %s",bdfp->bdf_pci_tbl[index].pci_signal_type);			prt_msg(msg_str);
}

void bdf_info(QSP_ARG_DECL  Image_File *ifp)
{
	sprintf(msg_str,"File %s:  %d segments, %d channels",ifp->if_name,
		ifp->if_dp->dt_frames,ifp->if_dp->dt_rows);
	prt_msg(msg_str);
}

int convert_fixed_string(char *ptr, int size )
{
	char buf[LLEN];

	if( size > (LLEN-1) ){
		sprintf(DEFAULT_ERROR_STRING,"convert_fixed_string:  can't honor request for buf len %d, max is %d",
			size,LLEN-1);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

	memcpy(buf,ptr,size);
	buf[size]=0;	/* null terminate */
	return( atoi(buf) );
}

#define BDF_MAGIC	"\377BIOSEMI"
#define FORMAT_24BIT	"24BIT"

int bdf_to_dp(Data_Obj *dp,BDF_info *bdfp)
{
	int size,n_records,n_chan,duration;

	/* Verify magic number */
	if( strcmp(bdfp->bdf_magic,BDF_MAGIC) ){
		NWARN("Bad magic number in BDF file");
		return(-1);
	}

	if( strcmp(bdfp->bdf_formatVersion,FORMAT_24BIT) ){
		NWARN("Sorry, only know how to deal with 24 bit BDF format, this is something else");
		return(-1);
	}

	/* these strings don't seem to be null-terminated... */
	/* copy the header length string into our own buffer */
/*
//sprintf(DEFAULT_ERROR_STRING,"header size string is \"%s\"",buf);
//advise(DEFAULT_ERROR_STRING);
*/
advise(DEFAULT_ERROR_STRING);

	size = atoi(bdfp->bdf_headerSize);
	n_records = atoi(bdfp->bdf_numDataRecord);
	duration = atoi(bdfp->bdf_dataDuration);
	n_chan = atoi(bdfp->bdf_numChan);

	/* duration is 1 - and Stan & company said that the data is blocked into 1-second
	 * "frames"...  n_records is a couple hundred, and I am guessing that that is the number of
	 * frames.  And then there are n_chan signals in a frame.
	 *
	 * The sampling rate is supposedly 512 Hz, is this given in the header?
	 */

	/* Ultimately we will convert these to long, but for now we will read the 24 bit data
	 * in as bytes...
	 */

	/* set dp->dt_prec */
	dp->dt_prec = PREC_BY;
	dp->dt_comps = 3;		/* the three bytes that make up a 24 bit word */
	/*
	//dp->dt_cols = n_chan;
	//dp->dt_rows = duration * 512; */	/* BUG get sampling rate from header */
	dp->dt_cols = duration * 512;		/* BUG get sampling rate from header */
	dp->dt_rows = n_chan;
	dp->dt_frames = n_records;
	dp->dt_seqs = 1;


	/* set dp->dt_cols, rows, etc */

	dp->dt_n_type_elts = dp->dt_comps * dp->dt_cols * dp->dt_rows
			* dp->dt_frames * dp->dt_seqs;

	set_shape_flags(&dp->dt_shape,dp);

	return(size-sizeof(BDF_info_text));
}

#define SET_OFFSET( field, channel_index )					\
										\
	{									\
	Per_Channel_Text pct;							\
	offset = n_channels * (((u_long)&pct.field) - ((u_long)&pct.pct_channel_name));		\
	offset += channel_index * sizeof(pct.field);				\
	}

#define COPY_STRING( field, len, channel_index )				\
										\
	{									\
	int offset, j;								\
	SET_OFFSET( field, channel_index )					\
	strncpy( str, &text_buf[offset], len );					\
	j=len-1;								\
	while(j>=0 && str[j]==' ') j--;						\
	str[j+1]=0;								\
	}



static void convert_per_channel_info(Per_Channel_Info *pci_p,char *text_buf,long n_channels,int channel_index)
{
	char str[81];

	/* For each field, copy the string and convert if necessary */
	COPY_STRING( pct_channel_name, 16, channel_index )
	pci_p->pci_channel_name = savestr(str);

	COPY_STRING( pct_channel_desc, 80, channel_index )
	pci_p->pci_channel_desc = savestr(str);

	COPY_STRING( pct_channel_unit, 8, channel_index )
	pci_p->pci_channel_unit = savestr(str);

	COPY_STRING( pct_min_val, 8, channel_index )
	pci_p->pci_min_val = atoi(str);

	COPY_STRING( pct_max_val, 8, channel_index )
	pci_p->pci_max_val = atoi(str);

	COPY_STRING( pct_adc_min, 8, channel_index )
	pci_p->pci_adc_min = atoi(str);

	COPY_STRING( pct_adc_max, 8, channel_index )
	pci_p->pci_adc_max = atoi(str);

	COPY_STRING( pct_filter_desc, 80, channel_index )
	pci_p->pci_filter_desc = savestr(str);

	COPY_STRING( pct_sample_rate, 8, channel_index )
	pci_p->pci_sample_rate = atoi(str);

	COPY_STRING( pct_signal_type, 32, channel_index )
	pci_p->pci_signal_type = savestr(str);
}

#define CONVERT_BDF_STRING( dest, src, len )						\
	strncpy(buf,src,len);								\
	i=len-1;									\
	while(i>=0&&buf[i]==' ')i--;							\
	buf[i+1]=0;									\
	dest = savestr(buf);


static void convert_bdf_strings(BDF_info *bdf_info_p, BDF_info_text *bdft_p)
{
	char buf[81];		/* one more than the longest possible string */
	int i;

	CONVERT_BDF_STRING(bdf_info_p->bdf_magic,bdft_p->bdft_magic,8)
	CONVERT_BDF_STRING(bdf_info_p->bdf_subject,bdft_p->bdft_subject,80)
	CONVERT_BDF_STRING(bdf_info_p->bdf_recording,bdft_p->bdft_recording,80)
	CONVERT_BDF_STRING(bdf_info_p->bdf_startdate,bdft_p->bdft_startdate,8)
	CONVERT_BDF_STRING(bdf_info_p->bdf_starttime,bdft_p->bdft_starttime,8)
	CONVERT_BDF_STRING(bdf_info_p->bdf_headerSize,bdft_p->bdft_headerSize,8)
	CONVERT_BDF_STRING(bdf_info_p->bdf_formatVersion,bdft_p->bdft_formatVersion,44)
	CONVERT_BDF_STRING(bdf_info_p->bdf_numDataRecord,bdft_p->bdft_numDataRecord,8)
	CONVERT_BDF_STRING(bdf_info_p->bdf_dataDuration,bdft_p->bdft_dataDuration,8)
	CONVERT_BDF_STRING(bdf_info_p->bdf_numChan,bdft_p->bdft_numChan,4)
}

FIO_OPEN_FUNC( bdf_open )
{
	Image_File *ifp;

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_BDF);
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	ifp->if_hd = getbuf( sizeof(BDF_info) );
#ifdef CAUTIOUS
	if( ifp->if_hd == NULL )
		ERROR1("CAUTIOUS:  error allocating BDF info struct");
#endif /* CAUTIOUS */

	if( IS_READABLE(ifp) ){
		BDF_info_text info;
		int size;

		if( fread(&info,sizeof(info),1,ifp->if_fp) != 1 ){
			sprintf(ERROR_STRING,"error reading header from bdf file %s",ifp->if_pathname);
			WARN(ERROR_STRING);
			DEL_IMG_FILE(name);
			return(NO_IMAGE_FILE);
		}
		convert_bdf_strings(ifp->if_hd,&info);
		size = bdf_to_dp(ifp->if_dp,ifp->if_hd);
		if( size < 0 ){
			WARN("bdf_open:  problem parsing BDF header");
			DEL_IMG_FILE(name);
			return(NO_IMAGE_FILE);
		}
		if( size > 0 ){
			char *per_channel_info_buf;
			int i;

			/* now read the "header", and throw it away */
			per_channel_info_buf = (char *)getbuf(size);
#ifdef CAUTIOUS
			if( per_channel_info_buf == NULL )
				ERROR1("CAUTIOUS:  bdf_open:  unable to allocate buffer for per-channel info text");
#endif	/* CAUTIOUS */
			if( fread(per_channel_info_buf,size,1,ifp->if_fp) != 1 ){
				sprintf(error_string,"bdf_open %s:  error reading %d bytes of per-channel info",ifp->if_name,size);
				WARN(error_string);
				DEL_IMG_FILE(name);
				return(NO_IMAGE_FILE);
			}

			/* now parse the per-channel info */
			HDR_P->bdf_n_channels = ifp->if_dp->dt_rows;
			HDR_P->bdf_pci_tbl = (Per_Channel_Info *)getbuf(HDR_P->bdf_n_channels * sizeof(Per_Channel_Info) );
#ifdef CAUTIOUS
			if( HDR_P->bdf_pci_tbl == NULL )
				ERROR1("CAUTIOUS:  bdf_open:  unable to allocate buffer for per-channel info table");
#endif	/* CAUTIOUS */
			for(i=0;i<HDR_P->bdf_n_channels;i++)
				convert_per_channel_info(&HDR_P->bdf_pci_tbl[i],per_channel_info_buf,HDR_P->bdf_n_channels,i);

			givbuf(per_channel_info_buf);

#ifdef FOOBAR
			/* now see where we are - this is just a sanity check */
			n = ftell(ifp->if_fp);
			sprintf(error_string,"After reading header data, file offset is %d (0x%x 0%o)",
				n,n,n);
			advise(error_string);
#endif /* FOOBAR */
		}
	}

#ifdef FOO
	else {
		/* can we write?? */
	}
#endif
	return(ifp);
} /* end bdf_open() */


FIO_CLOSE_FUNC( bdf_close )
{
	/* can we write multiple frames to bdf??? */

	/* TIFFClose(ifp->if_bdf); */
	GENERIC_IMGFILE_CLOSE(ifp);
}

#ifdef FOOBAR
int dp_to_bdf(BDF_info *bdfp,Data_Obj *dp)
{
#ifdef FOOBAR
	dimension_t size;
#endif /* FOOBAR */

	uint32 w,h,d;
	uint16 pc=1;
	uint16 ph=PHOTOMETRIC_MINISBLACK;
	uint16 dtype;
	uint16 bps;
	/* COMPRESSION_LZW not available due to Unisys patent enforcement */
	/* uint16 comp=COMPRESSION_LZW; */


	w = dp->dt_cols;
	h = dp->dt_rows;
	d = dp->dt_comps;
sprintf(error_string,"dp_to_bdf:  dimensions are %ld x %ld x %ld",h,w,d);
advise(error_string);

	/* num_frame set when when write request given */

	switch( MACHINE_PREC(dp) ){
		case PREC_UBY: bps=8; goto set_uint;
		case PREC_UIN: bps =16; goto set_uint;
		case PREC_UDI: bps=32;
set_uint:
			dtype = SAMPLEFORMAT_UINT;
			break;
		case PREC_BY: bps=8; goto set_int;
		case PREC_IN: bps =16; goto set_int;
		case PREC_DI: bps=32;
set_int:
			dtype = SAMPLEFORMAT_INT;
			break;
		case PREC_SP:  bps=32; goto set_flt;
		case PREC_DP:  bps=64;
set_flt:
			dtype = SAMPLEFORMAT_IEEEFP;
			break;
		default:
			NWARN("dp_to_bdf:  Unhandled precision");
			break;
	}

	if( TIFFSetField(bdf,TIFFTAG_SAMPLEFORMAT,dtype) != 1 )
		NWARN("error setting TIFF sample format");

	if( TIFFSetField(bdf,TIFFTAG_BITSPERSAMPLE,bps) != 1 )
		NWARN("error setting TIFF bits per sample");

	if( TIFFSetField(bdf,TIFFTAG_IMAGEWIDTH,w) != 1 )
		NWARN("error setting TIFF width tag");

	if( TIFFSetField(bdf,TIFFTAG_IMAGELENGTH,h) != 1 )
		NWARN("error setting TIFF length tag");

/*
//	if( TIFFSetField(bdf,TIFFTAG_IMAGEDEPTH,d) != 1 )
//		NWARN("error setting TIFF depth tag");
*/

	if( TIFFSetField(bdf,TIFFTAG_SAMPLESPERPIXEL,d) != 1 )
		NWARN("error setting TIFF depth tag");

	if( TIFFSetField(bdf,TIFFTAG_PLANARCONFIG,pc) != 1 )
		NWARN("error setting TIFF planar_config tag");

	if( TIFFSetField(bdf,TIFFTAG_PHOTOMETRIC,ph) != 1 )
		NWARN("error setting TIFF photometric tag");

	/*
	if( TIFFSetField(bdf,TIFFTAG_COMPRESSION,comp) != 1 )
		NWARN("error setting TIFF bits per sample");
	*/

	return(0);
}
#else /* ! FOOBAR */

int dp_to_bdf(BDF_info *bdfp,Data_Obj *dp)
{
	NERROR1("Sorry, dp_to_bdf not implemented");
	return(-1);
}

#endif /* ! FOOBAR */

FIO_SETHDR_FUNC( set_bdf_hdr )
{
	if( dp_to_bdf(ifp->if_hd,ifp->if_dp) < 0 ){
		bdf_close(QSP_ARG  ifp);
		return(-1);
	}
	return(0);
}

FIO_WT_FUNC( bdf_wt )
{
	if( ifp->if_dp == NO_OBJ ){	/* first time? */

		/* set the rows & columns in our file struct */
		setup_dummy(ifp);
		copy_dimensions(ifp->if_dp, dp);

		ifp->if_dp->dt_frames = ifp->if_frms_to_wt;
		ifp->if_dp->dt_seqs = 1;

		if( set_bdf_hdr(QSP_ARG  ifp) < 0 ) return(-1);

	} else if( !same_type(QSP_ARG  dp,ifp) ) return(-1);

	/* now write the data */
#ifdef FOOBAR
// We get an error from TIFFWriteScanline about setting PlanarConfig...
	datap = dp->dt_data;
	for(row=0;row<dp->dt_rows;row++){
		if( TIFFWriteScanline(ifp->if_bdf,datap,row,0) != 1 )
			NWARN("error writing TIFF scanline");
		datap += siztbl[MACHINE_PREC(dp)] * dp->dt_rowinc;
	}
#endif /* FOOBAR */

	ifp->if_nfrms ++ ;
	check_auto_close(QSP_ARG  ifp);
	return(0);
}

FIO_RD_FUNC( bdf_rd )
{
	if( x_offset != 0 || y_offset != 0  || t_offset != 0 ){
		sprintf(error_string,"bdf_rd %s:  Sorry, don't know how to handle non-zero offsets",
			ifp->if_name);
		NWARN(error_string);
		return;
	}

	if( fread(dp->dt_data,1,dp->dt_n_type_elts,ifp->if_fp) != dp->dt_n_type_elts ){
		sprintf(error_string,"bdf_rd %s:  error reading data",ifp->if_name);
		NWARN(error_string);
	}
}

int bdf_unconv(void *hdr_pp,Data_Obj *dp)
{
	NWARN("bdf_unconv not implemented");
	return(-1);
}

int bdf_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("bdf_conv not implemented");
	return(-1);
}

static int check_value_args(Data_Obj *dp, Image_File *ifp, const char *routine_name)
{
	if( dp == NO_OBJ ) return(-1);
	if( ifp == NO_IMAGE_FILE ) return(-1);

	if( ifp->if_type != IFT_BDF ){
		sprintf(DEFAULT_ERROR_STRING,"%s:  image file %s (%s) is not a %s file!?",
			routine_name,ifp->if_name,ft_tbl[ifp->if_type].ft_name,
			ft_tbl[IFT_BDF].ft_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(DEFAULT_ERROR_STRING,"%s:  object %s must be contiguous",routine_name,dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

	if( dp->dt_prec != PREC_DI ){
		sprintf(DEFAULT_ERROR_STRING,"%s:  object %s (%s) should have %s precision",
			routine_name,dp->dt_name,name_for_prec(dp->dt_prec),name_for_prec(PREC_DI));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}


	if( HDR_P->bdf_n_channels <= 0 ){
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  %s:  BDF file %s doesn't have any channels!?",routine_name,ifp->if_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	return(0);
}

#define GET_VALUE_ROUTINE( name, desc, field )						\
											\
COMMAND_FUNC( name )									\
{											\
	Data_Obj *dp;									\
	Image_File *ifp;								\
	long *lp;									\
	int i;										\
	char pmpt[LLEN];								\
											\
	sprintf(pmpt,"object for %s value array",desc);					\
	dp = PICK_OBJ(pmpt);								\
	ifp = PICK_IMG_FILE("bdf file");						\
											\
	if( check_value_args(dp,ifp,#name) < 0 ) return;				\
											\
	lp = (long *) dp->dt_data;							\
	for(i=0;i<HDR_P->bdf_n_channels;i++)					\
		*lp++ = HDR_P->bdf_pci_tbl[i].field;				\
}

GET_VALUE_ROUTINE( get_bdf_min_values, "min", pci_min_val )
GET_VALUE_ROUTINE( get_bdf_max_values, "max", pci_max_val )
GET_VALUE_ROUTINE( get_bdf_adc_min_values, "adc_min", pci_adc_min )
GET_VALUE_ROUTINE( get_bdf_adc_max_values, "adc_max", pci_adc_max )

static Command bdf_ctbl[]={
{ "get_min_values",	get_bdf_min_values,	"transfer channel min values to data object"		},
{ "get_max_values",	get_bdf_max_values,	"transfer channel max values to data object"		},
{ "get_adc_minima",	get_bdf_adc_min_values,	"transfer channel adc min values to data object"	},
{ "get_adc_maxima",	get_bdf_adc_max_values,	"transfer channel adc max values to data object"	},
{ "quit",		popcmd,			"exit submenu"						},
{ NULL_COMMAND												}
};

COMMAND_FUNC( bdf_menu )
{
	PUSHCMD(bdf_ctbl,"bdf");
}

