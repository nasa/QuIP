
#include "quip_config.h"

#include <stdio.h>		/* fileno() */

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>		/* stat() */
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>		/* stat() */
#endif

//#ifdef SOLARIS
//#define _POSIX_C_SOURCE	1		/* force fileno() in stdio.h */
//#endif /* SOLARIS */

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#ifdef HAVE_TIME_H
#include <time.h>
#endif

#ifdef HAVE_MATH_H
#include <math.h>		/* floor() */
#endif

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif


#include "quip_prot.h"
#include "query_bits.h"	// LLEN - BUG
#include "fio_prot.h"

#ifdef HAVE_JPEG_SUPPORT

//#include "filetype.h" /* ft_tbl */

#include "fiojpeg.h"
#include "jpeg_private.h"
#include "cdjpeg.h"
#include "markers.h"
#include "debug.h"

#include "jpegint.h"

int jpeg_debug=0;

#define JPEG_INFO_MAGIC_STRING	"JPGinfo2"
#define OLD_MAGIC_STRING	"JPEGinfo"

#ifdef FOOBAR
/* local prototypes */
static int good_scan_data(Image_File *ifp);
static void report_marker( JPEG_MARKER mrkr );
static void report_len(int len);
static short getBEshort(FILE *fp);
static int32_t getlong(FILE *fp);
static int32_t jgetlong( j_decompress_ptr cinfop );
static short jgetshort( j_decompress_ptr cinfop );
static int scan_markers( Image_File *ifp );
static void put_BE_short( j_compress_ptr cinfo, unsigned int s );
static void put_string( j_compress_ptr cinfo, const char *str );
static void put_short( j_compress_ptr cinfo, unsigned int s );
static void put_long( j_compress_ptr cinfo, uint32_t l );
static int rd_jpeg_hdr( Image_File *ifp );
static void complete_compressor_setup( Image_File *ifp );
METHODDEF(void) write_LML_file_header( Image_File *ifp );
static void _put_long(FILE *fp,uint32_t l);
#endif // FOOBAR

#include "cderror.h"
static const char * const cdjpeg_message_table[] = {
#include "cderror.h"
  NULL
};

#define HDR_P(ifp)	((Jpeg_Hdr *)ifp->if_hdr_p)

/* LML stuff... */

#define IS_LML(ifp)		( FT_CODE(IF_TYPE(ifp)) == IFT_LML )

#define OLD_APP3_LENGTH		0x18
#define NEW_APP3_LENGTH		0x2c

static int32_t file_offset=0;
static char mrkstr[128];
static Image_File *jpeg_ifp=NO_IMAGE_FILE;

/* code borrowed from jcmarker.c */

static void emit_byte (j_compress_ptr cinfo, int val)
/* Emit a byte */
{
  struct jpeg_destination_mgr * dest = cinfo->dest;

  *(dest->next_output_byte)++ = (JOCTET) val;
  if (--dest->free_in_buffer == 0) {
    if (! (*dest->empty_output_buffer) (cinfo))
      ERREXIT(cinfo, JERR_CANT_SUSPEND);
  }
}

/* from jdatadst.c */
#define OUTPUT_BUF_SIZE	4096

/* Expanded data destination object for stdio output */

typedef struct {
  struct jpeg_destination_mgr pub; /* public fields */

  FILE * outfile;		/* target stream */
  JOCTET * buffer;		/* start of buffer */
} my_destination_mgr;

typedef my_destination_mgr * my_dest_ptr;

static void flush_output( j_compress_ptr cinfo )
{
  size_t n_to_flush;
  my_dest_ptr dest = (my_dest_ptr) cinfo->dest;

  n_to_flush = OUTPUT_BUF_SIZE - dest->pub.free_in_buffer;
  if( n_to_flush==0 ) return;

  if (JFWRITE(dest->outfile, dest->buffer, n_to_flush) !=
      (size_t) n_to_flush)
    ERREXIT(cinfo, JERR_FILE_WRITE);

  fflush(dest->outfile);

  dest->pub.next_output_byte = dest->buffer;
  dest->pub.free_in_buffer = OUTPUT_BUF_SIZE;
}

static void report_marker(JPEG_MARKER mrkr)
{
	const char *ms;

	switch(mrkr){
		case M_SOI:	ms = "M_SOI";	break;
		case M_APP0:	ms = "M_APP0";	break;
		case M_APP1:	ms = "M_APP1";	break;
		case M_APP2:	ms = "M_APP2";	break;
		case M_APP3:	ms = "M_APP3";	break;
		case M_APP4:	ms = "M_APP4";	break;
		case M_APP5:	ms = "M_APP5";	break;
		case M_APP6:	ms = "M_APP6";	break;
		case M_APP7:	ms = "M_APP7";	break;
		case M_APP8:	ms = "M_APP8";	break;
		case M_APP9:	ms = "M_APP9";	break;
		case M_APP10:	ms = "M_APP10";	break;
		case M_APP11:	ms = "M_APP11";	break;
		case M_APP12:	ms = "M_APP12";	break;
		case M_APP13:	ms = "M_APP13";	break;
		case M_APP14:	ms = "M_APP14";	break;
		case M_APP15:	ms = "M_APP15";	break;
		case M_DQT:	ms = "M_DQT";	break;
		case M_DHT:	ms = "M_DHT";	break;
		case M_SOF0:	ms = "M_SOF0";	break;
		case M_SOF1:	ms = "M_SOF1";	break;
		case M_SOF2:	ms = "M_SOF2";	break;
		case M_SOF3:	ms = "M_SOF3";	break;
		case M_SOF5:	ms = "M_SOF5";	break;
		case M_SOF6:	ms = "M_SOF6";	break;
		case M_SOF7:	ms = "M_SOF7";	break;
		case M_SOS:	ms = "M_SOS";	break;
		case M_EOI:	ms = "M_EOI";	break;
		case M_COM:	ms = "M_COM";	break;
		default:
			sprintf(mrkstr,"(unknown marker 0x%x)",mrkr);
			ms=mrkstr;
			break;
	}
	sprintf(DEFAULT_ERROR_STRING,"0x%x:\t%s (0x%x)",file_offset-2,ms,mrkr);
	NADVISE(DEFAULT_ERROR_STRING);
}

/* big-endian */
static short getBEshort(FILE *fp)
{
	short s;

	s=getc(fp);
	s <<= 8;
	s += getc(fp);
	return(s);
}

static int32_t getlong(FILE *fp)
{
	int32_t l;
	l  = getc(fp);
	l += getc(fp) << 8;
	l += getc(fp) << 16;
	l += getc(fp) << 24;
	return(l);
}

/* little endian */

static short getshort(FILE *fp)
{
	short s;
	
	s=getc(fp);
	s += getc(fp) << 8;
	return(s);
}


static void report_len(int len)
{
	sprintf(DEFAULT_ERROR_STRING,"\t\t\t%d (0x%x)",len,len);
	NADVISE(DEFAULT_ERROR_STRING);
}

static int good_scan_data(Image_File *ifp)
{
	int c;
	FILE *fp;
	int nskipped=0;

	fp = ifp->if_fp;

	do {
		c=getc(fp);
		file_offset++;

		if( c == 0xff ){	/* marker ? */
			c = getc(fp);
			file_offset++;
			if( c < 0 ){	/* EOF */
				goto eof_seen;
			}
			if( c == M_EOI ){
				if( verbose ){
	sprintf(DEFAULT_ERROR_STRING,"%d scan data bytes skipped",nskipped);
					NADVISE(DEFAULT_ERROR_STRING);
					report_marker((JPEG_MARKER)c);
				}
				return(1);
			}
			nskipped += 2;
		} else nskipped++;
	} while( c>= 0 );

eof_seen:
	sprintf(DEFAULT_ERROR_STRING,
		"unexpected EOF in scan section of jpeg file %s",ifp->if_name);
	NWARN(DEFAULT_ERROR_STRING);
	return(0);
}

/* scan one frame's worth of markers */

static int scan_markers(Image_File *ifp)
{
	FILE *fp;
	int c1,c2;
	int len,nc;
	int32_t n_before;
	short dummy,xSize,ySize,numComp /*, samplePrecision,compCode,sampFactorHV,Tq*/ ;
	int i;
	int32_t appId,frameNo,sec,usec,frameSeqNo,frameSize,colorEncoding,videoStream,timeDecimation;

	fp = ifp->if_fp;

	if( verbose ) NADVISE("");

	/* SOI should be the first */

	c1=getc(fp);	file_offset++;
	if( c1 < 0 ) return(0);

	c2=getc(fp);	file_offset++;
	if( c2 < 0 ){
		NWARN("unexpected EOF");
		return(0);
	}

	while( c2 == 0xff ){
		c2=getc(fp);
		file_offset++;
		if( c2 < 0 ){
			/* EOF */
			return(0);
		}
	}

	if( c1!=0xff && c2!=M_SOI ){
		NADVISE("");
		sprintf(DEFAULT_ERROR_STRING,"0x%x:  expected 0x%x SOI (0x%x), saw 0x%x 0x%x",
			file_offset,
			0xff,M_SOI,c1,c2);
		NWARN(DEFAULT_ERROR_STRING);
	}
	goto process_marker;

next_marker:
	c1=getc(fp);
	file_offset++;
	n_before = file_offset;
	while( c1 != 0xff ){
		c1=getc(fp);
		if( c1 < 0 ) return(0);	/* EOF */
		file_offset++;
	}
	if( verbose && (file_offset-n_before)>0 ){
		sprintf(DEFAULT_ERROR_STRING, "%d (0x%x) pre-marker bytes skipped",
			file_offset-n_before,file_offset-n_before);
		NADVISE(DEFAULT_ERROR_STRING);
	}

	c2=getc(fp);
	file_offset++;
	if( c2==0 ){
		if( verbose ) NADVISE("skipping null byte");
		goto next_marker;
	}
	if( c2 < 0 ) return(0);	/* EOF */

	while( c2 == 0xff ){
		c2=getc(fp);
		file_offset++;
	}

	if( c2 < 0 ){
		NADVISE("EOF");
		return(0);
	}
	if( c1 != 0xff ){
		sprintf(DEFAULT_ERROR_STRING,"c1 = 0x%x, expected 0xff",c1);
		NWARN(DEFAULT_ERROR_STRING);
	}
process_marker:
	if( verbose )
		report_marker((JPEG_MARKER)c2);

	/* now eat up whatever characters we need to for this section... */
	switch(c2){
		case M_EOI:
			return(1);
			break;
		case M_SOI: break;
		default:
			report_marker((JPEG_MARKER)c2);
			NWARN("unhandled case in scan_markers");
			break;/* MRA next byte may not contain an offset
			          to the next marker*/
		case M_DQT:
		case M_DHT:
		case M_DRI:
		case M_APP0:
		case M_APP1:
		case M_APP2:
		case M_APP4:
		case M_APP5:
		case M_APP6:
		case M_APP7:
		case M_APP8:
		case M_APP9:
		case M_APP10:
		case M_APP11:
		case M_APP12:
		case M_APP13:
		case M_APP14:
		case M_COM:
			/* skip_variable */
			len = (unsigned int) getc(fp);
			len <<= 8;
			len += (unsigned int) getc(fp);
			len -= 2;			/* we've already skipped the 2 len bytes */

			if( verbose ) report_len(len);

			if( fseek(fp,len,SEEK_CUR) < 0 ){
				perror("fseek");
			}
			file_offset += len+2;
			break;

		case M_APP3:		/* scan_markers:  LML special data */
			/* BUG - check appId? */

			/* this is not getshort, this is big-endian */
			len = getBEshort(fp);

			if( len != OLD_APP3_LENGTH && len != NEW_APP3_LENGTH ){
				sprintf(DEFAULT_ERROR_STRING,
					"strange APP3 len 0x%x, expected 0x%x or 0x%x",
						len, OLD_APP3_LENGTH, NEW_APP3_LENGTH);
				NWARN(DEFAULT_ERROR_STRING);
			}

			if( verbose ) report_len(len);


			appId = getlong(fp);
			frameNo = getshort(fp);
			sec = getlong(fp);
			usec = getlong(fp);
			frameSize = getlong(fp);
			frameSeqNo = getlong(fp);
			if( len == NEW_APP3_LENGTH ){
				colorEncoding = getlong(fp);
				videoStream = getlong(fp);
				timeDecimation = getshort(fp);
				/* get 10 filler bytes */
				nc=10;
				while(nc--)
					if( (c1=getc(fp)) == EOF ){
						NWARN("premature jpeg EOF");
						// suppress compiler warning...
						sprintf(DEFAULT_ERROR_STRING,
					"appId = %d",appId);
						NADVISE(DEFAULT_ERROR_STRING);
						nc=0;
					}
			} else {		/* old style file */
				colorEncoding = 1;	/* ntsc */
				videoStream = 1;	/* D1 */
				timeDecimation = 1;	/* 60 fps */
			}


			if( HDR_P(ifp)->jpeg_comps == (-1) ){
				/* copy over on first frame only */
				HDR_P(ifp)->lml_frameNo = frameNo;
				HDR_P(ifp)->lml_sec = sec;
				HDR_P(ifp)->lml_usec = usec;
				HDR_P(ifp)->lml_frameSize = frameSize;
				HDR_P(ifp)->lml_frameSeqNo = frameSeqNo;
				HDR_P(ifp)->lml_fieldNo = 0;
				HDR_P(ifp)->lml_colorEncoding = colorEncoding;
				HDR_P(ifp)->lml_videoStream = videoStream;
				HDR_P(ifp)->lml_timeDecimation = timeDecimation;
			}

			file_offset += len;

			break;

		/* not sure about these other SOF's??? */
		case M_SOF1:
		case M_SOF2:
		case M_SOF3:
		case M_SOF5:
		case M_SOF6:
		case M_SOF7:

		case M_SOF0:
			len = (unsigned int) getc(fp);
			len <<= 8;
			len += (unsigned int) getc(fp);

			if( verbose ) report_len(len);

			dummy /*samplePrecision*/ = getc(fp);
			ySize = getBEshort(fp);
			xSize = getBEshort(fp);
			numComp = getc(fp);
//fprintf(stderr,"M_SOFX:  dy = %d, dx = %d, depth = %d\n",
//ySize,xSize,numComp);
//fflush(stderr);

			if( HDR_P(ifp)->jpeg_comps == (-1) ){
				/* first time */
				HDR_P(ifp)->jpeg_comps = numComp;
				HDR_P(ifp)->jpeg_height = ySize;
				HDR_P(ifp)->jpeg_width = xSize;
			} else {
				/* make sure nothing has changed */
				if( HDR_P(ifp)->jpeg_comps != numComp ||
				    HDR_P(ifp)->jpeg_width != xSize    ||
				    HDR_P(ifp)->jpeg_height != ySize ){
					sprintf(DEFAULT_ERROR_STRING,"Frame size mismatch!?  (JPEG file %s)",
						ifp->if_name);
					NWARN(DEFAULT_ERROR_STRING);
	sprintf(DEFAULT_ERROR_STRING,"old sizes:  %d comps, %d cols, %d rows",
		HDR_P(ifp)->jpeg_comps,HDR_P(ifp)->jpeg_width,HDR_P(ifp)->jpeg_height);
					NADVISE(DEFAULT_ERROR_STRING);
	sprintf(DEFAULT_ERROR_STRING,"new sizes:  %d comps, %d cols, %d rows",
						numComp,xSize,ySize);
					NADVISE(DEFAULT_ERROR_STRING);
				}
			}
			for(i=0;i<numComp;i++){
				dummy /*compCode*/ = getc(fp);
				dummy /*sampFactorHV*/ = getc(fp);	/* 4 bits for h and v ... */
				dummy /*Tq*/ = getc(fp);
				// These are all unused, and so generate compiler warnings...
				// by using dummy, at least we go from three warnings to one...
			}
			file_offset += len;
			break;

		case M_SOS:
			len = (unsigned int) getc(fp);
			len <<= 8;
			len += (unsigned int) getc(fp);

			if( verbose ) report_len(len);

			nc = getc(fp);		/* # comps in scan */
			if( len != (2*nc+6) ){
				sprintf(DEFAULT_ERROR_STRING,"SOS:  len = %d, n= %d",len,nc);
				NWARN(DEFAULT_ERROR_STRING);
			}
			while(nc--){
				c1=getc(fp);	/* compCode */
				c1=getc(fp);	/* hVfactorr */
			}
			c1=getc(fp);		/* Ss/Se */
			c1=getc(fp);		/* Ah */
			c1=getc(fp);		/* Al */
			file_offset += len;

			/* Now we need to skip over the data... */

			/* this function was added because we ran
			 * into some files where an 0xff pre-marker
			 * byte occurred in the scan data...
			 */

			return( good_scan_data(ifp) );

			break;
	}
	goto next_marker;
}

/*
 * Marker processor for COM and interesting APPn markers.
 * This replaces the library's built-in processor, which just skips the marker.
 * We want to print out the marker as text, to the extent possible.
 * Note this code relies on a non-suspending data source.
 */

LOCAL(unsigned int)
jpeg_getc (j_decompress_ptr cinfop)
/* Read next byte */
{
  struct jpeg_source_mgr * datasrc = cinfop->src;

  if (datasrc->bytes_in_buffer == 0) {
    if (! (*datasrc->fill_input_buffer) (cinfop))
      ERREXIT(cinfop, JERR_CANT_SUSPEND);
  }
  datasrc->bytes_in_buffer--;
  return GETJOCTET(*datasrc->next_input_byte++);
}



METHODDEF(boolean)
print_text_marker (j_decompress_ptr cinfop)
{
  boolean traceit = (cinfop->err->trace_level >= 1);
  int32_t length;
  unsigned int ch;
  unsigned int lastch = 0;

  length = jpeg_getc(cinfop) << 8;
  length += jpeg_getc(cinfop);
  length -= 2;			/* discount the length word itself */

  if (traceit) {
    if (cinfop->unread_marker == JPEG_COM){
      sprintf(DEFAULT_ERROR_STRING, "Comment, length %d:\n", length);
      NADVISE(DEFAULT_ERROR_STRING);
    } else {			/* assume it is an APPn otherwise */
      sprintf(DEFAULT_ERROR_STRING, "APP%d, length %d:\n",
	      cinfop->unread_marker - JPEG_APP0, length);
      NADVISE(DEFAULT_ERROR_STRING);
    }
  }

  while (--length >= 0) {
    ch = jpeg_getc(cinfop);
    if (traceit) {
      /* Emit the character in a readable form.
       * Nonprintables are converted to \nnn form,
       * while \ is converted to \\.
       * Newlines in CR, CR/LF, or LF form will be printed as one newline.
       */
      if (ch == '\r') {
	fprintf(stderr, "\n");
      } else if (ch == '\n') {
	if (lastch != '\r')
	  fprintf(stderr, "\n");
      } else if (ch == '\\') {
	fprintf(stderr, "\\\\");
      } else if (isprint(ch)) {
	putc(ch, stderr);
      } else {
	fprintf(stderr, "\\%03o", ch);
      }
      lastch = ch;
    }
  }

  if (traceit)
    fprintf(stderr, "\n");

  return TRUE;
}

static int32_t jgetlong(j_decompress_ptr cinfop)
{
	int32_t l;
	l  = jpeg_getc(cinfop);
	l += jpeg_getc(cinfop) << 8;
	l += jpeg_getc(cinfop) << 16;
	l += jpeg_getc(cinfop) << 24;
	return(l);
}

static short jgetshort(j_decompress_ptr cinfop)
{
	short s;
	s  = jpeg_getc(cinfop);
	s += jpeg_getc(cinfop) << 8;
	return(s);
}

METHODDEF(boolean)
process_lml_marker(j_decompress_ptr cinfop)
{
	int32_t length;
	int32_t app_id;
	int n,dummy;

	length = jpeg_getc(cinfop) << 8;
	length += jpeg_getc(cinfop);

	app_id = jgetlong(cinfop);

#define LML_APP_ID	0x4c4d4cL	/* 'L' 'M' 'L' */

	if( app_id != LML_APP_ID ){
		sprintf(DEFAULT_ERROR_STRING,
			"APP3 id code 0x%x does not match expected LML code 0x%lx",
			app_id,LML_APP_ID);
		NWARN(DEFAULT_ERROR_STRING);
	}

	if( length != OLD_APP3_LENGTH && length != NEW_APP3_LENGTH ){
		sprintf(DEFAULT_ERROR_STRING,"APP3 length is 0x%x, expected 0x%x or 0x%x!?",
			length,OLD_APP3_LENGTH,NEW_APP3_LENGTH);
		NWARN(DEFAULT_ERROR_STRING);
	}

	HDR_P(jpeg_ifp)->lml_frameNo  = jgetshort(cinfop);
	HDR_P(jpeg_ifp)->lml_sec = jgetlong(cinfop);
	HDR_P(jpeg_ifp)->lml_usec = jgetlong(cinfop);
	HDR_P(jpeg_ifp)->lml_frameSize = jgetlong(cinfop);
	HDR_P(jpeg_ifp)->lml_frameSeqNo = jgetlong(cinfop);

	if( length == NEW_APP3_LENGTH ){
		HDR_P(jpeg_ifp)->lml_colorEncoding = jgetlong(cinfop);
		HDR_P(jpeg_ifp)->lml_videoStream = jgetlong(cinfop);
		HDR_P(jpeg_ifp)->lml_timeDecimation = jgetshort(cinfop);
		n=5;
		while(n--) dummy=jgetshort(cinfop);	/* read pad bytes */
	}

	HDR_P(jpeg_ifp)->lml_fieldNo = 0;		/* APP3 occurs once per frame */

	if( verbose ){
		sprintf(DEFAULT_MSG_STR,"LML extension: sec=%ld usec=%ld size=%d seq=%d no=%d",
			HDR_P(jpeg_ifp)->lml_sec,
			HDR_P(jpeg_ifp)->lml_usec,
			HDR_P(jpeg_ifp)->lml_frameSize,
			HDR_P(jpeg_ifp)->lml_frameSeqNo,
			HDR_P(jpeg_ifp)->lml_frameNo);
		_prt_msg(DEFAULT_QSP_ARG  DEFAULT_MSG_STR);
	}

	return TRUE;
}

int jpeg_to_dp(Data_Obj *dp,Jpeg_Hdr *jpeg_hp)
{
	/* at one time was short --- WHY??? */
	SET_OBJ_PREC_PTR(dp,PREC_FOR_CODE(PREC_UBY));

	SET_OBJ_COMPS(dp,jpeg_hp->jpeg_comps);
	SET_OBJ_COLS(dp,jpeg_hp->jpeg_width);
	SET_OBJ_ROWS(dp,jpeg_hp->jpeg_height);
	SET_OBJ_FRAMES(dp,jpeg_hp->jpeg_frames);
	SET_OBJ_SEQS(dp,1);

	SET_OBJ_COMP_INC(dp,1);
	SET_OBJ_PXL_INC(dp,1);
	SET_OBJ_ROW_INC(dp,OBJ_PXL_INC(dp)*OBJ_COLS(dp));
	SET_OBJ_FRM_INC(dp,OBJ_ROW_INC(dp)*OBJ_ROWS(dp));
	SET_OBJ_SEQ_INC(dp,OBJ_FRM_INC(dp)*OBJ_FRAMES(dp));

	SET_OBJ_PARENT(dp, NO_OBJ);
	SET_OBJ_CHILDREN(dp, NO_LIST);

	SET_OBJ_AREA(dp, ram_area_p);		/* the default */
	SET_OBJ_DATA_PTR(dp, NULL);
	SET_OBJ_N_TYPE_ELTS(dp, OBJ_COMPS(dp) * OBJ_COLS(dp) * OBJ_ROWS(dp)
			* OBJ_FRAMES(dp) * OBJ_SEQS(dp));

	auto_shape_flags(OBJ_SHAPE(dp),dp);

	return(0);
}



/* the unconvert routine creates a disk header */

int jpeg_unconv(void *hdr_pp,Data_Obj *dp)
{
	NWARN("jpeg_unconv() not implemented!?");
	return(-1);
}

int lml_unconv( void *hd_pp, Data_Obj *dp )
{
	return jpeg_unconv(hd_pp,dp);
}

int lml_conv(Data_Obj *dp, void *hd_pp)
{
	return jpeg_conv(dp,hd_pp);
}

int jpeg_conv(Data_Obj *dp,void *hd_pp)
{
	NWARN("jpeg_conv not implemented");
	return(-1);
}

static void make_jpi_name(char buf[LLEN], Image_File *ifp, const char * suffix)
{
	int l,i;

	strcpy(buf,ifp->if_pathname);
	/* strip the suffix */
	l=strlen(ifp->if_pathname);
	i=l-1;
	while( i > 0 && buf[i] != '.' )
		i--;
	if( buf[i]=='.' ){
		if( (!strcmp(&buf[i+1],"jpeg")) || (!strcmp(&buf[i+1],"jpg")) ||
				(!strcmp(&buf[i+1],"JPEG")) || (!strcmp(&buf[i+1],"JPG")) ){
			buf[i]=0;
		}
	}
	strcat(buf,".jp");
	strcat(buf,suffix);
}

static FILE * remove_info_if_stale(const char *info_name,FILE *info_fp,const char *src_name, FILE *src_fp)
{
	struct stat info_statb, file_statb;

	/* if the info file is older than the file itself, then it should be unlinked and recomputed */
	/* need to stat both files... */
	if( fstat(fileno(info_fp),&info_statb) < 0 ){
		_tell_sys_error(DEFAULT_QSP_ARG  "check_jpeg_info:  fstat:");
		NERROR1("unable to stat jpeg info file");
	}
	if( fstat(fileno(src_fp),&file_statb) < 0 ){
		_tell_sys_error(DEFAULT_QSP_ARG  "check_jpeg_info:  fstat:");
		NERROR1("unable to stat jpeg data file");
	}
	/* now compare mod times */
	if( file_statb.st_mtime > info_statb.st_mtime ){
		sprintf(DEFAULT_ERROR_STRING,"Existing jpeg info file %s is older than file %s, will unlink and recompute",
				info_name,src_name);
		NADVISE(DEFAULT_ERROR_STRING);
		fclose(info_fp);

		if( unlink(info_name) < 0 ){
			_tell_sys_error(DEFAULT_QSP_ARG  "remove_info_if_stale:  unlink:");
			NWARN("unable to remove stale jpeg info file");
			/* may not have permission */
		}
		return(NULL);
	}
	return(info_fp);
}

#define JPI_ERROR_MSG(msg)					\
								\
	{							\
	sprintf(DEFAULT_ERROR_STRING,				\
	"read_jpeg_info_top (%s):  %s",ifp->if_name,msg);	\
	NWARN(DEFAULT_ERROR_STRING);				\
	}

/* This stuff is common for ascii or binary tables */

static int read_jpeg_info_top(FILE *info_fp, const char *magic_string, Image_File *ifp )
{
	char buf[16];
	dimension_t depth, cols, rows, frms;
	int c;

	/* first check the magic number */
	if( fread(buf,1,9,info_fp) != 9 ){
		JPI_ERROR_MSG("missing magic number data")
		return(0);
	}
	if( buf[8] != '\n' ){
		JPI_ERROR_MSG("bad magic number terminator");
		return(0);
	}
	buf[8]=0;
	if( strcmp(buf,magic_string) ){
		if( !strcmp(buf,OLD_MAGIC_STRING) ){
JPI_ERROR_MSG("info file is old format (32 bit offsets)");
			NERROR1("Please delete info file.");
			return 0;
		}

		JPI_ERROR_MSG("bad magic number data");
		sprintf(DEFAULT_ERROR_STRING,"read_jpeg_info_top:  expected \"%s\" but encountered \"%s\"",
			magic_string,buf);
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}
	/* now we know that the file starts out in the right format */
	if( fscanf(info_fp,"%d",&depth) != 1 ){
		JPI_ERROR_MSG("bad depth data");
		return(0);
	}
	if( fscanf(info_fp,"%d",&cols) != 1 ){
		JPI_ERROR_MSG("bad column count data");
		return(0);
	}
	if( fscanf(info_fp,"%d",&rows) != 1 ){
		JPI_ERROR_MSG("bad row count data")
		return(0);
	}
	if( fscanf(info_fp,"%d",&frms) != 1 ){
		JPI_ERROR_MSG("bad frame count data");
		return(0);
	}
	/* now read the newline character */
	c=fgetc(info_fp);
	if( c != '\n' ){
		JPI_ERROR_MSG("missing newline after frame count")
		return(0);
	}

	HDR_P(ifp)->jpeg_comps=depth;
	HDR_P(ifp)->jpeg_width=cols;
	HDR_P(ifp)->jpeg_height=rows;
	HDR_P(ifp)->jpeg_frames=frms;

	return(frms);		/* all OK */
}

static int read_binary_jpeg_info(Image_File *ifp, FILE *info_fp)
{
	seek_tbl_type *tbl;
	u_int frms;
	u_int n;

	/* now read the info */

	if( (frms=read_jpeg_info_top(info_fp,JPEG_INFO_MAGIC_STRING,ifp)) == 0 ){
		sprintf(DEFAULT_ERROR_STRING,"read_binary_jpeg_info:  problem with top of info file");
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}

	/* now allocate the seek table */
	tbl = (seek_tbl_type *)getbuf( sizeof(seek_tbl_type) * frms );
	if( (n=fread(tbl,sizeof(seek_tbl_type),frms,info_fp)) != frms ){
		sprintf(DEFAULT_ERROR_STRING,"read_binary_jpeg_info:  error reading seek table data for file %s",ifp->if_name);
		NWARN(DEFAULT_ERROR_STRING);
		sprintf(DEFAULT_ERROR_STRING,"Expected %d addresses, got %d",frms,n);
		NADVISE(DEFAULT_ERROR_STRING);
/*
sprintf(DEFAULT_ERROR_STRING,"sizeof(u_int) = %d",sizeof(u_int));
NADVISE(DEFAULT_ERROR_STRING);
sprintf(DEFAULT_ERROR_STRING,"sizeof(u_short) = %d",sizeof(u_short));
NADVISE(DEFAULT_ERROR_STRING);
sprintf(DEFAULT_ERROR_STRING,"sizeof(u_int) = %d",sizeof(unsigned int));
NADVISE(DEFAULT_ERROR_STRING);
*/
		return(0);
	}

	HDR_P(ifp)->jpeg_seek_table = tbl;

	return(1);
}

static int read_ascii_jpeg_info(Image_File *ifp, FILE *info_fp)
{
	seek_tbl_type *tbl;
	u_int frms;
	u_int i;

	// BUG?  this used to be a different string ("JPEGINFO")

	if( (frms=read_jpeg_info_top(info_fp,JPEG_INFO_MAGIC_STRING,ifp)) == 0 ){
		sprintf(DEFAULT_ERROR_STRING,"read_binary_jpeg_info:  problem with top of info file");
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}

	/* now read the info */

	/* now allocate the seek table */
	tbl = (seek_tbl_type *)getbuf( sizeof(seek_tbl_type) * frms );
	for(i=0;i<frms;i++){
		if( fscanf(info_fp,"%" SEEK_TBL_SCN_FMT ,&tbl[i]) != 1 )
			NWARN("error reading ascii seek table data");
		return(0);
	}

	HDR_P(ifp)->jpeg_seek_table = tbl;

	return(1);
}

/* Check for a .jpa or .jpi file, and if it exists, read it!
 *
 */

static int check_jpeg_info(Image_File *ifp)
{
	char jpi_name[LLEN];
	FILE *fp;

	make_jpi_name(jpi_name,ifp,"i");	/* look for (fast) binary version first */
	fp=fopen(jpi_name,"r");
	if( !fp ) {
		make_jpi_name(jpi_name,ifp,"a");	/* look for new ascii version if binary doesn't exist */
		fp=fopen(jpi_name,"r");
		if( !fp ) return(0);
		fp = remove_info_if_stale(jpi_name,fp,ifp->if_pathname,ifp->if_fp);
		if( !fp ) return(0);
		return( read_ascii_jpeg_info(ifp,fp) );
	} else {
		fp = remove_info_if_stale(jpi_name,fp,ifp->if_pathname,ifp->if_fp);
		if( !fp ) return(0);
		return( read_binary_jpeg_info(ifp,fp) );
	}
	/* NOT_REACHED */
	return(1);
}

static void save_jpeg_info_binary(Image_File *ifp)
{
	char jpi_name[LLEN];
	FILE *fp;

	make_jpi_name(jpi_name,ifp,"i");
	fp = fopen(jpi_name,"w");
	if( !fp ){
		sprintf(DEFAULT_ERROR_STRING,"save_jpeg_info:  couldn't open jpeg info file %s for writing",jpi_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	fprintf(fp,"%s\n",JPEG_INFO_MAGIC_STRING);
	fprintf(fp,"%d\n",HDR_P(ifp)->jpeg_comps);
	fprintf(fp,"%d\n",HDR_P(ifp)->jpeg_width);
	fprintf(fp,"%d\n",HDR_P(ifp)->jpeg_height);
	fprintf(fp,"%d\n",HDR_P(ifp)->jpeg_frames);
	if( fwrite(HDR_P(ifp)->jpeg_seek_table,sizeof(seek_tbl_type),HDR_P(ifp)->jpeg_frames,fp) !=
			(dimension_t)HDR_P(ifp)->jpeg_frames ){
		sprintf(DEFAULT_ERROR_STRING,"error writing seek table data to jpeg info file");
		NWARN(DEFAULT_ERROR_STRING);
		/* BUG at this point we should unlink the file? */
	}
	fclose(fp);
}

static void save_jpeg_info_ascii(Image_File *ifp)
{
	char jpi_name[LLEN];
	FILE *fp;
	int i;

	make_jpi_name(jpi_name,ifp,"a");
	fp = fopen(jpi_name,"w");
	if( !fp ){
		sprintf(DEFAULT_ERROR_STRING,"save_jpeg_info_ascii:  couldn't open jpeg info file %s for writing",jpi_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	fprintf(fp,"%s\n",JPEG_INFO_MAGIC_STRING);
	fprintf(fp,"%d\n",HDR_P(ifp)->jpeg_comps);
	fprintf(fp,"%d\n",HDR_P(ifp)->jpeg_width);
	fprintf(fp,"%d\n",HDR_P(ifp)->jpeg_height);
	fprintf(fp,"%d\n",HDR_P(ifp)->jpeg_frames);
	for(i=0;i<HDR_P(ifp)->jpeg_frames;i++)
		fprintf(fp,"%llu\n",
			(unsigned long long)HDR_P(ifp)->jpeg_seek_table[i]);
	fclose(fp);
}

static void save_jpeg_info(Image_File *ifp)
{
	if( default_jpeg_info_format == JPEG_INFO_FORMAT_ASCII )
		save_jpeg_info_ascii(ifp);
	else
		save_jpeg_info_binary(ifp);
}


/* This is really a misnomer... MJPEG files don't have headers, they
 * are just a stream of images...  We fake this, by scanning the file,
 * counting the number of images, then we rewind the file...
 */

static int rd_jpeg_hdr(Image_File *ifp)
{
	/* int n; */
	incr_t offset;
	dimension_t nf=0;
	seek_tbl_type *tbl;

	/* Very long jpeg movies take a long time to scan, so we'd like to save
	 * the header information.  We look for a file in the same directory as the file
	 * with a .jpi (JPeg Information) suffix...
	 */

	if( check_jpeg_info(ifp) )
		return(0);

	/* initialize the header with impossible values
	 * to indicate that we don't know anything yet.
	 */

	HDR_P(ifp)->jpeg_comps=(-1);
	HDR_P(ifp)->jpeg_width=(-1);
	HDR_P(ifp)->jpeg_height=(-1);

	offset = ftell(ifp->if_fp);
	if( verbose ){
		sprintf(DEFAULT_ERROR_STRING,"scanning file %s from offset %d, to count frames",
			ifp->if_name,offset);
		NADVISE(DEFAULT_ERROR_STRING);
	}

	/* While we are counting the frames, we remember their offsets so
	 * we can seek later...  We need to store the offsets in a table, but
	 * we don't know how big a table we are going to need!?  We allocate
	 * a big one, then we copy the data into a properly sized buffer
	 * when we are done.
	 *
	 * How big is big enough?  In the atc expt, we have 100 second trials (max),
	 * which translates to 6k fields...  we round up to 8k.
	 * For the FAA project, we merge four movies of 54000 frames, which creates
	 * a 4*54000=216000 frames movie
	 */
/*#define MAX_SEEK_TBL_SIZE 18000 */
/* #define MAX_SEEK_TBL_SIZE 220000 */
#define MAX_SEEK_TBL_SIZE 300000
	
	tbl = (seek_tbl_type *)getbuf( MAX_SEEK_TBL_SIZE * sizeof(seek_tbl_type) );

	tbl[0]=0;
	/* scan_markers() fills in the image dimensions & depth... */
	while( scan_markers(ifp) ){
		nf++;
		if( verbose && (nf % 60) == 0 ) _prt_msg_frag(DEFAULT_QSP_ARG  ".");
		if( nf < MAX_SEEK_TBL_SIZE )
			tbl[nf] = ftell(ifp->if_fp);
		/* if this is exceeded, we'll print a warning
		 * after we've finished scanning the file.
		 */
	}
	if( verbose ) _prt_msg(DEFAULT_QSP_ARG  "");

	if( nf > MAX_SEEK_TBL_SIZE ){
		sprintf(DEFAULT_ERROR_STRING,
	"rd_jpeg_hdr:  file %s has %d frames, which exceeds MAX_SEEK_TBL_SIZE (%d)",
			ifp->if_name,nf,MAX_SEEK_TBL_SIZE);
		NWARN(DEFAULT_ERROR_STRING);
		NADVISE("Consider recompiling");
		givbuf(tbl);
		tbl=NULL;
	}
	if( nf == 0 ){
		sprintf(DEFAULT_ERROR_STRING,"jpeg file %s has no frames!?",ifp->if_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

	HDR_P(ifp)->jpeg_frames=nf;

	/* now copy the table to the final resting place */ 
	if( tbl != NULL ){
		HDR_P(ifp)->jpeg_seek_table = (seek_tbl_type *)getbuf( nf * sizeof(seek_tbl_type) );
		memcpy(HDR_P(ifp)->jpeg_seek_table,tbl,nf*sizeof(seek_tbl_type));
		givbuf(tbl);

		/* we save the info for later! */
		if( nf > 140 )	/* used to be 256, but we want to use for editing files... */
			save_jpeg_info(ifp);
	} else {
		HDR_P(ifp)->jpeg_seek_table = NULL;
	}

	rewind(ifp->if_fp);
	return(0);
}

static void put_long(j_compress_ptr cinfo,uint32_t l)
{
	/* MSB last */
	emit_byte(cinfo,l&0xff);
	emit_byte(cinfo,(l>>8)&0xff);
	emit_byte(cinfo,(l>>16)&0xff);
	emit_byte(cinfo,(l>>24)&0xff);
}

static void _put_long(FILE *fp,uint32_t l)
{
	/* MSB last */
	putc(l&0xff,fp);
	putc((l>>8)&0xff,fp);
	putc((l>>16)&0xff,fp);
	putc((l>>24)&0xff,fp);
}

static void put_BE_short(j_compress_ptr cinfo,unsigned int s)
{
	/* MSB first */
	emit_byte(cinfo,(s>>8)&0xff);
	emit_byte(cinfo,s&0xff);
}

static void put_short(j_compress_ptr cinfo,unsigned int s)
{
	/* MSB last */
	emit_byte(cinfo,s&0xff);
	emit_byte(cinfo,(s>>8)&0xff);
}

static void put_string(j_compress_ptr cinfo,const char *str)
{
	do {
		emit_byte(cinfo,*str);
	} while( *str++ );
}

METHODDEF(void)
write_LML_file_header(Image_File *ifp)
{
	unsigned int datalen;
	int32_t frameSize,frameSeqNo;
	short frameNo;
	struct timeval tv;
	j_compress_ptr cinfop;
#ifndef OLD_LML_MARKER
	int n;
#endif /* OLD_LML_MARKER */

	cinfop = &(HDR_P(ifp)->u.c_cinfo);

	/*
	(*orig_header_method)(cinfop);
	*/

	/* Now write an APP3 marker... */
	emit_byte(cinfop,0xff);
	emit_byte(cinfop,M_APP3);

#ifdef OLD_LML_MARKER

	/* This code worked with the old (pre-LINVS) driver */

	datalen = 5*sizeof(int32_t) + 2*sizeof(short);	/* should be 24??? 0x18 */

	put_BE_short(cinfop,datalen);

	/* Where will we get the frame size?
	 * We may need to write the file out, check where the pointer
	 * is, then seek back and rewrite this data...
	 */
	frameNo = 0;
	frameSize = 0;
	frameSeqNo = 0;
	if( gettimeofday(&tv,NULL) < 0 ){
		perror("gettimeofday");
	}

	put_string(cinfop,"LML");
	put_short(cinfop,frameNo);
	put_long(cinfop,tv.tv_sec);
	put_long(cinfop,tv.tv_usec);

	/* remember the offset of the frameSize, so we can rewrite it when we know it!
	 * We first flush any buffered output, so ftell() will give us a valid answer...
	 */
	flush_output(cinfop);

	HDR_P(ifp)->jpeg_size_offset = ftell(((my_dest_ptr)cinfop->dest)->outfile);

	put_long(cinfop,frameSize);
	put_long(cinfop,frameSeqNo);
#else /* ! OLD_LML_MARKER */

	/* This code will hopefully work with the new LINVS driver */

	datalen = 5*sizeof(int32_t) + 2*sizeof(short);	/* the old header */
	datalen += 2*sizeof(int32_t) + 6 * sizeof(short);	/* + extra new stuff */

	put_BE_short(cinfop,datalen);

	/* Where will we get the frame size?
	 * We may need to write the file out, check where the pointer
	 * is, then seek back and rewrite this data...
	 */
	frameNo = 0;
	frameSize = 0;
	frameSeqNo = 0;
	if( gettimeofday(&tv,NULL) < 0 ){
		perror("gettimeofday");
	}

	put_string(cinfop,"LML");
	put_short(cinfop,frameNo);
	put_long(cinfop,tv.tv_sec);
	put_long(cinfop,tv.tv_usec);

	/* remember the offset of the frameSize, so we can rewrite it when we know it!
	 * We first flush any buffered output, so ftell() will give us a valid answer...
	 */
	flush_output(cinfop);

	HDR_P(ifp)->jpeg_size_offset = ftell(((my_dest_ptr)cinfop->dest)->outfile);

	put_long(cinfop,frameSize);
	put_long(cinfop,frameSeqNo);

	/* here is the new stuff */
	put_long(cinfop,1);	/* NTSC color encoding */
	put_long(cinfop,1);	/* D1 video stream */
	put_short(cinfop,1);	/* time decimation? */
	/* the rest are just pad bytes */
	n=5;
	while(n--)
		put_short(cinfop,0);
#endif /* ! OLD_LML_MARKER */
}

static void init_jpeg_hdr(Image_File *ifp)
{
//#ifdef CAUTIOUS
//	if( FT_CODE(IF_TYPE(ifp)) != IFT_JPEG && FT_CODE(IF_TYPE(ifp)) != IFT_LML ){
//		sprintf(DEFAULT_ERROR_STRING,
//		"CAUTIOUS:  init_jpeg_hdr:  file %s should be type jpeg or lml!?",
//			ifp->if_name);
//		NERROR1(DEFAULT_ERROR_STRING);
//	}
//#endif /* CAUTIOUS */
	assert( FT_CODE(IF_TYPE(ifp)) == IFT_JPEG || FT_CODE(IF_TYPE(ifp)) == IFT_LML );

	HDR_P(ifp)->jpeg_comps = 0;
	HDR_P(ifp)->jpeg_width = 0;
	HDR_P(ifp)->jpeg_height = 0;
	HDR_P(ifp)->jpeg_frames = 0;
	HDR_P(ifp)->jpeg_size_offset = 0;
	HDR_P(ifp)->jpeg_last_offset = 0;
	HDR_P(ifp)->jpeg_seek_table = NULL;

	HDR_P(ifp)->lml_sec = 0;
	HDR_P(ifp)->lml_usec = 0;
	HDR_P(ifp)->lml_frameSize = 0;
	HDR_P(ifp)->lml_frameSeqNo = 0;
	HDR_P(ifp)->lml_colorEncoding = 1;	/* NTSC */
	HDR_P(ifp)->lml_videoStream = 1;		/* D1 */
	HDR_P(ifp)->lml_timeDecimation = 1;	/* no decimation */
	HDR_P(ifp)->lml_frameNo = 0;
	HDR_P(ifp)->lml_fieldNo = 0;
} /* end init_jpeg_hdr */

static Image_File *finish_jpeg_open(QSP_ARG_DECL  Image_File *ifp)
{
	file_offset = 0; /* BUG? should this be per file?  is it even used? */

	ifp->if_hdr_p = getbuf( sizeof(Jpeg_Hdr) );

	if( IS_READABLE(ifp) ){
		struct jpeg_decompress_struct *cip;

		cip = &(HDR_P(ifp)->u.d_cinfo);

		/* initialize the decompression object */

		/* Initialize the JPEG decompression object
		 * with default error handling.
		 */
		cip->err = jpeg_std_error(&(HDR_P(ifp)->jerr));

		/* This is throwing a message about version number - why? */
		jpeg_create_decompress(cip);

		/* Add some application-specific error messages (from cderror.h) */
		HDR_P(ifp)->jerr.addon_message_table = cdjpeg_message_table;
		HDR_P(ifp)->jerr.first_addon_message = JMSG_FIRSTADDONCODE;
		HDR_P(ifp)->jerr.last_addon_message = JMSG_LASTADDONCODE;

		/* Insert custom marker processor for COM and APP12.
		 * APP12 is used by some digital camera makers for
		 * textual info, so we provide the ability to display
		 * it as text.  If you like, additional APPn marker
		 * types can be selected for display, but don't try
		 * to override APP0 or APP14 this way (see libjpeg.doc).
		 */
		jpeg_set_marker_processor(cip, JPEG_COM, print_text_marker);
		jpeg_set_marker_processor(cip, JPEG_APP0+12, print_text_marker);
		jpeg_set_marker_processor(cip, JPEG_APP0+3, process_lml_marker);

		/* Now safe to enable signal catcher. */
#ifdef NEED_SIGNAL_CATCHER
		enable_signal_catcher((j_common_ptr) cip);
#endif

		/* Specify data source for decompression */
		jpeg_stdio_src(cip, ifp->if_fp);

#ifdef PROGRESS_REPORT
		start_progress_monitor((j_common_ptr) cip, &HDR_P(ifp)->progress);
#endif

		if( rd_jpeg_hdr( ifp ) < 0 ){
			jpeg_close(QSP_ARG  ifp);
			return(NO_IMAGE_FILE);
		}
		jpeg_to_dp(ifp->if_dp,ifp->if_hdr_p);

	} else {
		struct jpeg_compress_struct *cip;

		cip = &(HDR_P(ifp)->u.c_cinfo);

		init_jpeg_hdr(ifp);

		/* Initialize the JPEG compression object
		 * with default error handling.
		 */

		cip->err = jpeg_std_error(&(HDR_P(ifp)->jerr));
		jpeg_create_compress(cip);
		/* Add some application-specific error messages (from cderror.h) */
		HDR_P(ifp)->jerr.addon_message_table = cdjpeg_message_table;
		HDR_P(ifp)->jerr.first_addon_message = JMSG_FIRSTADDONCODE;
		HDR_P(ifp)->jerr.last_addon_message = JMSG_LASTADDONCODE;

		/* Now safe to enable signal catcher. */
#ifdef NEED_SIGNAL_CATCHER
		enable_signal_catcher((j_common_ptr) cip);
#endif

		/* Initialize JPEG parameters.
		 * Much of this may be overridden later.
		 * In particular, we don't yet know the input file's color space,
		 * but we need to provide some value for jpeg_set_defaults() to work.
		 */

		cip->in_color_space = JCS_RGB; /* arbitrary guess */
		jpeg_set_defaults(cip);

#ifdef PROGRESS_REPORT
		start_progress_monitor((j_common_ptr) cip, &HDR_P(ifp)->progress);
#endif

		/* there is more stuff to do, but we defer until
		 * we know something about the input data...
		 */

	}
	return(ifp);
}

FIO_OPEN_FUNC( jpeg )
{
	Image_File *ifp;

	ifp = IMG_FILE_CREAT(name,rw,FILETYPE_FOR_CODE(IFT_JPEG));
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	return( finish_jpeg_open(QSP_ARG  ifp) );
}

FIO_OPEN_FUNC( lml )
{
	Image_File *ifp;

	ifp = IMG_FILE_CREAT(name,rw,FILETYPE_FOR_CODE(IFT_LML));
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	return( finish_jpeg_open(QSP_ARG  ifp) );
}

static void complete_compressor_setup(Image_File *ifp)
{
	struct jpeg_compress_struct *cip;

	cip = &(HDR_P(ifp)->u.c_cinfo);

	/* Now that we know input colorspace, fix colorspace-dependent defaults */
	jpeg_default_colorspace(cip);

	/* Adjust default compression parameters by re-parsing the options */
	install_cjpeg_params(cip);

	/* Specify data destination for compression */
	jpeg_stdio_dest(cip, ifp->if_fp);

	cip->write_JFIF_header = FALSE;	/* we might want to do this sometimes??? */

	/* the marker writer get's initialized by start_compress,
	 * so we can't do this here!?
	 */

	/*
	orig_header_method = cip->marker->write_file_header;
	cip->marker->write_file_header = write_LML_file_header;
	*/
}

FIO_CLOSE_FUNC( lml )
{
	jpeg_close(QSP_ARG  ifp);
}

FIO_CLOSE_FUNC( jpeg )
{
	/* First do the jpeg library cleanup */
	if( IS_READABLE(ifp) ){
		jpeg_destroy_decompress(&(HDR_P(ifp)->u.d_cinfo));
#ifdef PROGRESS_REPORT
		end_progress_monitor((j_common_ptr) &(HDR_P(ifp)->u.d_cinfo) );
#endif
	} else {
		jpeg_destroy_compress(&(HDR_P(ifp)->u.c_cinfo));
#ifdef PROGRESS_REPORT
		end_progress_monitor((j_common_ptr) &(HDR_P(ifp)->u.c_cinfo) );
#endif
	}

	if( HDR_P(ifp)->jpeg_seek_table != NULL )
		givbuf(HDR_P(ifp)->jpeg_seek_table);
	if( ifp->if_hdr_p != NULL )
		givbuf(ifp->if_hdr_p);
	GENERIC_IMGFILE_CLOSE(ifp);
}

FIO_RD_FUNC( lml )
{
	jpeg_rd(QSP_ARG  dp, ifp, x_offset, y_offset, t_offset );
}

FIO_RD_FUNC( jpeg )
{
	JDIMENSION num_scanlines;
	JSAMPLE *data_ptr;
	struct jpeg_decompress_struct	*cip;

	/* make sure that the sizes match */
	if( ! dp_same_dim(QSP_ARG  dp,ifp->if_dp,0,"jpeg_rd") ) return;
	if( ! dp_same_dim(QSP_ARG  dp,ifp->if_dp,1,"jpeg_rd") ) return;
	if( ! dp_same_dim(QSP_ARG  dp,ifp->if_dp,2,"jpeg_rd") ) return;

	jpeg_ifp = ifp;		/* so LML marker processor can xfer data to header */
	cip = &(HDR_P(ifp)->u.d_cinfo);

	HDR_P(ifp)->lml_fieldNo ++;	/* lml_marker_processor will zero this */

	/* Read file header, set default decompression parameters */
	(void) jpeg_read_header(cip, TRUE);

	/* Initialize the output module now to let it override any crucial
	 * option settings (for instance, GIF wants to force color quantization).
	 */

	/* This is where we need to copy in the user's custom settings... */

	/* Why is this commented out??? */
	/*
	install_djpeg_params(cip);
	*/

#ifdef QUIP_DEBUG
	if( debug & jpeg_debug ){
		/* BUG should set trace level to something? */
		(cip)->err->trace_level++;
		(cip)->err->trace_level++;
		(cip)->err->trace_level++;
	} else {
		(cip)->err->trace_level=0;
	}
#endif /* QUIP_DEBUG */


	/* Start decompressor */
	(void) jpeg_start_decompress(cip);

	/* Process data */

	/* BUG should check that dp is the correct size... */

	data_ptr = (JSAMPLE *)OBJ_DATA_PTR(dp);
	while (cip->output_scanline < cip->output_height) {
		num_scanlines = jpeg_read_scanlines(cip, &data_ptr,1);
		// suppress compiler warning
		if( num_scanlines < 1 ){
			WARN("Non-positive number of scanlines from jpeg_read_scanlines!?");
		}
		data_ptr += OBJ_ROW_INC(dp);
	}

#ifdef PROGRESS_REPORT
	/* Hack: count final pass as done in case finish_output does an extra pass.
	 * The library won't have updated completed_passes.
	 */
	HDR_P(ifp)->progress.pub.completed_passes
		= HDR_P(ifp)->progress.pub.total_passes;
#endif

	(void) jpeg_finish_decompress(cip);

	ifp->if_nfrms ++;

	if( FILE_FINISHED(ifp) ){
		if( verbose ){
			sprintf(DEFAULT_ERROR_STRING,
				"closing file \"%s\" after reading %d frames",
				ifp->if_name,ifp->if_nfrms);
			NADVISE(DEFAULT_ERROR_STRING);
		}
		jpeg_close(QSP_ARG  ifp);
		/* (*ft_tbl[ifp->if_type].close_func)(ifp); */
	}
}

/* Just like jpeg_wt, but we insist on the parameters being appropriate for LML33 */

FIO_WT_FUNC( lml )
{
	struct jpeg_compress_struct	*cinfop;
	int ci;
	int hfactor[3]={2,1,1},vfactor[3]={1,1,1};
	int oops=0;

	cinfop = &(HDR_P(ifp)->u.c_cinfo);

	/* first check image sizes */

	if( OBJ_COMPS(dp) != 3 ) {
		sprintf(DEFAULT_ERROR_STRING,
			"Number of components (%d) of image %s must be 3 for an LML file",
			OBJ_COMPS(dp),OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( OBJ_ROWS(dp) != 240 ){
		sprintf(DEFAULT_ERROR_STRING,
			"Number of rows (%d) of image %s must be 240 for an LML file",
			OBJ_ROWS(dp),OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( OBJ_COLS(dp) != 720 ){
		sprintf(DEFAULT_ERROR_STRING,
			"Number of columns (%d) of image %s must be 720 for an LML file",
			OBJ_COLS(dp),OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

	/* now make sure the compressor sample factors are correct */
	oops=0;
	for(ci=0;ci<3;ci++){
		if( cinfop->comp_info[ci].h_samp_factor != hfactor[ci] ){
			sprintf(DEFAULT_ERROR_STRING,
	"file %s:  resetting horizontal sampling factor for component %d from %d to %d",
				ifp->if_name,
				ci,cinfop->comp_info[ci].h_samp_factor,hfactor[ci]);
			NADVISE(DEFAULT_ERROR_STRING);
			oops++;
		}
		if( cinfop->comp_info[ci].v_samp_factor != vfactor[ci] ){
			sprintf(DEFAULT_ERROR_STRING,
	"file %s:  resetting vertical sampling factor for component %d from %d to %d",
				ifp->if_name,
				ci,cinfop->comp_info[ci].v_samp_factor,vfactor[ci]);
			NADVISE(DEFAULT_ERROR_STRING);
			oops++;
		}
	}
	if( oops ){
		set_my_sample_factors(hfactor,vfactor);
		install_cjpeg_params(cinfop);
	}

	return( jpeg_wt(QSP_ARG  dp,ifp) );
}

FIO_WT_FUNC( jpeg )
{
	struct jpeg_compress_struct	*cinfop;
	JSAMPLE *data_ptr;		/* what is JSAMPLE?  better be char... */

	cinfop = &(HDR_P(ifp)->u.c_cinfo);

	cinfop->image_height = OBJ_ROWS(dp);
	cinfop->image_width = OBJ_COLS(dp);
	cinfop->input_components = OBJ_COMPS(dp);	/* BUG make sure 3 or 1 */
	cinfop->num_components = OBJ_COMPS(dp);	/* BUG make sure 3 or 1 */

	if( ifp->if_dp == NO_OBJ ){	/* first time? */
		if( OBJ_COMPS(dp) == 1 ){
			cinfop->in_color_space = JCS_GRAYSCALE;
		} else if( OBJ_COMPS(dp) == 3 ){
			cinfop->in_color_space = JCS_RGB;
		} else {
			sprintf(DEFAULT_ERROR_STRING,
	"Object %s has bad number of components (%d) for jpeg",
				OBJ_NAME(dp),OBJ_COMPS(dp));
			NWARN(DEFAULT_ERROR_STRING);
			return(-1);
		}
		if( OBJ_PREC(dp) != PREC_BY && OBJ_PREC(dp) != PREC_UBY ){
			sprintf(DEFAULT_ERROR_STRING,"jpeg_wt:  image %s (%s) should have %s or %s precision",
				OBJ_NAME(dp),PREC_NAME(OBJ_PREC_PTR(dp)),
				PREC_NAME(PREC_FOR_CODE(PREC_BY)),
				PREC_NAME(PREC_FOR_CODE(PREC_UBY)) );
			NWARN(DEFAULT_ERROR_STRING);
			return(-1);
		}
		/* ifp->if_dp = dp; */
		setup_dummy(ifp);
		copy_dimensions(ifp->if_dp, dp);
		/* BUG?  where do we set the desired number of frames? */


		complete_compressor_setup(ifp);
	} else {
		/* BUG need to make sure that this image matches if_dp */
	}

	if( OBJ_COMPS(dp) == 1 && cinfop->jpeg_color_space != JCS_GRAYSCALE ){
		sprintf(DEFAULT_ERROR_STRING,
	"Object %s has one component, should use grayscale JPEG colorspace",
			OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	/* BUG?  We might like to support other jpeg colorspaces, e.g. RGB */
	if( OBJ_COMPS(dp) == 3 && cinfop->jpeg_color_space != JCS_YCbCr ){
		sprintf(DEFAULT_ERROR_STRING,
	"Object %s has 3 components, should use YCbCr JPEG colorspace",
			OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}


	/* In order to play back an mjpeg file on the LML board,
	 * the jpeg stream needs to have an APP3 marker to tell
	 * the driver the frameSize...
	 */

	/* Start compressor */
	jpeg_start_compress(cinfop, TRUE);

	if( IS_LML(ifp) && HDR_P(ifp)->lml_fieldNo == 0 )
		write_LML_file_header(ifp);

	/* Do we ever call write_file_header() or write_frame_header() ?
	 * The library code (jcmarker.c) says that we need to call emit_marker
	 * relative to these...
	 */

	/* write_marker_header() adds 2 to datalen, so that the len includes
	 * the 2 bytes for the length itself...
	 */

	/* Process data */
	data_ptr = (JSAMPLE *)OBJ_DATA_PTR(dp);
	while (cinfop->next_scanline < cinfop->image_height) {
		/* determine the input data address */
		(void) jpeg_write_scanlines(cinfop, &data_ptr, 1);
		data_ptr += OBJ_ROW_INC(dp);
	}

	/* Finish compression and release memory */
	jpeg_finish_compress(cinfop);

	if( IS_LML(ifp) )
		HDR_P(ifp)->lml_fieldNo ++;

	if( HDR_P(ifp)->lml_fieldNo > 1 ){
		size_t current_offset;
		int32_t frameSize;
		my_dest_ptr dest = (my_dest_ptr) cinfop->dest;

		HDR_P(ifp)->lml_fieldNo =0;	/* reset field counter */

		/* now fix the frame size */

		/* we don't flush the output here, because jpeg_finish_compress()
		 * did it; you wouldn't know that from looking at the dest object,
		 * though, because it doesn't reset n_free_in_buffer!
		 */

		current_offset = ftell(dest->outfile);

		frameSize = current_offset - HDR_P(ifp)->jpeg_last_offset;

		if( fseek(dest->outfile,HDR_P(ifp)->jpeg_size_offset,SEEK_SET) < 0 ){
			perror("fseek");
			NWARN("error seeking to frameSize");
			return(-1);
		}

		/* we don't want to use put_long() here, because we don't want
		 * to use the jpeg lib's i/o stuff...  it's been made invalid
		 * by jpeg_finish_compress!
		 */

		_put_long(dest->outfile,frameSize);
		fflush(dest->outfile);

		if( fseek(dest->outfile,current_offset,SEEK_SET) < 0 ){
			perror("fseek");
			NWARN("error seeking back to end of frame");
			return(-1);
		}
		HDR_P(ifp)->jpeg_last_offset = current_offset;
	}

	ifp->if_nfrms ++;
	if( ifp->if_nfrms == ifp->if_frms_to_wt ){
		if( verbose ){
	sprintf(ERROR_STRING, "closing file \"%s\" after writing %d frames",
			ifp->if_name,ifp->if_nfrms);
			NADVISE(ERROR_STRING);
		}
		close_image_file(QSP_ARG  ifp);
	}
	return(0);
}

FIO_INFO_FUNC( jpeg )
{
	// This is called after the default info func is called?

	// print any jpeg-specific information here...

	//WARN("jpeg_info_func:  not implemented!?");
}

FIO_INFO_FUNC( lml )
{
	int32_t ms;

	sprintf(msg_str,"\tField %d",HDR_P(ifp)->lml_fieldNo);
	prt_msg(msg_str);

	sprintf(msg_str,"\t%s",ctime(&HDR_P(ifp)->lml_sec));
	prt_msg_frag(msg_str);	/* ctime appends a newline */

	/*
	sprintf(msg_str,"\t%d secs",HDR_P(ifp)->lml_sec);
	prt_msg(msg_str);
	*/

	ms = floor(HDR_P(ifp)->lml_usec/1000.0);
	sprintf(msg_str,"\t%d ms.",ms);
	prt_msg(msg_str);

	sprintf(msg_str,"\tsize=%d",HDR_P(ifp)->lml_frameSize);
	prt_msg(msg_str);

	sprintf(msg_str,"\tseq=%d, no=%d",HDR_P(ifp)->lml_frameSeqNo,HDR_P(ifp)->lml_frameNo);
	prt_msg(msg_str);

	if( HDR_P(ifp)->lml_colorEncoding == 1 ){
		sprintf(msg_str,"\tcolor encoding:  NTSC");
	} else if( HDR_P(ifp)->lml_colorEncoding == 2 ){
		sprintf(msg_str,"\tcolor encoding:  PAL");
	} else if( HDR_P(ifp)->lml_colorEncoding == 3 ){
		sprintf(msg_str,"\tcolor encoding:  SECAM");
	} else {
		sprintf(msg_str,"\tunrecognized color encoding code 0x%x!?",
			HDR_P(ifp)->lml_colorEncoding);
	}
	prt_msg(msg_str);

	if( HDR_P(ifp)->lml_videoStream == 1 ){
		sprintf(msg_str,"\tvideo stream:  D1");
	} else if( HDR_P(ifp)->lml_videoStream == 3 ){
		sprintf(msg_str,"\tvideo stream:  CIF");
	} else if( HDR_P(ifp)->lml_videoStream == 4 ){
		sprintf(msg_str,"\tvideo stream:  QCIF");
	} else {
		sprintf(msg_str,"\tunrecognized video stream code 0x%x!?",
			HDR_P(ifp)->lml_videoStream);
	}
	prt_msg(msg_str);

	sprintf(msg_str,"\ttime decimation:  %d",HDR_P(ifp)->lml_timeDecimation);
	prt_msg(msg_str);
}

FIO_SEEK_FUNC( lml )
{
	return jpeg_seek_frame( QSP_ARG  ifp, n );
}

FIO_SEEK_FUNC( jpeg )
{
	seek_tbl_type *tbl;
	struct jpeg_decompress_struct	*cip;

	tbl = HDR_P(ifp)->jpeg_seek_table;
	if( tbl == NULL || fseek(ifp->if_fp,tbl[n],SEEK_SET) < 0 )
		return(-1);
	/* Now make sure that the jpeg library doesn't try to read anything that
	 * was already in it's buffer
	 */
	cip = &(HDR_P(ifp)->u.d_cinfo);
	cip->src->bytes_in_buffer = 0;

	return(0);
}

double get_lml_seconds(QSP_ARG_DECL  Image_File *ifp, dimension_t frame)
{
	if( ! IS_LML(ifp) ){
		sprintf(ERROR_STRING,"get_lml_seconds:  image file %s is not type lml, can't get timestamp",
				ifp->if_name);
		WARN(ERROR_STRING);
		return(-1.0);
	}
if( frame != 0 ) WARN("get_lml_seconds:  Sorry, don't know how to get timestamps for frames other than 0...");

	return( (double) HDR_P(ifp)->lml_sec );
}

double get_lml_milliseconds(QSP_ARG_DECL  Image_File *ifp, dimension_t frame)
{
	if( ! IS_LML(ifp) ){
		sprintf(ERROR_STRING,"get_lml_seconds:  image file %s is not type lml, can't get timestamp",
				ifp->if_name);
		WARN(ERROR_STRING);
		return(-1.0);
	}
if( frame != 0 ) WARN("get_lml_milliseconds:  Sorry, don't know how to get timestamps for frames other than 0...");

	return( (double) HDR_P(ifp)->lml_usec/1000.0 );
}

double get_lml_microseconds(QSP_ARG_DECL  Image_File *ifp, dimension_t frame)
{
	if( ! IS_LML(ifp) ){
		sprintf(ERROR_STRING,"get_lml_seconds:  image file %s is not type lml, can't get timestamp",
				ifp->if_name);
		WARN(ERROR_STRING);
		return(-1.0);
	}
if( frame != 0 ) WARN("get_lml_microseconds:  Sorry, don't know how to get timestamps for frames other than 0...");

	return( (double) HDR_P(ifp)->lml_usec );
}


#endif /* ! HAVE_JPEG_SUPPORT */

