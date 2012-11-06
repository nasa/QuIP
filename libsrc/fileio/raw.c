#include "quip_config.h"

char VersionId_fio_raw[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#include "fio_prot.h"
#include "debug.h"
#include "raw.h"
#include "savestr.h"		/* do we really need savestr??? */
#include "data_obj.h"
#include "filetype.h"
#include "uio.h"

static dimension_t raw_dim[N_DIMENSIONS]={1,0,0,0,1};
#define	raw_comps	raw_dim[0]
#define	raw_cols	raw_dim[1]
#define	raw_rows	raw_dim[2]
#define raw_frames	raw_dim[3]
#define	raw_seqs	raw_dim[4]

prec_t raw_prec=PREC_UBY;

/* BUG doesn't write multiple sequences */

void wt_raw_gaps(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp)
{
	incr_t sinc,finc,rinc,pinc,cinc;
	int size;
	char *sbase,*fbase,*rowbase,*pbase,*cbase;
	dimension_t s,f,row,col,comp;

	if( ! USES_STDIO(ifp) ){
		WARN("sorry, non-contiguous writes must use stdio");
		return;
	}
#ifdef CRAY
	if( dp->dt_prec != PREC_UBY ){
		WARN("Sorry, can only write unsigned byte images on CRAY");
		return;
	}
#endif /* CRAY */

	size=siztbl[dp->dt_prec];
	sinc = dp->dt_sinc*size;
	finc = dp->dt_finc*size;
	rinc = dp->dt_rowinc*size;
	pinc = dp->dt_pinc*size;
	cinc = dp->dt_cinc*size;

	sbase = (char *)dp->dt_data;
	for(s=0;s<dp->dt_seqs;s++){
		fbase = sbase;
		for(f=0;f<dp->dt_frames;f++){
			rowbase=fbase;
			for(row=0;row<dp->dt_rows;row++){
				pbase = rowbase;
				for(col=0;col<dp->dt_cols;col++){
					cbase=pbase;
					for(comp=0;comp<dp->dt_comps;comp++){
						/* write this pixel */
						if( fwrite(cbase,size,1,ifp->if_fp)
							!= 1 ){
					WARN("error writing pixel component");
							SET_ERROR(ifp);
							close_image_file(QSP_ARG  ifp);
							return;
						}
						cbase += cinc;
					}
					pbase += pinc;
				}
				rowbase += rinc;
			}
			fbase += finc;
		}
		sbase += sinc;
	}
}

void wt_raw_contig(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp)
{
	int n;
	dimension_t size;
	dimension_t npixels;
	int ipixels;
#ifdef FOOBAR
	u_int os;
#endif /* FOOBAR */
	int actual;

	size = siztbl[ dp->dt_prec ];
	size *= dp->dt_comps;

	npixels= dp->dt_seqs *
		 dp->dt_frames *
		 dp->dt_rows *
		 dp->dt_cols;

	/* BUG questionable cast */
	ipixels = (int) npixels;
	if( (dimension_t) ipixels != npixels )
		ERROR1("wt_raw_contig:  too much data, problem with short ints");


#ifdef CRAY
	if( dp->dt_prec == PREC_SP ){		/* convert to IEEE format */
		float *cbuf, *p;
		int goofed=0;

		n=CONV_LEN<npixels?CONV_LEN:npixels;
		cbuf = getbuf( 4 * n );		/* 4 is size of IEEE float */
		p = dp->dt_data;

		while( npixels > 0 ){
			n = CONV_LEN<npixels ? CONV_LEN : npixels ;
			cray2ieee(cbuf,p,n);
			if( USES_STDIO(ifp) ){
				if( fwrite(cbuf,4,n,ifp->if_fp) != n ){
					WARN("error #1 writing pixel data");
					goofed=1;
					goto ccdun;
				}
			} else {
				if( write(ifp->if_fd,cbuf,4*n) != 4*n ){
					WARN("error #2 writing pixel data");
					goofed=1;
					goto ccdun;
				}
			}
			p += n;
			npixels -= n;
		}
ccdun:		givbuf(cbuf);
		if( goofed ){
			SET_ERROR(ifp);
			close_image_file(ifp);
		}
		return;
	} else if( dp->dt_prec != PREC_UBY ){
		WARN("Sorry, can only write float or unsigned byte images on CRAY now...");
		return;
	}
#endif /* CRAY */
		
		


	if( USES_STDIO(ifp) ){
		if( (actual=fwrite(dp->dt_data,
			(size_t)size,(size_t)ipixels,ifp->if_fp))
			!= ipixels ){

			WARN("error #3 writing pixel data");
			sprintf(error_string,
				"%d bytes actually written",actual);
			advise(error_string);
			SET_ERROR(ifp);
			close_image_file(QSP_ARG  ifp);
		}
	} else {
		n = ipixels*size;
		if( (actual=write(ifp->if_fd,((char *)dp->dt_data),n)) != n ){
			sprintf(error_string,
				"%d bytes requested, %d bytes actually written",n,actual);
			WARN(error_string);
			SET_ERROR(ifp);
			close_image_file(QSP_ARG  ifp);
			return;
		}

	}
}

void wt_raw_data(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp)		/** output next frame */
{
	dimension_t totfrms;

#ifdef DEBUG
if( debug & debug_fileio ){
sprintf(error_string,"wt_raw_data, %s to file %s",
dp->dt_name,ifp->if_name);
advise(error_string);
}
#endif /* DEBUG */

	if( !same_type(QSP_ARG  dp,ifp) ) return;

	totfrms = dp->dt_frames * dp->dt_seqs;

	if( ifp->if_nfrms + totfrms > ifp->if_frms_to_wt ){
		sprintf(error_string,
		"Can't append object %s (%d frames) to file %s (too many frames, has %d, wants %d)",
			dp->dt_name,totfrms,ifp->if_name,
			ifp->if_nfrms,ifp->if_frms_to_wt);
		WARN(error_string);
		LONGLIST(dp);
		LONGLIST(ifp->if_dp);
		return;
	}
	if( !IS_CONTIGUOUS(dp) ){
		wt_raw_gaps(QSP_ARG  dp,ifp);
	} else {
		wt_raw_contig(QSP_ARG  dp,ifp);
	}

	/* The write functions can have an error if the file system
	 * is full; in this case, they will automatically close the file
	 * and release the ifp structure.  The memory associated
	 * with the item is not freed, it is just marked as available,
	 * therefore subsequent references to *ifp will not cause
	 * a segmentation violation.  Nevertheless, it is no longer
	 * a valid file!?  We need to check here whether an error occurred.
	 */
	
	if( HAD_ERROR(ifp) ){
#ifdef DEBUG
if( debug & debug_fileio ){
sprintf(error_string,"wt_raw_data, returning after error on file %s",
ifp->if_name);
advise(error_string);
}
#endif /* DEBUG */
		return;
	}

	ifp->if_nfrms += totfrms;

	check_auto_close(QSP_ARG  ifp);
}

FIO_WT_FUNC( raw_wt )
{

	if( ifp->if_dp == NO_OBJ ){	/* first time? */
		setup_dummy(ifp);
		copy_dimensions(ifp->if_dp, dp);
		ifp->if_dp->dt_frames = ifp->if_frms_to_wt;
		ifp->if_dp->dt_seqs = 1;
	}

	if( ifp->if_nfrms == 0 ){	/* this is the first */
		dimension_t nf;

		/* BUG what about seqs??? */
		nf = ifp->if_dp->dt_frames;
		copy_dimensions(ifp->if_dp , dp );
		ifp->if_dp->dt_frames = nf;
	}

	wt_raw_data(QSP_ARG  dp,ifp);

	return(0);
}

void set_raw_sizes( dimension_t arr[N_DIMENSIONS] )
{
	int i;

	for(i=0;i<N_DIMENSIONS;i++){
		raw_dim[i] = arr[i];
	}
}

void set_raw_prec(prec_t p)
{
	raw_prec = p;
}


/* used to pass this the ifp, but now we're makeing this uniform... */
int raw_to_dp( Data_Obj *dp, void *vp )
{
	Image_File *ifp;

	ifp = (Image_File *)vp;

	if( raw_rows <= 0 || raw_cols <= 0 )
		NWARN("size of raw image file not specified!?");
		
	dp->dt_prec = raw_prec;
	dp->dt_comps = raw_comps;
	dp->dt_cols = raw_cols;
	dp->dt_rows = raw_rows;
	if( raw_frames <= 0 ){
		/* get the number of frames by dividing the total file size by the image size */
		off_t s;

		/* first get the total file size */
		s = lseek(ifp->if_fd,0,SEEK_END);
		if( s == ((off_t)-1) ){
			tell_sys_error("raw_to_dp:  lseek");
			dp->dt_frames = 1;
		} else {
			dimension_t frm_size;

			frm_size = dp->dt_comps * dp->dt_cols * dp->dt_rows * siztbl[ MACHINE_PREC(dp) ];
			dp->dt_frames = s / frm_size;

			if( (s % frm_size) != 0 ){
				sprintf(DEFAULT_ERROR_STRING,
		"Number of bytes (%d) in raw file %s is not an integral multiple of the frame size (%d)",
					s,ifp->if_name,frm_size);
				NWARN(DEFAULT_ERROR_STRING);
			}
		}
		s = lseek(ifp->if_fd,0,SEEK_SET);
		if( s == ((off_t)-1) ){
			tell_sys_error("raw_to_dp:  lseek");
			NWARN("error rewinding file");
		}
	} else {
		dp->dt_frames = raw_frames;
	}
	dp->dt_seqs = raw_seqs;
	return 0;
}


FIO_OPEN_FUNC( raw_open )
{
	Image_File *ifp;

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_RAW);

	/* image_file_open creates dummy if_dp only if readable */

	if( ifp==NO_IMAGE_FILE ) return(ifp);

	if( rw == FILE_READ ){
		raw_to_dp(ifp->if_dp,ifp);
	}
#ifdef FOOBAR
	else {
		/* We used to set the number of frames in if_dp,
		 * here, but now we don't pass the number of frames
		 * to the open routine any more...
		 * Do we still need to set up the dummy struct???
		 */
		setup_dummy(ifp);
	}
#endif /* FOOBAR */
	return(ifp);
}

int raw_unconv( void *hd_pp, Data_Obj *dp )
{
	NWARN("raw_unconv not implemented");
	return(-1);
}

int raw_conv( Data_Obj *dp, void *hd_pp )
{
	NWARN("raw_conv not implemented");
	return(-1);
}

