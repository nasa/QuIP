#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#include "fio_prot.h"
#include "quip_prot.h"
#include "debug.h"
#include "img_file/raw.h"
//#include "savestr.h"		/* do we really need savestr??? */
#include "data_obj.h"
//#include "filetype.h"
//#include "uio.h"

static dimension_t raw_dim[N_DIMENSIONS]={1,0,0,0,1};
#define	raw_comps	raw_dim[0]
#define	raw_cols	raw_dim[1]
#define	raw_rows	raw_dim[2]
#define raw_frames	raw_dim[3]
#define	raw_seqs	raw_dim[4]

// BUG - not thread safe!?
static Precision* raw_prec_p=NULL;

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

	size=PREC_SIZE( OBJ_PREC_PTR(dp) );
	sinc = OBJ_SEQ_INC(dp)*size;
	finc = OBJ_FRM_INC(dp)*size;
	rinc = OBJ_ROW_INC(dp)*size;
	pinc = OBJ_PXL_INC(dp)*size;
	cinc = OBJ_COMP_INC(dp)*size;

	sbase = (char *)OBJ_DATA_PTR(dp);
	for(s=0;s<OBJ_SEQS(dp);s++){
		fbase = sbase;
		for(f=0;f<OBJ_FRAMES(dp);f++){
			rowbase=fbase;
			for(row=0;row<OBJ_ROWS(dp);row++){
				pbase = rowbase;
				for(col=0;col<OBJ_COLS(dp);col++){
					cbase=pbase;
					for(comp=0;comp<OBJ_COMPS(dp);comp++){
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
	size_t actual;

	size = PREC_SIZE( OBJ_PREC_PTR(dp) );
	size *= OBJ_COMPS(dp);

	npixels= OBJ_SEQS(dp) *
		 OBJ_FRAMES(dp) *
		 OBJ_ROWS(dp) *
		 OBJ_COLS(dp);

	/* BUG questionable cast */
	ipixels = (int) npixels;
	if( (dimension_t) ipixels != npixels )
		error1("wt_raw_contig:  too much data, problem with short ints");


#ifdef CRAY
	if( dp->dt_prec == PREC_SP ){		/* convert to IEEE format */
		float *cbuf, *p;
		int goofed=0;

		n=CONV_LEN<npixels?CONV_LEN:npixels;
		cbuf = getbuf( 4 * n );		/* 4 is size of IEEE float */
		p = OBJ_DATA_PTR(dp);

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
		if( (actual=fwrite(OBJ_DATA_PTR(dp),
			(size_t)size,(size_t)ipixels,ifp->if_fp))
			!= ipixels ){

			WARN("error #3 writing pixel data");
			sprintf(ERROR_STRING,
				"%ld bytes actually written",(long)actual);
			advise(ERROR_STRING);
			SET_ERROR(ifp);
			close_image_file(QSP_ARG  ifp);
		}
	} else {
		n = ipixels*size;
		if( (actual=write(ifp->if_fd,((char *)OBJ_DATA_PTR(dp)),n)) != n ){
			sprintf(ERROR_STRING,
				"%d bytes requested, %ld bytes actually written",n,(long)actual);
			WARN(ERROR_STRING);
			SET_ERROR(ifp);
			close_image_file(QSP_ARG  ifp);
			return;
		}

	}
}

void wt_raw_data(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp)		/** output next frame */
{
	dimension_t totfrms;

#ifdef QUIP_DEBUG
if( debug & debug_fileio ){
sprintf(ERROR_STRING,"wt_raw_data, %s to file %s",
OBJ_NAME(dp),ifp->if_name);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	if( !same_type(QSP_ARG  dp,ifp) ) return;

	totfrms = OBJ_FRAMES(dp) * OBJ_SEQS(dp);

	if( ifp->if_nfrms + totfrms > ifp->if_frms_to_wt ){
		sprintf(ERROR_STRING,
		"Can't append object %s (%d frames) to file %s (too many frames, has %d, wants %d)",
			OBJ_NAME(dp),totfrms,ifp->if_name,
			ifp->if_nfrms,ifp->if_frms_to_wt);
		WARN(ERROR_STRING);
		longlist(dp);
		longlist(ifp->if_dp);
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
#ifdef QUIP_DEBUG
if( debug & debug_fileio ){
sprintf(ERROR_STRING,"wt_raw_data, returning after error on file %s",
ifp->if_name);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		return;
	}

	ifp->if_nfrms += totfrms;

	check_auto_close(QSP_ARG  ifp);
}

FIO_WT_FUNC( raw )
{

	if( ifp->if_dp == NULL ){	/* first time? */
		setup_dummy(ifp);
		copy_dimensions(ifp->if_dp, dp);
		SET_OBJ_FRAMES(ifp->if_dp, ifp->if_frms_to_wt);
		SET_OBJ_SEQS(ifp->if_dp, 1);
	}

	if( ifp->if_nfrms == 0 ){	/* this is the first */
		dimension_t nf;

		/* BUG what about seqs??? */
		nf = OBJ_FRAMES(ifp->if_dp);
		copy_dimensions(ifp->if_dp , dp );
		SET_OBJ_FRAMES(ifp->if_dp, nf);
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

void set_raw_prec(Precision * prec_p)
{
	raw_prec_p = prec_p;
}


/* used to pass this the ifp, but now we're makeing this uniform... */
int raw_to_dp( Data_Obj *dp, void *vp )
{
	Image_File *ifp;

	ifp = (Image_File *)vp;

	if( raw_rows <= 0 || raw_cols <= 0 )
		NWARN("size of raw image file not specified!?");
		
	if( raw_prec_p == NULL ){
		sprintf(DEFAULT_ERROR_STRING,
			"Pixel precision for raw file %s not specified!?",
			ifp->if_name);
		NWARN(DEFAULT_ERROR_STRING);
		raw_prec_p = PREC_FOR_CODE(PREC_UBY);
		sprintf(DEFAULT_ERROR_STRING,
			"Assuming default value of %s.",PREC_NAME(raw_prec_p));
		NADVISE(DEFAULT_ERROR_STRING);
	}

	SET_OBJ_PREC_PTR(dp, raw_prec_p);
	SET_OBJ_COMPS(dp, raw_comps);
	SET_OBJ_COLS(dp, raw_cols);
	SET_OBJ_ROWS(dp, raw_rows);
	if( raw_frames <= 0 ){
		/* get the number of frames by dividing the total file size by the image size */
		off_t s;

		/* first get the total file size */
		s = lseek(ifp->if_fd,0,SEEK_END);
		if( s == ((off_t)-1) ){
			_tell_sys_error(DEFAULT_QSP_ARG  "raw_to_dp:  lseek");
			SET_OBJ_FRAMES(dp, 1);
		} else {
			dimension_t frm_size;

			frm_size = OBJ_COMPS(dp) * OBJ_COLS(dp) * OBJ_ROWS(dp) * PREC_SIZE( OBJ_MACH_PREC_PTR(dp) );
			SET_OBJ_FRAMES(dp, s / frm_size);

			if( (s % frm_size) != 0 ){
				sprintf(DEFAULT_ERROR_STRING,
		"Number of bytes (%ld) in raw file %s is not an integral multiple of the frame size (%ld)",
					(long)s,ifp->if_name,(long)frm_size);
				NWARN(DEFAULT_ERROR_STRING);
			}
		}
		s = lseek(ifp->if_fd,0,SEEK_SET);
		if( s == ((off_t)-1) ){
			_tell_sys_error(DEFAULT_QSP_ARG  "raw_to_dp:  lseek");
			NWARN("error rewinding file");
		}
	} else {
		SET_OBJ_FRAMES(dp, raw_frames);
	}
	SET_OBJ_SEQS(dp, raw_seqs);
	return 0;
}


FIO_OPEN_FUNC( raw )
{
	Image_File *ifp;

	ifp = IMG_FILE_CREAT(name,rw,filetype_for_code(QSP_ARG  IFT_RAW));

	/* img_file_creat creates dummy if_dp only if readable */

	if( ifp==NULL ) return(ifp);

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

