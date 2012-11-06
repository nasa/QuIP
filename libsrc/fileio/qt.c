
#include "quip_config.h"

char VersionId_fio_qt[] = QUIP_VERSION_STRING;

#ifdef HAVE_QUICKTIME

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>		/* memcpy */
#endif

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

#include "img_file.h"
#include "debug.h"		/* verbose */


#include "filetype.h" /* ft_tbl */

#define hdr	if_hd.qt_hd_p

/* prototypes */
static Image_File *finish_qt_open(Image_File *ifp);
void qt_close( Image_File *ifp );

static int qt_to_dp( Data_Obj *dp, Qt_Hdr *qt_hp )
{
	/* dp->dt_prec = PREC_IN; */	/* short */	/* WHY??? */
	dp->dt_prec = PREC_UBY;

	/* BUG need to set these from the header??? */
	/*
	dp->dt_tdim=qt_hp->qt_comps;
	dp->dt_cols=qt_hp->qt_width;
	dp->dt_rows=qt_hp->qt_height;
	dp->dt_frames=qt_hp->qt_frames;
	*/
	warn("Sorry, don't yet know how to determine the size of a quicktime file");

	dp->dt_seqs=1;

	dp->dt_cinc=1;
	dp->dt_pinc=1;
	dp->dt_rinc=dp->dt_pinc*dp->dt_cols;
	dp->dt_finc=dp->dt_rinc*dp->dt_rows;
	dp->dt_sinc=dp->dt_finc*dp->dt_frames;

	dp->dt_parent = NO_OBJ;
	dp->dt_children = NO_LIST;

	dp->dt_ap = ram_area;		/* the default */
	dp->dt_data = NULL;
	dp->dt_nelts = dp->dt_tdim * dp->dt_cols * dp->dt_rows
			* dp->dt_frames * dp->dt_seqs;

	set_shape_flags(&dp->dt_shape,dp);

	return(0);
}


/* the unconvert routine creates a disk header */

int qt_unconv( void *hdr_pp, Data_Obj *dp )
{
	warn("qt_unconv() not implemented!?");
	return(-1);
}

int qt_conv( Data_Obj *dp, void *hd_pp )
{
	warn("qt_conv not implemented");
	return(-1);
}

/* This is really a misnomer... MJPEG files don't have headers, they
 * are just a stream of images...  We fake this, by scanning the file,
 * counting the number of images, then we rewind the file...
 */

static int rd_qt_hdr( Image_File *ifp )
{
	/* int n; */
	long offset;
	long nf=0;
	long *tbl;

	/* initialize the header with impossible values
	 * to indicate that we don't know anything yet.
	 */

	ifp->hdr->qt_comps=(-1);
	ifp->hdr->qt_width=(-1);
	ifp->hdr->qt_height=(-1);

	offset = ftell(ifp->if_fp);
	if( verbose ){
		sprintf(error_string,"scanning file %s to count frames",ifp->if_name);
		advise(error_string);
	}

	/* While we are counting the frames, we remember their offsets so
	 * we can seek later...  We need to store the offsets in a table, but
	 * we don't know how big a table we are going to need!?  We allocate
	 * a big one, then we copy the data into a properly sized buffer
	 * when we are done.
	 *
	 * How big is big enough?  In the atc expt, we have 100 second trials (max),
	 * which translates to 6k fields...  we round up to 8k.
	 */
#define MAX_SEEK_TBL_SIZE 8192
	
	tbl = getbuf( MAX_SEEK_TBL_SIZE * sizeof(long) );

	tbl[0]=0;
	/* BUG need to scan the file??? */

	if( nf > MAX_SEEK_TBL_SIZE ){
		sprintf(error_string,
	"rd_qt_hdr:  file %s has %ld frames, which exceeds MAX_SEEK_TBL_SIZE (%d)",
			ifp->if_name,nf,MAX_SEEK_TBL_SIZE);
		warn(error_string);
		advise("Consider recompiling");
		givbuf(tbl);
		tbl=NULL;
	}
	if( nf == 0 ){
		sprintf(error_string,"qt file %s has no frames!?",ifp->if_name);
		warn(error_string);
		return(-1);
	}

	ifp->hdr->qt_frames=nf;

	/* now copy the table to the final resting place */ 
	if( tbl != NULL ){
		ifp->hdr->qt_seek_table = getbuf( nf * sizeof(long) );
		memcpy(ifp->hdr->qt_seek_table,tbl,nf*sizeof(long));
		givbuf(tbl);
	} else {
		ifp->hdr->qt_seek_table = NULL;
	}

	rewind(ifp->if_fp);
	return(0);
}

static void init_qt_hdr(Image_File *ifp)
{
#ifdef CAUTIOUS
	if( ifp->if_type != IFT_JPEG && ifp->if_type != IFT_LML ){
		sprintf(error_string,
		"CAUTIOUS:  init_qt_hdr:  file %s should be type qt or lml!?",
			ifp->if_name);
		error1(error_string);
	}
#endif /* CAUTIOUS */

	ifp->hdr->qt_comps = 0;
	ifp->hdr->qt_width = 0;
	ifp->hdr->qt_height = 0;
	ifp->hdr->qt_frames = 0;
	ifp->hdr->qt_size_offset = 0;
	ifp->hdr->qt_last_offset = 0;
	ifp->hdr->qt_seek_table = NULL;
}

Image_File * qt_open(const char *name,int rw)
{
	Image_File *ifp;

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_JPEG);
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	return( finish_qt_open(ifp) );
}

static Image_File *finish_qt_open(Image_File *ifp)
{
	u_long file_offset;

	file_offset = 0; /* BUG? should this be per file?  is it even used? */

	ifp->hdr = (Qt_Hdr *)getbuf( sizeof(Qt_Hdr) );

	if( IS_READABLE(ifp) ){
		/* need to do some quicktime stuff here! */

		if( rd_qt_hdr( ifp ) < 0 ){
			qt_close(ifp);
			return(NO_IMAGE_FILE);
		}
		qt_to_dp(ifp->if_dp,ifp->hdr);

	} else {
		warn("oops, don't know how to write quicktime");
	}
	return(ifp);
}


void qt_close( Image_File *ifp )
{
	/* First do the qt library cleanup */

	if( ifp->hdr->qt_seek_table != NULL )
		givbuf(ifp->hdr->qt_seek_table);
	if( ifp->hdr != NULL )
		givbuf(ifp->hdr);
	GENERIC_IMGFILE_CLOSE(ifp);
}

void qt_rd( Data_Obj *dp, Image_File *ifp, index_t x_offset, index_t y_offset,index_t t_offset )
{
	/* make sure that the sizes match */
	if( ! dp_same_dim(dp,ifp->if_dp,0) ) return;
	if( ! dp_same_dim(dp,ifp->if_dp,1) ) return;
	if( ! dp_same_dim(dp,ifp->if_dp,2) ) return;

	/* Process data */
	/* BUG should check that dp is the correct size... */

	/*
	data_ptr = dp->dt_data;
	while (cip->output_scanline < cip->output_height) {
		num_scanlines = qt_read_scanlines(cip, &data_ptr,1);
		data_ptr += dp->dt_rowinc;
	}
	*/

	warn("need to implement quicktime read function!");

	ifp->if_nfrms ++;

	if( FILE_FINIHED(ifp) ){
		if( verbose ){
			sprintf(error_string,
				"closing file \"%s\" after reading %ld frames",
				ifp->if_name,ifp->if_nfrms);
			advise(error_string);
		}
		qt_close(ifp);
		/* (*ft_tbl[ifp->if_type].close_func)(ifp); */
	}
}

int qt_wt( Data_Obj *dp, Image_File *ifp )
{
	warn("Sorry, can't write quicktime files yet");
	return(-1);
}

void qt_info( Image_File *ifp )
{
	/* print any quicktime-specific data here */
	warn("Sorry, don't know how to print quicktime-specific data yet");
}

int qt_seek_frame( Image_File *ifp, index_t n )
{
	long *tbl;

	tbl = ifp->hdr->qt_seek_table;
	if( tbl == NULL || fseek(ifp->if_fp,tbl[n],SEEK_SET) < 0 )
		return(-1);

	return(0);
}

#endif /* ! HAVE_QUICKTIME */

