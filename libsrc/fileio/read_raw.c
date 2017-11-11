#include "quip_config.h"

#include <stdio.h>
#include <fcntl.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif /* HAVE_UNISTD_H */

#include "quip_prot.h"
#include "fio_prot.h"
#include "img_file/raw.h"
#include "data_obj.h"

//#include "uio.h"

#define TRASH_BUF_SIZE	0x2000		/* 8k */

static char write_only_buffer[TRASH_BUF_SIZE];

/* We used to read() in chunks - WHY???
 * The code had a bug, it counted pixels and assumed that the number of bytes
 * in a chunk was evenly divisible by the pixel size - but for RGB byte images the size
 * is 3 - so CHUNK_SIZE needs to be divisible by 3!?
 *
 * I don't really see why we need to read in chunks at all!!?
 * Maybe a hack for the old dos port?
 */
#define CHUNK_SIZE	0x7000			/* a kluge */


#define TALL_TOLD	(ifp->if_flags&FILE_TALL)
#define SHORT_TOLD	(ifp->if_flags&FILE_SHORT)
#define THICK_TOLD	(ifp->if_flags&FILE_THICK)
#define THIN_TOLD	(ifp->if_flags&FILE_THIN)

#define TELL_TALL	ifp->if_flags |= FILE_TALL
#define TELL_SHORT	ifp->if_flags |= FILE_SHORT
#define TELL_THICK	ifp->if_flags |= FILE_THICK
#define TELL_THIN	ifp->if_flags |= FILE_THIN

void rd_raw_gaps(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp)
{
	incr_t sinc,finc,rinc,pinc,cinc;
	int size;
	char *sbase,*fbase,*rowbase,*pbase,*cbase;
	dimension_t s,f,row,col,comp;

	if( ! USES_STDIO(ifp) ){
		WARN("sorry, non-contiguous reads must use stdio");
		return;
	}

	size=PREC_SIZE(OBJ_PREC_PTR(dp));
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
						if( fread(cbase,size,1,ifp->if_fp)
							!= 1 ){
					WARN("error reading pixel component");
					SET_ERROR(ifp);
					(*FT_CLOSE_FUNC(IF_TYPE(ifp)))(QSP_ARG  ifp);
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

void _read_object(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp)
{
	dimension_t n, size;
	size_t n2;
	uint32_t npixels;
#ifdef CRAY
	int goofed=0;
#endif /* CRAY */

	if( !same_type(dp,ifp) ) return;

	/* this wants nframes the same too!?
	 * same_size() checks only rows & cols
	 */
	if( !same_size(QSP_ARG  dp,ifp) ) return;

	if( !IS_CONTIGUOUS(dp) ){
		rd_raw_gaps(QSP_ARG  dp,ifp);
		return;
	}

	npixels=OBJ_SEQS(dp) * OBJ_FRAMES(dp) * OBJ_ROWS(dp) * OBJ_COLS(dp) ;

	size = PREC_SIZE(OBJ_PREC_PTR(dp));
	size *= OBJ_COMPS(dp);

#ifdef CRAY
	/* CRAY float's are 8 bytes instead of 4,
	 * so we have to convert from IEEE format
	 */

	if( OBJ_PREC(dp) == PREC_SP ){
		float *cbuf, *p;

		n=CONV_LEN<npixels?CONV_LEN:npixels;
		cbuf = getbuf( 4 * n );		/* 4 is size of IEEE float */
		p = OBJ_DATA_PTR(dp);

		while( npixels > 0 ){
			n = CONV_LEN<npixels ? CONV_LEN : npixels ;
			if( USES_STDIO(ifp) ){
				if( (n2=fread(cbuf,4,n,ifp->if_fp)) != n ){
					sprintf(ERROR_STRING,
				"read_object %s from file %s:  %d pixels requested, %d pixels read",
						OBJ_NAME(dp),ifp->if_name,n,n2);
					WARN(ERROR_STRING);
					goofed=1;
					goto ccdun;
				}
			} else {
				// hips1 etc
				if( (n2=read(ifp->if_fd,cbuf,4*n)) != 4*n ){
					sprintf(ERROR_STRING,
				"read_object %s from file %s:  %d bytes requested, %d bytes read",
						OBJ_NAME(dp),ifp->if_name,4*n,n2);
					WARN(ERROR_STRING);
					goofed=1;
					goto ccdun;
				}
			}

			ieee2cray(p,cbuf,n);

			p += n;
			npixels -= n;
		}
ccdun:		givbuf(cbuf);
		if( goofed ) (*FT_CLOSE_FUNC(IF_TYPE(ifp)))(ifp);
dun2:		return;
	} else if( OBJ_PREC(dp) != PREC_UBY ){
		WARN("Sorry, can only read float or unsigned byte images on CRAY now...");
		goto dun2;
	}
#endif /* CRAY */
		
		
	if( USES_STDIO(ifp) ){
		if( (n2=fread(OBJ_DATA_PTR(dp),(size_t)size,(size_t)npixels,ifp->if_fp))
			!= (size_t)npixels ){

			sprintf(ERROR_STRING,
		"read_object %s from file %s:  %d pixels requested, but %ld pixels read",
				OBJ_NAME(dp),ifp->if_name,npixels,(long)n2);
			WARN(ERROR_STRING);
		}
	} else {
#ifdef FOOBAR
		uint32_t os;

		n=CHUNK_SIZE;
		os=0;
		while( npixels > 0 ){
			int n_actual;

			if( npixels*size < n )
				n=(npixels*size);

			/* BUG these casts are done for PC,
			 * should check for data loss!?
			 */

sprintf(ERROR_STRING,"%d pixels remaining, requesting %d bytes at offset %d (size=%d)",npixels,n,os,size);
advise(ERROR_STRING);
			if( (n_actual=read(ifp->if_fd,((char *)OBJ_DATA_PTR(dp))+os,(u_int)n))
				!= (int)n ){
				sprintf(ERROR_STRING,
					"error read()'ing pixel data, %d requested, %d actually read",n,n_actual);
				WARN(ERROR_STRING);
			} else {
				sprintf(ERROR_STRING,"%d pixels read successfully",n);
				advise(ERROR_STRING);
			}
			os += n;
			npixels -= n/size;
		}
#endif /* FOOBAR */

		size_t n_actual;

		n=npixels*size;
		if( (n_actual=read(ifp->if_fd,((char *)OBJ_DATA_PTR(dp)),(u_int)n)) != (int) n ){
			sprintf(ERROR_STRING,
				"error reading pixel data, %d requested, %ld actually read",n,(long)n_actual);
			WARN(ERROR_STRING);
		}
	}
}



/* read an image which is too small or too big for the target.
 * The offsets are offsets into the target data object.
 */

#define frag_read(dp,ifp,x_offset,y_offset,t_offset) _frag_read(QSP_ARG  dp,ifp,x_offset,y_offset,t_offset)

static int _frag_read(QSP_ARG_DECL  Data_Obj *dp,Image_File *ifp,index_t x_offset,index_t y_offset,index_t t_offset)
{
	dimension_t x_fill, y_fill;	/* number of cols,rows to draw */
	dimension_t dx,dy;		/* dimensions of input file */
	dimension_t x_skip;		/* extra data cols to right of image */
	dimension_t x_dump, y_dump;	/* data to read and throw away */
	dimension_t i;
	char *p;
	dimension_t size, n_elements_to_discard, n_trash_elements_per_buf;

	size = PREC_SIZE(OBJ_PREC_PTR(dp));
	size *= OBJ_COMPS(dp);
	x_fill=(OBJ_COLS(dp)-x_offset);
	y_fill=(OBJ_ROWS(dp)-y_offset);
	if( x_fill <= 0 || y_fill <= 0 ){
		warn("frag_read:  offset too great for this object");
		return(-1);
	}
	dx=OBJ_COLS(ifp->if_dp);
	dy=OBJ_ROWS(ifp->if_dp);
	if( dx > x_fill ){	/* image wider than data area */
		//x_skip=x_fill-dx;
		x_dump=dx-x_fill;
		//x_skip=0;
if( ! THICK_TOLD ){
sprintf(ERROR_STRING,"image in file %s too wide (%d) for object %s (%d)",
ifp->if_name,dx,OBJ_NAME(dp),x_fill);
warn(ERROR_STRING);
/*
//sprintf(ERROR_STRING,"xskip = %d    x_fill = %d    dx = %d    x_dump = %d\n",x_skip,x_fill,dx,x_dump);
//advise(ERROR_STRING);
*/
TELL_THICK;
}
	} else {		/* image thinner that data area? */

if( dx < x_fill && (! THIN_TOLD) ){
sprintf(ERROR_STRING,"image in file %s too thin (%d) for object %s (%d)",
ifp->if_name,dx,OBJ_NAME(dp),x_fill);
warn(ERROR_STRING);
TELL_THIN;
}
		//x_skip=x_fill-dx;
		x_fill=dx;
		x_dump=0;
	}
	if( dy > y_fill ){	/* image taller than data area */

if( ! TALL_TOLD ){
sprintf(ERROR_STRING,"image in file %s too tall (%d) for object %s (%d)",
ifp->if_name,dy,OBJ_NAME(dp),y_fill);
warn(ERROR_STRING);
TELL_TALL;
}
		x_skip=x_fill-dx;
		y_dump=dy-y_fill;
	} else {		/* image shorter than data area */

if( dy < y_fill && (! SHORT_TOLD) ){
sprintf(ERROR_STRING,"image in file %s too short (%d) for object %s (%d)",
ifp->if_name,dy,OBJ_NAME(dp),y_fill);
warn(ERROR_STRING);
TELL_SHORT;
}
		x_skip=x_fill-dx;
		y_fill=dy;
		y_dump=0;
	}
	p=(char *)OBJ_DATA_PTR(dp) + t_offset;
	p += y_offset * size * OBJ_COLS(dp);

	n_trash_elements_per_buf = TRASH_BUF_SIZE/(OBJ_COMPS(ifp->if_dp)*ELEMENT_SIZE(ifp->if_dp));

	for(i=0;i<y_fill;i++){
		p += (x_offset*size);
		if( USES_STDIO(ifp) ){
			if( fread(p,(size_t)size,(size_t)x_fill,ifp->if_fp)
				!= (size_t)x_fill )
				return(-1);
			/* now read the trash... */
			n_elements_to_discard = x_dump;
			do {
				dimension_t n_to_read;
				if( n_elements_to_discard > n_trash_elements_per_buf )
					n_to_read = n_trash_elements_per_buf;
				else
					n_to_read = n_elements_to_discard;

				if( fread(write_only_buffer,(size_t)size,(size_t)n_to_read,ifp->if_fp) != (size_t)n_to_read )
					return(-1);

				n_elements_to_discard -= n_to_read;
			} while( n_elements_to_discard > 0 );
		} else {
			/* BUG casting for pc */
			if( read(ifp->if_fd,p,(u_int)(size*x_fill))
				!= (int)(size*x_fill) )
				return(-1);
			/* now read the trash... */
			n_elements_to_discard = x_dump;
			do {
				dimension_t n_to_read;
				if( n_elements_to_discard > n_trash_elements_per_buf )
					n_to_read = n_trash_elements_per_buf;
				else
					n_to_read = n_elements_to_discard;

				if( read(ifp->if_fd,write_only_buffer,(u_int)(size*n_to_read)) != (int)(size*n_to_read) )
					return(-1);
				n_elements_to_discard -= n_to_read;
			} while( n_elements_to_discard > 0 );
		}
		p += ( (x_fill+x_skip) *size);
	}
	for(i=0;i<y_dump;i++)
		assert(dx > n_trash_elements_per_buf );
		/* BUG need to limit read size to write_only_buffer */
		if( USES_STDIO(ifp) ){
			if( fread(write_only_buffer,(size_t)size,(size_t)dx,ifp->if_fp)
				!= (size_t)dx )
				return(-1);
		} else {
			if( read(ifp->if_fd,write_only_buffer,(u_int)(size*dx))
				!= (int)(size*dx) )
				return(-1);
		}

	return(0);
}

FIO_RD_FUNC( raw )
{
	uint32_t totfrms;

	if( !same_type(dp,ifp) ) return;

	if( t_offset >= OBJ_FRAMES(dp) ){
		sprintf(ERROR_STRING,
			"raw_rd:  ridiculous frame offset %d (max %d)",
			t_offset,OBJ_FRAMES(dp)-1);
		WARN(ERROR_STRING);
		return;
	}

	t_offset *=	( OBJ_ROWS(dp)
			 * OBJ_COLS(dp)
			 * OBJ_COMPS(dp)
			 * PREC_SIZE(OBJ_PREC_PTR(dp)) );


	totfrms = OBJ_FRAMES(dp) * OBJ_SEQS(dp);

	if( 	OBJ_ROWS(dp)==OBJ_ROWS(ifp->if_dp) &&
		OBJ_COLS(dp)==OBJ_COLS(ifp->if_dp) &&
		x_offset==0 && y_offset==0 ){

		read_object(dp,ifp);
	} else {
		if( frag_read(dp,ifp,x_offset,y_offset,t_offset) < 0 )
			goto readerr;
	}

	ifp->if_nfrms += totfrms;

	/* We added the NO_AUTO_CLOSE flag so we could reverse
	 * a movie by reading the last frame first...
	 * BUG we need to add this functionality to the other filetypes.
	 */

	if( FILE_FINISHED(ifp) ){

		if( verbose ){
			sprintf(ERROR_STRING,
				"closing file \"%s\" after reading %d frames",
				ifp->if_name,ifp->if_nfrms);
			advise(ERROR_STRING);
		}
		(*FT_CLOSE_FUNC(IF_TYPE(ifp)))(QSP_ARG  ifp);
	}
	return;
readerr:
	sprintf(ERROR_STRING, "error reading pixel data from file \"%s\"",
		ifp->if_name);
	WARN(ERROR_STRING);
	SET_ERROR(ifp);
	(*FT_CLOSE_FUNC(IF_TYPE(ifp)) )(QSP_ARG  ifp);
	return;
}
