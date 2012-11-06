#include "quip_config.h"

char VersionId_fio_viff[] = QUIP_VERSION_STRING;


#include <stdio.h>
#include "filetype.h"
#include "getbuf.h"
#include "data_obj.h"
#include "debug.h"
#include "savestr.h"

#ifdef HAVE_KHOROS
#include "my_viff.h"

#include "vinclude.h"

#define hdr	if_hd.xvi_p

int viff_to_dp( Data_Obj *dp, struct xvimage *xvi_p )
{
	short type_dim=1;
	short prec;

	switch( xvi_p->data_storage_type ){
		case VFF_TYP_BIT:
			warn("sorry, 1 bit images not supported at this time");
			return(-1);
			break;
		case VFF_TYP_1_BYTE:  prec=PREC_BY; break;
		case VFF_TYP_2_BYTE:  prec=PREC_IN; break;
		case VFF_TYP_4_BYTE:  prec=PREC_DI; break;
		case VFF_TYP_FLOAT:   prec=PREC_SP; break;
		case VFF_TYP_COMPLEX: prec=PREC_SP; type_dim=2; break;
		default:
			sprintf(error_string,
		"viff_to_dp:  unsupported data storage type %d",
				xvi_p->data_storage_type);
			warn(error_string);
			return(-1);
			break;
	}
	dp->dt_prec = prec;
	dp->dt_tdim = type_dim;
	dp->dt_cols = xvi_p->row_size;
	dp->dt_rows = xvi_p->col_size;
	dp->dt_frames = 1;
	dp->dt_seqs = 1;

	dp->dt_cinc = 1;
	dp->dt_pinc = 1;
	dp->dt_rowinc = xvi_p->row_size*type_dim;
	dp->dt_finc = dp->dt_rowinc * xvi_p->col_size;
	dp->dt_sinc = dp->dt_rowinc * xvi_p->col_size;

	dp->dt_parent = NO_OBJ;
	dp->dt_children = NO_LIST;

	dp->dt_ap = ram_area;		/* the default */
	dp->dt_data = xvi_p->imagedata;
	dp->dt_nelts = dp->dt_tdim * dp->dt_cols * dp->dt_rows * dp->dt_frames;

	set_shape_flags(&dp->dt_shape,dp);

	return(0);
}

void rd_viff_hdr( int fdesc, struct xvimage *xvi_p, char *name )
{
	if( read(fdesc,xvi_p,sizeof(*xvi_p)) < 0 ){
		tell_sys_error("rd_viff_hdr");
		warn("error reading viff header");
	}
}

void viff_close( Image_File *ifp )
{
	if( ifp->hdr != NULL )
		givbuf(ifp->hdr);
	GENERIC_IMGFILE_CLOSE(ifp);
}

Image_File *		/**/
viff_open(const char *name,int rw)		/**/
{
	Image_File *ifp;

	ifp = IMAGE_FILE_OPEN(name,rw,IFT_VIFF);
	if( ifp==NO_IMAGE_FILE ) return(ifp);

	ifp->hdr = (struct xvimage *)getbuf( sizeof(struct xvimage) );

	if( IS_READABLE(ifp) ){
		/* BUG: should check for error here */
		rd_viff_hdr( ifp->if_fd, ifp->hdr, ifp->if_name );
		viff_to_dp(ifp->if_dp,ifp->hdr);
	}
	return(ifp);
}


int dp_to_viff( struct xvimage *xvi_p, Data_Obj *dp )
{
	dimension_t size;

	/* num_frame set when when write request given */

	xvi_p->identifier = XV_FILE_MAGIC_NUM;
	xvi_p->file_type = XV_FILE_TYPE_XVIFF;
	xvi_p->release = XV_IMAGE_REL_NUM;
	xvi_p->version = XV_IMAGE_VER_NUM;
	/*This should be machtype(NULL) BUG*/
	xvi_p->machine_dep=0x02;       
	
	xvi_p->col_size = dp->dt_rows;
	xvi_p->row_size = dp->dt_cols;

	if( dp->dt_frames != 1 ){
		warn("Sorry, viff files can only contain 1 frame");
		return(-1);
	}

	xvi_p->startx = VFF_NOTSUB;
	xvi_p->starty = VFF_NOTSUB;
	xvi_p->pixsizx = 1.0;
	xvi_p->pixsizy = 1.0;
	xvi_p->location_type = VFF_LOC_IMPLICIT;
	xvi_p->location_dim = 0;
	xvi_p->data_encode_scheme = VFF_DES_RAW;

	xvi_p->num_of_images=1;
	xvi_p->num_data_bands=dp->dt_tdim;

	xvi_p->map_row_size=0;
	xvi_p->map_col_size=0;
	xvi_p->map_scheme = VFF_MS_NONE;
	xvi_p->map_storage_type = VFF_MAPTYP_NONE;
	
	xvi_p->map_subrow_size = 0;
	xvi_p->map_enable = VFF_MAP_OPTIONAL;
	xvi_p->maps_per_cycle = 0;      /* Don't care */
	xvi_p->color_space_model = VFF_CM_NONE;
	xvi_p->ispare1 = 0;
	xvi_p->ispare2 = 0;
	xvi_p->fspare1 = 0;
	xvi_p->fspare2 = 0;


	switch( dp->dt_prec ){
		case PREC_BY:
			xvi_p->data_storage_type = VFF_TYP_1_BYTE;
			size=1;
			break;
		case PREC_IN:
			xvi_p->data_storage_type = VFF_TYP_2_BYTE;
			size=2;
			break;
		case PREC_DI:
			xvi_p->data_storage_type = VFF_TYP_4_BYTE;
			size=4;
			break;
		case PREC_SP:
			if( dp->dt_tdim == 1 ){
				xvi_p->data_storage_type = VFF_TYP_FLOAT;
				size=sizeof(float);
			} else if( dp->dt_tdim == 2 ){
				xvi_p->data_storage_type = VFF_TYP_COMPLEX;
				size=2*sizeof(float);
			} else if( dp->dt_tdim > 2 ){
				warn("sorry, VIFF does not support multidimensional data types");
				return(-1);
			} else error1("dp_to_viff:  bad data type");
			break;
		default:
			warn("dp_to_viff:  unsupported pixel format");
			return(-1);
	}

	xvi_p->imagedata=dp->dt_data;

	return(0);
}

void wt_viff_hdr( int fdesc, struct xvimage *xvi_p, char *name )
{
	if( write(fdesc,xvi_p,sizeof(*xvi_p)) < 0 ){
		tell_sys_error("wt_viff_hdr");
		warn("error writing viff header");
	}
}

int set_viff_hdr( Image_File *ifp )
{
	if( dp_to_viff(ifp->hdr,ifp->if_dp) < 0 ){
		viff_close(ifp);
		return(-1);
	}
	wt_viff_hdr(ifp->if_fd,ifp->hdr,ifp->if_name);	/* write it out */
	return(0);
}

int viff_wt( Data_Obj *dp, Image_File *ifp )
{
	long n, size;
	long npixels, os;
	long actual;

	if( ifp->if_dp == NO_OBJ ){
		setup_dummy(ifp);
		copy_dimensions(ifp->if_dp, dp);
		/* ifp->if_dp->dt_frames = ifp->hdr->num_frame; */

		/* this is a BUG if viff can actually support multiple frames */
		ifp->if_dp->dt_frames=1;

		if( set_viff_hdr(ifp) < 0 ) return(-1);
	} else if( !same_type(dp,ifp) ) return(-1);

	raw_wt(dp,ifp);
	return(0);
}


void viff_rd( Data_Obj *dp, Image_File *ifp, index_t x_offset, index_t y_offset, index_t t_offset )
{
	raw_rd(dp,ifp,x_offset,y_offset,t_offset);
}


int viff_unconv( struct xvimage **hd_pp, Data_Obj *dp )
{
	/* allocate space for new header */

	*hd_pp = getbuf( sizeof(struct xvimage) );
	if( *hd_pp == NULL ) return(-1);

	dp_to_viff(*hd_pp,dp);

	return(0);
}

int viff_conv(void)
{
	warn("viff_conv not implemented");
	return(-1);
}

#endif /* HAVE_KHOROS */

