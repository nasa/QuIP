
#include "quip_config.h"

/*#define BPP_3		 */
/*#define QUIP_DEBUG_USING_SDL */

#include <stdio.h>
#include <stdlib.h>


#ifdef QUIP_DEBUG_USING_SDL
#include <SDL/SDL.h> 
#endif /* QUIP_DEBUG_USING_SDL */

#include "img_file.h"
#include "debug.h"

#ifdef HAVE_MPEG

#include "filetype.h" /* ft_tbl */

#define hdr	if_hd.mpeg_hd_p


#undef QUIP_DEBUG

#ifdef QUIP_DEBUG_USING_SDL
SDL_Surface *screen;
#endif /* QUIP_DEBUG_USING_SDL */
 

#ifdef QUIP_DEBUG
/* declared in wrapper.c of mpeg lib */
extern int totNumFrames;
#endif /* QUIP_DEBUG */
//extern int get_n_of_frms(char *filename);


static int mpeg_to_dp(Data_Obj *dp,Mpeg_Hdr *mpeg_hp)
{

	dp->dt_prec = PREC_UBY;

	dp->dt_tdim = mpeg_hp->depth;
	dp->dt_cols = mpeg_hp->width;
	dp->dt_rows = mpeg_hp->height;
	dp->dt_frames = mpeg_hp->frames;
	dp->dt_seqs = 1;

	dp->dt_cinc = 1;
	dp->dt_pinc = 1;
	dp->dt_rinc = dp->dt_pinc*dp->dt_cols;
	dp->dt_finc = dp->dt_rinc*dp->dt_rows;
	dp->dt_sinc = dp->dt_finc*dp->dt_frames;

	dp->dt_parent = NULL;
	dp->dt_children = NULL;

	dp->dt_ap = ram_area;		/* the default */
	dp->dt_data = NULL;
	dp->dt_nelts = dp->dt_tdim * dp->dt_cols * dp->dt_rows
			* dp->dt_frames * dp->dt_seqs;

	auto_shape_flags(&dp->dt_shape);

	return(0);
}



/* the unconvert routine creates a disk header */

int mpeg_unconv(void *hdr_pp,Data_Obj *dp)
{
	warn("mpeg_unconv() not implemented!?");
	return(-1);
}

int mpeg_conv(Data_Obj *dp,void *hd_pp)
{
	warn("mpeg_conv not implemented");
	return(-1);
}


#ifdef QUIP_DEBUG

static void prt_wt_img(ImVfb *wt_img)
{

	printf("    vfb_width:%d\n",wt_img->vfb_width);			/* cols */
	printf("    vfb_height: %d\n",wt_img->vfb_height );		/* rows */
	printf("    vfb_fields: %d\n",wt_img->vfb_fields );		/* mask specifying what is in each pixel */
	printf("    vfb_nbytes: %d\n",wt_img->vfb_nbytes );		/* bytes per pixel */
	printf("ImClt  *vfb_clt: 0x%x\n",wt_img->vfb_clt);		/* points to attached CLT, if any */
	printf("    vfb_roff: %d\n",wt_img->vfb_roff );			/* offset (in bytes) to reach red */
	printf("    vfb_goff: %d\n",wt_img->vfb_goff );			/* green */
	printf("    vfb_boff: %d\n",wt_img->vfb_boff );			/* blue	*/
	printf("    vfb_aoff: %d\n",wt_img->vfb_aoff );			/* alpha-value */
	printf("    vfb_i8off: %d\n",wt_img->vfb_i8off );		/* color index */
	printf("    vfb_wpoff: %d\n",wt_img->vfb_wpoff );		/*write protect offset*/
	printf("    vfb_i16off: %d\n",wt_img->vfb_i16off );		/* color index */
	printf("    vfb_zoff: %d\n",wt_img->vfb_zoff );			/* z-value */
	printf("    vfb_moff: %d\n",wt_img->vfb_moff );			/* mono	*/
	printf("    vfb_fpoff: %d\n",wt_img->vfb_fpoff );		/* floating point */
	printf("    vfb_ioff: %d\n",wt_img->vfb_ioff );			/* integer */
	printf("ImVfbPtr vfb_pfirst: 0x%x\n",wt_img->vfb_pfirst );	/* points to first pixel */
	printf("ImVfbPtr vfb_plast: 0x%x\n",wt_img->vfb_plast );	/* points to last pixel */

}

#endif /* QUIP_DEBUG */


static ImVfb *dp_to_ImVfb(Data_Obj *dp)
{
	static ImVfb *wt_img = NULL;
	ImVfbPtr dst;
	int row, col, comp;
	u_char *src;


	/* allocates mem and initializes ImVfb struct */	
	if(dp->dt_tdim == 1)
		wt_img = MPEGe_ImVfbAlloc(dp->dt_cols, dp->dt_rows, IMVFBGRAY, TRUE);
	else if(dp->dt_tdim == 3)	
		wt_img = MPEGe_ImVfbAlloc(dp->dt_cols, dp->dt_rows, IMVFBRGB, TRUE);
	else if(dp->dt_tdim == 4) {
		warn("4 component images have yet to be tested ... This may be buggy");
		wt_img = MPEGe_ImVfbAlloc(dp->dt_cols, dp->dt_rows, 
						(IMVFBRGB | IMVFBALPHA) , TRUE);
	} else {
		sprintf(error_string, "ERROR: Unsupported number of components(%ld)",
									dp->dt_tdim);
		advise(error_string);
		return NULL;
	}

	if(!wt_img) {
		advise("ERROR: Unable to allocate mem and initialize ImVfb Struct");
		return NULL;
	}

	wt_img->vfb_nbytes = dp->dt_tdim;

	dst = wt_img->vfb_pfirst;
	src = (u_char *)dp->dt_data;

	for (row = 0;  row < dp->dt_rows;  row++ )
		for (col = 0;  col < dp->dt_cols;  col++ )
			for(comp=0; comp<dp->dt_tdim; comp++)
				*dst++ = *src++;

	return wt_img;
}




static int get_n_frs(Image_File *ifp)
{
	int n_frames=0;
   	char *pixels;
	ImageDesc *idp;
	Boolean moreframes = TRUE;
	
#ifdef QUIP_DEBUG
printf("get_n_frs: IN\n");
#endif /* QUIP_DEBUG */
	
	idp = ifp->hdr->idp;
	
	pixels = (char *)malloc(idp->Size * sizeof(char));

	/* get frames until the movie ends */
	while (moreframes) {
		/* GetMPEGFrame returns FALSE after last frame is decoded */
		moreframes = GetMPEGFrame(pixels);
		n_frames++;
	}

	free(pixels);	
	
	if(verbose) printf("n_frames calculated: %d\n", n_frames);

	RewindMPEG (ifp->if_fp, idp);

#ifdef QUIP_DEBUG
printf("get_n_frs: OUT\n");
#endif /* QUIP_DEBUG */
	
	return n_frames;	
}


Image_File * mpeg_open(const char *name,int rw)
{
	Image_File *ifp;
	ImageDesc *idp;

#ifdef QUIP_DEBUG
printf("mpeg_open: IN\n");
#endif /* QUIP_DEBUG */
	
	ifp = IMG_FILE_CREAT(name,rw,IFT_MPEG);
	if( ifp==NULL ) return(ifp);

	ifp->hdr = (Mpeg_Hdr *)getbuf( sizeof(Mpeg_Hdr) );


	if( IS_READABLE(ifp) ) {
	
		ifp->hdr->idp = (ImageDesc *)malloc( sizeof(ImageDesc) );
		idp = ifp->hdr->idp;

		SetMPEGOption (MPEG_DITHER, FULL_COLOR_DITHER);

		if (!OpenMPEG(ifp->if_fp, idp)) {
			advise("ERROR: OpenMPEG failed");
			exit(1);
		}

		/* fill in some basic header info */
		ifp->hdr->width = idp->Width;
		ifp->hdr->height = idp->Height;
#ifdef BPP_3
		ifp->hdr->depth = idp->Depth/8;
#else
		ifp->hdr->depth = idp->PixelSize/8;
#endif
		/* BUG: we need a more efficient way of counting frames! */
		ifp->hdr->frames = get_n_frs(ifp);

//		ifp->hdr->frames = get_n_of_frms(ifp->if_pathname);

		mpeg_to_dp(ifp->if_dp,ifp->hdr);

		
#ifdef QUIP_DEBUG_USING_SDL
{
		char buf[32];
		/* Initialize SDL */
		
		if ((SDL_Init(SDL_INIT_VIDEO) < 0) || !SDL_VideoDriverName(buf, 1)) {
			sprintf(error_string,"Couldn't init SDL video: %s\n",
				SDL_GetError());
			warn(error_string);
		}

		//screen = SDL_SetVideoMode(idp->Width, idp->Height, 0, 0);
		screen = SDL_SetVideoMode(idp->Width, idp->Height, 
				ifp->hdr->depth*8, 
				SDL_HWSURFACE/*video surface in hardware mem*/);

		if ( screen == NULL ) {
			fprintf(stderr, "Unable to set %dx%d video mode: %s\n",
				idp->Width, idp->Height, SDL_GetError());
			return NULL;
		}
}
#endif /* QUIP_DEBUG_USING_SDL */


	} else {

		ifp->hdr->enc_opts = (MPEGe_options *)malloc( sizeof(MPEGe_options) );
		
		/* Set default options. These can be changed later, if desired. */
		MPEGe_default_options(ifp->hdr->enc_opts);

		/* Start the library with default options */
		if( !MPEGe_open(ifp->if_fp, ifp->hdr->enc_opts) ) {
			warn("MPEGe lib initialization failure");
			return NULL;
		}

		/* It doesn't make sense to initialize the header for writing. */
	}
	
#ifdef QUIP_DEBUG
printf("mpeg_open: OUT\n");
#endif /* QUIP_DEBUG */

	return(ifp);
}


void mpeg_close(Image_File *ifp)
{
	/* First do the mpeg library cleanup */
	if( IS_READABLE(ifp) ){

		if( ifp->hdr->idp != NULL ) 
			free(ifp->hdr->idp);
	} else {
		
		/* Create MPEG end sequences and close output file. */
		if( !MPEGe_close(ifp->hdr->enc_opts) ) {
			sprintf(error_string,"ERROR: %s", ifp->hdr->enc_opts->error);
			advise(error_string);
		}

		if( ifp->hdr->enc_opts != NULL )
			free(ifp->hdr->enc_opts);
	}
	
	if( ifp->hdr != NULL )
		givbuf(ifp->hdr);

	generic_imgfile_close(ifp);
}


void mpeg_rd(Data_Obj *dp,Image_File *ifp,index_t x_offset, index_t y_offset,index_t t_offset)
{
	static char *data_ptr;
   	char *pixels;
	int i;
#ifdef BPP_3
	int actual_fr_size; 
#endif /* BPP_3 */

#ifdef QUIP_DEBUG
printf("mpeg_rd: IN\n");
#endif /* QUIP_DEBUG */

	/* make sure that the sizes match */
	if( ! dp_same_dim(dp,ifp->if_dp,0) ) return;	/* same # components? */
	if( ! dp_same_dim(dp,ifp->if_dp,1) ) return;	/* same # columns? */
	if( ! dp_same_dim(dp,ifp->if_dp,2) ) return;	/* same # rows? */

#ifdef FOOBAR
#ifdef BPP_3
	/* we don't need to alloate mem for the unused byte, hence 0.75 */
	actual_fr_size = ifp->hdr->idp->Size * 0.75 * sizeof(char);
	data_ptr = dp->dt_data = (char *)malloc(actual_fr_size);
#else	
	data_ptr = dp->dt_data = (char *)malloc(ifp->hdr->idp->Size);
#endif /* BPP_3 */	
#endif /* FOOBAR */

	/* Saad must have written that code resetting dp->dt_data!? */
	/* note that it was never free'd */
	data_ptr = (char *)dp->dt_data;
	
	/* mem for decoded image */
	pixels = (char *)malloc(ifp->hdr->idp->Size * sizeof(char));

	/* Get a single MPEG frame */
	GetMPEGFrame(pixels);

	/* Copy frame to the data object */
	for(i=0; i<ifp->hdr->idp->Size; i+=4) {
		/* For full colour dithering, there are four bytes per pixel: 
		 * red, green, blue, and unused.
		 */

		*(data_ptr) = *(pixels+i+2);
		*(data_ptr+1) = *(pixels+i+1);
		*(data_ptr+2) = *(pixels+i+0);
#ifdef BPP_3
		data_ptr+=3;
#else		
		data_ptr+=4;
#endif /* BPP_3 */		
	}

#ifdef QUIP_DEBUG_USING_SDL
#ifdef BPP_3
	screen->pixels = (char *)memcpy(screen->pixels, dp->dt_data, actual_fr_size);
#else	
	screen->pixels = (char *)memcpy(screen->pixels, dp->dt_data, ifp->hdr->idp->Size);
#endif /* BPP_3 */		

	SDL_UpdateRect(screen, 0, 0, 0, 0);
#endif /* QUIP_DEBUG_USING_SDL */
	
//	dp->dt_data = (char *)memcpy(dp->dt_data, pixels, ifp->hdr->idp->Size);

	free(pixels);

	ifp->if_nfrms ++;

	if( FILE_FINISHED(ifp) ){	/* BUG?  if autoclose is set to no, do we make sure we don't read past eof? */
		if( verbose ){
			sprintf(error_string,
				"closing file \"%s\" after reading %ld frames",
				ifp->if_name,ifp->if_nfrms);
			advise(error_string);
		}
	
		mpeg_close(ifp);
		
	}
	
#ifdef QUIP_DEBUG
printf("mpeg_rd: OUT\n");
#endif /* QUIP_DEBUG */
}


int mpeg_wt(Data_Obj *dp,Image_File *ifp)
{
	ImVfb *wt_img=NULL;

	if( ifp->if_dp == NULL ){	/* first time? */
		/* set header params if necessary? */
		setup_dummy(ifp);
		copy_dimensions(ifp->if_dp, dp);
		/* BUG?  where do we set the desired number of frames? */
	} else {
		/* BUG need to make sure that this image matches if_dp */
	}

	if( (wt_img = dp_to_ImVfb(dp))==NULL )
		return -1;
	
	/* Append the image to the MPEG stream. */
	if( !MPEGe_image(wt_img, ifp->hdr->enc_opts) ) {
		warn ("MPEGe_image failure");
		return -1;
	}
	
	ifp->if_nfrms ++;

	if( ifp->if_nfrms == ifp->if_frms_to_wt ){
		if( verbose ){
			sprintf(error_string, "closing file \"%s\" after writing %ld frames",
				ifp->if_name,ifp->if_nfrms);
			advise(error_string);
		}
		close_image_file(ifp);
	}
	return(0);
}


void report_mpeg_info(Image_File *ifp)
{
	sprintf(msg_str,"\timage depth (bits) %d", ifp->hdr->idp->Depth);
	prt_msg(msg_str);
	
	sprintf(msg_str,"\tbits per pixel %d", ifp->hdr->idp->PixelSize);
	prt_msg(msg_str);
	
	sprintf(msg_str,"\tsize of image %d bytes", ifp->hdr->idp->Size);
	prt_msg(msg_str);
}


#endif /* ! HAVE_MPEG */

