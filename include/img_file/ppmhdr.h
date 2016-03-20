

#ifndef PPM_HDR

#ifdef INC_VERSION
char VersionId_inc_ppmhdr[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

typedef struct ppm_hdr { 
	int format;	/* 5 = 1 component, 6 = 3 components */
	int rows,cols;
	int somex;	/* 255??? what is this??? */
	void *img_data;
} Ppm_Header;

#define PPM_HDR

typedef struct dis_hdr {
	int format;
	int rows,cols;
	int frames;
	int somex;
	void *img_data;
} Dis_Header;

#endif /* ! PPM_HDR */

