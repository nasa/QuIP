#include "quip_config.h"

char VersionId_fio_rdoldhdr[] = QUIP_VERSION_STRING;

/*
 * Copyright (c) 1991 Michael Landy
 *
 * Disclaimer:  No guarantees of performance accompany this software,
 * nor is any responsibility assumed on the part of the authors.  All the
 * software has been tested extensively and every effort has been made to
 * insure its reliability.
 */

/*
 * fread_oldhdr.c - HIPS read old (HIPS-1) format header
 *
 * Michael Landy - 2/1/82
 * modified to use read/write 4/26/82
 * modified for HIPS 2 - msl - 1/3/91
 */

#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "fio_prot.h"
#include "getbuf.h"
#include "perr.h"
#include "savestr.h"
#include "xparam.h"
#include "rdoldhdr.h"

#include "hips2.h"
#define LINES 100
#define MSBFIRST 1	/* HIPS-1 bit_packing value for PFMSBF */
#define LSBFIRST 2	/* HIPS-1 bit_packing value for PFLSBF */

static char *ssave[LINES];
static int slmax[LINES];
static int lalloc = 0;

/* local prototypes */

//int fread_oldhdr(FILE *,Hips2_Header *, char *, Filename);
static Bool hips_getline(FILE *fp,char **s,int *l);



int fread_oldhdr(FILE *fp,Hips2_Header *hd,char *firsts,Filename fname)
{
	int lineno,len,i,bit_packing,bpp;
	char *s;
	char *buf;

	/*
	hips_oldhdr = TRUE;
	jbm */
	if(lalloc<1) {
		ssave[0] = (char *)getbuf(LINELENGTH);
		slmax[0] = LINELENGTH;
		lalloc = 1;
	}
	hd->orig_name = savestr(firsts);
	if (hfgets(ssave[0],LINELENGTH,fp)) return(perr(HE_HDRREAD,fname));
	hd->seq_name = savestr(ssave[0]);
	if (fscanf(fp,"%d",&(hd->num_frame)) != 1)
		return(perr(HE_HDRREAD,fname));
	if (swallownl(fp))
		return(perr(HE_HDRREAD,fname));
	if (hfgets(ssave[0],LINELENGTH,fp)) return(perr(HE_HDRREAD,fname));
	hd->orig_date = savestr(ssave[0]);
	if (fscanf(fp,"%d",&(hd->rows)) != 1) return(perr(HE_HDRREAD,fname));
	if (fscanf(fp,"%d",&(hd->cols)) != 1) return(perr(HE_HDRREAD,fname));
	hd->orows = hd->rows;
	hd->ocols = hd->cols;
	hd->frow = 0;
	hd->fcol = 0;
	if (fscanf(fp,"%d",&bpp)!=1)
		return(perr(HE_HDRREAD,fname));
	if (fscanf(fp,"%d",&(bit_packing)) != 1) return(perr(HE_HDRREAD,fname));
	if (fscanf(fp,"%d",&(hd->pixel_format)) != 1)
		return(perr(HE_HDRREAD,fname));
	if (swallownl(fp))
		return(perr(HE_HDRREAD,fname));
	if (hd->pixel_format == PFBYTE && bit_packing == MSBFIRST)
		hd->pixel_format = PFMSBF;
	else if (hd->pixel_format == PFBYTE && bit_packing == LSBFIRST)
		hd->pixel_format = PFLSBF;
	hd->numcolor = 1;
	hd->numpix = hd->orows * hd->ocols;
	hd->sizepix = hsizepix(hd->pixel_format);
	hd->sizeimage = hd->sizepix * hd->numpix;
	if (hd->pixel_format == PFMSBF || hd->pixel_format == PFLSBF)
		hd->sizeimage = hd->orows * ((hd->ocols+7) / 8) * sizeof(h_byte);
	hd->imdealloc = FALSE;
	lineno = 0;
	len = 1;
	if (hips_getline(fp,&ssave[0],&slmax[0])) return(perr(HE_HDRREAD,fname));
	s = ssave[0];
	while(*(s += strlen(s)-3) == '|') {
		len += strlen(ssave[lineno]);
		lineno++;
		if (lineno >= LINES)
			return(perr(HE_HDRREAD,fname));
		if(lineno >= lalloc) {
			ssave[lineno] = (char *)getbuf(LINELENGTH);
			slmax[lineno] = LINELENGTH;
			lalloc++;
		}
		if (hips_getline(fp,&ssave[lineno],&slmax[lineno]))
			return(perr(HE_HDRREAD,fname));
		s = ssave[lineno];
	}
	len += strlen(ssave[lineno]);
	hd->sizehist = len-1;
	buf = (char *)getbuf(len);
	buf[0] = '\0';
	hd->histdealloc = TRUE;
	for (i=0;i<=lineno;i++)
		strcat(buf,ssave[i]);
	hd->seq_history = buf;
	lineno = 0;
	len = 1;
	while (1) {
		if (hips_getline(fp,&ssave[lineno],&slmax[lineno]))
			return(perr(HE_HDRREAD,fname));
		if (strcmp(ssave[lineno],".\n") == 0)
			goto founddot;
		len += strlen(ssave[lineno]);
		lineno++;
		if (lineno >= LINES)
			return(perr(HE_HDRREAD,fname));
		if(lineno >= lalloc) {
			ssave[lineno] = (char *)getbuf(LINELENGTH);
			slmax[lineno] = LINELENGTH;
			lalloc++;
		}
	}
founddot:
	hd->sizedesc = len-1;
	buf = (char *)getbuf(len);
	hd->seqddealloc = TRUE;
	*buf = '\0';
	for (i=0;i<lineno;i++)
		strcat(buf,ssave[i]);
	hd->seq_desc = buf;

	hd->numparam = 0;
	hd->paramdealloc = TRUE;
	hd->params = NULLPAR;
	if (hd->pixel_format == PFHIST) {
		if (fread(&i,sizeof(int),1,fp) != 1)
			return(perr(HE_HDRBREAD,fname));
		setparam(hd,"binleft",PFINT,1,i);
		if (fread(&i,sizeof(int),1,fp) != 1)
			return(perr(HE_HDRBREAD,fname));
		setparam(hd,"numbin",PFINT,1,i);
		setparam(hd,"binwidth",PFINT,1,256/i);
		setparam(hd,"imagepixfmt",PFINT,1,PFBYTE);
	}
	else if (hd->pixel_format == PFINTPYR ||
	    hd->pixel_format == PFFLOATPYR) {
#ifdef FOOBAR
		if (fread(&i,sizeof(int),1,fp) != 1)
			return(perr(HE_HDRBREAD,fname));
		setparam(hd,"toplev",PFINT,1,i);
		hd->numpix = pyrnumpix(i,hd->rows,hd->cols);
		hd->sizeimage = hd->numpix * hd->sizepix;
#else
		NWARN("sorry, pyramids not supported");
#endif
	}
	else if (hd->pixel_format == PFAHC3) {
		if (fread(&i,sizeof(int),1,fp) != 1)
			return(perr(HE_HDRBREAD,fname));
		setparam(hd,"stackrow",PFINT,1,i);
		if (fread(&i,sizeof(int),1,fp) != 1)
			return(perr(HE_HDRBREAD,fname));
		setparam(hd,"stackcol",PFINT,1,i);
		if (fread(&i,sizeof(int),1,fp) != 1)
			return(perr(HE_HDRBREAD,fname));
		setparam(hd,"stackdepth",PFINT,1,i);
	}
	else if (hd->pixel_format == PFBQ) {
		if (fread(&i,sizeof(int),1,fp) != 1)
			return(perr(HE_HDRBREAD,fname));
		setparam(hd,"stackspa",PFINT,1,i);
		if (fread(&i,sizeof(int),1,fp) != 1)
			return(perr(HE_HDRBREAD,fname));
		setparam(hd,"stackdepth",PFINT,1,i);
	}
	else if (hd->pixel_format == PFOCT) {
		if (fread(&i,sizeof(int),1,fp) != 1)
			return(perr(HE_HDRBREAD,fname));
		setparam(hd,"stacks",PFINT,1,i);
	}
	else if (hd->pixel_format == PFBT) {
		if (fread(&i,sizeof(int),1,fp) != 1)
			return(perr(HE_HDRBREAD,fname));
		setparam(hd,"height",PFINT,1,i);
		if (fread(&i,sizeof(int),1,fp) != 1)
			return(perr(HE_HDRBREAD,fname));
		setparam(hd,"width",PFINT,1,i);
		if (fread(&i,sizeof(int),1,fp) != 1)
			return(perr(HE_HDRBREAD,fname));
		setparam(hd,"vdom",PFINT,1,i);
	}
	return(HIPS_OK);
}

static Bool hips_getline(FILE *fp,char **s,int *l)
{
	int m;
	char c,*s1,*s2;

	s1 = *s;
	m = *l;
	while(fread(&c,1,1,fp) == 1 && c != '\n') {
		if (m-- <= 2) {
			s2 = (char *)getbuf(LINELENGTH+*l);
			strcpy(s2,*s);
			*s = s2;
			*l += LINELENGTH;
			m = LINELENGTH;
			s1 = s2 + strlen(s2);
		}
		*s1++ = c;
	}
	if (c == '\n') {
		*s1++ = '\n';
		*s1 = '\0';
		return(FALSE);
	}
	return(TRUE);
}

Bool swallownl(FILE *fp)
{
	int i;

	while ((i = getc(fp)) != '\n') {
		if (i == EOF)
			return(TRUE);
	}
	return(FALSE);
}

Bool hfgets(char *s,int n,FILE *fp)
{
	int i;

	fgets(s,n,fp);
	i=strlen(s);
	if (i==0 || s[i-1]!='\n')
		return(TRUE);
	return(FALSE);
}
