#include "quip_config.h"

/*
 * Copyright (c) 1991 Michael Landy
 *
 * Disclaimer:  No guarantees of performance accompany this software,
 * nor is any responsibility assumed on the part of the authors.  All the
 * software has been tested extensively and every effort has been made to
 * insure its reliability.
 */

/*
 * read_header.c - HIPS Picture Format Header read
 *
 * Michael Landy - 2/1/82
 * modified to use read/write 4/26/82
 * modified for HIPS2 1/3/91
 */

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "fio_api.h"
#include "quip_prot.h"
#include "getbuf.h"
#include "hips/perr.h"
#include "hips/rdoldhdr.h"
#include "hips/hips2.h"
#include "hips/readhdr.h"

int rd_hips2_hdr(FILE *fp,Hips2_Header *hd,const Filename fname)
{
	char inp[LINELENGTH],ptypes[20];
	int i,v,sizebin,curroffset;
#ifdef HIPS_PYRAMIDS
	int one=1;
	int toplev;
#endif /* HIPS_PYRAMIDS */
	struct extpar *xp,*lastxp;
	char *buf;

	/*
	hips_oldhdr = FALSE;
	jbm */
	if (hfgets(inp,LINELENGTH,fp)) return(perr(HE_HDRREAD,fname));
	if (strcmp(inp,"HIPS\n")!=0)
		return(fread_oldhdr(fp,hd,inp,fname));
	if (hfgets(inp,LINELENGTH,fp)) return(perr(HE_HDRREAD,fname));
	hd->orig_name = savestr(inp);
	if (hfgets(inp,LINELENGTH,fp)) return(perr(HE_HDRREAD,fname));
	hd->seq_name = savestr(inp);
	if (fscanf(fp,"%d",&(hd->num_frame)) != 1)
		return(perr(HE_HDRREAD,fname));
	if (swallownl(fp))
		return(perr(HE_HDRREAD,fname));
	if (hfgets(inp,LINELENGTH,fp)) return(perr(HE_HDRREAD,fname));
	hd->orig_date = savestr(inp);
	if (fscanf(fp,"%d",&(hd->orows)) != 1) return(perr(HE_HDRREAD,fname));
	if (fscanf(fp,"%d",&(hd->ocols)) != 1) return(perr(HE_HDRREAD,fname));
	if (fscanf(fp,"%d",&(hd->rows)) != 1) return(perr(HE_HDRREAD,fname));
	if (fscanf(fp,"%d",&(hd->cols)) != 1) return(perr(HE_HDRREAD,fname));
	if (fscanf(fp,"%d",&(hd->frow)) != 1) return(perr(HE_HDRREAD,fname));
	if (fscanf(fp,"%d",&(hd->fcol)) != 1) return(perr(HE_HDRREAD,fname));
	if (fscanf(fp,"%d",&(hd->pixel_format)) != 1)
		return(perr(HE_HDRREAD,fname));
	if ((hd->pixel_format == PFMSBF || hd->pixel_format == PFLSBF) &&
		(hd->fcol % 8 != 0))
			return(perr(HE_ROI8F,"fread_header",fname));
	if (fscanf(fp,"%d",&(hd->numcolor)) != 1)
		return(perr(HE_HDRREAD,fname));
	hd->numpix = hd->orows * hd->ocols;
	hd->sizepix = hsizepix(hd->pixel_format);
	hd->sizeimage = hd->sizepix * hd->numpix;
	if (hd->pixel_format == PFMSBF || hd->pixel_format == PFLSBF)
		hd->sizeimage = hd->orows * ((hd->ocols+7) / 8) * sizeof(h_byte);
	hd->imdealloc = FALSE;
	if (fscanf(fp,"%d",&(hd->sizehist)) != 1)
		return(perr(HE_HDRREAD,fname));
	if (swallownl(fp))
		return(perr(HE_HDRREAD,fname));
	if ((buf = (char *) getbuf(1 + (hd->sizehist))) == (char *) 0)
		return(perr(HE_ALLOCSUBR,"fread_header"));
	if (hd->sizehist) {
		if (fread(buf,hd->sizehist,1,fp) != 1)
			return(perr(HE_HDRREAD,fname));
	}
	buf[hd->sizehist] = '\0';
	hd->seq_history = buf;
	hd->histdealloc = TRUE;

	if (fscanf(fp,"%d",&(hd->sizedesc)) != 1)
		return(perr(HE_HDRREAD,fname));
	if (swallownl(fp))
		return(perr(HE_HDRREAD,fname));

	if ((buf = (char *) getbuf(1 + (hd->sizedesc))) == (char *) 0)
		return(perr(HE_ALLOCSUBR,"fread_header"));
	if (hd->sizedesc) {
		if (fread(buf,hd->sizedesc,1,fp) != 1)
			return(perr(HE_HDRREAD,fname));
	}
	buf[hd->sizedesc] = '\0';
	hd->seq_desc = buf;

	hd->seqddealloc = TRUE;
	if (fscanf(fp,"%d",&(hd->numparam)) != 1)
		return(perr(HE_HDRREAD,fname));
	if (swallownl(fp))
		return(perr(HE_HDRREAD,fname));
	hd->paramdealloc = TRUE;
	hd->params = NULLPAR;
	lastxp = NULLPAR;		/* suppress compiler warning */
	for (i=0;i<hd->numparam;i++) {
		xp = (struct extpar *)getbuf(sizeof(*xp));
		if (i==0)
			lastxp = hd->params = xp;
		else {
			lastxp->nextp = xp;
			lastxp = xp;
		}
		xp->nextp = NULLPAR;
		if (fscanf(fp,"%s %s %d",inp,ptypes,&(xp->count)) != 3)
			return(perr(HE_HDRPREAD,fname));
		xp->name = savestr(inp);
		switch(ptypes[0]) {
			case 'c': xp->format = PFASCII; break;
			case 'b': xp->format = PFBYTE; break;
			case 'i': xp->format = PFINT; break;
			case 'f': xp->format = PFFLOAT; break;
			case 's': xp->format = PFSHORT; break;
			default: return(perr(HE_HDRPTYPES,ptypes,fname));
		}
		if (xp->count == 1) {
			switch(xp->format) {
			case PFASCII:
			case PFBYTE:	if (fscanf(fp,"%d",&v) != 1)
						return(perr(HE_HDRPREAD,fname));
					xp->val.v_b = (h_byte) v;
					break;
			case PFSHORT:	if (fscanf(fp,"%d",&v) != 1)
						return(perr(HE_HDRPREAD,fname));
					xp->val.v_s = (short) v;
					break;
			case PFINT:	if (fscanf(fp,"%d",&(xp->val.v_i))
					    != 1)
						return(perr(HE_HDRPREAD,fname));
					break;
			case PFFLOAT:	if (fscanf(fp,"%f",&(xp->val.v_f))
					    != 1)
						return(perr(HE_HDRPREAD,fname));
					break;
			}
		}
		else {	/*  *** temporarily store offset in dealloc *** */
			if (fscanf(fp,"%d",&(xp->dealloc)) != 1)
				return(perr(HE_HDRPREAD,fname));
		}
		if (swallownl(fp))
			return(perr(HE_HDRREAD,fname));
	}
	if (fscanf(fp,"%d",&sizebin) != 1) return(perr(HE_HDRREAD,fname));
	if (swallownl(fp))
		return(perr(HE_HDRREAD,fname));
	curroffset = 0;
	xp = hd->params;
	while (xp != NULLPAR) {
		if (xp->count > 1) {
			if (xp->dealloc != curroffset)
				return(perr(HE_XINC,xp->name,xp->format,
				    xp->count,xp->dealloc,curroffset,fname));
			i = xp->count *
				(int)((xp->format == PFASCII) ? sizeof(char) :
				hsizepix(xp->format));
			i = (i+3) & ~03;
			if ((xp->val.v_pb = (h_byte *) getbuf(i)) == (h_byte *) 0)
				return(perr(HE_ALLOCSUBR,"fread_header"));
			xp->dealloc = TRUE;
			if (fread(xp->val.v_pb,i,1,fp) != 1)
				return(perr(HE_HDRBREAD,fname));
			curroffset += i;
		}
		else
			xp->dealloc = FALSE;
		xp = xp->nextp;
	}
	if (curroffset != sizebin)
		return(perr(HE_HDRXOV,fname,sizebin,curroffset));
	if (hd->pixel_format == PFINTPYR || hd->pixel_format == PFFLOATPYR) {
#ifdef HIPS_PYRAMIDS
		if (getparam(hd,"toplev",PFINT,&one,&toplev) == HIPS_ERROR)
			return(HIPS_ERROR);
		hd->numpix = pyrnumpix(toplev,hd->rows,hd->cols);
		hd->sizeimage = hd->numpix * hd->sizepix;
#else
		NWARN("sorry HIPS pyramids not supported");
#endif
	}
	return(HIPS_OK);
}

void rls_hips2_hd(Hips2_Header *hdrp)	/* free strings */
{
	if( hdrp->orig_name != NULL ) rls_str( hdrp->orig_name);
	if( hdrp->seq_name != NULL ) rls_str( hdrp->seq_name);
	if( hdrp->orig_date != NULL ) rls_str( hdrp->orig_date);
	if( hdrp->seq_history != NULL ) givbuf((void *)(hdrp->seq_history));
	if( hdrp->seq_desc != NULL ) givbuf((void *)(hdrp->seq_desc));

	if (hdrp->numparam > 0 ) /* BUG */
		NWARN("NOT releasing xparam structures");
}

void null_hips2_hd(Hips2_Header *hdrp)	/* set string pointers to NULL */
{
	hdrp->orig_name = NULL;
	hdrp->seq_name = NULL;
	hdrp->orig_date = NULL;
	hdrp->seq_history = NULL;
	hdrp->seq_desc = NULL;

	hdrp->numparam = 0;
}

