#include "quip_config.h"

char VersionId_fio_writehdr[] = QUIP_VERSION_STRING;

/*
 * Disclaimer:  No guarantees of performance accompany this software,
 * nor is any responsibility assumed on the part of the authors.  All the
 * software has been tested extensively and every effort has been made to
 * insure its reliability.
 */

/*
 * write_header.c - HIPL image header write
 *
 * Michael Landy - 2/1/82
 * modified to use read/write - 4/26/82
 * modified for HIPS 2 - 1/3/91
 */

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "fio_prot.h"

#include "hips2.h"
#include "perr.h"
#include "wsubs.h"

int wt_hips2_hdr(FILE *fp,Hips2_Header *hd,const Filename fname)
{
	int i,j,offset;
	char s[LINELENGTH];
	struct extpar *xp;

	fprintf(fp,"HIPS\n");
	i = 5;
	i += wnocr(fp,hd->orig_name);
	i += wnocr(fp,hd->seq_name);
	if ((j = dfprintf(fp,hd->num_frame,fname)) == HIPS_ERROR)
		return(HIPS_ERROR);
	i += j;
	i += wnocr(fp,hd->orig_date);
	if ((j = dfprintf(fp,hd->orows,fname)) == HIPS_ERROR)
		return(HIPS_ERROR);
	i += j;
	if ((j = dfprintf(fp,hd->ocols,fname)) == HIPS_ERROR)
		return(HIPS_ERROR);
	i += j;
	if ((j = dfprintf(fp,hd->rows,fname)) == HIPS_ERROR)
		return(HIPS_ERROR);
	i += j;
	if ((j = dfprintf(fp,hd->cols,fname)) == HIPS_ERROR)
		return(HIPS_ERROR);
	i += j;
	if ((j = dfprintf(fp,hd->frow,fname)) == HIPS_ERROR)
		return(HIPS_ERROR);
	i += j;
	if ((j = dfprintf(fp,hd->fcol,fname)) == HIPS_ERROR)
		return(HIPS_ERROR);
	i += j;
	if ((j = dfprintf(fp,hd->pixel_format,fname)) == HIPS_ERROR)
		return(HIPS_ERROR);
	i += j;
	if ((j = dfprintf(fp,hd->numcolor,fname)) == HIPS_ERROR)
		return(HIPS_ERROR);
	i += j;
	if ((j = dfprintf(fp,hd->sizehist,fname)) == HIPS_ERROR)
		return(HIPS_ERROR);
	i += j;
	i += hd->sizehist;
	if (fwrite(hd->seq_history,hd->sizehist,1,fp) != 1)
		return(perr(HE_HDRWRT,fname));
	if (hd->sizedesc == 0 || hd->seq_desc[hd->sizedesc - 1] != '\n') {
		if ((j = dfprintf(fp,hd->sizedesc + 1,fname)) == HIPS_ERROR)
			return(HIPS_ERROR);
		i += j;
		i += hd->sizedesc + 1;
		if (hd->sizedesc != 0) {
			if (fwrite(hd->seq_desc,hd->sizedesc,1,fp) != 1)
				return(perr(HE_HDRWRT,fname));
		}
		putc('\n',fp);
	}
	else {
		if ((j = dfprintf(fp,hd->sizedesc,fname)) == HIPS_ERROR)
			return(HIPS_ERROR);
		i += j;
		i += hd->sizedesc;
		if (fwrite(hd->seq_desc,hd->sizedesc,1,fp) != 1)
			return(perr(HE_HDRWRT,fname));
	}
	if ((j = dfprintf(fp,hd->numparam,fname)) == HIPS_ERROR)
		return(HIPS_ERROR);
	i += j;
	xp = hd->params;
	offset = 0;
	while (xp != NULLPAR) {
		if (xp->count == 1) {
			switch (xp->format) {
			case PFASCII:	sprintf(s,"%s c 1 %d\n",xp->name,
						(int) xp->val.v_b); break;
			case PFBYTE:	sprintf(s,"%s b 1 %d\n",xp->name,
						(int) xp->val.v_b); break;
			case PFSHORT:	sprintf(s,"%s s 1 %d\n",xp->name,
						(int) xp->val.v_s); break;
			case PFINT:	sprintf(s,"%s i 1 %d\n",xp->name,
						xp->val.v_i); break;
			case PFFLOAT:	sprintf(s,"%s f 1 %f\n",xp->name,
						xp->val.v_f); break;
			default:	return(perr(HE_HDWPTYPE,xp->format,
						fname));
			}
		}
		else {
			switch (xp->format) {
			case PFASCII:	sprintf(s,"%s c %d %d\n",xp->name,
						xp->count,offset);
					offset += xp->count * sizeof(h_byte);
					break;
			case PFBYTE:	sprintf(s,"%s b %d %d\n",xp->name,
						xp->count,offset);
					offset += xp->count * sizeof(h_byte);
					break;
			case PFSHORT:	sprintf(s,"%s s %d %d\n",xp->name,
						xp->count,offset);
					offset += xp->count * sizeof(short);
					break;
			case PFINT:	sprintf(s,"%s i %d %d\n",xp->name,
						xp->count,offset);
					offset += xp->count * sizeof(int);
					break;
			case PFFLOAT:	sprintf(s,"%s f %d %d\n",xp->name,
						xp->count,offset);
					offset += xp->count * sizeof(float);
					break;
			default:	return(perr(HE_HDWPTYPE,xp->format,
						fname));
			}
			offset = (offset+3) & (~03); /* round up to 4 bytes */
		}
		j = strlen(s);
		i += j;
		if (fwrite(s,j,1,fp) != 1)
			return(perr(HE_HDRPWRT,fname));
		xp = xp->nextp;
	}
	sprintf(s,"%d\n",offset);	/* the size of the binary area */
	j = strlen(s);
	i += j;
	while ((i & 03) != 0) {		/* pad binary area size line with
					   blanks so that entire header
					   comes out to be an even multiple
					   of 4 bytes - the binary area itself
					   is guaranteed to be an even
					   multiple because each individual
					   entry is padded */
		putc(' ',fp);
		i++;
	}
	if (fwrite(s,j,1,fp) != 1)	/* write size of binary area */
		return(perr(HE_HDRBWRT,fname));
	xp = hd->params;
	offset = 0;
	while (xp != NULLPAR) {
		if (xp->count > 1) {
			switch (xp->format) {
			case PFASCII:
			case PFBYTE:	i = xp->count * sizeof(h_byte); break;
			case PFSHORT:	i = xp->count * sizeof(short); break;
			case PFINT:	i = xp->count * sizeof(int); break;
			case PFFLOAT:	i = xp->count * sizeof(float); break;
			default:	return(perr(HE_HDWPTYPE,xp->format,
						fname));
			}
			if (fwrite(xp->val.v_pb,i,1,fp) != 1)
				return(perr(HE_HDRBWRT,fname));
			offset += i;
			while ((offset & 03) != 0) {
				putc('\0',fp);
				offset++;
			}
		}
		xp = xp->nextp;
	}
	return(HIPS_OK);
}
