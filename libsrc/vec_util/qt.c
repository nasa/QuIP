#include "quip_config.h"

char VersionId_vec_util_qt[] = QUIP_VERSION_STRING;

#include "vec_util.h"
#include "quadtree.h"
#include "getbuf.h"

static void giv_pixels(QuadTree *qtp,dimension_t onpixels);


/*
 * Build the heirarchical representation of an input image
 */

QuadTree *new_qtp(void)
{
	QuadTree *qtp;

	qtp=(QuadTree *)getbuf(sizeof(*qtp));
	if( qtp == NO_QUADTREE ) NERROR1("out of memory");
	return(qtp);
}
	
void del_qt(QuadTree *qtp)
{
	if( qtp->qt_size > 1 ){
		del_qt(qtp->qt_child[0]);
		del_qt(qtp->qt_child[1]);
		del_qt(qtp->qt_child[2]);
		del_qt(qtp->qt_child[3]);
	}
	free(qtp);
}

QuadTree *mk_qt(Data_Obj *dp,dimension_t size,dimension_t x,dimension_t y)
{
	QuadTree *qtp;

	qtp = new_qtp();
	if( size == 1 ){
		qtp->qt_child[0] = NO_QUADTREE;
		qtp->qt_child[1] = NO_QUADTREE;
		qtp->qt_child[2] = NO_QUADTREE;
		qtp->qt_child[3] = NO_QUADTREE;
		qtp->qt_val = *( ((float *) dp->dt_data) + x +y*dp->dt_rinc );
	} else {
		qtp->qt_child[0] = mk_qt(dp,size/2,x,y);
		qtp->qt_child[1] = mk_qt(dp,size/2,x+size/2,y);
		qtp->qt_child[2] = mk_qt(dp,size/2,x,y+size/2);
		qtp->qt_child[3] = mk_qt(dp,size/2,x+size/2,y+size/2);
		qtp->qt_val = (qtp->qt_child[0]->qt_val
				+ qtp->qt_child[1]->qt_val
				+ qtp->qt_child[2]->qt_val
				+ qtp->qt_child[3]->qt_val ) / 4;
	}
	qtp->qt_size = size;
	return(qtp);
}

QuadTree *build_qt(Data_Obj *dp)
{
	QuadTree *qtp;
	dimension_t size;

	size = dp->dt_cols;
	if( size != dp->dt_rows ){
		NWARN("image must be square");
		return(NO_QUADTREE);
	}

	qtp=mk_qt(dp,size,0,0);
	return(qtp);
}

void fill_output(Data_Obj *dp,QuadTree *qtp,dimension_t x,dimension_t y)
{
	if( qtp->qt_size == 1 ){
		if( qtp->qt_onpix > 0 )
			*((float *)dp->dt_data + x + y*dp->dt_rinc) = 1;
		else
			*((float *)dp->dt_data + x + y*dp->dt_rinc) = -1;
	} else {
		fill_output(dp,qtp->qt_child[0],x,y);
		fill_output(dp,qtp->qt_child[1],x+qtp->qt_size/2,y);
		fill_output(dp,qtp->qt_child[2],x,y+qtp->qt_size/2);
		fill_output(dp,qtp->qt_child[3],x+qtp->qt_size/2,
			y+qtp->qt_size/2);
	}
}

void qt_dither(Data_Obj *dpto, Data_Obj *dpfr )
{
	QuadTree *qtp;
	dimension_t totpixels, onpixels;
	

	qtp=build_qt(dpfr);

	totpixels = qtp->qt_size * qtp->qt_size;
	/* assume values go from -1 to 1 */
	onpixels = (dimension_t)(((qtp->qt_val+1.0)* (float)totpixels/2.0) + 0.5);

	giv_pixels(qtp,onpixels);

	/* now set up the output image */

	fill_output(dpto,qtp,0,0);

	del_qt(qtp);
}

static void giv_pixels(QuadTree *qtp,dimension_t onpixels)
{
	float sum;
	dimension_t i;
	long np[4];

	if( qtp->qt_size > 1 ){
		float rowsum,colsum,interaction;
		long rowq, colq;
#ifdef DEBUG
		int die=0;
#endif /* DEBUG */

		/* divide the pixels between the four children */

		/* add the two's to make things go from 0 to 2 */
		rowsum = 2+qtp->qt_child[0]->qt_val+qtp->qt_child[1]->qt_val;
		colsum = 2+qtp->qt_child[0]->qt_val+qtp->qt_child[2]->qt_val;
		interaction = qtp->qt_child[0]->qt_val+qtp->qt_child[3]->qt_val
			-(qtp->qt_child[1]->qt_val+qtp->qt_child[2]->qt_val);

		sum = rowsum +
			2+qtp->qt_child[2]->qt_val+qtp->qt_child[3]->qt_val;
		rowsum /= sum;
		colsum /= sum;
		interaction /= sum;

		/* now quantize the row and column sums */

		rowq = (long)(rowsum*(float)onpixels + 0.5);
		colq = (long)(colsum*(float)onpixels + 0.5);

		np[0] = (long) (((interaction-1)/2.0+rowsum+colsum)*onpixels/2.0 + 0.5);

top:
		if( np[0] < 0 ) np[0]=0;
		if( np[0] > colq ) np[0]=colq;
		if( np[0] > rowq ) np[0]=rowq;

		np[1] = rowq - np[0];
		np[2] = colq - np[0];
		np[3] = (long)onpixels - (np[0]+np[1]+np[2]);
		if( np[3] < 0 ) {
			np[0]++;
			goto top;
		}

#ifdef DEBUG
		if( np[3] < 0 || np[2] < 0 || np[1]<0 || np[0]<0
			|| (np[0]+np[1]+np[2]+np[3]) != (long)onpixels )
			debug=die=1;
		if( die ) NERROR1("roundoff error!?");
#endif /* DEBUG */
		for(i=0;i<4;i++)
			giv_pixels(qtp->qt_child[i],np[i]);
	}
	qtp->qt_onpix = onpixels;
}


