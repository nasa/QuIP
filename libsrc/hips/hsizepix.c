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
 * hsizepix - compute the size (in bytes) of a single image pixel
 */

#include "fio_api.h"
#include "quip_prot.h"
#include "hips/hips2.h"

hsize_t hbitsperpixel(int pfmt);

hsize_t hsizepix(int pfmt)
{
	hsize_t s;

	switch(pfmt) {
	case PFBYTE:	s = sizeof(h_byte); break;
	case PFSHORT:	s = sizeof(short); break;
	case PFINT:	s = sizeof(int); break;
	case PFFLOAT:	s = sizeof(float); break;
	case PFCOMPLEX:	s = 2*sizeof(float); break;
	case PFDOUBLE:	s = sizeof(double); break;
	case PFDBLCOM:	s = 2*sizeof(double); break;
	case PFSBYTE:	s = sizeof(sbyte); break;
	case PFUSHORT:	s = sizeof(h_ushort); break;
	case PFUINT:	s = sizeof(h_uint); break;
	case PFRGB:
	case PFBGR:	s = 3*sizeof(h_byte); break;
	case PFRGBZ:
	case PFZRGB:
	case PFBGRZ:
	case PFZBGR:	s = 4*sizeof(h_byte); break;
	case PFSTEREO:	s = sizeof(h_byte); break;
	case PFINTPYR:	s = sizeof(int); break;
	case PFFLOATPYR:s = sizeof(float); break;
	default:	s = 0; break;
	}
	return(s);
}

hsize_t hbitsperpixel(int pfmt)
{
	switch(pfmt) {
	case PFBYTE:	return(8);
	case PFSHORT:	return(16);
	case PFINT:	return(32);
	case PFFLOAT:	return(32);
	case PFCOMPLEX:	return(64);
	case PFDOUBLE:	return(64);
	case PFDBLCOM:	return(128);
	case PFSBYTE:	return(8);
	case PFUSHORT:	return(16);
	case PFUINT:	return(32);
	case PFRGB:
	case PFBGR:
	case PFRGBZ:
	case PFZRGB:
	case PFBGRZ:
	case PFZBGR:	return(24);
	case PFSTEREO:	return(8);
	case PFINTPYR:	return(32);
	case PFFLOATPYR:return(32);
	default:	return(HIPS_OK);
	}
}
