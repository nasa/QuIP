
#ifndef _JPEG_HDR_H_
#define _JPEG_HDR_H_

#ifdef INC_VERSION
char VersionId_inc_jpeg_hdr[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

#ifdef HAVE_INTTYPES_H
#include <inttypes.h>
#else
#error "inttypes.h not present, need to define print/scan formats for seek_tbl_type!?"
#endif

//typedef uint64_t seek_tbl_type;
#define seek_tbl_type uint64_t
#define SEEK_TBL_PRI_FMT	PRId64
#define SEEK_TBL_SCN_FMT	SCNd64

struct jpeg_hdr;
typedef struct jpeg_hdr Jpeg_Hdr;


#endif /* _JPEG_HDR_H_ */

