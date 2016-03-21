
#ifndef NO_GET_HDR

#include "data_obj.h"
#include "hipl_fmt.h"

extern   int fh_err(Header *hd,const char *s);
extern   int read_chr(int fd);
extern   char *gtline(int fd);
extern   int dfscanf(int fd,Header *hd);
extern   char *catlines(const char *s1,const char *s2);
extern   int ftch_header(int fd,Header *hd);

#define NO_GET_HDR
#endif /* NO_GET_HDR */

