

#ifndef NO_STRINGBUF

#include "typedefs.h"

/* string buffer structure */

typedef struct string_buffer {
	char *		sb_buf;
	u_int		sb_size;
} String_Buf;


#define NO_STRINGBUF ((String_Buf *)NULL)


void enlarge_buffer(String_Buf *sbp,u_int size);
void copy_string(String_Buf *sbp,const char *str);
void copy_strbuf(String_Buf *dst_sbp,String_Buf *src_sbp);
void cat_string(String_Buf *sbp,const char *str);


#endif /* ! NO_STRINGBUF */

