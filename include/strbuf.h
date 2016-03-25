

#ifndef NO_STRINGBUF

#include "typedefs.h"

/* string buffer structure */

typedef struct string_buffer {
	char *		sb_buf;
	size_t		sb_size;
} String_Buf;


#define NO_STRINGBUF ((String_Buf *)NULL)

/* String_Buf */
/* For now this is not an object... */
#define NEW_STRINGBUF		new_stringbuf()
#define SB_BUF(sbp)		sbp->sb_buf
#define SB_SIZE(sbp)		sbp->sb_size
#define SET_SB_BUF(sbp,s)	sbp->sb_buf = s
#define SET_SB_SIZE(sbp,n)	sbp->sb_size = n


extern void enlarge_buffer(String_Buf *sbp,size_t size);
extern void copy_string(String_Buf *sbp,const char *str);
extern void copy_strbuf(String_Buf *dst_sbp,String_Buf *src_sbp);
extern void cat_string(String_Buf *sbp,const char *str);
extern void copy_string_n(String_Buf *sbp,const char *str,int n);
extern void cat_string_n(String_Buf *sbp,const char *str, int n);

extern String_Buf *new_stringbuf(void);
extern void rls_stringbuf(String_Buf *);

#endif /* ! NO_STRINGBUF */

