#ifndef _STRING_REF_H_

typedef struct string_ref {
	const char *	sr_string;
	int		sr_count;
} String_Ref;

extern String_Ref *save_stringref(const char *s);
extern void rls_stringref(String_Ref *);

#define SR_STRING(srp)		(srp)->sr_string
#define SR_COUNT(srp)		(srp)->sr_count

#define INC_SR_COUNT(srp)	(srp)->sr_count ++
#define DEC_SR_COUNT(srp)	(srp)->sr_count --

#define _STRING_REF_H_

#endif // _STRING_REF_H_

