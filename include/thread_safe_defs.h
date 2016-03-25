// for the unix version, this file includes autoconf's config.h...

#ifndef _THREAD_SAFE_DEFS_H_
#define _THREAD_SAFE_DEFS_H_

struct query_stack;

#ifdef THREAD_SAFE_QUERY

#define QSP_DECL		Query_Stack *qsp;
#define THIS_QSP		qsp
#define SINGLE_QSP_ARG_DECL	struct query_stack *qsp
#define TMP_SINGLE_QSP_ARG_DECL	void *qsp
#define QSP_ARG_DECL		Query_Stack *qsp,
#define TMP_QSP_ARG_DECL	struct query_stack *qsp,
#define TMP_QSP_ARG		(void *)qsp,
#define QSP_ARG			qsp,
#define SINGLE_QSP_ARG		qsp
#define DEFAULT_QSP_ARG		DEFAULT_QSP,
#define NULL_QSP_ARG		NULL,
#define SGL_DEFAULT_QSP_ARG	DEFAULT_QSP
#define SELF_QSP		self
#define SELF_QSP_ARG		self,
#define FGETS			qpfgets
extern char *qpfgets(TMP_QSP_ARG_DECL  void *buf, int size, void *fp);

#ifdef HAVE_PTHREADS
#define LOCK_ITEM_TYPE(itp)	// BUG put correct code here
#define UNLOCK_ITEM_TYPE(itp)	// BUG put correct code here
#else /* ! HAVE_PTHREADS */
#define LOCK_ITEM_TYPE(itp)	// do nothing
#define UNLOCK_ITEM_TYPE(itp)	// do nothing
#endif /* ! HAVE_PTHREADS */

#define DEFAULT_ERROR_STRING	DEFAULT_QSP->qs_error_string
#define ERROR_STRING		qsp->qs_error_string
#define DEFAULT_MSG_STR		(DEFAULT_QSP->qs_msg_str)
#define MSG_STR			(qsp->qs_msg_str)


#else /* ! THREAD_SAFE_QUERY */

#define QSP_DECL
#define THIS_QSP		DEFAULT_QSP
#define SINGLE_QSP_ARG_DECL	void
#define TMP_SINGLE_QSP_ARG_DECL	void
#define QSP_ARG_DECL
#define TMP_QSP_ARG_DECL
#define TMP_QSP_ARG
#define QSP_ARG
#define SINGLE_QSP_ARG
#define DEFAULT_QSP_ARG
#define NULL_QSP_ARG
#define SGL_DEFAULT_QSP_ARG
#define SELF_QSP
#define SELF_QSP_ARG
#define FGETS			fgets

#define DEFAULT_ERROR_STRING	DEFAULT_QSP->qs_error_string
#define ERROR_STRING		DEFAULT_ERROR_STRING
#define DEFAULT_MSG_STR		(DEFAULT_QSP->qs_msg_str)
#define MSG_STR			DEFAULT_MSG_STR

#endif /* ! THREAD_SAFE_QUERY */


#define /* ! _THREAD_SAFE_DEFS_H_ */




