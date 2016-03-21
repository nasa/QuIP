
#ifndef _DV_GLOBALS_H_
#define _DV_GLOBALS_H_

#include <pthread.h>
#include "quip_prot.h"

// C++ has bool...
#ifndef __cplusplus
//typedef int bool;		// now this seems to be coming from somewhere else - where?
#endif

#define FALSE	0
#define TRUE	1


extern List *g_output_queue;
extern List *g_buffer_queue;

extern int queue_size(List *);
extern void *queue_front(List *);
extern void queue_pop_front(List *);
extern void queue_push_back(List *,void *);

//extern volatile bool	g_reader_active;
extern bool	g_reader_active;
extern bool	g_buffer_underrun;
extern int	g_card;
extern int	g_channel;
extern pthread_mutex_t	g_mutex;
extern char *g_dv1394;


// ieee1394io.c

extern void *read_frames(void *);

// error_util.c

//void real_fail_null(const void *eval, const char *eval_str, const char *func, const char *file, int line);

#endif /* _DV_GLOBALS_H_ */

