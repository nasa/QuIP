
#ifndef ASYNC_H
#define ASYNC_H

#ifdef INC_VERSION
char VersionId_inc_async[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

extern COMMAND_FUNC( do_fork );
extern COMMAND_FUNC( wait_child );

#endif /* ! ASYNC_H */

