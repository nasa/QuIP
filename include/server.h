#ifndef _SERVER_H_
#define _SERVER_H_

// BUG - this needs to also support the old w3c library?

#include "quip_config.h"
#include "quip_prot.h"

extern COMMAND_FUNC( do_http_menu );

extern String_Buf *get_url_contents(QSP_ARG_DECL  const char *url);
extern void write_file_from_url( QSP_ARG_DECL  FILE *fp_out, const char *url );
extern void init_http_subsystem(SINGLE_QSP_ARG_DECL);

#endif /* ! _SERVER_H_ */

