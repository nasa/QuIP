#ifndef _SERVER_H_
#define _SERVER_H_

// BUG - this needs to also support the old w3c library?

#include "quip_config.h"
#include "quip_prot.h"

extern COMMAND_FUNC( do_http_menu );

extern String_Buf *_get_url_contents(QSP_ARG_DECL  const char *url);
extern void _write_file_from_url( QSP_ARG_DECL  FILE *fp_out, const char *url );
extern void _init_http_subsystem(SINGLE_QSP_ARG_DECL);

#define get_url_contents(url) _get_url_contents(QSP_ARG  url)
#define write_file_from_url( fp_out, url ) _write_file_from_url( QSP_ARG  fp_out, url )
#define init_http_subsystem() _init_http_subsystem(SINGLE_QSP_ARG)

#endif /* ! _SERVER_H_ */

