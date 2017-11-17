#include "quip_config.h"

#include "quip_prot.h"
#include "server.h"
#include "my_encryption.h"
#include "quip_menu.h"

static COMMAND_FUNC( do_read_file_from_server )
{
	const char *url;
	String_Buf *file_content_sbp;

	url=NAMEOF("remote file name or URL");

	file_content_sbp = get_url_contents(QSP_ARG  url);

	if( file_content_sbp == NULL ) return;

	// Before we push the file contents,
	// check the filename to see if it is an encrypted file...

	if( has_encryption_suffix(url) ){
		String_Buf *sbp;
		sbp = decrypt_text(sb_buffer(file_content_sbp));
		rls_stringbuf(file_content_sbp);
		if( sbp == NULL ){
			WARN("error decrypting URL text");
			return;
		}
		file_content_sbp = sbp;
	}

	push_text( sb_buffer(file_content_sbp), url );

	exec_quip(SINGLE_QSP_ARG);				// interpret the commands!

	// free the memory here!
	// BUG?  make sure not HALTING?
	rls_stringbuf(file_content_sbp);

	return;
}

static COMMAND_FUNC( do_get_file_from_server )
{
	const char *url;
	const char *s;
	FILE *fp_out=NULL;

	url=NAMEOF("remote file name or URL");
	s=NAMEOF("output filename");

	fp_out = try_open(s,"w");
	if( fp_out == NULL ) return;

	write_file_from_url( QSP_ARG  fp_out, url );

#ifndef BUILD_FOR_IOS
	// in iOS, data is received asynchronously, so we have to
	// return and wait for that to happen before closing the file.
	// This means we have to be careful in our scripts, we can't
	// assume the data is available after the command executes.

	fclose(fp_out);
#endif /* BUILD_FOR_IOS */
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(http_menu,s,f,h)

MENU_BEGIN(http)
ADD_CMD( save_url,	do_get_file_from_server,	download and store a URL )
ADD_CMD( read_url,	do_read_file_from_server,	download and interpret a URL )
MENU_END(http)

COMMAND_FUNC( do_http_menu )
{
	static int inited=0;

	if( ! inited ){
		init_http_subsystem(SINGLE_QSP_ARG);
		inited=1;
	}

	CHECK_AND_PUSH_MENU(http);
}

