#include "quip_config.h"

#include "quip_prot.h"
#include "server.h"
#include "my_encryption.h"
#include "strbuf.h"
#include "quip_menu.h"

static COMMAND_FUNC( do_read_file_from_server )
{
	const char *url;
	String_Buf *file_contents;

	url=NAMEOF("remote file name or URL");

	file_contents = get_url_contents(QSP_ARG  url);

	if( file_contents == NULL ) return;

	// Before we push the file contents,
	// check the filename to see if it is an encrypted file...

	if( has_encryption_suffix(url) ){
		String_Buf *sbp;
		sbp = decrypt_text(file_contents->sb_buf);
		if( sbp == NULL ){
			WARN("error decrypting URL text");
			return;
		}
		file_contents = sbp;
	}

	PUSH_TEXT( file_contents->sb_buf, url );

	exec_quip(SINGLE_QSP_ARG);				// interpret the commands!

	// BUG free the memory here!
	return;
}

static COMMAND_FUNC( do_get_file_from_server )
{
	const char *url;
	const char *s;
	FILE *fp_out=NULL;

	url=NAMEOF("remote file name or URL");
	s=NAMEOF("output filename");

	fp_out = TRY_OPEN(s,"w");
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

	PUSH_MENU(http);
}

