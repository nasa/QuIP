#include <math.h>
#include <string.h>
#include <ctype.h>
#include "quip_config.h"
#include "quip_prot.h"
#include "my_encryption.h"
#include "fileck.h"
#include "getbuf.h"
#include "strbuf.h"

#ifdef HAVE_SECRET_KEY

#include "secret_key.c"

#else

void gen_string( char *buf, int buflen )
{
	int i;
	const char *s;

	if( buflen < 33 ){
		sprintf(DEFAULT_ERROR_STRING,"gen_string:  buffer length (%d) should be at least 33",buflen);
		NWARN(DEFAULT_ERROR_STRING);
		for(i=0;i<buflen;i++){
			buf[i] = 'a' + i;
		}
		buf[i]=0;
		return;
	}
	/* Check the environment */
	/* would be better to get this from a script variable (which could be loaded from the environment) */
	s=getenv("ENCRYPTION_KEY");
	if( s != NULL ){
		int j;
		i=j=0;
		while( i<strlen(s) && i < (buflen-1) ){
			buf[i] = s[i];
			i++;
		}
		while(i<buflen){
			buf[i++] = 0;
		}
		return;
	}

	/* Use something arbitrary */
	i=0;
	while(i<(buflen-1)){
		buf[i] = 'a' + i;
		i++;
	}
	buf[i]=0;
}

#endif /* ! HAVE_SECRET_KEY */

/* Make the key */

int init_my_symm_key(void **key_ptr)
{
	/* For an AES key, we need 128 bits (16 bytes).
	 * Our 'secret' string is 33 characters, so
	 * we add pairs to get 16 bytes.
	 */

#define STRBUF_LEN	40
#define KEYARR_LEN	16
	char rawcryptokeyarr[KEYARR_LEN];
	char str[STRBUF_LEN];
	int i;
	void *my_symm_key;
	int klen;

	gen_string(str,STRBUF_LEN);
	for(i=0;i<KEYARR_LEN;i++)
		rawcryptokeyarr[i] = str[i] + str[KEYARR_LEN+i];

	klen = hash_my_key(&my_symm_key,rawcryptokeyarr,KEYARR_LEN);
	if( klen < encryption_key_size() ){
		NWARN("init_my_symm_key:  hash length less than required key length!?");
		my_symm_key=NULL;
		return -1;
	}

	*key_ptr = my_symm_key;
	return 0;
}

/* utilities for encryption and decryption of data.
 */

/* utilities for converting hex ascii to binary */

static int value_of_hex_digit(int c)
{
	int d;

	if( isdigit(c) ){
		d=c-'0';
	} else if( islower(c) && c<='f' ){
		d= 10 + c - 'a';
	} else if( isupper(c) && c<='F' ){
		d= 10 + c - 'A';
	} else {
		sprintf(DEFAULT_ERROR_STRING,
	"value_of_hex_digit:  character 0x%x is not a valid hex digit!?",
			c);
		NWARN(DEFAULT_ERROR_STRING);
		d = -1;
	}
	return d;
}

#define NEXT_DIGIT(var)						\
								\
	{							\
		var=value_of_hex_digit(*s);			\
		if( var < 0 ) return var;	/* error */	\
		s++;						\
	}

// convert from hex assumes a string with no white space

static int convert_from_hex(uint8_t *buf, const char *s)
{
	int n_converted=0;

	while( *s ){
		int v,d1,d2;

		NEXT_DIGIT(d1)

		if( ! (*s) ){
	NWARN("convert_from_hex:  input string has an odd # chars!?");
			return -1;
		}

		NEXT_DIGIT(d2)

		v = (d1 << 4) | d2;
		*buf ++ = (uint8_t)v;
		n_converted++;
	}
	return n_converted;
}

/*********** convert an encrypted buffer to a printable string *****/

#define HEX_CHAR( d )	( (d) < 10 ? '0'+(d) : 'a'+(d)-10 )

static void format_hex(char *s,const uint8_t *input, size_t len)
{
	while( len-- ){
		*s++ = HEX_CHAR( ((*input)>>4) & 0xf );
		*s++ = HEX_CHAR( (*input) & 0xf );
		input++;
	}
	*s=0;
}

#define MAX_BYTES_PER_LINE	32

static void print_hex_data(FILE *fp, const uint8_t *buf, size_t len)
{
	size_t n_remaining, n_this_line;
	char linechars[2*MAX_BYTES_PER_LINE+2];

	n_remaining=len;
	while( n_remaining ){
		n_this_line = n_remaining > MAX_BYTES_PER_LINE ?
				MAX_BYTES_PER_LINE : n_remaining;
		format_hex(linechars,buf,(int)n_this_line);
		linechars[n_this_line*2]='\n';
		linechars[1+n_this_line*2]=0;
		if( fputs(linechars,fp) == EOF ){
			NWARN("Error writing hex line");
			return;
		}
		buf += n_this_line;
		n_remaining -= n_this_line;
	}
}

int has_encryption_suffix(const char *name)
{
	const char *s;

	s = name + strlen(name);	// start at the end...
	while( s >= name && *s != '.' )
		s--;
	if( *s == '.' ) s++;
	if( !strcmp(s,ENCRYPTED_SCRIPT_SUFFIX) )
		return 1;
	return 0;
}

#define PRINT_ERR1(fmt,arg)				\
	{						\
		sprintf(DEFAULT_ERROR_STRING,fmt,arg);	\
		NWARN(DEFAULT_ERROR_STRING);		\
	}

#define PRINT_ERR2(fmt,arg1,arg2)				\
	{						\
		sprintf(DEFAULT_ERROR_STRING,fmt,arg1,arg2);	\
		NWARN(DEFAULT_ERROR_STRING);		\
	}

/* The original motivation for this is to be able
 * to conceal the scripts in the bundle that implement
 * the administrator back-door.  The basic strategy
 * is that we have a fixed key that the program uses;
 * we generate underlying string programmatically, so
 * that the string doesn't exist in the executable file.
 */

static const char *encrypt_string(const char *input_string)
{
	size_t n,l,max_raw_size;
	uint8_t *rawbuf;

	l=(int)strlen(input_string);

	// may have to pad - ensure the output buffer can hold
	// an integral number of blocks.

	SET_OUTPUT_SIZE( max_raw_size , l );

	rawbuf = (uint8_t *)getbuf(max_raw_size);

	n=encrypt_char_buf(input_string,l,rawbuf,max_raw_size);

	if( n <= 0 ){
		NWARN("encryption failed!?");
		givbuf(rawbuf);
		return NULL;
	} else {
		char *asciibuf;

		asciibuf = (char *)getbuf(2*(1+max_raw_size));
		format_hex(asciibuf,rawbuf,n);
		givbuf(rawbuf);
		return asciibuf;
	}
}

static const char *decrypt_string(const char *input_string)
{
	uint8_t *buf;
	size_t buflen;
	char *asciibuf;
	size_t n;

	if( strlen(input_string) & 1 ){
		NWARN(
	"decrypt_string:  input string has an odd number of chars!?");
		return NULL;
	}

	buflen=strlen(input_string)/2;
	buf = (uint8_t *)getbuf(buflen);
	if( convert_from_hex(buf,input_string) < 0 ){
		NWARN("error converting hex string for decryption");
		return NULL;
	}
	asciibuf = (char *)getbuf(buflen+1);

	n=decrypt_char_buf(buf,buflen,asciibuf,buflen);
	givbuf(buf);

//#ifdef CAUTIOUS
//	if( n > buflen ){
//		NWARN("CAUTIOUS:  too many decrypted chars!?");
//		return NULL;
//	}
//#endif /* CAUTIOUS */
	assert( n <= buflen );

	if( n <= 0 ){
		NWARN("decryption failed!?");
		givbuf(asciibuf);
		return NULL;
	} else {
		asciibuf[n]=0;	// terminate string
		return asciibuf;
	}

} // end decrypt_string

static void encrypt_file(QSP_ARG_DECL  FILE *fp_in, FILE *fp_out )
{
	long n_in;
	char *inbuf;
	uint8_t *outbuf;
	long max_out_len;
	long n_converted;

	// We use stat to get the size of the file, and then
	// create a buffer to hold the whole thing for encryption.
	// This is probably OK as long as we are just encrypting
	// small-to-medium sized script files.  If we start doing
	// images we will have to rethink, or go to a streaming
	// coder.  (Although we are probably OK if we process chunks
	// which are multiples of the blocksize.)

	n_in = (long) fp_content_size(QSP_ARG  fp_in);

//#ifdef CAUTIOUS
//	if( n_in < 0 ){
//		NWARN("CAUTIOUS:  encrypt_file:  couldn't determine input file size!?");
//		return;
//	}
//#endif /* CAUTIOUS */
	assert( n_in >= 0 );

	if( n_in == 0 ){
		NWARN("encrypt_file:  input file is empty!?");
		return;
	}

	inbuf = getbuf(n_in);

	if( fread(inbuf,1,n_in,fp_in) != n_in ){
		WARN("do_encrypt_file:  error reading input data");
		return;
	}

	/* Now we have the data */

	SET_OUTPUT_SIZE( max_out_len , n_in );

	outbuf = getbuf(max_out_len);
	n_converted=encrypt_char_buf(inbuf,n_in,outbuf,max_out_len);
	if( n_converted < 0 ) goto cleanup;

	print_hex_data(fp_out,outbuf,n_converted);

cleanup:
	givbuf(outbuf);
	givbuf(inbuf);
} // end encrypt_file

// Should we pass the max buffer size?
// We know the size of the text when we allocate the data buffer,
// So as long as we only call here we should be OK.

#define MAX_LINE_SIZE	512

static long convert_lines_from_hex(uint8_t *data, const char *text)
{
	char linebuf[MAX_LINE_SIZE];
	const char *line_end, *line_start;
	long total_converted=0;
	long n_converted;
	long n_to_copy;

	// We find the newline's in the text,
	// copy each line to our local buffer,
	// then convert it.

	line_start = text;
	while( *line_start ){
		// find the end of the line
		line_end = strstr(line_start,"\n");
		if( line_end == NULL ){		// no newline found
			// do something special here?
			NWARN("convert_lines_from_hex:  missing final newline");
			line_end = line_start+strlen(line_start);
		}
		n_to_copy = line_end - line_start;
//#ifdef CAUTIOUS
//		if( n_to_copy <= 0 ){
//			NWARN("CAUTIOUS:  convert_lines_from_hex:  empty line!?");
//			return -1;
//		}
//#endif /* CAUTIOUS */
		assert( n_to_copy > 0 );

		if( n_to_copy >= MAX_LINE_SIZE ){
			NWARN("convert_lines_from_hex:  line too long!?");
			return -1;
		}
		strncpy(linebuf,line_start,n_to_copy);
		linebuf[n_to_copy] = 0;

		n_converted = convert_from_hex(data,linebuf);
		if( n_converted < 0 ){
	NWARN("decrypt_file:  error converting hex string for decryption");
			return -1;
		}
		data += n_converted;
		total_converted += n_converted;
		line_end++;
		line_start = line_end;
	}
	return total_converted;
}

String_Buf *decrypt_text( const char *text )
{
	uint8_t *data;
	size_t buf_size,n_bytes,n_decrypted;
	String_Buf *out_sbp=NULL;
		
	buf_size=1+(size_t)ceil(strlen(text)/2);
	data = getbuf(buf_size);
	n_bytes = convert_lines_from_hex(data,text);
	if( n_bytes > 0 ) {
		out_sbp = new_stringbuf();
		if( n_bytes > out_sbp->sb_size )
			enlarge_buffer(out_sbp,n_bytes);

		n_decrypted=decrypt_char_buf(data,n_bytes,out_sbp->sb_buf,n_bytes);
		if( n_decrypted <= 0 ){
			rls_stringbuf(out_sbp);
			out_sbp = NULL;
		}
	}
	givbuf(data);
	return out_sbp;
}

char *decrypt_file_contents(QSP_ARG_DECL  FILE *fp_in,
					size_t *count_p)
{
	long n_in;
	uint8_t *inbuf;
	char *outbuf;
	size_t buf_size;
	long total_converted=0;
	long n_decrypted;
	char *text_buf;


	// We assume the file is hex-encoded, so we
	// have twice as many chars as we need...
	// We also have newlines that we don't care
	// about, but for now we don't worry about
	// them

	n_in = (long) fp_content_size(QSP_ARG  fp_in);

//#ifdef CAUTIOUS
//	if( n_in < 0 ){
//		NWARN("CAUTIOUS:  decrypt_file_contents:  couldn't determine input file size!?");
//		return NULL;
//	}
//#endif /* CAUTIOUS */
	assert( n_in >= 0 );

	if( n_in == 0 ){
		WARN("decrypt_file_contents:  file is empty!?");
		return NULL;
	}

	// We probably have extra chars because of newlines,
	// but we don't assume it here - although our use of
	// fgets assumes the file is line-oriented...

	buf_size=1+(size_t)ceil(n_in/2);
	inbuf = getbuf(buf_size);

#ifdef READ_LINE_BY_LINE
	char line_buf[MAX_LINE_SIZE];	// the actual line should be much smaller
	long n_converted;

	uint8_t *buf=inbuf;
	while( fgets(line_buf,MAX_LINE_SIZE,fp_in) != NULL ){
		int i;
		i=strlen(line_buf)-1;
		if( line_buf[i] == '\n' ) line_buf[i]=0;

		// convert from hex
		n_converted = convert_from_hex(buf,line_buf);
		if( n_converted < 0 ){
	NWARN("decrypt_file:  error converting hex string for decryption");
			goto cleanup1;
		}
		buf += n_converted;
		total_converted += n_converted;
	}
#else
	// read entire file as a block

// When the file is large, this buffer can get big...
	text_buf = getbuf(1+n_in);
	if( fread(text_buf,1,n_in,fp_in) != n_in ){
		WARN("decrypt_file_contents:  error reading file contents!?");
		goto cleanup1;
	}
	text_buf[n_in]=0;	// make sure null-terminated string

	total_converted = convert_lines_from_hex(inbuf,text_buf);
	givbuf(text_buf);

	if( total_converted <= 0 ){
		WARN("decrypt_file_contents:  error converting hex lines!?");
		goto cleanup1;
	}

#endif /* ! READ_LINE_BY_LINE */


	// BUG?  we should to a CAUTIOUS check here to confirm
	// that the number of converted bytes is a multiple of
	// the blocksize...

	outbuf = getbuf(total_converted+1);	// will add nul termination
	n_decrypted=decrypt_char_buf(inbuf,total_converted,
					outbuf,total_converted);
	if( n_decrypted < 0 ) goto cleanup2;
	outbuf[n_decrypted]=0;	// terminate string

	givbuf(inbuf);
	*count_p = n_decrypted;
	return outbuf;

cleanup2:
	givbuf(outbuf);
cleanup1:
	givbuf(inbuf);
	*count_p = 0;
	return NULL;

} // end decrypt_file_contents

static void decrypt_file(QSP_ARG_DECL  FILE *fp_in, FILE *fp_out )
{
	char *outbuf;
	size_t n_converted;

	outbuf = decrypt_file_contents(QSP_ARG  fp_in,&n_converted);
	if( outbuf == NULL ){
		WARN("decrypt_file:  failed!?");
		return;
	}

	// Now all the data is decrypted...
	if( fwrite(outbuf,1,n_converted,fp_out) != n_converted ){
		WARN("do_decrypt_file:  error writing input data");
	}

	givbuf(outbuf);
}

COMMAND_FUNC( do_encrypt_string )
{
	const char *vn;
	const char *s;
	const char *e;

	vn=NAMEOF("variable name for result");
	s=NAMEOF("string to encrypt");

	e = encrypt_string(s);

	if( e != NULL ){
		ASSIGN_VAR(vn,e);
		rls_str(e);
	} else
		WARN("Encryption failed.");
}

COMMAND_FUNC( do_decrypt_string )
{
	const char *vn;
	const char *s;
	const char *d;

	vn=NAMEOF("variable name for result");
	s=NAMEOF("string to decrypt");

	d = decrypt_string(s);

	if( d != NULL ){
		ASSIGN_VAR(vn,d);
		rls_str(d);
	} else
		WARN("Decryption failed.");
}

COMMAND_FUNC( do_encrypt_file )
{
	const char *infile_name;
	const char *outfile_name;
	FILE *fp_in, *fp_out;

	infile_name = NAMEOF("input filename");
	outfile_name = NAMEOF("output filename");

	fp_in = TRY_OPEN(infile_name,"r");
	if( ! fp_in ) return;

	fp_out = TRY_OPEN(outfile_name,"w");
	if( !fp_out ){
		fclose(fp_in);
		return;
	}

	encrypt_file(QSP_ARG  fp_in, fp_out );

	// fall through

	fclose(fp_in);
	fclose(fp_out);
	return;
}

COMMAND_FUNC( do_decrypt_file )
{
	const char *infile_name;
	const char *outfile_name;
	FILE *fp_in, *fp_out;

	infile_name = NAMEOF("input filename");
	outfile_name = NAMEOF("output filename");

	fp_in = TRY_OPEN(infile_name,"r");
	if( ! fp_in ) return;

	fp_out = TRY_OPEN(outfile_name,"w");
	if( !fp_out ){
		fclose(fp_in);
		return;
	}

	decrypt_file(QSP_ARG  fp_in, fp_out );

	fclose(fp_in);
	fclose(fp_out);
	return;
}

COMMAND_FUNC( do_read_encrypted_file )
{
	FILE *fp;
	const char *s;
	char *outbuf;
	size_t n_converted;


	s=NAMEOF("input file name");
	fp=TRY_OPEN(s,"r");
	if( fp == NULL ) return;

	outbuf = decrypt_file_contents(QSP_ARG  fp,&n_converted);
	if( outbuf == NULL ){
		WARN("read_encrypted_file:  decryption failed!?");
		return;
	}
	PUSH_TEXT(outbuf,s);

	// Should we call exec_quip here?
	// We should not need to - we are in a command already!?
	// But if we don't then we can't release the buffer...
	// This assumes that PUSH_TEXT does not make a copy...
	exec_quip(SINGLE_QSP_ARG);

	// Now we should be done with the file contents
	// BUG?  can we be sure that we didn't halt because of an alert???
	// should we check status of HALTING???

	rls_str(outbuf);
}

