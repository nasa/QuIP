/* Linkage to libgcrypt */
#include "quip_config.h"
#include "quip_prot.h"
#include "query_stack.h"
#include "my_encryption.h"

#ifdef HAVE_LIBGCRYPT

#ifdef BUILD_FOR_WINDOWS
#include "win_gcrypt.h"
#else // ! BUILD_FOR_WINDOWS

#define GCRYPT_NO_DEPRECATED
#include <gcrypt.h>
#endif // ! BUILD_FOR_WINDOWS

/************* symmetric encryption *********************/

/* cipher */
static int the_crypt_algo = GCRY_CIPHER_AES128;

/* cipher modes - do we need this? */
static int the_crypt_mode = GCRY_CIPHER_MODE_CBC;
/* Possible block cipher modes:
	GCRY_CIPHER_MODE_AESWRAP
	GCRY_CIPHER_MODE_ECB
	GCRY_CIPHER_MODE_CBC
	GCRY_CIPHER_MODE_CFB
	GCRY_CIPHER_MODE_OFB
	GCRY_CIPHER_MODE_CTR
*/

static int the_crypt_flags=0;

static gcry_cipher_hd_t my_cipher_handle=NULL;


/****** hashing *********/

static gcry_md_hd_t my_hash_hdl=NULL;

static int the_hash_algo=GCRY_MD_SHA256;	// 32 byte message digest
//GCRY_MD_SHA512		// 64 byte message digest

static int the_hash_flags=0;

static int the_hash_len=32;

#define CHECK_STATUS(func,call)				\
	if( status != 0 ){				\
		report_gcrypt_error(#func,#call,status);	\
	}

static void report_gcrypt_error(const char *whence,
				const char *call, gcry_error_t status)
{
	sprintf(DEFAULT_ERROR_STRING,
		"%s:  %s:  %s", whence,call, gcry_strerror(status));
	NWARN(DEFAULT_ERROR_STRING);
}


static void init_gcrypt_subsystem(void)
{
	const char *s;
	gcry_error_t status;
	void *key;
	int key_len;

	s=gcry_check_version(GCRYPT_VERSION);	
	if( !s ){	// mismatch means wrong dynamic library
		sprintf(DEFAULT_ERROR_STRING,
			"Expected libgcrypt version %s!?", GCRYPT_VERSION);
		NWARN(DEFAULT_ERROR_STRING);
		NERROR1("libgcrypt version mismatch!?");
	}

	if( verbose ){
		sprintf(DEFAULT_ERROR_STRING,"libgcrypt version %s",s);
		NADVISE(DEFAULT_ERROR_STRING);
	}

#ifdef USE_SECURE_MEMORY

	/* We don't want to see any warnings, e.g. because we have not yet
	 * parsed program options which might be used to suppress such
	 * warnings.
	 */
NADVISE("libgcrypt will use secure memory...");
	gcry_control (GCRYCTL_SUSPEND_SECMEM_WARN);

	/* ... If required, other initialization goes here.  Note that the
	 * process might still be running with increased privileges and that
	 * the secure memory has not been intialized.
	 */

	/* Allocate a pool of 16k secure memory.  This make the secure memory
	 * available and also drops privileges where needed.
	 */
	gcry_control (GCRYCTL_INIT_SECMEM, 16384, 0);


	/* It is now okay to let Libgcrypt complain when there was/is
	 * a problem with the secure memory.
	 */
	gcry_control (GCRYCTL_RESUME_SECMEM_WARN);

	/* ... If required, other initialization goes here.  */

	/* Tell Libgcrypt that initialization has completed. */
	gcry_control (GCRYCTL_INITIALIZATION_FINISHED, 0);

#else /* ! USE_SECURE_MEMORY */

//NADVISE("libgcrypt will NOT use secure memory...");
	// This initialization assumes that the environment
	// is secure, so that secure memory does not need to
	// be used for key storage...

	/* Disable secure memory.  */
	gcry_control (GCRYCTL_DISABLE_SECMEM, 0);

	/* ... If required, other initialization goes here.  */

	/* Tell Libgcrypt that initialization has completed. */
	gcry_control (GCRYCTL_INITIALIZATION_FINISHED, 0);

#endif /* ! USE_SECURE_MEMORY */

	/* self-test fails on euler!? */
	status = gcry_control (GCRYCTL_SELFTEST);
	CHECK_STATUS(init_gcrypt_subsystem,GCRYCTL_SELFTEST)

	status = gcry_md_open(&my_hash_hdl,the_hash_algo, the_hash_flags);
	CHECK_STATUS(init_gcrypt_subsystem,gcry_md_open)


	/* Now initialize for encryption */

	status = gcry_cipher_open( &my_cipher_handle,
			the_crypt_algo,the_crypt_mode,the_crypt_flags);
	CHECK_STATUS(init_gcrypt_subsystem,gcry_cipher_open)

// to release:
//	gcry_cipher_close(my_cipher_handle);

	/* get the key */
	init_my_symm_key(&key);
	key_len = encryption_key_size();

	status = gcry_cipher_setkey(my_cipher_handle,key,key_len);
	CHECK_STATUS(init_gcrypt_subsystem,gcry_cipher_setkey)

//	status = gcry_cipher_setiv(my_cipher_handle,iv,iv_len);
//	status = gcry_cipher_reset(my_cipher_handle);

}

/* We need to implement the interface in my_encryption.h */

// How to deal with padding?  We pad with nulls, and then
// we decrypt the same nulls...  For the time being we might
// assume that any null characters at the end of the buffer
// are pad chars - but what if we want to encrypt arbitrary
// data, where a null byte might be a legitimate value???

size_t decrypt_char_buf(const uint8_t *in_buf, size_t in_len, char *out_buf, size_t out_len )
{
	gcry_error_t status;
	int bs;

	if( my_cipher_handle == NULL )
		init_gcrypt_subsystem();

	bs = encryption_block_size();
	if( bs <= 0 ) return 0;	// when lib not present?

	if( (in_len % bs) != 0 ){
		sprintf(DEFAULT_ERROR_STRING,
"decrypt_char_buf:  input size (%ld) is not an integral number of blocks (bs = %d)!?",
			(long)in_len,bs);
		NWARN(DEFAULT_ERROR_STRING);
		return 0;
	}

	status = gcry_cipher_decrypt(my_cipher_handle,
			out_buf,out_len,in_buf,in_len);
	CHECK_STATUS(decrypt_char_buf,gcry_cipher_decrypt)
	// how do we know how many chars actually written?

	status = gcry_cipher_reset(my_cipher_handle);
	CHECK_STATUS(decrypt_char_buf,gcry_cipher_reset)

	// If we padded, there will be zeroes at the end of the
	// message - we might assume that they are invalid,
	// as we are generally encrypting C strings?

	while( out_len >= 1 && out_buf[out_len-1]==0 )
		out_len --;

	return out_len;
}

size_t encrypt_char_buf(const char *in_buf, size_t in_len, uint8_t *out_buf, size_t out_len)
{
	gcry_error_t status;
	size_t padded_len, n_blocks;
	char *pad_buf;
	int bs;

	if( my_cipher_handle == NULL )
		init_gcrypt_subsystem();

	/* gcrypt doesn't seem to offer padding, so the input
	 * must be an integral number of blocks...
	 */
	bs=encryption_block_size();
	if( bs <= 0 ) return 0;	// when lib not present?

	n_blocks = in_len/bs;
	if( (in_len % bs) != 0 ){
		int i;
		n_blocks++;
		padded_len = n_blocks * bs;
		pad_buf = getbuf(padded_len);
		memcpy(pad_buf,in_buf,in_len);
		i=in_len;
		while(i<padded_len)
			pad_buf[i++] = 0;
	} else {
		pad_buf = (char *)in_buf;
		padded_len=in_len;
	}
	if( out_len < padded_len ){
		NWARN("encrypt_char_buf:  output buffer too small for pad!?");
		return 0;
	}

	status = gcry_cipher_encrypt(my_cipher_handle,
			out_buf,out_len,pad_buf,padded_len);
	CHECK_STATUS(encrypt_char_buf,gcry_cipher_encrypt)
	// how do we know how many chars actually written?

	status = gcry_cipher_reset(my_cipher_handle);
	CHECK_STATUS(encrypt_char_buf,gcry_cipher_reset)

	// if we allocated a buffer, then free it.
	if( pad_buf != in_buf ) givbuf(pad_buf);

	return padded_len;
}

size_t encryption_block_size(void)
{
	size_t s;

	/* Retrieve the block length in bytes used with algorithm A. */
	s = gcry_cipher_get_algo_blklen (the_crypt_algo);
	return s;
}


size_t encryption_key_size(void)
{
	/* Retrieve the key length in bytes used with algorithm A. */
	return gcry_cipher_get_algo_keylen (the_crypt_algo);
}

int encryption_hash_size(void)
{
	return gcry_md_get_algo_dlen(the_hash_algo);
}

// this needs to hash the key and allocate the space for the hash

int hash_my_key(void **vpp,const char *key,int key_len)
{
	unsigned char *digest;
	unsigned char *storage;
	int i;
	int need_size;

	// get required digest size
	need_size = gcry_md_get_algo_dlen(the_hash_algo);
	storage = getbuf(need_size);

	for(i=0;i<key_len;i++){
		gcry_md_putc(my_hash_hdl,key[i]);
	}
	gcry_md_final(my_hash_hdl);
	digest = gcry_md_read(my_hash_hdl,0);
	memcpy(storage,digest,need_size);
	*vpp = storage;

	gcry_md_reset(my_hash_hdl);	// to compute a second hash
					// or we could close?

	// return the length of the hash
	// For SHA256, this is 32
	return the_hash_len;
}

#ifdef ELSEWHERE
	gcry_md_reset(my_hash_hdl);	// to compute a second hash
//	gcry_md_close(my_hash_hdl);

	gcry_md_write(my_hash_hdl,data_buf,data_len);
	gcry_md_putc(my_hash_hdl,c);
	gcry_md_final(my_hash_hdl);
	unsigned char *digest;
	digest = gcry_md_read(my_hash_hdl,0);

	// shortcut function:

	gcry_md_get_algo_dlen(the_hash_algo);	// get required digest size
	unsigned char digest_buf[DIGEST_SIZE];
	gcry_md_hash_buffer(the_hash_algo,digest_buf,inbuf,inlen);

	// to verify algo available for use:
	status = gcry_md_test_algo(the_hash_algo);
#endif // ELSEWHERE

#else /* ! HAVE_LIBGCRYPT */

/* dummy functions which allow the thing to link */

/* Linkage to libgcrypt */

size_t decrypt_char_buf(const uint8_t *in_buf, size_t in_len, char *out_buf, size_t out_len )
{
	NWARN("decrypt_char_buf:  libgcrypt not present!?");
	return 0;
}

size_t encrypt_char_buf(const char *in_buf, size_t in_len, uint8_t *out_buf, size_t out_len)
{
	NWARN("encrypt_char_buf:  libgcrypt not present!?");
	return 0;
}

size_t encryption_block_size(void)
{
	NWARN("encryption_block_size:  libgcrypt not present!?");
	return 0;
}


size_t encryption_key_size(void)
{
	NWARN("encryption_key_size:  libgcrypt not present!?");
	return 0;
}

int encryption_hash_size(void)
{
	NWARN("encryption_hash_size:  libgcrypt not present!?");
	return 0;
}

// this needs to hash the key and allocate the space for the hash

int hash_my_key(void **vpp,const char *key,int key_len)
{
	NWARN("hash_my_key:  libgcrypt not present!?");
	return 0;
}

#endif /* ! HAVE_LIBGCRYPT */

