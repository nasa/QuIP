
#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>
#include <Security/SecureTransport.h>	// errSSLCrypto
#include "quip_prot.h"
#include "ios_item.h"	// STRINGOBJ

/* Can't find this file on the system!? */
#include <CommonCrypto/CommonCryptor.h>
#include <CommonCrypto/CommonDigest.h>

#include "my_encryption.h"

#ifdef USE_CC_PADDING
#define PADDING_OPTION	kCCOptionPKCS7Padding
#else
#define PADDING_OPTION	0
#endif

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

static void report_cc_error(const char *whence,int status)
{
	const char *msg;

	switch( status ){
		case kCCParamError:
			msg = "Illegal parameter value."; break;
		case kCCBufferTooSmall:
	msg = "Insufficent buffer provided for specified operation."; break;
		case kCCMemoryFailure:
			msg = "Memory allocation failure."; break;
		case kCCAlignmentError:
			msg = "Input size was not aligned properly."; break;
		case kCCDecodeError:
	msg = "Input data did not decode or decrypt properly."; break;
		case kCCUnimplemented:
	msg = "Function not implemented for the current algorithm."; break;
		default:
	PRINT_ERR2("%s:  Unrecognized common crypto error status %d",
				whence,status);
			return;
			break;
	}
	PRINT_ERR2("%s:  %s",whence,msg);
}

/* The original motivation for this is to be able
 * to conceal the scripts in the bundle that implement
 * the administrator back-door.  The basic strategy
 * is that we have a fixed key that the program uses;
 * we generate underlying string programmatically, so
 * that the string doesn't exist in the executable file.
 */

/* Transform the key string into something unrecognizable.
 * This should be very difficult to reverse, but if the attacker
 * has the original string, then not so hard?
 */


int hash_my_key( void **hash_addr,
				const char *keytag, int taglen )
{
	CC_SHA256_CTX ctx;
	//int status;
	// This is probably longer than it needs to be
	static unsigned char hash_buf[CC_SHA256_DIGEST_LENGTH];

	// header file says funcs always return 1!?
	/*status =*/ CC_SHA256_Init(&ctx);
	/*status =*/ CC_SHA256_Update(&ctx,keytag,taglen);
	/*status =*/ CC_SHA256_Final(hash_buf,&ctx);

	*hash_addr = hash_buf;

	return CC_SHA256_DIGEST_LENGTH;
}

// initialization vector is only used in cipher block chaining (CBC).
// If none is provided, all zeroes is used.
// There isn't much documenation on what the size should be,
// but we guess it is the block size - i.e., it represents the first
// block from which to chain the first real block.

#ifdef USE_CBC
#define IV_SIZE 1024
#define INIT_VECTOR iv;
#else
#define INIT_VECTOR	NULL
#endif /* USE_CBC */

static void *my_symm_key=NULL;
static CCCryptorRef my_encryptor=NULL;
static CCCryptorRef my_decryptor=NULL;

size_t encryption_block_size(void)
{
	return kCCBlockSizeAES128 ;
}

size_t encryption_key_size(void)
{
	return kCCKeySizeAES128;
}

// We originally enabled padding, but we disable it to be compatible
// with our unix gcrypt implementation - which doesn't appear to
// support it!?

static int init_my_encryptor(void)
{
	CCCryptorStatus status;

	if( my_symm_key == NULL && init_my_symm_key(&my_symm_key) < 0 )
		return -1;

	status = CCCryptorCreate(
		kCCEncrypt,			// op
		kCCAlgorithmAES128,		// alg
		PADDING_OPTION,		// options
		my_symm_key, encryption_key_size(),
		INIT_VECTOR,
		&my_encryptor);

	if( status != kCCSuccess ){
		report_cc_error("CCCryptorCreate",status);
		return -1;
	}
	return 0;
}


static int init_my_decryptor(void)
{
	CCCryptorStatus status;

	if( my_symm_key == NULL && init_my_symm_key(&my_symm_key) < 0 )
		return -1;

	status = CCCryptorCreate(
		kCCDecrypt,			// op
		kCCAlgorithmAES128,		// alg
		PADDING_OPTION,		// options
		my_symm_key, encryption_key_size(),
		INIT_VECTOR,
		&my_decryptor);

	if( status != kCCSuccess ){
		report_cc_error("CCCryptorCreate",status);
		return -1;
	}
	return 0;
}

static size_t crypt_buffer(
	CCCryptorRef the_cryptor,
	const uint8_t *in_buf,size_t in_len,
	uint8_t *out_buf, size_t max_out_len)
{
	/* If the input length is larger than the block size,
	 * call the Update function repeatedly
	 */
	size_t n_remaining, n_to_update;
	size_t n_out_available, n_written, total_written=0;
	size_t blocksize;
	CCCryptorStatus status;

	n_remaining = in_len;
	n_out_available = max_out_len;
	blocksize = kCCBlockSizeAES128 ;

	while( n_remaining ){
		n_to_update = n_remaining >= blocksize ?
			blocksize : n_remaining;
		status = CCCryptorUpdate(the_cryptor, in_buf,
			n_to_update, out_buf, n_out_available,
			&n_written);
		if( status != kCCSuccess ){
			report_cc_error("CCCryptorUpdate",status);
			return -1;
		}
		n_out_available -= n_written;
		out_buf += n_written;
		total_written += n_written;
		in_buf += n_to_update;
		n_remaining -= n_to_update;
	}
	status = CCCryptorFinal(the_cryptor, out_buf,
		n_out_available, &n_written);
	if( status != kCCSuccess ){
		report_cc_error("CCCryptorFinal",status);
		return -1;
	}

	total_written += n_written;

	status = CCCryptorReset(the_cryptor, NULL);
	if( status != kCCSuccess ){
		report_cc_error("CCCryptorReset",status);
		return -1;
	}

	return total_written;
}

size_t encrypt_char_buf(const char *in_buf,size_t in_len,
	uint8_t *out_buf, size_t max_out_len)
{
	if( my_encryptor == NULL ){
		if( init_my_encryptor() < 0 ){
			my_encryptor = NULL;	// make sure
			NWARN("Couldn't create encryptor");
			return -1;
		}
	}

#ifndef USE_CC_PADDING

	size_t bs;
	char *pad_buf;
	size_t padded_len;
	size_t n_blocks;
	size_t retval;

	/* gcrypt doesn't seem to offer padding, so the input
	 * must be an integral number of blocks...
	 *
	 * CommonCrypto does support padding, but we aren't using it
	 * in order to be compatible with gcrypt.  Here we do our
	 * own padding...
	 */
	bs=encryption_block_size();
	n_blocks = in_len/bs;
	if( (in_len % bs) != 0 ){
		size_t i;
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
	if( max_out_len < padded_len ){
		NWARN("encrypt_char_buf:  output buffer too small for pad!?");
		return 0;
	}

	retval = crypt_buffer(my_encryptor,
		(const uint8_t *)pad_buf,padded_len,out_buf,max_out_len);

	// free the buffer if we allocated one...
	if( pad_buf != in_buf ) givbuf(pad_buf);

	return retval;

#else // USE_CC_PADDING

	return crypt_buffer(my_encryptor,
		(const uint8_t *)in_buf,in_len,out_buf,max_out_len);

#endif // USE_CC_PADDING
}


size_t decrypt_char_buf(const uint8_t *in_buf,size_t in_len,
	char *out_buf, size_t max_out_len)
{
	size_t n_decrypted;

	if( my_decryptor == NULL ){
		if( init_my_decryptor() < 0 ){
			my_decryptor = NULL;	// make sure
			NWARN("Couldn't create decryptor");
			return -1;
		}
	}

	n_decrypted = crypt_buffer(my_decryptor,
			in_buf,in_len,(uint8_t *)out_buf,max_out_len);

#ifndef USE_CC_PADDING
	// If we padded, there will be zeroes at the end of the
	// message - we might assume that they are invalid,
	// as we are generally encrypting C strings?

	while( n_decrypted >= 1 && out_buf[n_decrypted-1]==0 )
		n_decrypted --;
#endif /* ! USE_CC_PADDING */

	return n_decrypted;
}

