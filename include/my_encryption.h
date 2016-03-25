
// Macros for calculating size of encryption buffers

#define ROUND_UP_TO_BLOCK(t,n)				\
	{						\
		size_t bs=encryption_block_size();	\
		if( bs != 0 ){				\
			t = ((n) % bs)==0 ? n :		\
				n + bs - (n%bs);	\
		} else {				\
			NERROR1("Encryption block size is zero!?");	\
			t = n; /* quiet compiler */	\
		}					\
	}

#define INCREASE_BY_BLOCKSIZE(t,n)			\
	{						\
		size_t bs=encryption_block_size();		\
		t = n + bs ;				\
	}


// We can't help wondering whether the requirement
// that the buffer size exceed the input size by the
// block length is really a programmer error?

// the libgcrypt implementation just wants to have
// an integral number of blocks.

#ifdef BUILD_FOR_OBJC
#define SET_OUTPUT_SIZE(t,n)	INCREASE_BY_BLOCKSIZE(t,n)
#else
#define SET_OUTPUT_SIZE(t,n)	ROUND_UP_TO_BLOCK(t,n)
#endif

//#define USE_THESE_UTILITIES

extern size_t encrypt_char_buf(const char *in_buf,size_t in_len,
					uint8_t *out_buf, size_t out_len);

extern size_t decrypt_char_buf(const uint8_t *in_buf,size_t in_len,
					char *out_buf, size_t out_len);

extern size_t encryption_block_size(void);
extern size_t encryption_key_size(void);
extern int encryption_hash_size(void);

extern int hash_my_key(void **vpp,const char *key,int key_len);
extern int init_my_symm_key(void **vpp);

// should be static, but here for testing
void gen_string(char *buf,int buf_len);

// used to be static, now used in main on iPad...
extern char *decrypt_file_contents(QSP_ARG_DECL  FILE *fp_in,
					size_t *count_p);

#define ENCRYPTED_SCRIPT_SUFFIX	"enc"

extern int has_encryption_suffix(const char *name);
extern String_Buf *decrypt_text( const char *buffer );

