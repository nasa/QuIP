//
//  macos_main.m
//  coq
//

#import "quip_config.h"
#import "quip_prot.h"
#import "my_encryption.h"

// utility function to take the full pathname of a resource
// and extract the resource directory, and the parent bundle
// directory, saving both to variables.

#define RESOURCE_DIR_VARNAME	"RESOURCE_DIR"
#define BUNDLE_DIR_VARNAME	"BUNDLE_DIR"


static void note_path(QSP_ARG_DECL  const char *whence)
{
	static int path_noted=0;
	/* We temporarily save the pathname so that we can trim the trailing components... */
	char *path_string;
	
	if( path_noted ) return;
	path_noted=1;

	path_string = (char *) savestr(whence);

	int i=(int)strlen(path_string)-1;

	// path_string holds the full pathname of the startup file,
	// so we need to strip the last component...

	while( i>=0 && path_string[i] != '/' )
		i--;
	if( i <= 0 ){
		WARN("Error finding resource directory path!?");
	} else {
		path_string[i]=0;
		ASSIGN_RESERVED_VAR( RESOURCE_DIR_VARNAME ,path_string );
	}

	// Now strip another component to get the main bundle directory
	while( i>=0 && path_string[i] != '/' )
		i--;
	if( i <= 0 ){
		WARN("Error finding bundle directory path!?");
	} else {
		path_string[i]=0;
		ASSIGN_RESERVED_VAR( BUNDLE_DIR_VARNAME,path_string );
	}

	rls_str(path_string);
}

// The startup file may be encrypted...  We look first for startup.enc, if that exists
// we decrypt and push onto the input stack.  Note that we don't call check_quip here,
// so we don't want to use read_encrypted_file...
//
// Because the files are pushed onto a stack the second one pushed is
// the first one read...  So we push the plain-text file first, so that
// the encrypted file is the first one read...

int macos_read_global_startup(SINGLE_QSP_ARG_DECL)
{
	NSBundle *main_bundle;

	main_bundle = [NSBundle mainBundle];

	if( main_bundle == NULL ){
		WARN("unable to locate main bundle!?");
		return -1;
	}

	NSString *startup_path;

	// Now we do the same thing for a plaintext file...

	startup_path = [main_bundle pathForResource:@"mac_startup" ofType:@"scr"];
	if( startup_path != NULL ){
		FILE *fp;
		fp = fopen(startup_path.UTF8String,"r");
		if( fp == NULL ){
			WARN("error opening global startup file!?");
			return -1;
		}
advise("mac_startup.scr found in resource bundle...");
		// called from the main thread...
fprintf(stderr,"macos_read_global_startup:  redirecting to startup file\n");
		redir(QSP_ARG  fp, startup_path.UTF8String );

		note_path(QSP_ARG  startup_path.UTF8String);
	}


	startup_path = [main_bundle pathForResource:@"mac_startup" ofType:@"enc"];
	if( startup_path != NULL ){
		// startup.enc exists!
		FILE *fp;
		fp = fopen(startup_path.UTF8String,"r");
		if( fp == NULL ){
			WARN("error opening encrypted startup file!?");
			return -1;
		}
//advise("startup.enc found in resource bundle...");
		size_t n;
		char *s=decrypt_file_contents(QSP_ARG  fp, &n);
		if( s == NULL ){
			WARN("error decrypting startup file!?");
			return -1;
		}
fprintf(stderr,"macos_read_global_startup:  pushing text of encrypted file\n");
		PUSH_TEXT(s,startup_path.UTF8String);	//

		// BUG  We should free the file contents
		// and the saved name eventually?

		note_path(QSP_ARG  startup_path.UTF8String);
	}

	return 0;
}

