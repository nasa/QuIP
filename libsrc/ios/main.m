//
//  main.m
//  coq
//
//  Created by Jeff Mulligan on 9/11/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <AdSupport/ASIdentifierManager.h>

#import "quip_config.h"
#import "quip_prot.h"
#import "my_encryption.h"
#import "ios_item.h"		// STRINGOBJ

// utility function to take the full pathname of a resource
// and extract the resource directory, and the parent bundle
// directory, saving both to variables.

#define RESOURCE_DIR_VARNAME	"RESOURCE_DIR"
#define BUNDLE_DIR_VARNAME	"BUNDLE_DIR"
#define DOCUMENTS_DIR_VARNAME	"DOCUMENTS_DIR"

static void find_doc_dir(void)
{
	//NSError *error;
	NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory,
						 NSUserDomainMask, YES); 
	NSString *documentsDirectory = [paths objectAtIndex:0]; // Get documents folder

	/*
NSString *dataPath = [documentsDirectory stringByAppendingPathComponent:@"/MyFolder"];

if (![[NSFileManager defaultManager] fileExistsAtPath:dataPath])
    [[NSFileManager defaultManager] createDirectoryAtPath:dataPath withIntermediateDirectories:NO attributes:nil error:&error]; //Create folder
	*/

//fprintf(stderr,"Documents directory:  %s\n",documentsDirectory.UTF8String);
	ASSIGN_RESERVED_VAR( DOCUMENTS_DIR_VARNAME, documentsDirectory.UTF8String );
}


#ifdef FOOBAR
// note_path is called when we have found the startup file...

static void note_path(QSP_ARG_DECL  const char *whence)
{
	static int path_noted=0;
	/* We temporarily save the pathname so that we can trim the trailing components... */
	char *path_string;

	if( path_noted ) return;
	path_noted=1;
find_doc_dir();

	path_string = (char *) savestr(whence);

	int i=(int)strlen(path_string)-1;

	// path_string holds the full pathname of the startup file,
	// so we need to strip the last component...

fprintf(stderr,"note_path( %s ), will strip last component...\n",
whence);

	while( i>=0 && path_string[i] != '/' )
		i--;
	if( i <= 0 ){
		WARN("Error finding resource directory path!?");
	} else {
		path_string[i]=0;
		ASSIGN_RESERVED_VAR( RESOURCE_DIR_VARNAME ,path_string );
fprintf(stderr,"%s set to %s\n",RESOURCE_DIR_VARNAME,path_string);
	}

	// Now strip another component to get the main bundle directory
	while( i>=0 && path_string[i] != '/' )
		i--;
	if( i <= 0 ){
		WARN("Error finding bundle directory path!?");
	} else {
		path_string[i]=0;
		ASSIGN_RESERVED_VAR( BUNDLE_DIR_VARNAME,path_string );
fprintf(stderr,"%s set to %s\n",BUNDLE_DIR_VARNAME,path_string);
	}

	rls_str(path_string);
}
#endif // FOOBAR

// The startup file may be encrypted...  We look first for startup.enc, if that exists
// we decrypt and push onto the input stack.  Note that we don't call check_quip here,
// so we don't want to use read_encrypted_file...
//
// Because the files are pushed onto a stack the second one pushed is
// the first one read...  So we push the plain-text file first, so that
// the encrypted file is the first one read...

// defined in platform.h?  but why is that included here?
//#define STRINGIFY(v)	_STRINGIFY(v)
//#define _STRINGIFY(v)	#v

#ifndef STARTUP_FILE
#define STARTUP_FILE	startup
#endif // ! STARTUP_FILE

int ios_read_global_startup(SINGLE_QSP_ARG_DECL)
{
	NSBundle *main_bundle;
	char *startup_filename;

	main_bundle = [NSBundle mainBundle];

	if( main_bundle == NULL ){
		WARN("unable to locate main bundle!?");
		return -1;
	}

	NSString *startup_path;

	// Now we do the same thing for a plaintext file...
	//STARTUP_FILE = "foobar";
//fprintf(stderr,"ios_read_global_startup:  STARTUP_FILE = %s\n",STRINGIFY(STARTUP_FILE));
	startup_filename=STRINGIFY(STARTUP_FILE);	// default value

	startup_path = [main_bundle pathForResource:STRINGOBJ(startup_filename)
								ofType:@"scr"];
	if( startup_path != NULL ){
		FILE *fp;
		fp = fopen(startup_path.UTF8String,"r");
		if( fp == NULL ){
			WARN("error opening global startup file!?");
			return -1;
		}
/*
sprintf(ERROR_STRING,
"Startup file %s.scr found in resource bundle...",startup_filename);
advise(ERROR_STRING);
*/
		// called from the main thread...
		redir(QSP_ARG  fp, startup_path.UTF8String );

		//note_path(QSP_ARG  startup_path.UTF8String);
	}


	startup_path = [main_bundle pathForResource:STRINGOBJ(startup_filename)
								ofType:@"enc"];
	if( startup_path != NULL ){
		// startup.enc exists!
		FILE *fp;
		fp = fopen(startup_path.UTF8String,"r");
		if( fp == NULL ){
			WARN("error opening encrypted startup file!?");
			return -1;
		}
		/*
//advise("startup.enc found in resource bundle...");
sprintf(ERROR_STRING,
"Startup file %s.enc found in resource bundle...",startup_filename);
advise(ERROR_STRING);
sprintf(ERROR_STRING,
"Startup path is %s...",startup_path.UTF8String);
advise(ERROR_STRING);
*/
		size_t n;
		char *s=decrypt_file_contents(QSP_ARG  fp, &n);
		if( s == NULL ){
			WARN("error decrypting startup file!?");
			return -1;
		}
		PUSH_TEXT(s,startup_path.UTF8String);	//

		// BUG  We should free the file contents
		// and the saved name eventually?

		//note_path(QSP_ARG  startup_path.UTF8String);
	} else {
		sprintf(ERROR_STRING,
			"Failed to find startup file %s.scr or %s.enc!?",
			startup_filename,startup_filename);
		ERROR1(ERROR_STRING);
	}

	/* Apparently, this used to be where we looked up the UDID? */

		// This code is deprecated - for ios 6 and higher we will use
		// the advertising identifier...
		//
		// With xcode 5, this won't compile even when targeting iOS 6!?
		// use of respondsToSelector gets around this...
	
		// We have to do this somewhere ...

	/* end of old comments... */


	find_doc_dir();

	return 0;
}

static void write_test_file(void)
{
	NSError *error;
	NSString *stringToWrite = @"1\n2\n3\n4";
	NSString *filePath = [[NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) firstObject] stringByAppendingPathComponent:@"myTestFile.txt"];
fprintf(stderr,"writing test file\n");
	[stringToWrite writeToFile:filePath atomically:YES encoding:NSUTF8StringEncoding error:&error];
	

	NSString *msgString=@"this is a test message\n";
	NSArray *documentPaths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
	NSString *documentsDirectory = [documentPaths objectAtIndex:0];
	NSString *documentTXTPath = [documentsDirectory stringByAppendingPathComponent:@"myTestFile2.txt"];
	NSFileHandle *myHandle = [NSFileHandle fileHandleForUpdatingAtPath:documentTXTPath ];
	[myHandle seekToEndOfFile];
fprintf(stderr,"writing second test file\n");
	[myHandle writeData:  [msgString dataUsingEncoding:NSUTF8StringEncoding]];
	[myHandle closeFile];

}

#include "quip_start_menu.c"

int main(int argc,char *argv[])
{
#ifdef NOT_ARC
	NSAutoreleasePool *pool;

	pool = [NSAutoreleasePool new];
#endif /* NOT_ARC */

	int retVal;

// Use these two lines to make the output go to a file
// instead of the Xcode debugger window

/*
NSString *logPath = @"/tmp/xcode.log";
freopen([logPath fileSystemRepresentation], "w", stderr);
*/

	@autoreleasepool {
//NSLog(@"Calling start_quip...\n");
        init_quip_menu();
		write_test_file();
		start_quip_with_menu(argc,argv,quip_menu);

		retVal = UIApplicationMain(argc, argv, nil, @"quipAppDelegate" );
	}

	return retVal;

}

