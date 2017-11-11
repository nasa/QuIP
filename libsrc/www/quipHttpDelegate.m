#include "quip_config.h"
#include "quip_prot.h"
//#include "quipHttpDelegate.h"
#include "server.h"
#include "ios_item.h"

#ifdef USE_HTTP_DELEGATE

@implementation quipHttpDelegate

@synthesize receivedData;
@synthesize destination;
@synthesize fp;
@synthesize op_complete;
@synthesize condition;
@synthesize url;

- (void)connection:(NSURLConnection *)connection didReceiveResponse:(NSURLResponse *)response
{
	// This method is called when the server has determined that it
	// has enough information to create the NSURLResponse.

	// It can be called multiple times, for example in the case of a
	// redirect, so each time we reset the data.

	// receivedData is an instance variable declared elsewhere.
	[receivedData setLength:0];

	op_complete = YES;

	[condition lock];
	[condition signal];
	[condition unlock];
}


- (void)connection:(NSURLConnection *)connection didReceiveData:(NSData *)data
{
	// Append the new data to receivedData.
	// receivedData is an instance variable declared elsewhere.
	[receivedData appendData:data];
}

- (void) connection:(NSURLConnection *)connection
	didFailWithError:(NSError *)error
{
	// release the connection, and the data object
	//[connection release];

	// receivedData is declared as a method instance elsewhere
	//[receivedData release];

	// inform the user
//	NSLog(@"Connection failed! Error - %@ %@",
//		[error localizedDescription],
//		[[error userInfo] objectForKey:NSURLErrorFailingURLStringErrorKey]);

	op_complete = YES;

	[condition lock];
	[condition signal];
	[condition unlock];

}

- (void)connectionDidFinishLoading:(NSURLConnection *)connection
{
	// do something with the data
	switch( destination ){
	case DEST_FILE:
		if( fwrite( receivedData.bytes, 1, receivedData.length, fp)
			!= receivedData.length ){
//			NWARN("Error writing downloaded data!?");
//			advise("Error writing downloaded data!?");
		}
		fclose(fp);
		// Signal the interpreter that the download is finished???
		break;
	default:
//		NWARN("Unhandled case in connectionDidFinishLoading!?");
//		advise("Unhandled case in connectionDidFinishLoading!?");
		break;
	}

	// why the locks and unlocks?

	op_complete = YES;

	[condition lock];
	[condition signal];
	[condition unlock];

	// receivedData is declared as a method instance elsewhere
	//NSLog(@"Succeeded! Received %d bytes of data",[receivedData length]);
}

-(id) initWithURL:(NSString *) s
{
	self = [super init];
	
	// Make the condition
	condition = [[NSCondition alloc] init];
	// do we need to worry about the condition name field??? 

	url = s;
	op_complete = NO;

	return self;
}

#ifdef FOOBAR
-(void) start
{
	// Create the request.

	NSURLRequest *theRequest=
		[ NSURLRequest
			requestWithURL:[NSURL URLWithString:url]
			cachePolicy:NSURLRequestUseProtocolCachePolicy
			timeoutInterval:60.0
		];

	[ NSURLConnection sendSynchronousRequest:theRequest
			returningResponse:&theResponse
			error:&theError
	];

#ifdef FOOBAR
	// do we need to check for error here?

	// create the connection with the request
	// and start loading the data

	NSURLConnection *theConnection=
		[ [NSURLConnection alloc]
			initWithRequest:theRequest
			delegate:self
		];

	if (theConnection) {
		// Create the NSMutableData to hold the received data.
		// receivedData is an instance variable declared elsewhere.
		//receivedData = [[NSMutableData data] retain];
		receivedData = [[NSMutableData alloc] init];
	} else {
		// Inform the user that the connection failed.
//		NWARN("Failed to establish connection to server!?");
	}
#endif /* FOOBAR */
}
#endif /* FOOBAR */

@end
#endif /* USE_HTTP_DELEGATE */

void init_http_subsystem(void)
{
	// nop
}

// This function is started in a different queue...
#ifdef FOOBAR
static void _write_file_from_url_helper(void *arg)
{
	quipHttpDelegate *qhd;

//advise("helper func BEGIN");

	qhd = (__bridge quipHttpDelegate *)(arg);

	// The file will be closed when we return, so we shouldn't return
	// until all the data has been received and written.
	// BUT control won't be given up as long as we stay here!?!?

//advise("starting transfer...");
//	[qhd start];
}
#endif /* FOOBAR */

static NSData *fetch_url_contents(QSP_ARG_DECL  NSString *url)
{
	NSURLResponse *theResponse;
	NSError *theError;
	NSData *data;

	NSURLRequest *theRequest=
		[ NSURLRequest
			requestWithURL:[NSURL URLWithString:url]
			cachePolicy:NSURLRequestUseProtocolCachePolicy
			timeoutInterval:60.0
		];

	data = [ NSURLConnection sendSynchronousRequest:theRequest
			returningResponse:&theResponse
			error:&theError
	];

	return data;
}

void write_file_from_url( QSP_ARG_DECL  FILE *fp, const char *url )
{
	NSData *data;

	data = fetch_url_contents(QSP_ARG  STRINGOBJ(url) );

	if( data == NULL ){
		WARN("problem with synchronous connection");
		return;
	}
advise("got some data!");
	if( fwrite( data.bytes, 1, data.length, fp)
		!= data.length ){
		WARN("Error writing downloaded data!?");
	}
	fclose(fp);


#ifdef FOOBAR
	quipHttpDelegate *qhd;
	dispatch_queue_t queue;

	// Start a new thread to receive the data,
	// so that this one can block until it is done.

	qhd = [[quipHttpDelegate alloc] initWithURL: STRINGOBJ(url) ];
#ifdef CAUTIOUS
	if( qhd == NULL ){
		WARN("CAUTIOUS:  write_file_from_url:  error creating quipHttpDelegate!?");
		return;
	}
#endif /* CAUTIOUS */

	qhd.destination = DEST_FILE;
	qhd.fp = fp;

	queue = dispatch_queue_create("url_recv_queue", NULL);
	dispatch_async_f(queue,(__bridge void *)(qhd),_write_file_from_url_helper);

	[qhd.condition lock];
	while( ! qhd.op_complete ){
advise("waiting");
		[qhd.condition wait];
advise("back from wait");
	}
	[qhd.condition unlock];
advise("operation complete!");
#endif /* FOOBAR */
}

String_Buf *get_url_contents( QSP_ARG_DECL  const char *url )
{
	NSData *data;
	String_Buf *sbp;

	data = fetch_url_contents(QSP_ARG  STRINGOBJ(url) );

	sbp = new_stringbuf();

	if( sb_size(sbp) <= data.length )
		enlarge_buffer(sbp,1+data.length);
	memcpy(sb_buffer(sbp),data.bytes,data.length);
	*(sb_buffer(sbp) + data.length) = 0;
	return sbp;
}

