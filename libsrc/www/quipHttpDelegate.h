
@interface quipHttpDelegate : NSObject

typedef enum {
	DEST_NONE,
	DEST_FILE,
	DEST_BUFFER
} DestinationType;

@property (retain) NSMutableData *	receivedData;
@property (retain) NSCondition *	condition;
@property (retain) NSString *		url;
@property BOOL 				op_complete;
@property DestinationType 		destination;
@property FILE * 			fp;

- (void)connection:(NSURLConnection *)connection didReceiveResponse:(NSURLResponse *)response;
- (void)connection:(NSURLConnection *)connection didReceiveData:(NSData *)data;
- (void) connection:(NSURLConnection *)connection
	didFailWithError:(NSError *)error;
- (void)connectionDidFinishLoading:(NSURLConnection *)connection;
-(id) initWithURL:(NSString *) url;

@end

