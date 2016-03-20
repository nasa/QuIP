
@interface quipAlarm : NSObject

@property dispatch_source_t	timer;
@property const char *		script;
@property BOOL			ticking;;

- (id)initWithTimeout:(NSTimeInterval)timeout;
-(void) setDelay: (float) delay;

@end

