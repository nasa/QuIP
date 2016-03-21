
#include "ios_list.h"

@interface IOS_Stack: NSObject

@property (retain) IOS_List *list;

-(void) push : (id) obj;
-(id) pop ;
-(id) bottom;
-(id) top ;
-(int) depth;

@end

