#ifndef _QUIPTABLE_H_
#define _QUIPTABLE_H_

#include <UIKit/UIKit.h>

@interface quipTableViewController : UITableViewController <UITableViewDelegate,UITableViewDataSource>
@property (retain) NSMutableArray *	my_test_array;

@end

#endif /* _QUIPTABLE_H_ */
