/* provide support for reachability tests */

#include "quip_config.h"
#include "nports_api.h"
#import <SystemConfiguration/SystemConfiguration.h>
#include "ios_gui.h"

void test_reachability(QSP_ARG_DECL  const char *s)
{
	BOOL status;
	SCNetworkReachabilityRef target;
	SCNetworkReachabilityFlags flags;

	target = SCNetworkReachabilityCreateWithName (
		kCFAllocatorDefault, s );

	status = SCNetworkReachabilityGetFlags ( target, &flags );
    
    // BUG? - need to release object 'target' ???  no auto-release??

	if( ! status ){
		simple_alert(QSP_ARG  "Network test failed", "Unable to get flags!?");
		return;
	}

	if( flags & kSCNetworkReachabilityFlagsReachable ){
		sprintf(MSG_STR,"%s is reachable.",s);
		simple_alert(QSP_ARG  "Success!", MSG_STR );
	} else {
		sprintf(MSG_STR, "%s is NOT reachable;\nCheck wireless?",s);
		simple_alert(QSP_ARG  "Network test failed", MSG_STR );
	}
}


