#include "quip_config.h"
#include <UIKit/UIKit.h>
#include <CoreMotion/CoreMotion.h>
#include "quipAppDelegate.h"
#include "quip_prot.h"
#include "ios_prot.h"

// Someday we may want to use these event queues???

/*-(void)initMotionManager
{
	[super initMotionManager]

	self.motionManager = [[CMMotionManager alloc] init];
	self.motionManager.accelerometerUpdateInterval = 0.1;

	[self.motionManager
		startAccelerometerUpdatesToQueue:[NSOperationQueue currentQueue]
		withHandler:^(CMAccelerometerData  *accelerometerData, NSError *error) {
			[self outputAccelertionData:accelerometerData.acceleration];
			if(error){
				NSLog(@"%@", error);
			}
		}
	];
}*/

static CMMotionManager *mgr=NULL;
static int accelerometer_started=0;


#ifdef CAUTIOUS

#define INSURE_MOTION_MGR										\
if( mgr == NULL ) mgr = [[CMMotionManager alloc] init];			\
if( mgr == NULL ) ERROR1("CAUTIOUS:  set_accelerometer_interval:  Failed to create motion manager!?");

#else // ! CAUTIOUS

#define INSURE_MOTION_MGR										\
if( mgr == NULL ) mgr = [[CMMotionManager alloc] init];

#endif // ! CAUTIOUS

static COMMAND_FUNC( set_accelerometer_interval )
{
	float interval;

	interval = HOW_MUCH("interval in seconds between accelerometer reads");

	INSURE_MOTION_MGR

	mgr.accelerometerUpdateInterval = interval;
	//	self.motionManager.accelerometerUpdateInterval = interval;

	//[[UIAccelerometer sharedAccelerometer] setUpdateInterval: interval];
}

static COMMAND_FUNC( do_start )
{
	INSURE_MOTION_MGR
	if( accelerometer_started ){
		WARN("accelerometer already started!?");
		return;
	}

	mgr.accelerometerUpdateInterval = 0.1;
	[mgr startAccelerometerUpdates];
	accelerometer_started = 1;
}

static COMMAND_FUNC( do_stop )
{
	INSURE_MOTION_MGR

	if( ! accelerometer_started ){
		WARN("accelerometer not running!?");
		return;
	}

	[mgr stopAccelerometerUpdates];
	//[[UIAccelerometer sharedAccelerometer] setDelegate:NULL];
	accelerometer_started = 0;
}

static COMMAND_FUNC( do_read_accel )
{
	CMAccelerometerData *d;

	INSURE_MOTION_MGR

	if( ! accelerometer_started )
		do_start(SINGLE_QSP_ARG);	// bad name for this function!?

	d = mgr.accelerometerData;
#ifdef CAUTIOUS
	if( d == NULL ){
		WARN("CAUTIOUS:  do_read_accel:  null data!?");
		return;
	}
#endif // CAUTIOUS

	// check this too?

	// used to keep filtered values in accel array, but without regular
	// time sampling that makes no sense...
	//sprintf(ERROR_STRING,"accel:  %g %g %g	  %g %g %g\n",
	//	a->x,a->y,a->z,
	//	accel[0],accel[1],accel[2]);
	sprintf(ERROR_STRING,"accel:  %g %g %g\n",
		d.acceleration.x,d.acceleration.y,d.acceleration.z);
	advise(ERROR_STRING);

}


#define ADD_CMD(s,f,h)	ADD_COMMAND(accel_menu,s,f,h)
MENU_BEGIN(accel)
ADD_CMD( interval,	set_accelerometer_interval,	set time between accelerometer reads)
ADD_CMD( start,	do_start,	start processing of accelerometer events)
ADD_CMD( stop,	do_stop,	stop processing of accelerometer events)
ADD_CMD( read,	do_read_accel,	store current accelerometer in a variable )
MENU_END(accel)

COMMAND_FUNC(do_accel_menu)
{
	PUSH_MENU(accel);
}

