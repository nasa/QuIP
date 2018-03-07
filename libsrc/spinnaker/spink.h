#include "SpinnakerC.h"

// Compiler warning C4996 suppressed due to deprecated strcpy() and sprintf()
// functions on Windows platform.
#if defined WIN32 || defined _WIN32 || defined WIN64 || defined _WIN64
    #pragma warning(disable : 4996)
#endif

// This macro helps with C-strings.
#define MAX_BUFF_LEN 256

extern int spink_node_is_readable(spinNodeHandle hdl);
extern int spink_node_is_available(spinNodeHandle hdl);
extern int release_spink_cam_list( spinCameraList *hCamList_p );
extern int release_spink_cam(spinCamera hCam);

