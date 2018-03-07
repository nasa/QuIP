
//#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "spink.h"


// Example entry point; this function sets up the system and retrieves
// interfaces for the example.
int main(int argc, char** argv)
{
	spinSystem hSystem = NULL;
	spinInterfaceList hInterfaceList;
	spinCameraList hCameraList;
	size_t numCameras = 0;
	size_t numInterfaces = 0;

	spinError errReturn = SPINNAKER_ERR_SUCCESS;
	spinError err = SPINNAKER_ERR_SUCCESS;
	unsigned int i = 0;

	// Print application build information
	printf("Application build date: %s %s \n\n", __DATE__, __TIME__);

	if( get_spink_system(&hSystem) < 0 )
		exit(1);

	if( get_spink_interfaces(hSystem,&hInterfaceList,&numInterfaces) < 0 ) exit(1);
	if( get_spink_cameras(hSystem,&hCameraList,&numCameras) < 0 ) exit(1);

	printf("Number of cameras detected: %u\n\n", (unsigned int)numCameras);

	// Finish if there are no cameras
	if (numCameras == 0 || numInterfaces == 0)
	{
		// Clear and destroy camera list before releasing system
		if( release_spink_cam_list(&hCameraList) < 0 ) exit(1);
		if( release_spink_interface_list(hInterfaceList) < 0 ) exit(1);
		if( release_spink_system(hSystem) < 0 ) exit(1);

		printf("\nNot enough cameras/interfaces!\n");
		printf("Done! Press Enter to exit...\n");
		getchar();

		return -1;
	}

	printf("\n*** QUERYING INTERFACES ***\n\n");

	//
	// Run example on each interface
	//
	// *** NOTES ***
	// In order to run all interfaces in a loop, each interface needs to
	// retrieved using its index.
	//
	for (i = 0; i < numInterfaces; i++)
	{
		// Select interface
		spinInterface hInterface = NULL;

		if( get_spink_interface_from_list(&hInterface,hInterfaceList,i) < 0 )
			exit(1);

		// Run example
		if( query_spink_interface(hInterface) < 0 )
			exit(1);

		if( release_spink_interface(hInterface) < 0 )
			exit(1);
	}

	if( release_spink_cam_list(&hCameraList) < 0 ) exit(1);
	if( release_spink_interface_list(hInterfaceList) < 0 ) exit(1);
	if( release_spink_system(hSystem) < 0 ) exit(1);

	exit(0);
}

