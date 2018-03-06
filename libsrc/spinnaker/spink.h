#include "SpinnakerC.h"
#include <stdio.h>
#include <string.h>

#include "spink.h"

// Compiler warning C4996 suppressed due to deprecated strcpy() and sprintf()
// functions on Windows platform.
#if defined WIN32 || defined _WIN32 || defined WIN64 || defined _WIN64
    #pragma warning(disable : 4996)
#endif

// This macro helps with C-strings.
#define MAX_BUFF_LEN 256

// This function queries an interface for its cameras and then prints out
// device information.
spinError QueryInterface(spinInterface hInterface)
{
    spinError err = SPINNAKER_ERR_SUCCESS;
    unsigned int i = 0;

    //
    // Retrieve TL nodemap from interface
    //
    // *** NOTES ***
    // Each interface has a nodemap that can be retrieved in order to access
    // information about the interface itself, any devices connected, or
    // addressing information if applicable.
    //
    spinNodeMapHandle hNodeMapInterface = NULL;

    err = spinInterfaceGetTLNodeMap(hInterface, &hNodeMapInterface);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve interface nodemap. Aborting with error %d...\n\n", err);
        return err;
    }

    //
    // Print interface display name
    //
    // *** NOTES ***
    // Each interface has a nodemap that can be retrieved in order to access
    // information about the interface itself, any devices connected, or
    // addressing information if applicable.
    //
    spinNodeHandle hInterfaceDisplayName = NULL;

    // Retrieve node
    err = spinNodeMapGetNode(hNodeMapInterface, "InterfaceDisplayName", &hInterfaceDisplayName);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve node (interface display name). Aborting with error %d...\n\n", err);
        return err;
    }

    // Check availability
    bool8_t interfaceDisplayNameIsAvailable = False;

    err = spinNodeIsAvailable(hInterfaceDisplayName, &interfaceDisplayNameIsAvailable);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to check node availability (interface display name). Aborting with error %d...\n\n", err);
        return err;
    }

    // Check readability
    bool8_t interfaceDisplayNameIsReadable = False;

    err = spinNodeIsReadable(hInterfaceDisplayName, &interfaceDisplayNameIsReadable);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to check node readability (interface display name). Aborting with error %d...\n\n", err);
        return err;
    }

    // Print
    char interfaceDisplayName[MAX_BUFF_LEN];
    size_t lenInterfaceDisplayName = MAX_BUFF_LEN;

    if (interfaceDisplayNameIsAvailable && interfaceDisplayNameIsReadable)
    {
        err = spinStringGetValue(hInterfaceDisplayName, interfaceDisplayName, &lenInterfaceDisplayName);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to retrieve value (interface display name). Aborting with error %d...\n\n", err);
            return err;
        }
    }
    else
    {
        strcpy(interfaceDisplayName, "Interface display name not readable");
    }

    printf("%s\n", interfaceDisplayName);

    //
    // Retrieve list of cameras from the interface
    //
    // *** NOTES ***
    // Camera lists can be retrieved from an interface or the system object.
    // Camera lists retrieved from an interface, such as this one, only return
    // cameras attached on that specific interface whereas camera lists
    // retrieved from the system will return all cameras on all interfaces.
    //
    // *** LATER ***
    // Camera lists must be cleared manually. This must be done prior to
    // releasing the system and while the camera list is still in scope.
    //
    spinCameraList hCameraList = NULL;
    size_t numCameras = 0;

    // Create empty camera list
    err = spinCameraListCreateEmpty(&hCameraList);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to create camera list. Aborting with error %d...\n\n", err);
        return err;
    }

    // Retrieve cameras
    err = spinInterfaceGetCameras(hInterface, hCameraList);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve camera list. Aborting with error %d...\n\n", err);
        return err;
    }

    // Retrieve number of cameras
    err = spinCameraListGetSize(hCameraList, &numCameras);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve number of cameras. Aborting with error %d...\n\n", err);
        return err;
    }

    // Return if no cameras detected
    if (numCameras == 0)
    {
        printf("\tNo devices detected.\n\n");

        //
        // Clear and destroy camera list before losing scope
        //
        // *** NOTES ***
        // Camera lists do not automatically clean themselves up. This must be done
        // manually. The same is true of interface lists.
        //
        err = spinCameraListClear(hCameraList);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to clear camera list. Aborting with error %d...\n\n", err);
            return err;
        }

        err = spinCameraListDestroy(hCameraList);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to destroy camera list. Aborting with error %d...\n\n", err);
            return err;
        }

        return err;
    }

    // Print device vendor and model name for each camera on the interface
    for (i = 0; i < numCameras; i++)
    {
        //
        // Select camera
        //
        // *** NOTES ***
        // Each camera is retrieved from a camera list with an index. If the
        // index is out of range, an exception is thrown.
        //
        // *** LATER ***
        // Each camera handle needs to be released before losing scope or the
        // system is released.
        //
        spinCamera hCam = NULL;

        err = spinCameraListGet(hCameraList, i, &hCam);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to retrieve camera. Aborting with error %d...\n\n", err);
            return err;
        }

        // Retrieve TL device nodemap; please see NodeMapInfo_C example for
        // additional comments on transport layer nodemaps.
        spinNodeMapHandle hNodeMapTLDevice = NULL;

        err = spinCameraGetTLDeviceNodeMap(hCam, &hNodeMapTLDevice);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to retrieve TL device nodemap. Aborting with error %d...\n\n", err);
            return err;
        }

        //
        // Retrieve device vendor name
        //
        // *** NOTES ***
        // Grabbing node information requires first retrieving the node and
        // then retrieving its information. There are two things to keep in
        // mind. First, a node is distinguished by type, which is related
        // to its value's data type.  Second, nodes should be checked for
        // availability and readability/writability prior to making an
        // attempt to read from or write to the node.
        //
        spinNodeHandle hDeviceVendorName = NULL;
        bool8_t deviceVendorNameIsAvailable = False;
        bool8_t deviceVendorNameIsReadable = False;

        // Retrieve node
        err = spinNodeMapGetNode(hNodeMapTLDevice, "DeviceVendorName", &hDeviceVendorName);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to retrieve device information (vendor name node). Aborting with error %d...\n\n", err);
            return err;
        }

        // Check availability
        err = spinNodeIsAvailable(hDeviceVendorName, &deviceVendorNameIsAvailable);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to check node availability (vendor name node). Aborting with error %d...\n\n", err);
            return err;
        }

        // Check readability
        err = spinNodeIsReadable(hDeviceVendorName, &deviceVendorNameIsReadable);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to check node readability (vendor name node). Aborting with error %d...\n\n", err);
            return err;
        }

        //
        // Retrieve device model name
        //
        // *** NOTES ***
        // Because C has no try-catch blocks, each function returns an error
        // code to suggest whether an error has occurred. Errors can be
        // sufficiently handled with these return codes. Checking availability
        // and readability/writability makes for safer and more complete code;
        // however, keeping in mind example conciseness and legibility, only
        // this example and NodeMapInfo_C demonstrate checking node
        // availability and readability/writability while other examples
        // handle errors with error codes alone.
        //
        spinNodeHandle hDeviceModelName = NULL;
        bool8_t deviceModelNameIsAvailable = False;
        bool8_t deviceModelNameIsReadable = False;

        err = spinNodeMapGetNode(hNodeMapTLDevice, "DeviceModelName", &hDeviceModelName);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to retrieve device information (model name node). Aborting with error %d...\n\n", err);
            return err;
        }

        err = spinNodeIsAvailable(hDeviceModelName, &deviceModelNameIsAvailable);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to check node availability (model name node). Aborting with error %d...\n\n", err);
            return err;
        }

        err = spinNodeIsReadable(hDeviceModelName, &deviceModelNameIsReadable);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to check node readability (model name node). Aborting with error %d...\n\n", err);
            return err;
        }

        //
        // Print device vendor and model names
        //
        // *** NOTES ***
        // Generally it is best to check readability when it is required to read
        // information from a node and writability when it is required to write
        // to a node. For most nodes, writability implies readability while
        // readability does not imply writability.
        //
        char deviceVendorName[MAX_BUFF_LEN];
        size_t lenDeviceVendorName = MAX_BUFF_LEN;
        char deviceModelName[MAX_BUFF_LEN];
        size_t lenDeviceModelName = MAX_BUFF_LEN;

        // Print device vendor name
        if (deviceVendorNameIsAvailable && deviceVendorNameIsReadable)
        {
            err = spinStringGetValue(hDeviceVendorName, deviceVendorName, &lenDeviceVendorName);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                printf("Unable to retrieve device information (vendor name value). Aborting with error %d...\n\n", err);
                return err;
            }
        }
        else
        {
            strcpy(deviceVendorName, "Not readable");
        }

        // Print device model name
        if (deviceModelNameIsAvailable && deviceModelNameIsReadable)
        {
            err = spinStringGetValue(hDeviceModelName, deviceModelName, &lenDeviceModelName);
            if (err != SPINNAKER_ERR_SUCCESS)
            {
                printf("Unable to retrieve device information (model name value). Aborting with error %d...\n\n", err);
                return err;
            }
        }
        else
        {
            strcpy(deviceModelName, "Not readable");
        }

        printf("\tDevice %d %s %s\n\n", i, deviceVendorName, deviceModelName);

        //
        // Release camera before losing scope
        //
        // *** NOTES ***
        // Every handle that is created for a camera must be released before
        // the system is released or an exception will be thrown.
        //
        err = spinCameraRelease(hCam);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to release camera. Aborting with error %d...\n\n", err);
            return err;
        }
    }

    //
    // Clear and destroy camera list before losing scope
    //
    // *** NOTES ***
    // Camera lists do not automatically clean themselves up. This must be done
    // manually. The same is true of interface lists.
    //
    err = spinCameraListClear(hCameraList);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to clear camera list. Aborting with error %d...\n\n", err);
        return err;
    }

    err = spinCameraListDestroy(hCameraList);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to destroy camera list. Aborting with error %d...\n\n", err);
        return err;
    }

    return err;
}

// Example entry point; this function sets up the system and retrieves
// interfaces for the example.
int main(/*int argc, char** argv*/)
{
    spinError errReturn = SPINNAKER_ERR_SUCCESS;
    spinError err = SPINNAKER_ERR_SUCCESS;
    unsigned int i = 0;

    // Print application build information
    printf("Application build date: %s %s \n\n", __DATE__, __TIME__);

    //
    // Retrieve singleton reference to system object
    //
    // *** NOTES ***
    // Everything originates with the system object. It is important to notice
    // that it has a singleton implementation, so it is impossible to have
    // multiple system objects at the same time.
    //
    // *** LATER ***
    // The system object should be cleared prior to program completion.  If not
    // released explicitly, it will be released automatically.
    //
    spinSystem hSystem = NULL;

    err = spinSystemGetInstance(&hSystem);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve system instance. Aborting with error %d...\n\n", err);
        return err;
    }

    //
    // Retrieve list of interfaces from the system
    //
    // *** NOTES ***
    // Interface lists are retrieved from the system object.
    //
    // *** LATER ***
    // Interface lists must be cleared and destroyed manually. This must be
    // done prior to releasing the system and while the interface list is still
    // in scope.
    //
    spinInterfaceList hInterfaceList = NULL;
    size_t numInterfaces = 0;

    // Create empty interface list
    err = spinInterfaceListCreateEmpty(&hInterfaceList);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to create empty interface list. Aborting with error %d...\n\n", err);
        return err;
    }

    // Retrieve interfaces from system
    err = spinSystemGetInterfaces(hSystem, hInterfaceList);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve interface list. Aborting with error %d...\n\n", err);
        return err;
    }

    // Retrieve number of interfaces
    err = spinInterfaceListGetSize(hInterfaceList, &numInterfaces);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve number of interfaces. Aborting with err %d...\n\n", err);
        return err;
    }

    printf("Number of interfaces detected: %u\n\n", (unsigned int)numInterfaces);

    //
    // Retrieve list of cameras from the system
    //
    // *** NOTES ***
    // Camera lists can be retrieved from an interface or the system object.
    // Camera lists retrieved from the system, such as this one, return all
    // cameras available on the system.
    //
    // *** LATER ***
    // Camera lists must be cleared and destroyed manually. This must be done
    // prior to releasing the system and while the camera list is still in
    // scope.
    //
    spinCameraList hCameraList = NULL;
    size_t numCameras = 0;

    // Create empty camera list
    err = spinCameraListCreateEmpty(&hCameraList);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to create camera list. Aborting with error %d...\n\n", err);
        return err;
    }

    // Retrieve cameras from system
    err = spinSystemGetCameras(hSystem, hCameraList);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve camera list. Aborting with error %d...\n\n", err);
        return err;
    }

    // Retrieve number of cameras
    err = spinCameraListGetSize(hCameraList, &numCameras);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to retrieve number of cameras. Aborting with  error %d...\n\n", err);
        return err;
    }

    printf("Number of cameras detected: %u\n\n", (unsigned int)numCameras);

    // Finish if there are no cameras
    if (numCameras == 0 || numInterfaces == 0)
    {
        // Clear and destroy camera list before releasing system
        err = spinCameraListClear(hCameraList);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to clear camera list. Aborting with error %d...\n\n", err);
            return err;
        }

        err = spinCameraListDestroy(hCameraList);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to destroy camera list. Aborting with error %d...\n\n", err);
            return err;
        }

        // Clear and destroy interface list before releasing system
        err = spinInterfaceListClear(hInterfaceList);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to clear interface list. Aborting with error %d...\n\n", err);
            return err;
        }

        err = spinInterfaceListDestroy(hInterfaceList);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to destroy interface list. Aborting with error %d...\n\n", err);
            return err;
        }

        // Release system
        err = spinSystemReleaseInstance(hSystem);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to release system instance. Aborting with  error %d...\n\n", err);
            return err;
        }

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

        err = spinInterfaceListGet(hInterfaceList, i, &hInterface);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            printf("Unable to retrieve interface from list (error %d)...\n", err);
            errReturn = err;
            continue;
        }

        // Run example
        err = QueryInterface(hInterface);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            errReturn = err;
        }

        // Release interface
        err = spinInterfaceRelease(hInterface);
        if (err != SPINNAKER_ERR_SUCCESS)
        {
            errReturn = err;
        }
    }

    //
    // Clear and destroy camera list before releasing system
    //
    // *** NOTES ***
    // Camera lists are not shared pointers and do not automatically clean
    // themselves up and break their own references. Therefore, this must be
    // done manually. The same is true of interface lists.
    //
    err = spinCameraListClear(hCameraList);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to clear camera list. Aborting with error %d...\n\n", err);
        return err;
    }

    err = spinCameraListDestroy(hCameraList);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to destroy camera list. Aborting with error %d...\n\n", err);
        return err;
    }

    //
    // Clear and destroy interface list before releasing system
    //
    // *** NOTES ***
    // Interface lists are not shared pointers and do not automatically clean
    // themselves up and break their own references. Therefore, this must be
    // done manually. The same is true of camera lists.
    //
    err = spinInterfaceListClear(hInterfaceList);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to clear interface list. Aborting with error %d...\n\n", err);
        return err;
    }

    err = spinInterfaceListDestroy(hInterfaceList);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to destroy interface list. Aborting with error %d...\n\n", err);
        return err;
    }

    //
    // Release system
    //
    // *** NOTES ***
    // The system should be released, but if it is not, it will do so itself.
    // It is often at the release of the system (whether manual or automatic)
    // that unbroken references and still registered events will throw an
    // exception.
    //
    err = spinSystemReleaseInstance(hSystem);
    if (err != SPINNAKER_ERR_SUCCESS)
    {
        printf("Unable to release system instance. Aborting with  error %d...\n\n", err);
        return err;
    }

    printf("\nDone! Press Enter to exit...\n");
    getchar();

    return errReturn;
}
