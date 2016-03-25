#include "quip_config.h"

#ifdef HAVE_LIBUSB

#include "usb.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>

#include <libusb-1.0/libusb.h>

static struct libusb_context *nvstusb_usb_context = 0;
static const int nvstusb_usb_debug_level = 3;

struct nvstusb_usb_device {
	struct libusb_device_handle *handle;
};

	/* convert a libusb error to a readable string */
static const char * libusb_error_to_string( int error)
{
	switch((enum libusb_error) error) {
		case LIBUSB_SUCCESS:              return "Success (no error)";
		case LIBUSB_ERROR_IO:             return "Input/output error";
		case LIBUSB_ERROR_INVALID_PARAM:  return "Invalid parameter";
		case LIBUSB_ERROR_ACCESS:         return "Access denied (insufficient permissions)";
		case LIBUSB_ERROR_NO_DEVICE:      return "No such device(it may have been disconnected)";
		case LIBUSB_ERROR_NOT_FOUND:      return "Entity not found";
		case LIBUSB_ERROR_BUSY:           return "Resource busy";
		case LIBUSB_ERROR_TIMEOUT:        return "Operation timed out";
		case LIBUSB_ERROR_OVERFLOW:       return "Overflow";
		case LIBUSB_ERROR_PIPE:           return "Pipe error";
		case LIBUSB_ERROR_INTERRUPTED:    return "System call interrupted (perhaps due to signal)";
		case LIBUSB_ERROR_NO_MEM:         return "Insufficient memory";
		case LIBUSB_ERROR_NOT_SUPPORTED:  return "Operation not supported or unimplemented on this platform";
		case LIBUSB_ERROR_OTHER:          return "Other error";
	}
	return "Unknown error";
}

/* initialize usb */
bool nvstusb_usb_init(void)
{
	if (0 != nvstusb_usb_context) {
		return true;
	}

	struct libusb_context *ctx = 0;
	libusb_init(&ctx);
	if (0 == ctx) {
		fprintf(stderr, "nvstusb: Could not initialize libusb\n");
		return false;
	}

	libusb_set_debug(ctx, nvstusb_usb_debug_level);
	fprintf(stderr, "nvstusb: libusb initialized, debug level %d\n", nvstusb_usb_debug_level);

	nvstusb_usb_context = ctx;
	return true;
}

/* shutdown usb */
void nvstusb_usb_deinit(void)
{
	if (0 == nvstusb_usb_context) return;

	libusb_exit(nvstusb_usb_context);
	fprintf(stderr, "nvstusb: libusb deinitialized\n");

	nvstusb_usb_context = 0;
}

/* get the number of endpoints on a device */
static int nvstusb_usb_get_numendpoints( struct libusb_device_handle *handle)
{
	assert(handle != 0);

	struct libusb_device *dev = libusb_get_device(handle);
	struct libusb_config_descriptor *cfgDesc = 0;
	int res = libusb_get_active_config_descriptor(dev, &cfgDesc);
	if (res < 0) {
		fprintf(stderr, "nvstusb: Could not determine the number of endpoints... Error %d: %s\n", res, libusb_error_to_string(res));
		return res;
	}

	int num = cfgDesc->interface->altsetting->bNumEndpoints;
	libusb_free_config_descriptor(cfgDesc);
	fprintf(stderr, "nvstusb: Found %d endpoints...\n", num);
	return num;
}

static bool nvstusb_usb_needs_firmware( struct nvstusb_usb_device *dev)
{
	assert(dev != 0);
	assert(dev->handle != 0);

	return nvstusb_usb_get_numendpoints(dev->handle) == 0;
}


/* upload firmware file */
static int nvstusb_usb_load_firmware( struct nvstusb_usb_device *dev,
	const char *filename)
{
	assert(dev != 0);
	assert(dev->handle != 0);

	FILE *fw = fopen(filename, "rb");
	if (!fw) { perror(filename); return -1; }

	fprintf(stderr, "nvstusb: Loading firmware...\n");

	uint8_t lenPos[4];
	uint8_t buf[1024];

	while(fread(lenPos, 4, 1, fw) == 1) {
		uint16_t length = (lenPos[0]<<8) | lenPos[1];
		uint16_t pos    = (lenPos[2]<<8) | lenPos[3];

		if (fread(buf, length, 1, fw) != 1) {
			perror(filename);
			return LIBUSB_ERROR_OTHER;
		}

		int res = libusb_control_transfer(
			dev->handle,
			LIBUSB_REQUEST_TYPE_VENDOR,
			0xA0, /* 'Firmware load' */
			pos, 0x0000,
			buf, length,
			0
		);
		if (res < 0) {
			fprintf(stderr, "nvstusb: Error uploading firmware... Error %d: %s\n", res, libusb_error_to_string(res));
			return res;
		}
	}

	fclose(fw);
	return 0;
}

/* open 3d controller */
struct nvstusb_usb_device * nvstusb_usb_open_device( const char *firmware)
{
	assert(nvstusb_usb_context != 0);

	//int res;
	struct libusb_device_handle *handle =
		libusb_open_device_with_vid_pid(nvstusb_usb_context, 0x0955, 0x0007);

	if (0 == handle) {
		fprintf(stderr, "nvstusb: No NVIDIA 3d stereo controller found...\n");
		return 0;
	}

	fprintf(stderr, "nvstusb: Found NVIDIA 3d stereo controller...\n");

	struct nvstusb_usb_device *dev = (struct nvstusb_usb_device *) malloc(sizeof(*dev));
	dev->handle = handle;

	if (nvstusb_usb_needs_firmware(dev)) {
		if (nvstusb_usb_load_firmware(dev, firmware) < 0) {
			free(dev);
			return 0;
		}
		libusb_reset_device(dev->handle);
		libusb_close(dev->handle);
		usleep(250000);
		handle = dev->handle = libusb_open_device_with_vid_pid(nvstusb_usb_context, 0x0955, 0x0007);
		libusb_reset_device(dev->handle);
		usleep(250000);
	}
	libusb_set_configuration(dev->handle, 1); // TODO: error checking
	libusb_claim_interface(dev->handle, 0);   // TODO: error checking

	return dev;
}

/* close the device */
void nvstusb_usb_close_device( struct nvstusb_usb_device *dev)
{
	if (0 == dev) return;

	if (0 != dev->handle) {
		libusb_close(dev->handle);
	}
	free(dev);
}

/* send data to an endpoint, bulk transfer */
int nvstusb_usb_write_bulk( struct nvstusb_usb_device *dev,
	int endpoint, const void *data, int size)
{
	int sent = 0;

	assert(dev         != 0);
	assert(dev->handle != 0);

	return libusb_bulk_transfer(dev->handle, endpoint | LIBUSB_ENDPOINT_OUT, (unsigned char*)data, size, &sent, 0);
}

/* receive data from an endpoint */
int nvstusb_usb_read_bulk( struct nvstusb_usb_device *dev, int endpoint,
	void *data, int size)
{
	int recvd = 0;
	int res;

	assert(dev         != 0);
	assert(dev->handle != 0);

	res = libusb_bulk_transfer(dev->handle, endpoint | LIBUSB_ENDPOINT_IN,
			(unsigned char*) data, size, &recvd, 200);

//fprintf(stderr,"nvstusb_usb_read_bulk:  res = %d\n",res);
// figure out what the return value means,
// so we can test return value!

	return recvd;
}


#endif // HAVE_LIBUSB
