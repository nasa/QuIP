#ifndef __NVSTUSB_USB_H__
#define __NVSTUSB_USB_H__

#include <stdbool.h>
#include <stdint.h>

struct nvstusb_usb_device;

extern bool nvstusb_usb_init(void);
extern void nvstusb_usb_deinit(void);

extern struct nvstusb_usb_device *nvstusb_usb_open_device(const char *firmware);
extern void nvstusb_usb_close_device(struct nvstusb_usb_device *dev);

extern int nvstusb_usb_write_bulk(struct nvstusb_usb_device *dev, int endpoint, const void *data, int size);
extern int nvstusb_usb_read_bulk(struct nvstusb_usb_device *dev, int endpoint, void *data, int size);

#endif // __NVSTUSB_USB_H__
