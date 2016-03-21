

#ifdef HAVE_LINUX_IOCTL_H

#include<linux/ioctl.h>

#define VGAT_GET_REG 		_IOR('z', 1, int)
#define VGAT_VBL_WAIT 		_IOR('z', 2, int)

#endif /* HAVE_LINUX_IOCTL_H */


