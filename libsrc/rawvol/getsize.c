
#include "quip_config.h"

/*
 * getsize.c --- get the size of a partition.
 * 
 * Copyright (C) 1995 Theodore Ts'o.  This file may be
 * redistributed under the terms of the GNU Public License.
 */

#include <stdio.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#ifdef HAVE_SYS_MOUNT_H
#include <sys/mount.h>
#endif

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_LINUX_FS_H
#include <linux/fs.h>
#endif

#ifdef HAVE_LINUX_FD_H
#include <linux/fd.h>
#endif

#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif

#ifdef HAVE_SYS_DISKLABEL_H
#include <sys/disklabel.h>
#endif /* HAVE_SYS_DISKLABEL_H */


#include "quip_prot.h"		/* verbose */

/* #include <linux/ext2_fs.h> */
/* #include "ext2fs.h" */

#include "llseek.h"

static int valid_offset (int fd,off64_t offset)
{
	char ch;

#ifdef HAVE_LSEEK64
	if (lseek64 (fd, offset, 0) < 0)
		return 0;
#else
	if (lseek (fd, offset, 0) < 0)
		return 0;
#endif

	if (read (fd, &ch, 1) < 1)
		return 0;
	return 1;
}

/*
 * Returns the number of blocks in a partition
 */
errcode_t get_device_size(QSP_ARG_DECL  CONST char *file,int blocksize,blk_t *retblocks)
{
	int	fd;
#ifdef BLKGETSIZE
	long	size;
#endif /* BLKGETSIZE */

	off64_t high, low;
#ifdef FDGETPRM
	struct floppy_struct this_floppy;
#endif
#ifdef HAVE_SYS_DISKLABEL_H
	struct disklabel lab;
	struct partition *pp;
	char ch;
#endif /* HAVE_SYS_DISKLABEL_H */

	fd = open(file, O_RDONLY);
	if (fd < 0)
		return errno;

#ifdef BLKGETSIZE
	if( verbose ) advise("Using BLKGETSIZE to find disk size");

	/* what is the logic of this?
	 * Does it assume a system block size of 512?
	 */
	if (ioctl(fd, BLKGETSIZE, &size) >= 0) {
		close(fd);
		*retblocks = size / (blocksize / 512);
		return 0;
	}
#endif
#ifdef FDGETPRM
	if( verbose ) advise("Using FDGETPRM to find disk size");

	if (ioctl(fd, FDGETPRM, &this_floppy) >= 0) {
		close(fd);
		*retblocks = this_floppy.size / (blocksize / 512);
		return 0;
	}
#endif
#ifdef HAVE_SYS_DISKLABEL_H
	size = strlen(file) - 1;
	if (size >= 0) {
		ch = file[size];
		if (isdigit(ch))
			size = 0;
		else if (ch >= 'a' && ch <= 'h')
			size = ch - 'a';
		else
			size = -1;
	}
	if( verbose ) advise("Using DIOCGDINFO to find disk size");
	if (size >= 0 && (ioctl(fd, DIOCGDINFO, (char *)&lab) >= 0)) {
		pp = &lab.d_partitions[size];
		if (pp->p_size) {
			close(fd);
			*retblocks = pp->p_size / (blocksize / 512);
			return 0;
		}
	}
#endif /* HAVE_SYS_DISKLABEL_H */

	/*
	 * OK, we couldn't figure it out by using a specialized ioctl,
	 * which is generally the besy way.  So do binary search to
	 * find the size of the partition.
	 */
	if( verbose ) advise("Using binary search to find disk size");

	low = 0;
	for (high = 1024; valid_offset (fd, high); high *= 2)
		low = high;
	while (low < high - 1) {
		off64_t mid;
		
		mid = (low + high) / 2;

		if (valid_offset (fd, mid))
			low = mid;
		else
			high = mid;
	}
	valid_offset (fd, 0);
	close(fd);
	*retblocks = (low + 1) / blocksize;
	return 0;
}
