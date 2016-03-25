
#include "quip_config.h"

#ifdef HAVE_METEOR

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/types.h>

#include "ioctl_meteor.h"
#include "mmenu.h"
#include "quip_prot.h"

extern int fbfd;

int32_t last_mm_size=0;
int meteor_fd;
struct meteor_frame_offset meteor_off;

int checkChips(SINGLE_QSP_ARG_DECL) 
{
	FILE *file;
	char buffer[256];
	int count;

	file = fopen("/proc/pci", "r");
	if (file==NULL) {
		WARN("Failed to open /proc/pci.");
		advise("Assuming this is not Natoma.");
		return(0);
	}

	while ((count=fscanf(file, "%255s\n", buffer))>0) {
		if (!strncasecmp("Natoma", buffer, 255)) {
			fclose(file);
			advise("Looks like a Natoma chipset.  Using YUV_PACKED.");
			return(1);
		}
	}
	advise("Doesn't look like a Natoma chipset.  Using YUV_PLANAR.");
	fclose(file);
	return(0);
}

#ifdef FAKE_METEOR_HARDWARE

void meteor_mmap()
{
	int i;

	advise("meteor_mmap:  compiled for FAKE hardware!?");

	if( mmbuf!=NULL ){
		munmap(mmbuf,last_mm_size);
	}

	/* BUG we need some better fake values! */
	meteor_off.fb_size = 8*640*480*4;	/* ifdef FAKE_METEOR_HARDWARE */
	meteor_off.mem_off = 0;
	for (i=0; i<num_meteor_frames; i++)
		meteor_off.frame_offset[i]=i*640*480*4;	/* ifdef FAKE_METEOR_HARDWARE */

	mmbuf = getbuf(meteor_off.fb_size);
				
	last_mm_size=meteor_off.fb_size;

	_mm = (struct meteor_mem *)(mmbuf + meteor_off.mem_off);
}

#else /* ! FAKE_METEOR_HARDWARE */

void meteor_mmap(SINGLE_QSP_ARG_DECL)
{
	if( mmbuf!=NULL ){
		munmap(mmbuf,last_mm_size);
	}

	if (ioctl(meteor_fd, METEORGFROFF, &meteor_off) < 0){
		perror("ioctl FrameOffset failed");
		return;
	}

	/* should we make sure that we have initialized meteor_off? */

	if( verbose ){
		int i;
		sprintf(ERROR_STRING, "size: 0x%x", meteor_off.fb_size);
		advise(ERROR_STRING);
		sprintf(ERROR_STRING, "mem: 0x%x", meteor_off.mem_off);
		advise(ERROR_STRING);
		for (i=0; i<num_meteor_frames; i++){
			sprintf(ERROR_STRING,
				"frame %d: 0x%x", i, meteor_off.frame_offset[i]);
			advise(ERROR_STRING);
		}
	}


/*
	mmbuf=(char *) mmap((caddr_t)0, meteor_off.fb_size,
				 PROT_READ|PROT_WRITE,
				 MAP_FILE|MAP_SHARED,
				 meteor_fd, (off_t)0);
*/

	mmbuf=(char *) mmap((caddr_t)0, meteor_off.fb_size,
				 PROT_READ|PROT_WRITE,
#ifdef LINUX
				 MAP_FILE|
#endif /* LINUX */
				 MAP_SHARED,
				 fbfd, (off_t)0);

//sprintf(ERROR_STRING,"meteor_mmap:  mmbuf = 0x%lx, size = 0x%lx",(int_for_addr)mmbuf,meteor_off.fb_size);
//advise(ERROR_STRING);
	if (mmbuf == (char *)(-1)) {
		perror("mmap failed");
		exit(1);
	}

	last_mm_size=meteor_off.fb_size;

	if( verbose ){
		sprintf(ERROR_STRING,"meteor_mmap:  meteor_off.mem_off = 0x%x",meteor_off.mem_off);
		advise(ERROR_STRING);
	}

	_mm = (struct meteor_mem *)(mmbuf + meteor_off.mem_off);

//sprintf(ERROR_STRING,"meteor_mmap:  _mm = 0x%lx",(int_for_addr)_mm);
//advise(ERROR_STRING);
//sprintf(ERROR_STRING,"meteor_mmap:  _mm->frame_size = 0x%x",_mm->frame_size);
//advise(ERROR_STRING);

	if( verbose ){
		sprintf(ERROR_STRING,"meteor_mmap:  _mm->frame_size = 0x%x",_mm->frame_size);
		advise(ERROR_STRING);
	}
} /* end meteor_mmap() */

#endif /* ! FAKE_METEOR_HARDWARE */


#endif /* HAVE_METEOR */

