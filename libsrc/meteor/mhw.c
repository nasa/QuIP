
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

static int n_mapped_frames=0;

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


static int map_meteor_frame(QSP_ARG_DECL  uint32_t index)
{
	struct meteor_geomet geo;
	void *p;

	assert(index>=0 && index<MAX_NUM_FRAMES);

	// First tell the driver which frame we wish to map

	if( ioctl(meteor_fd, METEORSFRMIDX, &index) < 0 ){
		perror("ioctl METEORSFRMIDX");
		WARN("Error setting frame index");
		return -1;
	}

	if( ioctl(meteor_fd, METEORGETGEO, &geo) < 0 ){
		perror("ioctl METEORGETGEO");
		WARN("Error getting frame geometry");
		return -1;
	}
fprintf(stderr,"map_meteor_frame:  frame size is %d (0x%x)\n",geo.frame_size,geo.frame_size);
	// Get the frame size from the geometry...


	p = mmap((caddr_t)0, geo.frame_size,
				 PROT_READ|PROT_WRITE,
#ifdef LINUX
				 MAP_FILE|
#endif /* LINUX */
				 MAP_SHARED,
				 meteor_fd, (off_t)0);
	if( p == MAP_FAILED ){
		perror("mmap");
		WARN("Error mapping frame!?");
		return -1;
		// BUG unwind frames already mapped?
	}

	set_frame_address(index,p);

	return 0;

} /* end meteor_mmap() */

void *map_mem_data(SINGLE_QSP_ARG_DECL)
{
	void *p;
	uint32_t index=MAGIC_FRAME_INDEX;

	// Set the frame index to the magic value indicating mem_data
fprintf(stderr,"map_mem_data setting magic frame index %d\n",MAGIC_FRAME_INDEX);
	if( ioctl(meteor_fd, METEORSFRMIDX, &index) < 0 ){
		perror("ioctl METEORSFRMIDX");
		WARN("Error setting magic frame index for mem_data");
		return NULL;
	}

fprintf(stderr,"map_mem_data calling mmap\n");
	p = mmap((caddr_t)0, getpagesize(),
				 PROT_READ|PROT_WRITE,
#ifdef LINUX
				 MAP_FILE|
#endif /* LINUX */
				 MAP_SHARED,
				 meteor_fd, (off_t)0);
	if( p == MAP_FAILED ){
		perror("mmap");
		WARN("Error mapping frame!?");
		return NULL;
		// BUG unwind frames already mapped?
	}
fprintf(stderr,"map_mem_data: returning 0x%lx\n",(u_long)p);
	return p;
}

static int release_meteor_frame(QSP_ARG_DECL  int index)
{
	WARN("Sorry, release_meteor_frame not implemented!?");
	return -1;
}

static int release_mapped_frames(SINGLE_QSP_ARG_DECL)
{
	int i;

	for(i=0;i<n_mapped_frames;i++){
		if( release_meteor_frame(QSP_ARG  i) < 0 ){
			sprintf(ERROR_STRING,"release_mapped_frames:  error releasing frame %d!?",i);
			WARN(ERROR_STRING);
			return -1;
		}
	}
	return 0;
}

static int get_meteor_frames(SINGLE_QSP_ARG_DECL)
{
	uint32_t n;

	if( ioctl(meteor_fd, METEORGNBUFFRMS, &n) < 0 ){
		perror("ioctl METEORGNBUFFRMS");
		WARN("Error getting number of ring buffer frames!?");
		return -1;
	}
fprintf(stderr,"get_meteor_frames will return %d\n",n);
	return n;
}

// Originally, the frame buffer was allocated contiguously, and we would read the
// offsets of all of the frames within the area.  But now the driver allocates the frames
// individually, so we have to request the addresses one-by-one.
// We use a new ioctl to set the index of the requested frame.

void meteor_mmap(SINGLE_QSP_ARG_DECL)
{
	int i, n_frames;

	if( n_mapped_frames > 0 ){
		if( release_mapped_frames(SINGLE_QSP_ARG) < 0 ){
			WARN("meteor_mmap:  error releasing old frames!?");
			return;
		}
	}

	n_frames = get_meteor_frames(SINGLE_QSP_ARG);
	if( n_frames <= 0 ){
		WARN("meteor_mmap:  device has no frames!?");
		return;
	}

	for(i=0;i<n_frames;i++){
		if( map_meteor_frame(QSP_ARG  i) < 0 ){
			sprintf(ERROR_STRING,"meteor_mmap:  error mapping frame %d!?",i);
			WARN(ERROR_STRING);

			if( release_mapped_frames(SINGLE_QSP_ARG) < 0 ){
				WARN("meteor_mmap:  error releasing frames after mapping error!?");
			}
			return;
		}
	}
}

#endif /* ! FAKE_METEOR_HARDWARE */


#endif /* HAVE_METEOR */

