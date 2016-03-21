
#ifndef SIR_DISK_MAGIC

#ifdef INC_VERSION
char VersionId_inc_sir_disk[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

#define SUCCESS	0
#define FAILURE	1
#define SIR_DISK_MAGIC		01234L
#define SIR_DISK_VERSION	01L

typedef struct sir_disk_hdr { 
	long magic;
	long version;
	/* short ints or long ints??? */
	long video_width;
	long video_height;
	long video_packing;
	long video_format;
	long video_timing;
	long video_capture_type;
	long video_field_dominance;
	long block_size;
	long blocks_per_image;
	long video_start_block;
	long video_n_blocks;
	long audio_n_blocks;
} Sir_Disk_Hdr;

/* this is because of a +1LL in sgi's v2d, d2v code, which seeks an extra block!? */

#define EXTRABLOCK	512		/* make this 0 if no extra block */

#endif /* SIR_DISK_MAGIC */

