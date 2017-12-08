/* Flood fill
 *
 * The basic idea is very simple; each iteration we examine each pixel;
 * it the pixel borders a filled pixel, we test it, and if the test
 * succeeds, then we fill it, also setting a global flag that something
 * has changed.  We repeat this until nothing changes.
 *
 * That implementation turned out to be very slow... One problem is
 * that unfilled pixels have to examine all of their neighbors.
 * We will try a second implementation in which when a pixel is
 * filled, it marks it's un-filled neighbors.
 *
 * No difference.  Eliminating the flag checks after each kernel
 * launch reduces the time (for 100 iterations) from 11 msec to 7 msec!
 * This could probably be speeded up quite a bit if the control
 * logic could be run on the device instead of on the host...
 *
 * But can we launch a thread array from a device function?
 * Or should we launch the whole grid and have one special thread
 * which is the master?
 * The slow implementation has one thread per pixel in the image;
 * but many iterations are required... better perhaps to have one
 * thread per filled pixel with unchecked neighbors?
 *
 * We can only synchronize threads within a block, so we would have to
 * do this with a single block.  Let's say we have one thread per
 * filled pixel...  Each pixel has up to 4 fillable neighbors (although
 * only the first seed pixel with have all 4 unfilled).  So we have
 * an array in shared memory that we fill with the pixel values. (Need
 * to check how to avoid bank conflicts!)  Then we have a table of
 * of future pixels.  Each thread gets 4 slots.  After these have
 * been filled, we would like to prune duplicates; we won't have many
 * when filling parallel to a coordinate axis, but there will be lots
 * for an oblique front...  we could use a hash function?  Or use the
 * flag image.  We could use these values:
 * 0 - unchecked
 * 1 - filled
 * 2 - queued
 * 3 - rejected
 *
 *	0 0 0 0 0    0 0 0 0 0    0 0 2 0 0
 *	0 0 0 0 0    0 0 2 0 0    0 2 1 2 0
 *	0 0 2 0 0 -> 0 2 1 2 0 -> 2 1 1 1 2
 *	0 0 0 0 0    0 0 2 0 0    0 2 1 2 0
 *	0 0 0 0 0    0 0 0 0 0    0 0 2 0 0
 *
 * Shared memory per block is only 16k, so we can't put the whole image
 * there...
 *
 * We have an array of pixels to check, sized 4 times the max number
 * of threads in a block.  We have an array of active pixels, sized
 * the max number of threads.  After syncing the threads, we need to make
 * up the new active pixel list.  We may not have enough threads to do all
 * of the pixels, so we have several lists.  After processing each list,
 * we transfer new pixels to be checked to the list, marking them as queued.
 * If we run out of space, we will have to set a flag that says we
 * have unrecorded pixels that need to be queued; if that is set when
 * we are all done, we should scan the entire image again looking for them,
 * maybe using a special flag value to indicated un-fulfilled queue request?
 * If we can allocate 2048 queue request slots it ought to be enough
 * for a 512x512 image...
 *
 * We probably want to have the shared memory allocated at launch time...
 */

#include "quip_config.h"

#ifdef HAVE_CUDA

#define BUILD_FOR_CUDA

#include <stdio.h>
#include <curand.h>

#include "quip_prot.h"
#include "my_cu2.h"
#include "cuda_supp.h"			// describe_cuda_error

#include "cu2_fill_expanded.c"

#endif /* HAVE_CUDA */

