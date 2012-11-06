
#ifndef NO_QUADTREE

#ifdef INC_VERSION
char VersionId_inc_quadtree[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

#include "data_obj.h"

typedef struct quadtree {
	struct quadtree *qt_child[4];
	float qt_val;
	dimension_t qt_size,qt_onpix;
} QuadTree;

#define NO_QUADTREE ((QuadTree *) NULL)

#endif /* NO_QUADTREE */

