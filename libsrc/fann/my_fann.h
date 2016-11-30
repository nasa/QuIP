
#include "item_type.h"

#ifdef HAVE_FANN

// Unfortunately, we have to pick a type for FANN at compile time!?
// Perhaps we could compile multiple versions ourselves, but would there be namespace
// collisions?

//#include "doublefann.h"
#include "floatfann.h"

#endif // HAVE_FANN

typedef struct my_fann {
	Item		mf_item;
#ifdef HAVE_FANN
	struct fann*	mf_fann_p;
#endif // HAVE_FANN
} My_FANN;

#define FANN_NAME(mfp)	(mfp)->mf_item.item_name

ITEM_INTERFACE_PROTOTYPES(My_FANN,fann)

#define MAX_LAYERS	32		// BUG avoid fixed size...

struct net_params {
	int n_layers;
	int layer_size[MAX_LAYERS];
};

