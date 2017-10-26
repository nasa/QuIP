
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

#define new_fann(s)	_new_fann(QSP_ARG  s)
#define fann_of(s)	_fann_of(QSP_ARG  s)
#define del_fann(s)	_del_fann(QSP_ARG  s)
#define pick_fann(s)	_pick_fann(QSP_ARG  s)
#define list_fanns(fp)	_list_fanns(QSP_ARG  fp)

#define MAX_LAYERS	32		// BUG avoid fixed size...

struct net_params {
	int n_layers;
	unsigned int layer_size[MAX_LAYERS];
};

