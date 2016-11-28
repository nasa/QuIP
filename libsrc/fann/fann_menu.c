// provide an interface to the FANN neural network library
#include "quip_config.h"
#include "quip_prot.h"
#include "quip_menu.h"
#include "data_obj.h"
#include "my_fann.h"
#include <string.h>	// memcpy()

static COMMAND_FUNC(do_create_std)
{
	int n_hidden_layers;
	int n_input;
	int n_output;
	My_FANN *mfp;
	const char *s;
	int i;
#define MAX_HIDDEN_LAYERS	32		// BUG avoid fixed size...
	unsigned int layer_size[2+MAX_HIDDEN_LAYERS];

	s=NAMEOF("name for network");
	n_input = HOW_MANY("number of inputs");
	n_output = HOW_MANY("number of outputs");
	n_hidden_layers = HOW_MANY("number of hidden layers");

	if( n_hidden_layers > MAX_HIDDEN_LAYERS ){
		sprintf(ERROR_STRING,
	"do_create_std:  Sorry, max. number of hidden layers is hard-coded to %d.",
			MAX_HIDDEN_LAYERS);
		WARN(ERROR_STRING);
		// eat the args before returning
		for(i=0;i<n_hidden_layers;i++)
			layer_size[1] = HOW_MANY("dummy node count");
		return;
	}

	layer_size[0] = n_input;
	for(i=0;i<n_hidden_layers;i++){
		char pmpt[128];
		sprintf(pmpt,"number of nodes in hidden layer %d",i+1);
		layer_size[1+i] = HOW_MANY(pmpt);
		if( layer_size[1+i] <= 0 ){
			WARN("layer size must be positive");
			layer_size[1+i] = 1;
		}
	}
	layer_size[1+i] = n_output;

	// make sure not in use
	mfp = fann_of(QSP_ARG  s);
	if( mfp != NULL ){
		sprintf(ERROR_STRING,"do_create_std:  network name %s is already in use!?",s);
		WARN(ERROR_STRING);
		return;
	}
	mfp = new_fann(QSP_ARG  s);
	assert(mfp!=NULL);

#ifdef HAVE_FANN
	// BUG fann_create_standard uses varargs, so the number of neurons in each
	// layer must be specified...
	//mfp->mf_fann_p = fann_create_standard(n_layers, n_input, n_hidden, n_output);
	mfp->mf_fann_p = fann_create_standard_array(2+n_hidden_layers, layer_size);
	if( mfp->mf_fann_p == NULL ){
//		enum fann_errno_enum e;
//fprintf(stderr,"getting errno\n");
//		e = fann_get_errno( (struct fann_error *) NULL );
//printf(stderr,"errno = %d\n",e);
		// clean up
		del_fann(QSP_ARG  mfp);
		return;
	}

	// BUG we might want to provide menu commands to edit these parameters?
	fann_set_activation_steepness_hidden(mfp->mf_fann_p, 1);
	fann_set_activation_steepness_output(mfp->mf_fann_p, 1);

	fann_set_activation_function_hidden(mfp->mf_fann_p, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(mfp->mf_fann_p, FANN_SIGMOID_SYMMETRIC);

	fann_set_train_stop_function(mfp->mf_fann_p, FANN_STOPFUNC_BIT);
	fann_set_bit_fail_limit(mfp->mf_fann_p, 0.01f);

	fann_set_training_algorithm(mfp->mf_fann_p, FANN_TRAIN_RPROP);
	
#endif // HAVE_FANN
}

static int check_network(QSP_ARG_DECL  My_FANN *mfp, Data_Obj *input_dp, Data_Obj *output_dp)
{
	if( mfp == NULL || input_dp == NULL || output_dp == NULL ) return -1;

#ifdef HAVE_FANN
	// depth of objects should match the networks #'s of inputs and outputs!
	if( OBJ_COMPS(input_dp) != fann_get_num_input(mfp->mf_fann_p) ){
		sprintf(ERROR_STRING,"Network %s has %d inputs, but input data object %s has %d components!?",
			FANN_NAME(mfp),fann_get_num_input(mfp->mf_fann_p),OBJ_NAME(input_dp),OBJ_COMPS(input_dp));
		WARN(ERROR_STRING);
		return -1;
	}
	if( OBJ_COMPS(output_dp) != fann_get_num_output(mfp->mf_fann_p) ){
		sprintf(ERROR_STRING,"Network %s has %d outputs, but output data object %s has %d components!?",
			FANN_NAME(mfp),fann_get_num_output(mfp->mf_fann_p),OBJ_NAME(output_dp),OBJ_COMPS(output_dp));
		WARN(ERROR_STRING);
		return -1;
	}
#endif // HAVE_FANN

#define FANN_PREC	PREC_SP	// BUG - has to match what is included in my_fann.h

	// Do input and output have to have the same type?
	if( OBJ_MACH_PREC(input_dp) != FANN_PREC ){
		sprintf(ERROR_STRING,"Input data object %s (%s) must have %s precision!?",
			OBJ_NAME(input_dp),PREC_NAME(OBJ_PREC_PTR(input_dp)),NAME_FOR_PREC_CODE(FANN_PREC));
		WARN(ERROR_STRING);
		return -1;
	}

	if( OBJ_MACH_PREC(output_dp) != FANN_PREC ){
		sprintf(ERROR_STRING,"Output data object %s (%s) must have %s precision!?",
			OBJ_NAME(output_dp),PREC_NAME(OBJ_PREC_PTR(output_dp)),NAME_FOR_PREC_CODE(FANN_PREC));
		WARN(ERROR_STRING);
		return -1;
	}

	// Objects must not only have the same number of "pixels" - we insist that they must have the same shape...
	if( OBJ_COLS(output_dp) != OBJ_COLS(input_dp) || OBJ_ROWS(output_dp) != OBJ_ROWS(input_dp) ||
			OBJ_FRAMES(output_dp) != OBJ_FRAMES(input_dp) || OBJ_SEQS(output_dp) != OBJ_SEQS(input_dp) ){
		sprintf(ERROR_STRING,"Training objests %s and %s must have the same shape!?", OBJ_NAME(output_dp),OBJ_NAME(input_dp));
		WARN(ERROR_STRING);
		return -1;
	}

	if( ! IS_CONTIGUOUS(input_dp) ){
		sprintf(ERROR_STRING,"Input training data object %s must be contiguous!?",OBJ_NAME(input_dp));
		WARN(ERROR_STRING);
		return -1;
	}

	if( ! IS_CONTIGUOUS(output_dp) ){
		sprintf(ERROR_STRING,"Output training data object %s must be contiguous!?",OBJ_NAME(output_dp));
		WARN(ERROR_STRING);
		return -1;
	}
	return 0;
}

static COMMAND_FUNC(do_fann_train)
{
	const unsigned int max_epochs = 1000;
	const unsigned int epochs_between_reports = 10;
	const float desired_error = (const float) 0;
	My_FANN *mfp;
	Data_Obj *input_dp, *output_dp;
#ifdef HAVE_FANN
	struct fann_train_data *data;
	int n_data;
#endif // HAVE_FANN

	mfp = pick_fann(QSP_ARG  "");
	//s = NAMEOF("name of file containing training data");
	input_dp = PICK_OBJ("name of object containing input data");
	output_dp = PICK_OBJ("name of object containing output data");

	if( check_network(QSP_ARG  mfp, input_dp, output_dp) < 0 ) return;

#ifdef HAVE_FANN
	/*
	// The FANN data files consist of a header line, with n_points, n_inputs, n_outputs
	// Then, for each "point", there is one line with the input values followed by a line with the output values.
	// For QuIP, we prefer to provide the data in separate input and output objects
	data = fann_read_train_from_file(s);
	// error check?
	*/

	n_data = OBJ_N_MACH_ELTS(output_dp) / OBJ_COMPS(output_dp);

	/* fann_create_train_array seems to have gone away in v 2.2 !?
	data = fann_create_train_array(n_data,OBJ_COMPS(input_dp),(fann_type *)OBJ_DATA_PTR(input_dp),
		OBJ_COMPS(output_dp),(fann_type *)OBJ_DATA_PTR(output_dp));
		*/
	data = fann_create_train(n_data,OBJ_COMPS(input_dp),OBJ_COMPS(output_dp));

	// This block copy assumes that the block is contiguous...
	memcpy(data->input[0],OBJ_DATA_PTR(input_dp),n_data*OBJ_COMPS(input_dp)*sizeof(fann_type));
	memcpy(data->output[0],OBJ_DATA_PTR(output_dp),n_data*OBJ_COMPS(output_dp)*sizeof(fann_type));

	fann_train_on_data(mfp->mf_fann_p, data, max_epochs, epochs_between_reports, desired_error);

	fann_destroy_train(data);
#endif // HAVE_FANN
}

static COMMAND_FUNC(do_fann_run)
{
	My_FANN *mfp;
	Data_Obj *input_dp, *output_dp;
	dimension_t n_data;
	float *src, *dst, *result;

	mfp = pick_fann(QSP_ARG  "");
	//s = NAMEOF("name of file containing training data");
	input_dp = PICK_OBJ("source object containing input data");
	output_dp = PICK_OBJ("destination object for output data");

	if( check_network(QSP_ARG  mfp, input_dp, output_dp) < 0 ) return;

	n_data = OBJ_N_MACH_ELTS(input_dp) / OBJ_COMPS(input_dp);
	src = OBJ_DATA_PTR(input_dp);
	dst = OBJ_DATA_PTR(output_dp);

	while( n_data -- ){
#ifdef HAVE_FANN
		result = fann_run(mfp->mf_fann_p,src);
		memcpy(dst,result,sizeof(*dst)*OBJ_COMPS(output_dp));
		// BUG - do we need to free the result array???  memory leak?
#endif // HAVE_FANN

		src += OBJ_COMPS(input_dp);
		dst += OBJ_COMPS(output_dp);
	}
}


static COMMAND_FUNC(do_init_weights)
{
	My_FANN *mfp;
	Data_Obj *input_dp, *output_dp;
#ifdef HAVE_FANN
	struct fann_train_data *data;
	int n_data;
#endif // HAVE_FANN

	mfp = pick_fann(QSP_ARG  "");
	//s = NAMEOF("name of file containing training data");
	input_dp = PICK_OBJ("name of object containing input data");
	output_dp = PICK_OBJ("name of object containing output data");

	if( check_network(QSP_ARG  mfp, input_dp, output_dp) < 0 ) return;

#ifdef HAVE_FANN
	/*
	// The FANN data files consist of a header line, with n_points, n_inputs, n_outputs
	// Then, for each "point", there is one line with the input values followed by a line with the output values.
	// For QuIP, we prefer to provide the data in separate input and output objects
	data = fann_read_train_from_file(s);
	// error check?
	*/

	n_data = OBJ_N_MACH_ELTS(output_dp) / OBJ_COMPS(output_dp);
//	data = fann_create_train_array(n_data,OBJ_COMPS(input_dp),(fann_type *)OBJ_DATA_PTR(input_dp),
//		OBJ_COMPS(output_dp),(fann_type *)OBJ_DATA_PTR(output_dp));
	data = fann_create_train(n_data,OBJ_COMPS(input_dp),OBJ_COMPS(output_dp));

	// This block copy assumes that the block is contiguous...
	memcpy(data->input[0],OBJ_DATA_PTR(input_dp),n_data*OBJ_COMPS(input_dp)*sizeof(fann_type));
	memcpy(data->output[0],OBJ_DATA_PTR(output_dp),n_data*OBJ_COMPS(output_dp)*sizeof(fann_type));



	// Not clear why weight initialization requires the data???
	fann_init_weights(mfp->mf_fann_p, data);

	fann_destroy_train(data);
#endif // HAVE_FANN
}

static COMMAND_FUNC(do_list_fanns)
{
	list_fanns(SINGLE_QSP_ARG);
}

static COMMAND_FUNC(do_info_fann)
{
	My_FANN *mfp;

	mfp = pick_fann(QSP_ARG  "name of network");
	if( mfp == NULL ) return;

#ifdef HAVE_FANN
	fann_print_parameters(mfp->mf_fann_p);
#endif // HAVE_FANN
}

static COMMAND_FUNC(do_del_fann)
{
	My_FANN *mfp;

	mfp = pick_fann(QSP_ARG  "name of network");
	if( mfp == NULL ) return;

#ifdef HAVE_FANN
	fann_destroy(mfp->mf_fann_p);
#endif // HAVE_FANN

	del_fann(QSP_ARG  mfp);	// does this release the name???
}

//#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(fann_menu,s,f,h)

MENU_BEGIN(fann)
ADD_CMD(create_std,	do_create_std,		create standard network	)
ADD_CMD(init_weights,	do_init_weights,	initialize network weights )
ADD_CMD(train,		do_fann_train,		train a network)
ADD_CMD(run,		do_fann_run,		run a network)
ADD_CMD(list,		do_list_fanns,		list all networks)
ADD_CMD(info,		do_info_fann,		print info about a network)
ADD_CMD(delete,		do_del_fann,		delete a network)
MENU_END(fann)

COMMAND_FUNC(do_fann_menu)
{
	PUSH_MENU(fann)
}

#ifdef FOOBAR

int FANN_API test_callback(struct fann *ann, struct fann_train_data *train,
	unsigned int max_epochs, unsigned int epochs_between_reports, 
	float desired_error, unsigned int epochs)
{
	printf("Epochs     %8d. MSE: %.5f. Desired-MSE: %.5f\n", epochs, fann_get_MSE(ann), desired_error);
	return 0;
}

int main()
{
	fann_type *calc_out;
	const unsigned int num_input = 2;
	const unsigned int num_output = 1;
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 3;
	const float desired_error = (const float) 0;
	const unsigned int max_epochs = 1000;
	const unsigned int epochs_between_reports = 10;
	struct fann *ann;
	struct fann_train_data *data;

	unsigned int i = 0;
	unsigned int decimal_point;

	printf("Creating network.\n");
	ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);

	data = fann_read_train_from_file("xor.data");

	fann_set_activation_steepness_hidden(ann, 1);
	fann_set_activation_steepness_output(ann, 1);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

	fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
	fann_set_bit_fail_limit(ann, 0.01f);

	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);

	fann_init_weights(ann, data);
	
	printf("Training network.\n");
	fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error);

	printf("Testing network. %f\n", fann_test_data(ann, data));

	for(i = 0; i < fann_length_train_data(data); i++)
	{
		calc_out = fann_run(ann, data->input[i]);
		printf("XOR test (%f,%f) -> %f, should be %f, difference=%f\n",
			   data->input[i][0], data->input[i][1], calc_out[0], data->output[i][0],
			   fann_abs(calc_out[0] - data->output[i][0]));
	}

	printf("Saving network.\n");

	fann_save(ann, "xor_float.net");

	decimal_point = fann_save_to_fixed(ann, "xor_fixed.net");
	fann_save_train_to_fixed(data, "xor_fixed.data", decimal_point);

	printf("Cleaning up.\n");
	fann_destroy_train(data);
	fann_destroy(ann);

	return 0;
}

#endif // FOOBAR
