
#include "quip_config.h"
#include "quip_prot.h"

int main(int argc,char *argv[])
{
	input_on_stdin();
	start_quip(argc,argv);
	while( QS_LEVEL(DEFAULT_QSP) >= 0 ){
//		qs_do_cmd(SGL_DEFAULT_QSP_ARG);
		qs_do_cmd(DEFAULT_QSP);
	}

	return 0;

}


