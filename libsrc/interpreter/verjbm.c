#include "quip_config.h"

char VersionId_interpreter_verjbm[] = QUIP_VERSION_STRING;

/*
 * This file will get the static defines
 * of the ID strings of the include files.
 * This scheme is suboptimal, in that it does not guarantee that
 * all C files were compiled with the same version of a given
 * .h file.
 */

#include "version.h"
#include "verjbm.h"

void verjbm(SINGLE_QSP_ARG_DECL)
{
	auto_version(QSP_ARG  "INTERPRETER","VersionId_interpreter");
}

