#ifndef _FILECK_H_
#define _FILECK_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include "quip_fwd.h"

extern int path_exists(QSP_ARG_DECL  const char *name);
extern int directory_exists(QSP_ARG_DECL  const char *dirname);
extern int regfile_exists(QSP_ARG_DECL  const char *dirname);
extern int can_write_to(QSP_ARG_DECL  const char *name);
extern int can_read_from(QSP_ARG_DECL  const char *name);
extern int file_exists(QSP_ARG_DECL  const char *pathname);
extern /* long */ off_t file_content_size(QSP_ARG_DECL  const char *pathname);
extern /* long */ off_t fp_content_size(QSP_ARG_DECL  FILE *fp);

extern int check_file_access(QSP_ARG_DECL  const char *pathname);

#ifdef __cplusplus
}
#endif

#endif /* ! _FILECK_H_ */


