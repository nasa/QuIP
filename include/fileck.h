#ifndef _FILECK_H_
#define _FILECK_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include "quip_fwd.h"

extern int _path_exists(QSP_ARG_DECL  const char *name);
extern int _directory_exists(QSP_ARG_DECL  const char *dirname);
extern int _regfile_exists(QSP_ARG_DECL  const char *dirname);
extern int _can_write_to(QSP_ARG_DECL  const char *name);
extern int _can_read_from(QSP_ARG_DECL  const char *name);
extern int _file_exists(QSP_ARG_DECL  const char *pathname);
extern /* long */ off_t _file_content_size(QSP_ARG_DECL  const char *pathname);
extern /* long */ off_t _fp_content_size(QSP_ARG_DECL  FILE *fp);
extern int _check_file_access(QSP_ARG_DECL  const char *pathname);

#define path_exists(name)	_path_exists(QSP_ARG  name)
#define directory_exists(name)	_directory_exists(QSP_ARG  name)
#define regfile_exists(name)	_regfile_exists(QSP_ARG  name)
#define can_write_to(name)	_can_write_to(QSP_ARG  name)
#define can_read_from(name)	_can_read_from(QSP_ARG  name)
#define file_exists(name)	_file_exists(QSP_ARG  name)
#define file_content_size(name)	_file_content_size(QSP_ARG  name)
#define fp_content_size(fp)	_fp_content_size(QSP_ARG  fp)
#define check_file_access(name)	_check_file_access(QSP_ARG  name)


#ifdef __cplusplus
}
#endif

#endif /* ! _FILECK_H_ */


