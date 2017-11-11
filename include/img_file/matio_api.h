
#ifndef MATIO_API_H
#define MATIO_API_H

#ifdef HAVE_MATIO

#ifdef FOOBAR
#include <matio.h>
#endif // FOOBAR

#include "img_file.h"

//FIO_INTERFACE_PROTOTYPES( mat , matvar_t )
FIO_INTERFACE_PROTOTYPES( mat , Matio_Hdr )

#define set_mat_hdr(ifp)	_set_mat_hdr(QSP_ARG  ifp)

#endif /* HAVE_MATIO */
#endif /* MATIO_API_H */

