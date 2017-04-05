
define(`H_CALL_PROJ_2V',	SLOW_HOST_CALL($1,,,,2) )
define(`H_CALL_PROJ_2V_IDX',	SLOW_HOST_CALL($1,,,,2) )
define(`H_CALL_PROJ_3V',	SLOW_HOST_CALL($1,,,,3) )

define(`H_CALL_MM_NOCC',`

GENERIC_HOST_FAST_CALL($1,/*bitmap*/,/*typ*/,/*scalars*/,/*vectors*/)
GENERIC_HOST_EQSP_CALL($1,,,,)
GENERIC_HOST_SLOW_CALL($1,,,,)

/* h_call_mm_nocc calling generic_host_fast_switch... */
GENERIC_HOST_FAST_SWITCH($1,,,,NOCC)
')

