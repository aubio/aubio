
#include <m_pd.h>

char aubio_version[] = "aubio external for pd, version 0.1";

void aubio_setup (void);
extern void aubioonset_tilde_setup (void);
extern void aubiotempo_tilde_setup (void);
extern void aubiotss_tilde_setup (void);
extern void aubioquiet_tilde_setup (void);
extern void aubiopitch_tilde_setup (void);

void aubio_setup (void)
{
	post(aubio_version);
	aubioonset_tilde_setup();
	aubiotempo_tilde_setup();
	aubiotss_tilde_setup();
	aubioquiet_tilde_setup();
	aubiopitch_tilde_setup();
}
