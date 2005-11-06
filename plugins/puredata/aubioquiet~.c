/**
 *
 * a puredata wrapper for aubioquiet
 *
 * Thanks to Johannes M Zmolnig for writing the excellent HOWTO:
 *       http://iem.kug.ac.at/pd/externals-HOWTO/  
 *
 * */

#include <m_pd.h>
#include <aubio.h>

char aubioquiet_version[] = "aubioquiet~ version 0.1";

static t_class *aubioquiet_tilde_class;

void aubioquiet_tilde_setup (void);

typedef struct _aubioquiet_tilde 
{
	t_object x_obj;
	t_float threshold;
	t_int pos; /*frames%dspblocksize*/
	t_int bufsize;
	t_int hopsize;
	t_int wassilence;
	t_int issilence;
	fvec_t *vec;
	t_outlet *quietbang;
	t_outlet *noisybang;
} t_aubioquiet_tilde;

static t_int *aubioquiet_tilde_perform(t_int *w) 
{
	t_aubioquiet_tilde *x = (t_aubioquiet_tilde *)(w[1]);
	t_sample *in          = (t_sample *)(w[2]);
	int n                 = (int)(w[3]);
	int j;
	for (j=0;j<n;j++) {
		/* write input to datanew */
		fvec_write_sample(x->vec, in[j], 0, x->pos);
		/*time for fft*/
		if (x->pos == x->hopsize-1) {         
			/* block loop */
			if (aubio_silence_detection(x->vec, x->threshold)==1) {
				if (x->wassilence==1) {
					x->issilence = 1;
				} else {
					x->issilence = 2;
					outlet_bang(x->quietbang);
				}
				x->wassilence=1;
			} else { 
				if (x->wassilence<=0) {
					x->issilence = 0;
				} else {
					x->issilence = -1;
					outlet_bang(x->noisybang);
				}
				x->wassilence=0;
			}
			/* end of block loop */
			x->pos = -1; /* so it will be zero next j loop */
		}
		x->pos++;
	}
	return (w+4);
}

static void aubioquiet_tilde_dsp(t_aubioquiet_tilde *x, t_signal **sp)
{
	dsp_add(aubioquiet_tilde_perform, 3, x, sp[0]->s_vec, sp[0]->s_n);
}

static void aubioquiet_tilde_debug(t_aubioquiet_tilde *x)
{
	post("aubioquiet~ bufsize:\t%d", x->bufsize);
	post("aubioquiet~ hopsize:\t%d", x->hopsize);
	post("aubioquiet~ threshold:\t%f", x->threshold);
	post("aubioquiet~ audio in:\t%f", x->vec->data[0][0]);
}

static void *aubioquiet_tilde_new (t_floatarg f)
{
	t_aubioquiet_tilde *x = 
		(t_aubioquiet_tilde *)pd_new(aubioquiet_tilde_class);

	x->threshold = (f < -1000.) ? -70 : (f >= 0.) ? -70. : f;
	x->bufsize   = 1024;
	x->hopsize   = x->bufsize / 2;

	x->vec = (fvec_t *)new_fvec(x->hopsize,1);
	x->wassilence = 1;

  	floatinlet_new (&x->x_obj, &x->threshold);
	x->quietbang = outlet_new (&x->x_obj, &s_bang);
	x->noisybang = outlet_new (&x->x_obj, &s_bang);
	post(aubioquiet_version);
	return (void *)x;
}

void aubioquiet_tilde_setup (void)
{
	aubioquiet_tilde_class = class_new (gensym ("aubioquiet~"),
			(t_newmethod)aubioquiet_tilde_new,
			0, sizeof (t_aubioquiet_tilde),
			CLASS_DEFAULT, A_DEFFLOAT, 0);
	class_addmethod(aubioquiet_tilde_class, 
			(t_method)aubioquiet_tilde_dsp, 
			gensym("dsp"), 0);
	class_addmethod(aubioquiet_tilde_class, 
			(t_method)aubioquiet_tilde_debug,
        		gensym("debug"), 0);
	class_sethelpsymbol(aubioquiet_tilde_class, 
			gensym("help-aubioquiet~.pd"));
	CLASS_MAINSIGNALIN(aubioquiet_tilde_class, 
			t_aubioquiet_tilde, threshold);
}

