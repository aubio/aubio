/**
 *
 * a puredata wrapper for aubio onset detection functions 
 *
 * Thanks to Johannes M Zmolnig for writing the excellent HOWTO:
 *       http://iem.kug.ac.at/pd/externals-HOWTO/  
 *
 * */

#include <m_pd.h>
#include <aubio.h>

char aubioonset_version[] = "aubioonset~ version 0.1";

static t_class *aubioonset_tilde_class;

void aubioonset_tilde_setup (void);

typedef struct _aubioonset_tilde 
{
	t_object x_obj;
	t_float threshold;	
	t_int bufsize;
	t_int hopsize;
	aubio_onsetdetection_t *o;
	aubio_pvoc_t * pv;
	fvec_t *vec;
	fvec_t *onset;
	cvec_t *fftgrain;
	t_outlet *onsetval;
} t_aubioonset_tilde;

static t_int *aubioonset_tilde_perform(t_int *w) 
{
	t_aubioonset_tilde *x = (t_aubioonset_tilde *)(w[1]);
	t_sample *in          = (t_sample *)(w[2]);
	int n                 = (int)(w[3]);
	//t_sample f = (x->threshold < 0.) ? 0.2 : 
	//	(x->threshold > 10.) ? 10. : x->threshold;
	while (n--) //*(x->vec->data[0])++ = (*in++);
		x->vec->data[0][n] = in[n];
	aubio_pvoc_do (x->pv, x->vec, x->fftgrain);
	aubio_onsetdetection(x->o, x->fftgrain, x->onset);
	outlet_float(x->onsetval, x->onset->data[0][0]);
	return (w+4);
}

static void aubioonset_tilde_dsp(t_aubioonset_tilde *x, t_signal **sp)
{
	dsp_add(aubioonset_tilde_perform, 3, x, sp[0]->s_vec, sp[0]->s_n);
}

static void aubioonset_tilde_debug(t_aubioonset_tilde *x)
{
	post("aubioonset~ bufsize:\t%d", x->bufsize);
	post("aubioonset~ hopsize:\t%d", x->hopsize);
	post("aubioonset~ threshold:\t%f", x->threshold);
	post("aubioonset~ audio in:\t%f", x->vec->data[0][0]);
	post("aubioonset~ onset:\t%f", x->onset->data[0][0]);
	outlet_float(x->onsetval, x->threshold);
}

static void *aubioonset_tilde_new (t_floatarg f)
{
	t_aubioonset_tilde *x = 
		(t_aubioonset_tilde *)pd_new(aubioonset_tilde_class);

	x->threshold = (f < 1e-5) ? 0.1 : (f > 10.) ? 10. : f;
	/* should get from block~ size */
	x->bufsize   = 1024;
	x->hopsize   = x->bufsize / 2;

	x->o = new_aubio_onsetdetection(aubio_onset_complex, x->bufsize, 1);
	x->vec = (fvec_t *)new_fvec(x->hopsize,1);
	x->pv = new_aubio_pvoc(x->bufsize, x->hopsize, 1);
	x->fftgrain  = new_cvec(x->bufsize,1);
	x->onset = (fvec_t *)new_fvec(1,1);

  	floatinlet_new (&x->x_obj, &x->threshold);
	x->onsetval = outlet_new (&x->x_obj, &s_float);
	return (void *)x;
}

void aubioonset_tilde_setup (void)
{
	aubioonset_tilde_class = class_new (gensym ("aubioonset~"),
			(t_newmethod)aubioonset_tilde_new,
			0, sizeof (t_aubioonset_tilde),
			CLASS_DEFAULT, A_DEFFLOAT, 0);
	class_addmethod(aubioonset_tilde_class, 
			(t_method)aubioonset_tilde_dsp, 
			gensym("dsp"), 0);
	class_addmethod(aubioonset_tilde_class, 
			(t_method)aubioonset_tilde_debug,
        		gensym("debug"), 0);
	class_sethelpsymbol(aubioonset_tilde_class, 
			gensym("help-aubioonset~.pd"));
	CLASS_MAINSIGNALIN(aubioonset_tilde_class, 
			t_aubioonset_tilde, threshold);
	post("aubioonset~ v0.1 for puredata");
}

