
/**
 *
 * a puredata wrapper for aubio zero crossing rate function 
 *
 * Thanks to Johannes M Zmolnig for writing the excellent HOWTO:
 *       http://iem.kug.ac.at/pd/externals-HOWTO/  
 *
 * */

#include <m_pd.h>
#include <aubio.h>

char aubiozcr_version[] = "aubiozcr~ version 0.1";

static t_class *aubiozcr_tilde_class;

void aubiozcr_tilde_setup (void);

typedef struct _aubiozcr_tilde 
{
	t_object x_obj;
	t_int pos; /*frames%dspblocksize*/
	t_int bufsize;
	t_float f;
	fvec_t *vec;
	t_outlet *zcr;
} t_aubiozcr_tilde;

static t_int *aubiozcr_tilde_perform(t_int *w) 
{
	t_aubiozcr_tilde *x = (t_aubiozcr_tilde *)(w[1]);
	t_sample *in        = (t_sample *)(w[2]);
	int n               = (int)(w[3]);
	int j;
	for (j=0;j<n;j++) {
		/* write input to datanew */
		fvec_write_sample(x->vec, in[j], 0, x->pos);
		/*time for fft*/
		if (x->pos == x->bufsize-1) {         
			/* block loop */
			outlet_float(x->zcr, aubio_zero_crossing_rate(x->vec));
			/* end of block loop */
			x->pos = -1; /* so it will be zero next j loop */
		}
		x->pos++;
	}
	return (w+4);
}

static void aubiozcr_tilde_dsp(t_aubiozcr_tilde *x, t_signal **sp)
{
	dsp_add(aubiozcr_tilde_perform, 3, x, sp[0]->s_vec, sp[0]->s_n);
}

static void aubiozcr_tilde_debug(t_aubiozcr_tilde *x)
{
	post("aubiozcr~ bufsize:\t%d", x->bufsize);
	post("aubiozcr~ audio in:\t%f", x->vec->data[0][0]);
}

static void *aubiozcr_tilde_new (void)
{
	t_aubiozcr_tilde *x = 
		(t_aubiozcr_tilde *)pd_new(aubiozcr_tilde_class);

	x->bufsize   = 1024;

	x->vec = (fvec_t *)new_fvec(x->bufsize,1);

	x->zcr = outlet_new (&x->x_obj, &s_float);
	post(aubiozcr_version);
	return (void *)x;
}

void aubiozcr_tilde_setup (void)
{
	aubiozcr_tilde_class = class_new (gensym ("aubiozcr~"),
			(t_newmethod)aubiozcr_tilde_new,
			0, sizeof (t_aubiozcr_tilde),
			CLASS_DEFAULT, 0);
	class_addmethod(aubiozcr_tilde_class, 
			(t_method)aubiozcr_tilde_dsp, 
			gensym("dsp"), 0);
	class_addmethod(aubiozcr_tilde_class, 
			(t_method)aubiozcr_tilde_debug,
        		gensym("debug"), 0);
	CLASS_MAINSIGNALIN(aubiozcr_tilde_class, 
			t_aubiozcr_tilde, f);
}

