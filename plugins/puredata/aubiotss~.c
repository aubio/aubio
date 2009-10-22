/**
 *
 * a puredata wrapper for aubio tss detection functions 
 *
 * Thanks to Johannes M Zmolnig for writing the excellent HOWTO:
 *       http://iem.kug.ac.at/pd/externals-HOWTO/  
 *
 * */

#include <m_pd.h>
#define AUBIO_UNSTABLE 1
#include <aubio.h>

char aubiotss_version[] = "aubiotss~ version 0.1";

static t_class *aubiotss_tilde_class;

void aubiotss_tilde_setup (void);

typedef struct _aubiotss_tilde 
{
	t_object x_obj;
	t_float thres;	
	t_int pos; /*frames%dspblocksize*/
	t_int bufsize;
	t_int hopsize;
	aubio_pvoc_t * pv;
	aubio_pvoc_t * pvt;
	aubio_pvoc_t * pvs;
	aubio_tss_t * tss;
	fvec_t *vec;
	cvec_t *fftgrain;
	cvec_t *cstead;
	cvec_t *ctrans;
	fvec_t *trans;
	fvec_t *stead;
} t_aubiotss_tilde;

static t_int *aubiotss_tilde_perform(t_int *w) 
{
	t_aubiotss_tilde *x = (t_aubiotss_tilde *)(w[1]);
	t_sample *in          = (t_sample *)(w[2]);
	t_sample *outtrans    = (t_sample *)(w[3]);
	t_sample *outstead    = (t_sample *)(w[4]);
	int n                 = (int)(w[5]);
	int j;
	for (j=0;j<n;j++) {
		/* write input to datanew */
		fvec_write_sample(x->vec, in[j], 0, x->pos);
		/*time for fft*/
		if (x->pos == x->hopsize-1) {         
			/* block loop */
			/* test for silence */
			//if (!aubio_silence_detection(x->vec, x->threshold2))
			aubio_pvoc_do  (x->pv,  x->vec, x->fftgrain);
			aubio_tss_set_threshold ( x->tss, x->thres);
			aubio_tss_do   (x->tss, x->fftgrain, x->ctrans, x->cstead);
			aubio_pvoc_rdo (x->pvt, x->ctrans, x->trans);
			aubio_pvoc_rdo (x->pvs, x->cstead, x->stead);
			//}
			/* end of block loop */
			x->pos = -1; /* so it will be zero next j loop */
		}
		x->pos++;
		*outtrans++ = x->trans->data[0][x->pos];
		*outstead++ = x->stead->data[0][x->pos];
	}
	return (w+6);
}

static void aubiotss_tilde_dsp(t_aubiotss_tilde *x, t_signal **sp)
{
	dsp_add(aubiotss_tilde_perform, 5, x, 
			sp[0]->s_vec, sp[1]->s_vec, sp[2]->s_vec, sp[0]->s_n);
}

static void aubiotss_tilde_debug(t_aubiotss_tilde *x)
{
	post("aubiotss~ bufsize:\t%d", x->bufsize);
	post("aubiotss~ hopsize:\t%d", x->hopsize);
	post("aubiotss~ threshold:\t%f", x->thres);
	post("aubiotss~ audio in:\t%f", x->vec->data[0][0]);
	post("aubiotss~ audio out:\t%f", x->stead->data[0][0]);
}

static void *aubiotss_tilde_new (t_floatarg f)
	//, t_floatarg bufsize)
{
	t_aubiotss_tilde *x = 
		(t_aubiotss_tilde *)pd_new(aubiotss_tilde_class);

	x->thres    = (f < 1e-5) ? 0.01 : (f > 1.) ? 1. : f;
	x->bufsize  = 1024; //(bufsize < 64) ? 1024: (bufsize > 16385) ? 16385: bufsize;
	x->hopsize  = x->bufsize / 4;

	x->vec = (fvec_t *)new_fvec(x->hopsize,1);

	x->fftgrain  = (cvec_t *)new_cvec(x->bufsize,1);
	x->ctrans = (cvec_t *)new_cvec(x->bufsize,1);
	x->cstead = (cvec_t *)new_cvec(x->bufsize,1);

	x->trans = (fvec_t *)new_fvec(x->hopsize,1);
	x->stead = (fvec_t *)new_fvec(x->hopsize,1);

	x->pv  = (aubio_pvoc_t *)new_aubio_pvoc(x->bufsize, x->hopsize, 1);
	x->pvt = (aubio_pvoc_t *)new_aubio_pvoc(x->bufsize, x->hopsize, 1);
	x->pvs = (aubio_pvoc_t *)new_aubio_pvoc(x->bufsize, x->hopsize, 1);

	x->tss = (aubio_tss_t *)new_aubio_tss(x->bufsize, x->hopsize, 1);

  	floatinlet_new (&x->x_obj, &x->thres);
	outlet_new(&x->x_obj, gensym("signal"));
	outlet_new(&x->x_obj, gensym("signal"));
	post(aubiotss_version);
	return (void *)x;
}

void aubiotss_tilde_setup (void)
{
	aubiotss_tilde_class = class_new (gensym ("aubiotss~"),
			(t_newmethod)aubiotss_tilde_new,
			0, sizeof (t_aubiotss_tilde),
			CLASS_DEFAULT, A_DEFFLOAT, 0);
	class_addmethod(aubiotss_tilde_class, 
			(t_method)aubiotss_tilde_dsp, 
			gensym("dsp"), 0);
	class_addmethod(aubiotss_tilde_class, 
			(t_method)aubiotss_tilde_debug,
        		gensym("debug"), 0);
	CLASS_MAINSIGNALIN(aubiotss_tilde_class, 
			t_aubiotss_tilde, thres);
}
