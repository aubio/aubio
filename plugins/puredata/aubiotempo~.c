/**
 *
 * a puredata wrapper for aubio tempo detection functions 
 *
 * Thanks to Johannes M Zmolnig for writing the excellent HOWTO:
 *       http://iem.kug.ac.at/pd/externals-HOWTO/  
 *
 * */

#include <m_pd.h>
#include <aubio.h>

char aubiotempo_version[] = "aubiotempo~ version 0.1";

static t_class *aubiotempo_tilde_class;

void aubiotempo_tilde_setup (void);

typedef struct _aubiotempo_tilde 
{
	t_object x_obj;
	t_float threshold;	
	t_float threshold2;	
	t_int pos; /*frames%dspblocksize*/
	t_int bufsize;
	t_int hopsize;
	t_int pos2;
	t_int winlen;
	t_int step;
	aubio_onsetdetection_t *o;
	aubio_pvoc_t * pv;
	aubio_pickpeak_t * parms;
	aubio_beattracking_t * bt;
	fvec_t *out;
	fvec_t *dfframe;
	fvec_t *vec;
	fvec_t *onset;
	cvec_t *fftgrain;
	t_outlet *tempobang;
	t_outlet *onsetbang;
} t_aubiotempo_tilde;

static t_int *aubiotempo_tilde_perform(t_int *w) 
{
	t_aubiotempo_tilde *x = (t_aubiotempo_tilde *)(w[1]);
	t_sample *in          = (t_sample *)(w[2]);
	int n                 = (int)(w[3]);
	int winlen            = x->winlen;
	int step              = x->step;
	int j,i,isonset,istactus;
	smpl_t * btoutput = x->out->data[0];
	for (j=0;j<n;j++) {
		/* write input to datanew */
		fvec_write_sample(x->vec, in[j], 0, x->pos);
		/*time for fft*/
		if (x->pos == x->hopsize-1) {         
			/* block loop */
			aubio_pvoc_do (x->pv,x->vec, x->fftgrain);
			aubio_onsetdetection(x->o,x->fftgrain, x->onset);
			if (x->pos2 == step -1 ) {
				aubio_beattracking_do(x->bt,x->dfframe,x->out);
				/* rotate dfframe */
				for (i = 0 ; i < winlen - step; i++ ) 
					x->dfframe->data[0][i] = x->dfframe->data[0][i+step];
				for (i = winlen - step ; i < winlen; i++ ) 
					x->dfframe->data[0][i] = 0.;
				x->pos2 = -1;
			}
			x->pos2++;
			isonset = aubio_peakpick_pimrt_wt(x->onset,x->parms,&(x->dfframe->data[0][winlen - step + x->pos2]));
			/* end of second level loop */
			istactus = 0;
			i=0;
			for (i = 1; i < btoutput[0]; i++ ) { 
				/* test for silence */
				if (aubio_silence_detection(x->vec, x->threshold2)==1) {
					isonset  = 0; istactus = 0;
				} else {
					if (x->pos2 == btoutput[i]) {
						outlet_bang(x->tempobang);
					}
					if (isonset) {
						outlet_bang(x->onsetbang);
					}
				}
			}

			/* end of block loop */
			x->pos = -1; /* so it will be zero next j loop */
		}
		x->pos++;
	}
	return (w+4);
}

static void aubiotempo_tilde_dsp(t_aubiotempo_tilde *x, t_signal **sp)
{
	dsp_add(aubiotempo_tilde_perform, 3, x, sp[0]->s_vec, sp[0]->s_n);
}

static void aubiotempo_tilde_debug(t_aubiotempo_tilde *x)
{
	post("aubiotempo~ bufsize:\t%d", x->bufsize);
	post("aubiotempo~ hopsize:\t%d", x->hopsize);
	post("aubiotempo~ threshold:\t%f", x->threshold);
	post("aubiotempo~ audio in:\t%f", x->vec->data[0][0]);
	post("aubiotempo~ onset:\t%f", x->onset->data[0][0]);
}

static void *aubiotempo_tilde_new (t_floatarg f)
{
	t_aubiotempo_tilde *x = 
		(t_aubiotempo_tilde *)pd_new(aubiotempo_tilde_class);

	x->threshold = (f < 1e-5) ? 0.1 : (f > 10.) ? 10. : f;
	x->threshold2 = -70.;
	/* should get from block~ size */
	x->bufsize   = 1024;
	x->hopsize   = x->bufsize / 2;
  	x->winlen = 512*512/x->hopsize;
  	x->step = x->winlen/4;

	x->o = new_aubio_onsetdetection(aubio_onset_complex, x->bufsize, 1);
	x->vec = (fvec_t *)new_fvec(x->hopsize,1);
	x->pv = (aubio_pvoc_t *)new_aubio_pvoc(x->bufsize, x->hopsize, 1);
	x->fftgrain  = (cvec_t *)new_cvec(x->bufsize,1);
	x->onset = (fvec_t *)new_fvec(1,1);
  	x->parms = new_aubio_peakpicker(x->threshold);
  	x->bt = (aubio_beattracking_t *)new_aubio_beattracking(x->winlen,1);
  	x->dfframe = (fvec_t *)new_fvec(x->winlen,1);
  	x->out = (fvec_t *)new_fvec(x->step,1);

  	floatinlet_new (&x->x_obj, &x->threshold);
	x->tempobang = outlet_new (&x->x_obj, &s_bang);
	x->onsetbang = outlet_new (&x->x_obj, &s_bang);
	post(aubiotempo_version);
	return (void *)x;
}

void aubiotempo_tilde_setup (void)
{
	aubiotempo_tilde_class = class_new (gensym ("aubiotempo~"),
			(t_newmethod)aubiotempo_tilde_new,
			0, sizeof (t_aubiotempo_tilde),
			CLASS_DEFAULT, A_DEFFLOAT, 0);
	class_addmethod(aubiotempo_tilde_class, 
			(t_method)aubiotempo_tilde_dsp, 
			gensym("dsp"), 0);
	class_addmethod(aubiotempo_tilde_class, 
			(t_method)aubiotempo_tilde_debug,
        		gensym("debug"), 0);
	class_sethelpsymbol(aubiotempo_tilde_class, 
			gensym("help-aubiotempo~.pd"));
	CLASS_MAINSIGNALIN(aubiotempo_tilde_class, 
			t_aubiotempo_tilde, threshold);
}

