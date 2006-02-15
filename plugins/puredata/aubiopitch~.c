/**
 *
 * a puredata wrapper for aubio pitch detection functions 
 *
 * Thanks to Johannes M Zmolnig for writing the excellent HOWTO:
 *       http://iem.kug.ac.at/pd/externals-HOWTO/  
 *
 * */

#include <m_pd.h>
#include <aubio.h>

char aubiopitch_version[] = "aubiopitch~ version 0.1";

aubio_pitchdetection_type type_pitch = aubio_pitch_schmitt; // aubio_pitch_mcomb
aubio_pitchdetection_mode mode_pitch = aubio_pitchm_freq;

static t_class *aubiopitch_tilde_class;

void aubiopitch_tilde_setup (void);

typedef struct _aubiopitch_tilde 
{
	t_object x_obj;
	t_float threshold;	
	t_float threshold2;	
	t_int pos; /*frames%dspblocksize*/
	t_int bufsize;
	t_int hopsize;
	aubio_pitchdetection_t *o;
	fvec_t *vec;
	t_outlet *pitch;
} t_aubiopitch_tilde;

static t_int *aubiopitch_tilde_perform(t_int *w) 
{
	t_aubiopitch_tilde *x = (t_aubiopitch_tilde *)(w[1]);
	t_sample *in          = (t_sample *)(w[2]);
	int n                 = (int)(w[3]);
	int j;
	smpl_t pitch;
	for (j=0;j<n;j++) {
		/* write input to datanew */
		fvec_write_sample(x->vec, in[j], 0, x->pos);
		/*time for fft*/
		if (x->pos == x->hopsize-1) {         
			/* block loop */
			pitch = aubio_pitchdetection(x->o,x->vec);
			outlet_float(x->pitch, pitch);
			/* end of block loop */
			x->pos = -1; /* so it will be zero next j loop */
		}
		x->pos++;
	}
	return (w+4);
}

static void aubiopitch_tilde_dsp(t_aubiopitch_tilde *x, t_signal **sp)
{
	dsp_add(aubiopitch_tilde_perform, 3, x, sp[0]->s_vec, sp[0]->s_n);
}

static void aubiopitch_tilde_debug(t_aubiopitch_tilde *x)
{
	post("aubiopitch~ bufsize:\t%d", x->bufsize);
	post("aubiopitch~ hopsize:\t%d", x->hopsize);
	post("aubiopitch~ threshold:\t%f", x->threshold);
	post("aubiopitch~ audio in:\t%f", x->vec->data[0][0]);
}

//static void *aubiopitch_tilde_new (t_floatarg f)
static void *aubiopitch_tilde_new (void)
{
	t_aubiopitch_tilde *x = 
		(t_aubiopitch_tilde *)pd_new(aubiopitch_tilde_class);

	x->bufsize   = 2048;
	x->hopsize   = x->bufsize / 2;

	//FIXME: get the real samplerate
    	x->o = new_aubio_pitchdetection(x->bufsize*4, 
                    x->hopsize, 1, 44100., type_pitch, mode_pitch);
	x->vec = (fvec_t *)new_fvec(x->hopsize,1);

  	//floatinlet_new (&x->x_obj, &x->threshold);
	x->pitch = outlet_new (&x->x_obj, &s_float);

	post(aubiopitch_version);
	return (void *)x;
}

void aubiopitch_tilde_setup (void)
{
	aubiopitch_tilde_class = class_new (gensym ("aubiopitch~"),
			(t_newmethod)aubiopitch_tilde_new,
			0, sizeof (t_aubiopitch_tilde),
			CLASS_DEFAULT, A_DEFFLOAT, 0);
	class_addmethod(aubiopitch_tilde_class, 
			(t_method)aubiopitch_tilde_dsp, 
			gensym("dsp"), 0);
	class_addmethod(aubiopitch_tilde_class, 
			(t_method)aubiopitch_tilde_debug,
        		gensym("debug"), 0);
	class_sethelpsymbol(aubiopitch_tilde_class, 
			gensym("help-aubiopitch~.pd"));
	CLASS_MAINSIGNALIN(aubiopitch_tilde_class, 
			t_aubiopitch_tilde, threshold);
}

