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

char aubiotempo_version[] = "aubiotempo~ version 0.2";

static t_class *aubiotempo_tilde_class;

void aubiotempo_tilde_setup (void);

typedef struct _aubiotempo_tilde 
{
  t_object x_obj;
  t_float threshold;
  t_float silence;
  t_int pos; /*frames%dspblocksize*/
  t_int bufsize;
  t_int hopsize;
  aubio_tempo_t * t;
  fvec_t *vec;
  fvec_t *output;
  t_outlet *tempobang;
  t_outlet *onsetbang;
} t_aubiotempo_tilde;

static t_int *aubiotempo_tilde_perform(t_int *w) 
{
  t_aubiotempo_tilde *x = (t_aubiotempo_tilde *)(w[1]);
  t_sample *in          = (t_sample *)(w[2]);
  int n                 = (int)(w[3]);
  int j;
  for (j=0;j<n;j++) {
    /* write input to datanew */
    fvec_write_sample(x->vec, in[j], 0, x->pos);
    /*time for fft*/
    if (x->pos == x->hopsize-1) {         
      /* block loop */
      aubio_tempo (x->t, x->vec, x->output);
      if (x->output->data[0][0]) {
        outlet_bang(x->tempobang);
      }
      if (x->output->data[0][1]) {
        outlet_bang(x->onsetbang);
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
}

static void *aubiotempo_tilde_new (t_floatarg f)
{
  t_aubiotempo_tilde *x = 
    (t_aubiotempo_tilde *)pd_new(aubiotempo_tilde_class);

  x->threshold = (f < 1e-5) ? 0.1 : (f > 10.) ? 10. : f;
  x->silence = -70.;
  /* should get from block~ size */
  x->bufsize   = 1024;
  x->hopsize   = x->bufsize / 2;

  x->t = new_aubio_tempo (aubio_onset_complex, x->bufsize, x->hopsize, 1);
  aubio_tempo_set_silence(x->t,x->silence);
  aubio_tempo_set_threshold(x->t,x->threshold);
  x->output = (fvec_t *)new_fvec(2,1);
  x->vec = (fvec_t *)new_fvec(x->hopsize,1);

  floatinlet_new (&x->x_obj, &x->threshold);
  x->tempobang = outlet_new (&x->x_obj, &s_bang);
  x->onsetbang = outlet_new (&x->x_obj, &s_bang);
  post(aubiotempo_version);
  return (void *)x;
}

static void *aubiotempo_tilde_del(t_aubiotempo_tilde *x)
{
  if(x->t)      del_aubio_tempo(x->t);
  if(x->output) del_fvec(x->output);
  if(x->vec)    del_fvec(x->vec);
  return 0;
}

void aubiotempo_tilde_setup (void)
{
  aubiotempo_tilde_class = class_new (gensym ("aubiotempo~"),
      (t_newmethod)aubiotempo_tilde_new,
      (t_method)aubiotempo_tilde_del,
      sizeof (t_aubiotempo_tilde),
      CLASS_DEFAULT, A_DEFFLOAT, 0);
  class_addmethod(aubiotempo_tilde_class, 
      (t_method)aubiotempo_tilde_dsp, 
      gensym("dsp"), 0);
  class_addmethod(aubiotempo_tilde_class, 
      (t_method)aubiotempo_tilde_debug,
      gensym("debug"), 0);
  CLASS_MAINSIGNALIN(aubiotempo_tilde_class, 
      t_aubiotempo_tilde, threshold);
}

