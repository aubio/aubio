#define PYAUBIO_dct_doc \
    "dct(size=1024)\n"\
    "\n"\
    "Compute Discrete Fourier Transorms of Type-II.\n"\
    "\n"\
    "Parameters\n"\
    "----------\n"\
    "size : int\n"\
    "    size of the DCT to compute\n"\
    "\n"\
    "Example\n"\
    "-------\n"\
    ">>> d = aubio.dct(16)\n"\
    ">>> d.size\n"\
    "16\n"\
    ">>> x = aubio.fvec(np.ones(d.size))\n"\
    ">>> d(x)\n"\
    "array([4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n"\
    "      dtype=float32)\n"\
    ">>> d.rdo(d(x))\n"\
    "array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],\n"\
    "      dtype=float32)\n"\
    "\n"\
    "References\n"\
    "----------\n"\
    "`DCT-II in Discrete Cosine Transform\n"\
    "<https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II>`_\n"\
    "on Wikipedia.\n"

#define PYAUBIO_mfcc_doc \
    "mfcc(buf_size=1024, n_filters=40, n_coeffs=13, samplerate=44100)\n"\
    "\n"\
    "Compute Mel Frequency Cepstrum Coefficients (MFCC).\n"\
    "\n"\
    "`mfcc` creates a callable which takes a `cvec` as input.\n"\
    "\n"\
    "If `n_filters = 40`, the filterbank will be initialized with\n"\
    ":meth:`filterbank.set_mel_coeffs_slaney`. Otherwise, if `n_filters`\n"\
    "is greater than `0`, it will be initialized with\n"\
    ":meth:`filterbank.set_mel_coeffs` using `fmin = 0`,\n"\
    "`fmax = samplerate/`.\n"\
    "\n"\
    "Example\n"\
    "-------\n"\
    ">>> buf_size = 2048; n_filters = 128; n_coeffs = 13; samplerate = 44100\n"\
    ">>> mf = aubio.mfcc(buf_size, n_filters, n_coeffs, samplerate)\n"\
    ">>> fftgrain = aubio.cvec(buf_size)\n"\
    ">>> mf(fftgrain).shape\n"\
    "(13,)\n"\
    ""

#define PYAUBIO_notes_doc \
    "notes(method=\"default\", buf_size=1024, hop_size=512, samplerate=44100)\n"\
    "\n"\
    "Note detection\n"

#define PYAUBIO_onset_doc \
    "onset(method=\"default\", buf_size=1024, hop_size=512, samplerate=44100)\n"\
    "\n"\
    "Onset detection object. `method` should be one of method supported by\n"\
    ":class:`specdesc`.\n"

#define PYAUBIO_pitch_doc \
    "pitch(method=\"default\", buf_size=1024, hop_size=512, samplerate=44100)\n"\
    "\n"\
    "Pitch detection.\n"\
    "\n"\
    "Supported methods: `yinfft`, `yin`, `yinfast`, `fcomb`, `mcomb`,\n"\
    "`schmitt`, `specacf`, `default` (`yinfft`).\n"

#define PYAUBIO_sampler_doc \
    "sampler(hop_size=512, samplerate=44100)\n"\
    "\n"\
    "Sampler.\n"

#define PYAUBIO_specdesc_doc \
    "specdesc(method=\"default\", buf_size=1024)\n"\
    "\n"\
    "Spectral descriptor.\n"\
    "\n"\
    "Onset novelty methods: `energy`, `hfc`, `complex`, `phase`, `wphase`,\n"\
    "`specdiff`, `kl`, `mkl`, `specflux`.\n"\
    "\n"\
    "Shape description methods: `centroid`, `spread`, `skewness`,\n"\
    "`slope`, `decrease`, `rolloff`.\n"\
    "\n"\
    "Parameters\n"\
    "----------\n"\
    "method : str\n"\
    "    Onset novelty or shape description method.\n"\
    "buf_size : int\n"\
    "    Length of the input frame.\n"

#define PYAUBIO_tempo_doc \
    "tempo(method=\"default\", buf_size=1024, hop_size=512, samplerate=44100)\n"\
    "\n"\
    "Tempo detection and beat tracking.\n"

#define PYAUBIO_tss_doc \
    "tss(buf_size=1024, hop_size=512)\n"\
    "\n"\
    "Transient/Steady-state separation.\n"
