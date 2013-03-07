#! /usr/bin/env python

def get_keyboard_edges(firstnote = 21, lastnote = 108, y0 = 0, y1 = 1):
  octaves = 10

  # build template of white notes
  scalew  = 12/7.
  xw_temp = [i*scalew for i in range(0,7)]
  # build template of black notes
  scaleb  = 6/7.
  xb_temp = [i*scaleb for i in [1,3,7,9,11]]

  xb,xw = [],[]
  for octave in range(octaves-1): 
    for i in xb_temp:
      curnote = i+12*octave
      if  curnote > firstnote-1 and curnote < lastnote+1:
        xb = xb + [curnote] 
  for octave in range(octaves-1): 
    for i in xw_temp:
      curnote = i+12*octave
      if  curnote > firstnote-1 and curnote < lastnote+1:
        xw = xw + [curnote]

  xwdelta = [1/2. * scalew for i in range(len(xw))]
  yw      = [y0+(y1-y0)*1/2. for i in range(len(xw))]
  ywdelta = [(y1-y0)*1/2. for i in range(len(xw))]

  xbdelta = [2/3. * scaleb for i in range(len(xb))]
  yb      = [y0+(y1-y0)*2/3. for i in range(len(xb))]
  ybdelta = [(y1-y0)*1/3. for i in range(len(xb))]

  whites,white_height = xw,yw
  blacks,black_height = xb,yb

  return blacks,whites, 2/3. *scaleb, 1/2. * scalew

def create_keyboard_patches(firstnote, lastnote, ax = None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    import matplotlib.patches as mpatches

    blacks, whites, b_width, w_width = get_keyboard_edges(firstnote, lastnote)

    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    verts, codes = [], []
    for white in whites:
        verts += [ (white - w_width, 0), (white - w_width, 1), (white + w_width, 1),  (white + w_width, 0) ]
        verts += [ (white - w_width, 0) ]
        codes  += [Path.MOVETO] + [Path.LINETO] * 4
    path = Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor= 'white', edgecolor='black', lw=1)
    ax.add_patch(patch)

    verts, codes = [], []
    for black in blacks:
        verts +=  [ (black - b_width, 0.33), (black - b_width, 1), (black + b_width, 1),  (black + b_width, 0.33) ]
        verts += [ (black - b_width, 0.33) ]
        codes += [Path.MOVETO] + [Path.LINETO] * 4
    path = Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor= 'black', edgecolor='black', lw=1)
    ax.add_patch(patch)

    ax.axis(xmin = firstnote, xmax = lastnote)

if __name__ == '__main__':

  if 0:
    from aubio.gnuplot import gnuplot_create
    import Gnuplot
    whites  = Gnuplot.Data(blacks, yw,xwdelta,ywdelta,with_ = 'boxxyerrorbars')
    blacks  = Gnuplot.Data(whites, yb,xbdelta,ybdelta,with_ = 'boxxyerrorbars fill solid')
    g = gnuplot_create('','')
    #g('set style fill solid .5')
    #g('set xrange [60-.5:72+.5]')
    #g('set yrange [-0.1:1.1]')
    g.plot(whites,blacks)
  else:
    import matplotlib.pyplot as plt
    create_keyboard_patches(firstnote = 61, lastnote = 108)
    plt.show()
