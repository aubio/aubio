
def draw_keyboard(firstnote = 21, lastnote = 108, y0 = 0, y1 = 1):
  import Gnuplot
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

  whites  = Gnuplot.Data(xw,yw,xwdelta,ywdelta,with_ = 'boxxyerrorbars')
  blacks  = Gnuplot.Data(xb,yb,xbdelta,ybdelta,with_ = 'boxxyerrorbars fill solid')

  return blacks,whites

if __name__ == '__main__':
  from aubio.gnuplot import gnuplot_create
  blacks,whites = draw_keyboard(firstnote = 21, lastnote = 108)
  g = gnuplot_create('','')
  #g('set style fill solid .5')
  #g('set xrange [60-.5:72+.5]')
  #g('set yrange [-0.1:1.1]')

  g.plot(whites,blacks)
