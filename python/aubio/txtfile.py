"""Copyright (C) 2004 Paul Brossier <piem@altern.org>
print aubio.__LICENSE__ for the terms of use
"""

__LICENSE__ = """\
  Copyright (C) 2004-2009 Paul Brossier <piem@aubio.org>

  This file is part of aubio.

  aubio is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  aubio is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with aubio.  If not, see <http://www.gnu.org/licenses/>.
"""            

def read_datafile(filename,depth=-1):
    """read list data from a text file (columns of float)"""
    if filename == '--' or filename == '-':
        import sys
        fres = sys.stdin
    else:
        fres = open(filename,'ro')
    l = []
    while 1:
        tmp = fres.readline()
        if not tmp : break
        else: tmp = tmp.split()
        if depth > 0:
            for i in range(min(depth,len(tmp))):
                tmp[i] = float(tmp[i])
            l.append(tmp)
        elif depth == 0:
            l.append(float(tmp[0]))
        else:
            for i in range(len(tmp)):
                tmp[i] = float(tmp[i])
            l.append(tmp)
    return l

