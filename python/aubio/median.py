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

""" 
original author Tim Peters
modified by Paul Brossier <piem@altern.org>
inspired from http://www.ics.uci.edu/~eppstein/161/python/peters-selection.py
"""

def short_find(a, rank):
    """ find the rank-th value in sorted a """
    # copy to b before sorting
    b = a[:]
    b.sort()
    return b[rank - 1]

def percental(a, rank):
    """ Find the rank'th-smallest value in a, in worst-case linear time. """
    n = len(a)
    assert 1 <= rank <= n
    if n <= 7:
        return short_find(a, rank)

    ## Find median of median-of-7's.
    ##medians = [short_find(a[i : i+7], 4) for i in xrange(0, n-6, 7)]
    #median = find(medians, (len(medians) + 1) // 2)
    
    # modified to Find median
    median = short_find([a[0], a[-1], a[n//2]], 2)

    # Partition around the median.
    # a[:i]   <= median
    # a[j+1:] >= median
    i, j = 0, n-1
    while i <= j:
        while a[i] < median:
            i += 1
        while a[j] > median:
            j -= 1
        if i <= j:
            a[i], a[j] = a[j], a[i]
            i += 1
            j -= 1

    if rank <= i:
        return percental(a[:i], rank)
    else:
        return percental(a[i:], rank - i)

