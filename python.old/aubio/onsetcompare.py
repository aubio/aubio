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

""" this file contains routines to compare two lists of onsets or notes.
it somewhat implements the Receiver Operating Statistic (ROC).
see http://en.wikipedia.org/wiki/Receiver_operating_characteristic
"""

def onset_roc(ltru, lexp, eps):
    """ compute differences between two lists 
          orig = hits + missed + merged 
          expc = hits + bad + doubled
        returns orig, missed, merged, expc, bad, doubled 
    """
    orig, expc = len(ltru), len(lexp)
    # if lexp is empty
    if expc == 0 : return orig,orig,0,0,0,0
    missed, bad, doubled, merged = 0, 0, 0, 0
    # find missed and doubled ones first
    for x in ltru:
        correspond = 0
        for y in lexp:
            if abs(x-y) <= eps:    correspond += 1
        if correspond == 0:        missed += 1
        elif correspond > 1:       doubled += correspond - 1 
    # then look for bad and merged ones
    for y in lexp:
        correspond = 0
        for x in ltru:
            if abs(x-y) <= eps:    correspond += 1
        if correspond == 0:        bad += 1
        elif correspond > 1:       merged += correspond - 1
    # check consistancy of the results
    assert ( orig - missed - merged == expc - bad - doubled)
    return orig, missed, merged, expc, bad, doubled 

def onset_diffs(ltru, lexp, eps):
    """ compute differences between two lists 
          orig = hits + missed + merged 
          expc = hits + bad + doubled
        returns orig, missed, merged, expc, bad, doubled 
    """
    orig, expc = len(ltru), len(lexp)
    # if lexp is empty
    l = []
    if expc == 0 : return l 
    # find missed and doubled ones first
    for x in ltru:
        correspond = 0
        for y in lexp:
            if abs(x-y) <= eps:    l.append(y-x) 
    # return list of diffs
    return l 

def onset_rocloc(ltru, lexp, eps):
    """ compute differences between two lists 
          orig = hits + missed + merged 
          expc = hits + bad + doubled
        returns orig, missed, merged, expc, bad, doubled 
    """
    orig, expc = len(ltru), len(lexp)
    l = []
    labs = []
    mean = 0
    # if lexp is empty
    if expc == 0 : return orig,orig,0,0,0,0,l,mean
    missed, bad, doubled, merged = 0, 0, 0, 0
    # find missed and doubled ones first
    for x in ltru:
        correspond = 0
        for y in lexp:
            if abs(x-y) <= eps:    correspond += 1
        if correspond == 0:        missed += 1
        elif correspond > 1:       doubled += correspond - 1 
    # then look for bad and merged ones
    for y in lexp:
        correspond = 0
        for x in ltru:
            if abs(x-y) <= eps:    
	    	correspond += 1
            	l.append(y-x) 
            	labs.append(abs(y-x))
        if correspond == 0:        bad += 1
        elif correspond > 1:       merged += correspond - 1
    # check consistancy of the results
    assert ( orig - missed - merged == expc - bad - doubled)
    return orig, missed, merged, expc, bad, doubled, l, labs

def notes_roc (la, lb, eps):
    from numpy import transpose, add, resize 
    """ creates a matrix of size len(la)*len(lb) then look for hit and miss
    in it within eps tolerance windows """
    gdn,fpw,fpg,fpa,fdo,fdp = 0,0,0,0,0,0
    m = len(la)
    n = len(lb)
    x =           resize(la[:][0],(n,m))
    y = transpose(resize(lb[:][0],(m,n)))
    teps =  (abs(x-y) <= eps[0]) 
    x =           resize(la[:][1],(n,m))
    y = transpose(resize(lb[:][1],(m,n)))
    tpitc = (abs(x-y) <= eps[1]) 
    res = teps * tpitc
    res = add.reduce(res,axis=0)
    for i in range(len(res)) :
        if res[i] > 1:
            gdn+=1
            fdo+=res[i]-1
        elif res [i] == 1:
            gdn+=1
    fpa = n - gdn - fpa
    return gdn,fpw,fpg,fpa,fdo,fdp

def load_onsets(filename) :
    """ load onsets targets / candidates files in arrays """
    l = [];
    
    f = open(filename,'ro')
    while 1:
        line = f.readline().split()
        if not line : break
        l.append(float(line[0]))
    
    return l
