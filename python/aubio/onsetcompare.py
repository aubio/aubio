"""Copyright (C) 2004 Paul Brossier <piem@altern.org>
print aubio.__LICENSE__ for the terms of use
"""

__LICENSE__ = """\
     Copyright (C) 2004 Paul Brossier <piem@altern.org>

     This program is free software; you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation; either version 2 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License
     along with this program; if not, write to the Free Software
     Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
"""            

""" this file contains routines to compare two lists of onsets or notes.
it somewhat implements the Receiver Operating Statistic (ROC).
see http://en.wikipedia.org/wiki/Receiver_operating_characteristic
"""

from numarray import *

def onset_roc(la, lb, eps):
    """ thanks to nicolas wack for the rewrite"""
    """ compute differences between two lists """
    """ feature: scalable to huge lists """
    n, m = len(la), len(lb)
    if m == 0 :
        return 0,0,0,n,0
    missed, bad = 0, 0
    # find missed ones first
    for x in la:
        correspond = 0
        for y in lb:
            if abs(x-y) <= eps:
                correspond += 1
        if correspond == 0:
            missed += 1
    # then look for bad ones
    for y in lb:
        correspond = 0
        for x in la:
            if abs(x-y) <= eps:
               correspond += 1
        if correspond == 0:
            bad += 1
    ok    = n - missed
    hits  = m - bad
    total = n
    return ok,bad,missed,total,hits
    
    
def notes_roc (la, lb, eps):
    """ creates a matrix of size len(la)*len(lb) then look for hit and miss
    in it within eps tolerance windows """
    gdn,fpw,fpg,fpa,fdo,fdp = 0,0,0,0,0,0
    m = len(la)
    n = len(lb)
    x =           resize(la[:,0],(n,m))
    y = transpose(resize(lb[:,0],(m,n)))
    teps =  (abs(x-y) <= eps[0]) 
    x =           resize(la[:,1],(n,m))
    y = transpose(resize(lb[:,1],(m,n)))
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

"""
def onset_roc (la, lb, eps):
    \"\"\" build a matrix of all possible differences between two lists \"\"\"
    \"\"\" bug: not scalable to huge lists \"\"\"
        n, m        = len(la), len(lb)
    if m ==0 :
        return 0,0,0,n,0
        missed, bad = 0, 0
        x           = resize(la[:],(m,n))
        y           = transpose(resize(lb[:],(n,m)))
        teps        = (abs(x-y) <= eps)
        resmis      = add.reduce(teps,axis = 0)
        for i in range(n) :
        if resmis[i] == 0:
            missed += 1
    resbad = add.reduce(teps,axis=1)
    for i in range(m) : 
        if resbad[i] == 0:
            bad += 1
    ok    = n - missed
    hits  = m - bad
    total = n
    return ok,bad,missed,total,hits
"""

