#!/usr/bin/env python3
import numpy as np
import pylab

import curbd

dtData = 0.0641
number_units = 255
steps = 165
tData, xBump, hBump = curbd.bump_gen(number_units, dtData, steps)

out = curbd.trainMultiRegionRNN(hBump, dtData, trainType='currents',
                                verbose=True, dtFactor=10, nRunTrain=120,
                                tauRNN=0.01, g=1.5, ampInWN=0.1, nRunFree=1,
                                plotStatus=False)

mid = int(number_units/2)
fig = pylab.figure()
ax = fig.add_subplot(111)
ax.plot(out['tRNN'], out['RNN'][mid, :], label='RNN')
ax.plot(tData, hBump[mid, :], label='hBump', linestyle='dashed')
ax.plot(tData, xBump[mid, :], label='xBump')
ax.legend()
fig.show()
