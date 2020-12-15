#!/usr/bin/env python3
import pylab

import curbd

dtData = 0.0641
number_units = 255
steps = 165
trainType = 'rates'
tData, xBump, hBump = curbd.bump_gen(number_units, dtData, steps)

if trainType == 'currents':
    activity = hBump
else:
    activity = xBump

out = curbd.trainMultiRegionRNN(activity, dtData, trainType=trainType,
                                verbose=True, dtFactor=100, nRunTrain=40,
                                tauRNN=0.01, g=1.5, ampInWN=0.1, nRunFree=1,
                                plotStatus=True)

mid = int(number_units/2)
fig = pylab.figure()
ax = fig.add_subplot(111)
ax.plot(out['tRNN'], out['RNN'][mid, :], label='RNN')
ax.plot(tData, hBump[mid, :], label='current', linestyle='dashed')
ax.plot(tData, xBump[mid, :], label='rate')
ax.legend()
ax.set_title('trained on {}'.format(trainType))
fig.show()
