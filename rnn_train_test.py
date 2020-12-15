#!/usr/bin/env python3
import curbd

dtData = 0.0641
number_units = 500
steps = 165
tData, xBump, hBump = curbd.bump_gen(number_units, dtData, steps)

out = curbd.trainMultiRegionRNN(xBump, dtData, trainType='currents',
                                verbose=True, dtFactor=100, nRunTrain=120,
                                tauRNN=0.01, g=1.5, ampInWN=0.1, nRunFree=1,
                                plotStatus=True)
