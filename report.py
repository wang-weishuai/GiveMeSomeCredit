import pandas as pd
import numpy as np

# instream

tt = 0
ff = 0
ft = 0
tf = 0
good = 0
bad = 0
data = pd.read_csv('./ratio_2.csv')
tru = data['tru']
my = data['my']

for i in range(len(tru)):
    if tru[i] == 1 and my[i] == 1:
        tt += 1
        good += 1
    elif tru[i] == 1 and my[i] == 0:
        ft += 1
        good += 1
    elif tru[i] == 0 and my[i] == 1:
        tf += 1
        bad += 1
    elif tru[i] == 0 and my[i] == 0:
        ff += 1
        bad += 1

print(tt, ff, ft, tf)
print(ff / (ff + tf), tt / (tt + ft))
print(good, bad)
