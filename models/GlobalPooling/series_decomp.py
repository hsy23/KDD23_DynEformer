import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL


def s_decomp(s, type):
    stl = STL(s, period=24, robust=True)
    res_robust = stl.fit()
    # fig = res_robust.plot()
    # plt.show()
    if type == 'trend':
        return res_robust.trend
    else:
        return res_robust.seasonal