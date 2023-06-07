import numpy as np


class DeductionRate():
    def __init__(self, trend_class_num, seasonal_class_num):
        pass


def get_mac_deduction(all_mac_bd, mac_bd):
    app_bd = np.sum(all_mac_bd, axis=0)
    app_95_index = np.argsort(app_bd)[int(len(app_bd))*0.95]
    mac_app_95 = mac_bd[app_95_index]
    mac_95 = np.sort(mac_bd)[int(len(mac_bd))*0.95]
    return (mac_95-mac_app_95)/mac_95


def get_app_deduction(all_mac_bd):
    app_bd = np.sum(all_mac_bd, axis=0)
    app_95_index = np.argsort(app_bd)[int(len(app_bd)*0.95)]
    app_95 = app_bd[app_95_index]

    mac_95 = 0
    for mac_bd in all_mac_bd:
        mac_95 += np.sort(mac_bd)[int(len(mac_bd)*0.95)]
    return 1-app_95/mac_95