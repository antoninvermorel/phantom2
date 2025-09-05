"""This file is where the function get_valid_intervals used for tests is implemented"""

import numpy as np
from phantom_2 import tracking

def get_valid_intervals(ds_read, tracking_config, ideal_result, var_param, context, val_range, step):
    """
    Cherche les intervalles continus de valeurs d'un paramètre pour lesquels
    le tracking produit le résultat attendu.

    Parameters
    ----------
    dataset : Dataset
        Données sur lesquelles exécuter le tracking.
    tracking_config : dict
        Config de tracking initiale.
    ideal_result : dict
        Dictionnaire {trackedblob_id: set(parents)} correspondant au résultat attendu.
    var_param : str
        Nom du paramètre à tester.
    context : {"assoc", "interac"}
        Which context is considered (association or interaction).
    val_range : tuple
        (min, max) de valeurs à tester.
    step : float
        Pas pour le balayage.

    Returns
    -------
    intervals : list of tuples
        Liste des intervalles (min, max) de valeurs du paramètre valides.
    """
    import copy
    
    valid_vals = []
    vals_to_test = np.linspace(val_range[0], val_range[1], num=int((val_range[1]-val_range[0])/step)+1)

    if f"{context}_euclidean_features" in tracking_config:
        subconfig_key = f"{context}_euclidean_features"
    elif f"{context}_mahalanobis_features" in tracking_config:
        subconfig_key = f"{context}_mahalanobis_features"
    else:
        raise KeyError(f"No valid subconfig found for context '{context}'")

    for val in vals_to_test:
        print(f"test with {var_param} = {np.round(val, 3)}")
        config_test = copy.deepcopy(tracking_config)
        config_test[subconfig_key][var_param] = val
        tracked_blobs_test = tracking(ds_read, config_test)

        result = {tb.id: tb.parents for tb in tracked_blobs_test}
        if result == ideal_result:
            valid_vals.append(val)        

    intervals = []
    if valid_vals:
        start = prev = valid_vals[0]
        for v in valid_vals[1:]:
            if abs(v - prev) > abs(step) * 1.1:
                intervals.append([np.round(start, 3), np.round(prev, 3)])
                start = v
            prev = v
        intervals.append([np.round(start, 3), np.round(prev, 3)])
    
    print("\n")
    print(f"valid_{var_param}_intervals :", intervals)
    print("\n")
    return intervals





















