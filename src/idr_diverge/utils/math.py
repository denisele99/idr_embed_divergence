import numpy as np


def transform_data(data_dict, type_):
    """
    Transforms a dictionary of lists based on the specified type.
    
    Supported types:
    - 'log': apply log10(x + 1) to each value
    - 'log_mean': apply log10(x + 1), then take the mean
    - 'geo_mean': apply log10(x + 1), then take 10 ** mean
    - 'mean': take the mean of each list
    """
    
    if type_ == 'log':
        return {k: np.log10([x + 1 for x in v]) for k, v in data_dict.items()}
    
    elif type_ == 'log_mean':
        return {k: np.mean(np.log10([x + 1 for x in v])) for k, v in data_dict.items()}
    
    elif type_ == 'geo_mean':
        return {k: 10 ** np.mean(np.log10([x + 1 for x in v])) for k, v in data_dict.items()}
    
    elif type_ == 'mean':
        return {k: np.mean(v) for k, v in data_dict.items()}
    
    elif type_ == 'median':
        return {k: np.median(v) for k, v in data_dict.items()}
    
    elif type_ == 'log_median':
        return {k: np.median(np.log10([x + 1 for x in v])) for k, v in data_dict.items()}
    
    elif type_ == 'geo_median':
        return {k: 10 ** np.median(np.log10([x + 1 for x in v])) for k, v in data_dict.items()}
    
    
    else:
        raise ValueError(f"Unsupported type: {type_}. Must be one of {'log','log_mean','geo_mean','mean', 'median','log_median','geo_median'}")