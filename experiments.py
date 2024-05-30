from nilmtk.api import API
import warnings
from nilmtk.disaggregate import CO, Hart85, Mean, FHMMExact

import h5py

filepath = 'ukdale.h5'
with h5py.File(filepath, 'r') as f:
    print(f.keys())

warnings.filterwarnings("ignore")

experiment = {
    'power': {'mains': ['apparent', 'active'], 'appliance': ['apparent', 'active']},
    'sample_rate': 60,
    'appliances': ['fridge', 'air conditioner', 'electric furnace', 'washing machine'],
    'artificial_aggregate': True,
    'chunksize': 20000,
    'DROP_ALL_NANS': False,
    'methods': {"Mean": Mean({}), "Hart85": Hart85({}), "FHMM_EXACT": FHMMExact({'num_of_states': 2}), "CO": CO({})},
    'train': {
        'datasets': {
            'Dataport': {
                'path': 'ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': '2015-01-28',
                        'end_time': '2015-02-12'
                    }
                    #,
                    #2: {
                    #    'start_time': '2015-01-28',
                    #    'end_time': '2015-02-12'
                    #},
                    #3: {
                    #    'start_time': '2015-04-30',
                    #    'end_time': '2015-05-14'
                    #}"""
                }
            }
        }
    },
    'test': {
        'datasets': {
            'Datport': {
                'path': 'ukdale.h5',
                'buildings': {
                    4: {
                        'start_time': '2015-04-30',
                        'end_time': '2015-05-07'
                    },
                    5: {
                        'start_time': '2014-01-26',
                        'end_time': '2014-02-03'
                    }
                }
            }
        },
        'metrics': ['mae', 'rmse']
    }
}

experiment_results = API(experiment)
