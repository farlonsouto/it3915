from nilmtk.api import API
from nilmtk.disaggregate import FHMMExact, Mean, CO
from hmmlearn.hmm import GaussianHMM
import numpy as np


class RegularizedFHMMExact(FHMMExact):

    def partial_fit(self, train_mains, train_submeters):
        super(RegularizedFHMMExact, self).partial_fit(train_mains, train_submeters)

        # Regularize covariance matrices
        for appliance in self.model:
            if isinstance(self.model[appliance], GaussianHMM):
                covars = self.model[appliance].covars_
                print(f"Before regularization: {covars}")
                try:
                    covars += 1e-4 * np.eye(covars.shape[-1])
                except ValueError as e:
                    print(f"Covariance regularization failed for appliance {appliance}: {e}")
                print(f"After regularization: {covars}")


experiment = {
    'power': {'mains': ['apparent', 'active'], 'appliance': ['apparent', 'active']},
    'sample_rate': 60,
    'appliances': ['kettle', 'toaster', 'microwave'],
    'artificial_aggregate': True,
    'chunksize': 20000,
    'DROP_ALL_NANS': True,
    'methods': {"Mean": Mean({}), "CO": CO({}), "FHMM_EXACT": RegularizedFHMMExact({'num_of_states': 2})},
    'train': {
        'datasets': {
            'Dataport': {
                'path': 'ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': '2014-01-01',
                        'end_time': '2015-02-15'
                    }
                }
            }
        }
    },
    'test': {
        'datasets': {
            'Dataport': {
                'path': 'ukdale.h5',
                'buildings': {
                    5: {
                        'start_time': '2014-01-01',
                        'end_time': '2015-02-15'
                    }
                }
            }
        },
        'metrics': ['mae', 'rmse']
    }
}

experiment_results = API(experiment)
