import warnings

from nilmtk import DataSet

from .generator import TimeSeriesDataGenerator

warnings.simplefilter(action='ignore', category=FutureWarning)


class TimeSeries:
    """
    Encapsulates UK Dale dataset handling intelligence with a focus on loading the data from a single HDF5 format file.
    """

    def __init__(self, dataset: DataSet, training_buildings: list, test_buildings: list, wandb_config):
        self.training_buildings = training_buildings
        self.test_buildings = test_buildings
        self.dataset = dataset
        self.wandb_config = wandb_config
        self.appliance = wandb_config.appliance
        # Calculated for UK Dale dataset and preserved
        self.normalization_params = {
            "aggregated": {
                "active": {
                    "mean": 356.242431640625,
                    "std": 481.7847900390625
                },
                "apparent": {
                    "mean": 415.4137878417969,
                    "std": 490.3323669433594
                }
            },
            "appliance": {
                "fridge": {
                    "mean": 40.021522521972656,
                    "std": 52.8307991027832
                },
                "kettle": {
                    "mean": 17.524539947509766,
                    "std": 200.2118377685547
                },
                "microwave": {
                    "mean": 11.71904468536377,
                    "std": 107.25408935546875
                },
                "dish washer": {
                    "mean": 27.594993591308594,
                    "std": 237.3380584716797
                }
            }
        }

    def getTrainingDataGenerator(self):
        return TimeSeriesDataGenerator(
            self.dataset, self.training_buildings, self.appliance, self.normalization_params, self.wandb_config,
            is_training=True
        )

    def getTestDataGenerator(self):
        return TimeSeriesDataGenerator(
            self.dataset, self.test_buildings, self.appliance, self.normalization_params, self.wandb_config,
            is_training=False
        )
