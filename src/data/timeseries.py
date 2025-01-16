import warnings

from nilmtk import DataSet

from .generator import TimeSeriesDataGenerator

warnings.simplefilter(action='ignore', category=FutureWarning)


class TimeSeries:
    """
    Encapsulates UK Dale dataset handling intelligence with a focus on loading the data from a single HDF5 format file.
    Data are available for each house as follows:
        House 1: start 09/11/2012, end 26/04/2017
        House 2: start 17/02/2013, end 10/10/2013
        House 3: start 27/02/2013, end 08/04/2013
        House 4: start 09/03/2013, end 01/10/2013
        House 5: start 29/06/2014, end 13/11/2014
    """

    def __init__(self, dataset: DataSet, training_buildings: list, test_buildings: list, wandb_config):
        self.training_buildings = training_buildings
        self.test_buildings = test_buildings
        self.dataset = dataset
        self.wandb_config = wandb_config
        self.appliance = wandb_config.appliance_name
        # Calculated for UK Dale dataset and preserved
        self.normalization_params = {
            "aggregated": {
                "active": {
                    "mean": 356.242431640625,
                    "std": 481.7847900390625,
                    "max": 8442.8701,
                    "min": 0.0
                },
                "apparent": {
                    "mean": 415.4137878417969,
                    "std": 490.3323669433594,
                    "max": 11564.25,
                    "min": 1.59
                }
            },
            "appliance": {
                "fridge": {
                    "mean": 40.021522521972656,
                    "std": 52.8307991027832,
                    "min": 0.0,
                    "max": 3323.0,
                    "ON_mean": 95.50467681884766,
                    "ON_std": 38.2950439453125,
                    "ON_duration_mean": 203.8301116627745,
                    "ON_duration_std": 156.27473717257544,
                    "possible_values": [
                        [300.0, 3], [365.0, 3], [412.0, 3], [535.0, 0], [540.0, 0], [559.0, 0],
                        [971.0, 2], [1303.0, 2], [1357.0, 2], [1594.0, 1], [1698.0, 1], [1790.0, 1],
                        [2420.0, 4], [2984.0, 4], [3232.0, 4]
                    ]
                },
                "kettle": {
                    "mean": 17.524539947509766,
                    "std": 200.2118377685547,
                    "min": 0.0,
                    "max": 3998.0,
                    "ON_mean": 700.00,  # 2436.310546875,
                    "ON_std": 1000.0,  # 242.7790069580078,
                    "ON_duration_mean": 17.33459317585302,
                    "ON_duration_std": 8.689201380737993,
                    "possible_values": [
                        [305.0, 0], [351.0, 0], [485.0, 0], [602.0, 2], [625.0, 2], [926.0, 2],
                        [1321.0, 4], [1357.0, 4], [1637.0, 4], [2245.0, 1], [2286.0, 1], [2415.0, 1],
                        [2453.0, 3], [2639.0, 3], [2710.0, 3], [3.0, 1], [4.0, 1], [327.0, 1],
                        [581.0, 2], [713.0, 2], [747.0, 2], [2815.0, 4], [2862.0, 4], [2900.0, 4],
                        [2930.0, 0], [3004.0, 0], [3019.0, 0], [3862.0, 3], [3958.0, 3], [189.0, 1],
                        [319.0, 1], [326.0, 1], [1288.0, 3], [1304.0, 3], [1305.0, 3], [2004.0, 0],
                        [2096.0, 0], [2263.0, 0], [2806.0, 2], [2882.0, 2], [2901.0, 2], [3406.0, 4],
                        [3482.0, 4], [3615.0, 4], [1.0, 3], [82.0, 3], [334.0, 3], [442.0, 1],
                        [644.0, 1], [679.0, 1], [1438.0, 4], [1885.0, 4], [2177.0, 4], [2804.0, 0],
                        [2964.0, 0], [2991.0, 0], [3821.0, 2], [3839.0, 2], [3843.0, 2]
                    ]
                },
                "microwave": {
                    "mean": 11.71904468536377,
                    "std": 107.25408935546875,
                    "min": 0.0,
                    "max": 3969.0,
                    "ON_mean": 1369.419921875,
                    "ON_std": 464.2821044921875,
                    "ON_duration_mean": 38.97576396206533,
                    "ON_duration_std": 30.437855682645633,
                    "possible_values": [
                        [271.0, 0], [363.0, 0], [409.0, 0], [650.0, 2], [871.0, 2], [974.0, 2],
                        [1251.0, 4], [1341.0, 4], [1602.0, 4], [2197.0, 1], [2415.0, 1], [2423.0, 1],
                        [2555.0, 3], [2857.0, 3], [3267.0, 3], [269.0, 0], [283.0, 0], [305.0, 0],
                        [409.0, 4], [515.0, 4], [567.0, 4], [1031.0, 2], [1049.0, 2], [1469.0, 2],
                        [2043.0, 1], [2105.0, 1], [2173.0, 1], [2965.0, 3], [3279.0, 3], [3835.0, 3],
                        [38.0, 0], [41.0, 0], [178.0, 0], [396.0, 4], [580.0, 4], [640.0, 4],
                        [836.0, 2], [838.0, 2], [856.0, 2], [1280.0, 1], [1288.0, 1], [1392.0, 1],
                        [1408.0, 3], [1448.0, 3], [1459.0, 3]
                    ]

                },
                "dish washer": {
                    "mean": 27.594993591308594,
                    "std": 237.3380584716797,
                    "min": 0.0,
                    "max": 3973.0,
                    "ON_mean": 1006.519775390625,
                    "ON_std": 1061.5968017578125,
                    "ON_duration_mean": 0,
                    "ON_duration_std": 0,
                    "possible_values": [
                        [310.0, 0], [370.0, 0], [490.0, 0], [842.0, 2], [1432.0, 2], [1491.0, 2],
                        [2294.0, 4], [2370.0, 4], [2389.0, 4], [2639.0, 1], [2766.0, 1], [2883.0, 1],
                        [3343.0, 3], [3474.0, 3], [3620.0, 3], [1.0, 5], [10.0, 6]
                    ]
                }
            }
        }

    def getTrainingDataGenerator(self):
        return TimeSeriesDataGenerator(self.dataset, self.training_buildings, self.appliance, self.normalization_params,
                                       self.wandb_config, is_training=True)

    def getTestDataGenerator(self):
        return TimeSeriesDataGenerator(self.dataset, self.test_buildings, self.appliance, self.normalization_params,
                                       self.wandb_config, is_training=False)
