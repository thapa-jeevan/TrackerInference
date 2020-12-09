from tracker.configs.settings import DETECTOR_ARCHITECTURE

# SSD ssd300
RATIOS = [[2, 3], [2, 3], [2, 3], [2], [2]]
SCALES = [0.2, 0.375, 0.55, 0.725, 0.9, 1.075]
FM_SIZES = [19, 10, 5, 3, 1]

# SSD ssd512
if DETECTOR_ARCHITECTURE == "ssd512":
    RATIOS = [[2, 3], [2, 3], [2, 3], [2], [2], [2]]
    SCALES = [0.07, 0.15, 0.34, 0.53, 0.71, 0.9, 1.0875]
    FM_SIZES = [32, 16, 8, 4, 2, 1]
