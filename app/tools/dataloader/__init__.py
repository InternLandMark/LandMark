from .city_dataset import CityDataset
from .matrixcity_dataset import MatrixCityDataset

dataset_dict = {
    "city": CityDataset,
    "matrixcity": MatrixCityDataset,
}
