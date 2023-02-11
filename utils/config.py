import yaml


class Config:
    def __init__(self, config_file='config.yml'):
        self.config_file = config_file
        with open(config_file, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def get_dataset_path(self, dataset: str = 'default'):
        datasets = self.config['datasets']
        if dataset not in datasets.keys():
            return datasets['default']
        return datasets[dataset]

    def write_dataset_path(self, dataset: str, path: str):
        self.config['datasets'][dataset] = path
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)
