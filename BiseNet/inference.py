import yaml

from model import BiSeNet

class MaskModel:
    def __init__(self):
        f = open("config.yaml", 'r')
        self.cfg = yaml.safe_load(f)["test"]
        f.close()

        self.model = BiSeNet