import yaml

from src import Implementation

if __name__ == "__main__":
    f = open("config.yaml", 'r')
    config = yaml.safe_load(f)
    f.close()

    i = Implementation(config)
    print("Ready!")
    while True:
        img_path = input()
        i.generate(img_path)
