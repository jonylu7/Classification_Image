from pathlib import Path
import requests
import tarfile

data_path=Path("data/")

url="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
name="CIFAR_100"
image_path=data_path/name
tar_path=data_path/(name+".tar.gz")





if image_path.is_dir():
    print(f"{image_path} exisits")
else:
    image_path.mkdir(parents=True,exist_ok=True)
if not(tar_path.is_file()):
    with open(tar_path,"wb") as f:
        requests=requests.get(url)
        print(f"Downloading {image_path}")
        f.write(requests.content)
        f.close()
print(tar_path)
print(list(image_path.glob('*')))
if (len(list(image_path.glob('*')))<1):
    with tarfile.open(tar_path,"r") as Tar:
        Tar.extractall(image_path)
        Tar.close()

