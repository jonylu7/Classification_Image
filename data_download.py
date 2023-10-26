from pathlib import Path
import requests
import tarfile

def downloadFile(url:str,image_path:Path,tar_path:Path):
    if image_path.is_dir():
        print(f"{image_path} exisits")
    else:
        image_path.mkdir(parents=True,exist_ok=True)
    if not(tar_path.is_file()):
        with open(tar_path,"wb") as f:
            request=requests.get(url)
            print(f"Downloading {image_path}")
            f.write(request.content)
            f.close()
    print(tar_path)
    print(list(image_path.glob('*')))
    if (len(list(image_path.glob('*')))<1):
        with tarfile.open(tar_path,"r") as Tar:
            Tar.extractall(image_path)
            Tar.close()




if __name__=="__main__":
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    name = "CIFAR_100"
    image_path =Path("data/") / name
    tar_path = Path("data/") / (name + ".tar.gz")
    downloadFile(url,image_path,tar_path)


