import gdown

if __name__ == "__main__":
    folder_url = "https://drive.google.com/drive/u/3/folders/1rGblqA0Wh0vhDFrjasMGvOYpyDiD3jVW"
    gdown.download_folder(id=folder_url, output="./", quiet=True)
