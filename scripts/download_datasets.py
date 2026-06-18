import argparse
import requests
import os
import zipfile
import tarfile

def download_file(url, dest):
    print(f"Téléchargement de {url} vers {dest}...")
    response = requests.get(url, stream=True)
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['flickr30k', 'mscoco', 'newsclippings'])
    args = parser.parse_args()
    
    os.makedirs('data/raw', exist_ok=True)
    
    if args.dataset == 'flickr30k':
        # NOTE: Flickr30k nécessite généralement une demande d'accès
        url = "https://github.com/BryanPlummer/flickr30k_entities/archive/refs/heads/master.zip" # Exemple partiel
        dest = "data/raw/flickr30k.zip"
        download_file(url, dest)
        with zipfile.ZipFile(dest, 'r') as zipf:
            zipf.extractall('data/raw/')
    elif args.dataset == 'mscoco':
        urls = [
            "http://images.cocodataset.org/zips/val2017.zip",
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        ]
        for url in urls:
            fname = url.split('/')[-1]
            dest = f"data/raw/{fname}"
            download_file(url, dest)
            with zipfile.ZipFile(dest, 'r') as zipf:
                zipf.extractall('data/raw/')

if __name__ == '__main__':
    main()
