import argparse
import os
import random
import shutil
from glob import glob

def generate_pairs(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    # Attend une structure : source_dir/images/*.jpg et source_dir/captions/*.txt
    image_paths = glob(os.path.join(source_dir, 'images', '*.jpg'))
    if not image_paths:
        print(f"Aucune image trouvée dans {os.path.join(source_dir, 'images')}")
        return

    captions = {}
    caption_files = glob(os.path.join(source_dir, 'captions', '*.txt'))
    for cap_file in caption_files:
        with open(cap_file, 'r', encoding='utf-8') as f:
            captions[os.path.basename(cap_file).replace('.txt','')] = f.read()
    
    coherent_pairs = []
    for img in image_paths:
        base_name = os.path.basename(img).replace('.jpg','')
        if base_name in captions:
            coherent_pairs.append((img, captions[base_name]))
    
    if not coherent_pairs:
        print("Aucune paire cohérente trouvée.")
        return

    # Génération incohérente par mélange (en évitant la propre légende)
    incoherent_pairs = []
    all_caption_values = list(captions.values())
    for img, own_caption in coherent_pairs:
        candidates = [c for c in all_caption_values if c != own_caption]
        if not candidates:
            candidates = all_caption_values
        other_cap = random.choice(candidates)
        incoherent_pairs.append((img, other_cap))
    
    all_pairs = [(p, 1) for p in coherent_pairs] + [(p, 0) for p in incoherent_pairs]
    random.shuffle(all_pairs)
    
    n = len(all_pairs)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    splits = {
        'train': all_pairs[:train_end],
        'validation': all_pairs[train_end:val_end],
        'test': all_pairs[val_end:]
    }
    
    for split_name, split_data in splits.items():
        for (img_path, caption), label in split_data:
            label_dir = 'coherent' if label == 1 else 'incoherent'
            dest_dir = os.path.join(output_dir, split_name, label_dir)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(img_path, os.path.join(dest_dir, os.path.basename(img_path)))
            with open(os.path.join(dest_dir, os.path.basename(img_path).replace('.jpg','.txt')), 'w', encoding='utf-8') as f:
                f.write(caption)
    print(f"Génération terminée dans {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    generate_pairs(args.source, args.output)
