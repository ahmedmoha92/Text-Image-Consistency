import json
import os
import shutil

def prepare_coco(json_path, images_src, output_dir):
    print(f"Préparation de COCO depuis {json_path}...")
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'captions'), exist_ok=True)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    
    # MSCOCO can have multiple captions per image. We'll take the first one.
    image_captions = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_captions:
            image_captions[img_id] = ann['caption']
    
    count = 0
    for img_id, caption in image_captions.items():
        if img_id in id_to_filename:
            fname = id_to_filename[img_id]
            src_img = os.path.join(images_src, fname)
            if os.path.exists(src_img):
                # Symlink to save space and time (use relative path for Docker compatibility)
                dest_img = os.path.join(output_dir, 'images', fname)
                if not os.path.exists(dest_img):
                    rel_src = os.path.relpath(src_img, os.path.dirname(dest_img))
                    os.symlink(rel_src, dest_img)
                
                # Write caption
                cap_fname = fname.replace('.jpg', '.txt')
                with open(os.path.join(output_dir, 'captions', cap_fname), 'w', encoding='utf-8') as f:
                    f.write(caption)
                count += 1
    
    print(f"Préparation terminée. {count} paires prêtes dans {output_dir}")

if __name__ == "__main__":
    # On utilise val2017 par défaut car c'est ce qui est présent
    prepare_coco(
        'data/raw/annotations/captions_val2017.json',
        'data/raw/val2017',
        'data/raw/coco_prepared'
    )
