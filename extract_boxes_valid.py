import json
import cv2
import os


data_dir = '../public_dataset/'
valid_dir = f'{data_dir}images_part1_valid/'
valid_metadata_json = f'{data_dir}images_part1_valid.json'

new_valid_dir = f'{data_dir}valid/'
if not os.path.isdir(new_valid_dir):
    os.mkdir(new_valid_dir)

with open(valid_metadata_json) as f:
    valid_meta = json.load(f)

new_img_shape = (224, 224)

for annotation in valid_meta['annotations']:
    annotation_id = annotation['id']
    bbox = [int(a) for a in annotation['bbox']]
    
    # find image by id
    for img_data in valid_meta['images']:
        if img_data['id'] == annotation['image_id']:
            break
    img_file = img_data['file_name']
            
    # find category by id
    for cat in valid_meta['categories']:
        if cat['id'] == annotation['category_id']:
            break
    category = cat['name']
    
    img = cv2.imread(f'{valid_dir}{img_file}', 1)
    bbox_img = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    
    resized = cv2.resize(bbox_img, new_img_shape)

    category_dir = f'{new_valid_dir}{category}/'
    if not os.path.isdir(category_dir):
        os.mkdir(category_dir)
    
    cv2.imwrite(f'{category_dir}ann{annotation_id}.png', resized)
