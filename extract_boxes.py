import json
import cv2
import os

data_dir = '../public_dataset/'
train_dir = f'{data_dir}reference_images_part1/'
valid_dir = f'{data_dir}images_part1_valid/'
train_metadata_json = f'{data_dir}reference_images_part1.json'
valid_metadata_json = f'{data_dir}images_part1_valid.json'

new_train_dir = f'{data_dir}train/'

with open(train_metadata_json) as f:
    train_meta = json.load(f)

new_img_shape = (224, 224)

for annotation in train_meta['annotations']:
    annotation_id = annotation['id']
    bbox = annotation['bbox']
    
    # find image by id
    for img_data in train_meta['images']:
        if img_data['id'] == annotation['image_id']:
            break
    img_file = img_data['file_name']
            
    # find category by id
    for cat in train_meta['categories']:
        if cat['id'] == annotation['category_id']:
            break
    category = cat['name']
    
    img = cv2.imread(f'{train_dir}{img_file}', 1)
    bbox_img = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    
    resized = cv2.resize(bbox_img, new_img_shape)

    if not os.path.isdir(new_train_dir):
        os.mkdir(new_train_dir)
    category_dir = f'{new_train_dir}{category}/'
    if not os.path.isdir(category_dir):
        os.mkdir(category_dir)
    
    cv2.imwrite(f'{category_dir}ann{annotation_id}.png', resized)
