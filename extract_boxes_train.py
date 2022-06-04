import json
import cv2
import os


data_dir = '../public_dataset/'
train_dir = f'{data_dir}reference_images_part1/'
train_metadata_json = f'{data_dir}reference_images_part1.json'

new_train_dir = f'{data_dir}train/'
if not os.path.isdir(new_train_dir):
    os.mkdir(new_train_dir)

with open(train_metadata_json) as f:
    train_meta = json.load(f)

categories = {}
for cat in train_meta['categories']:
    categories[cat['id']] = cat['name']

new_img_shape = (224, 224)

for annotation in train_meta['annotations']:
    annotation_id = annotation['id']
    bbox = annotation['bbox']
    
    # find image by id
    for img_data in train_meta['images']:
        if img_data['id'] == annotation['image_id']:
            break
    img_file = img_data['file_name']

    category = categories[annotation['category_id']]
    
    img = cv2.imread(f'{train_dir}{img_file}', 1)
    bbox_img = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

    max_dim_size = max(bbox[2], bbox[3])
    x_dim_shortage, y_dim_shortage = max_dim_size - bbox[2], max_dim_size - bbox[3]
    img_square = cv2.copyMakeBorder(
        bbox_img,
        top=int(y_dim_shortage / 2),
        bottom=int(y_dim_shortage / 2),
        left=int(x_dim_shortage / 2),
        right=int(x_dim_shortage / 2),
        borderType=cv2.BORDER_CONSTANT,
        value=[82, 82, 82]
    )
    
    resized = cv2.resize(img_square, new_img_shape)

    category_dir = f'{new_train_dir}{category}/'
    if not os.path.isdir(category_dir):
        os.mkdir(category_dir)
    
    cv2.imwrite(f'{category_dir}ann{annotation_id}.png', resized)
