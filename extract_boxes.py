import json
import cv2
import os


def preprocess_images(original_images_dir, metadata_json, preprocessed_dir, new_img_shape=(224, 224)):
    if not os.path.isdir(preprocessed_dir):
        os.mkdir(preprocessed_dir)

    with open(metadata_json) as f:
        meta_data = json.load(f)

    categories = {}
    for cat in meta_data['categories']:
        categories[cat['id']] = cat['name']

    for annotation in meta_data['annotations']:
        annotation_id = annotation['id']
        bbox = annotation['bbox']

        # find image by id
        for img_data in meta_data['images']:
            if img_data['id'] == annotation['image_id']:
                break
        img_file = img_data['file_name']

        category = categories[annotation['category_id']]

        img = cv2.imread(f'{original_images_dir}{img_file}', 1)
        bbox_img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

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

        category_dir = f'{preprocessed_dir}{category}/'
        if not os.path.isdir(category_dir):
            os.mkdir(category_dir)

        cv2.imwrite(f'{category_dir}ann{annotation_id}.png', resized)
