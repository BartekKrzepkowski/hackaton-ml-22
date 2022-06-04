from extract_boxes import preprocess_images


data_dir = '../public_dataset/'
valid_dir = f'{data_dir}images_part1_valid/'
valid_metadata_json = f'{data_dir}images_part1_valid.json'
new_valid_dir = f'{data_dir}valid/'

preprocess_images(valid_dir, valid_metadata_json, new_valid_dir)
