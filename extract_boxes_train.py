from extract_boxes import preprocess_images


data_dir = '../public_dataset/'
train_dir = f'{data_dir}reference_images_part1/'
train_metadata_json = f'{data_dir}reference_images_part1.json'
new_train_dir = f'{data_dir}train/'

preprocess_images(train_dir, train_metadata_json, new_train_dir)
