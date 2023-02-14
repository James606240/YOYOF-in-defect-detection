import os
import shutil
import cv2
import glob as glob

ROOT_DATA = os.path.join('input', 'data_root')
ROOT_IMAGE_PREFIX = os.path.join(ROOT_DATA, 'dataset')
ANNOT_FOLDER = os.path.join(ROOT_IMAGE_PREFIX, 'Annotations')
TEXT_FILE_IMAGE_SETS = os.path.join(ROOT_IMAGE_PREFIX, 'ImageSets', 'Main')
IMAGE_FOLDER = os.path.join(ROOT_IMAGE_PREFIX, 'JPEGImages')

os.makedirs(ROOT_DATA, exist_ok=True)
os.makedirs(ROOT_IMAGE_PREFIX, exist_ok=True)
os.makedirs(ANNOT_FOLDER, exist_ok=True)
os.makedirs(TEXT_FILE_IMAGE_SETS, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

TRAIN_IMAGES_PATH = os.path.join(
    'input', 
    'Aquarium Combined.v2-raw-1024.voc', 
    'train'
)
TRAIN_XML_PATH = os.path.join(
    'input', 
    'Aquarium Combined.v2-raw-1024.voc', 
    'train'
)
VALID_IMAGES_PATH = os.path.join(
    'input', 
    'Aquarium Combined.v2-raw-1024.voc', 
    'valid'
)
VALID_XML_PATH = os.path.join(
    'input', 
    'Aquarium Combined.v2-raw-1024.voc', 
    'valid'
)

# Give split as 'train' or 'val', or 'test'.
def create_text_file_sets(orig_image_folder, save_folder, split='train'):
    all_images = glob.glob(os.path.join(orig_image_folder, '*.jpg'))
    with open(os.path.join(save_folder, split+'.txt'), 'w') as f:
        for i, image_name in enumerate(all_images):
            file_name = image_name.split('.jpg')[0].split(os.path.sep)[-1]
            f.writelines(file_name+'\n')

# Copy the XML files to `Annotations` folder.
def xml_to_annotations_folder(
    orig_xml_train_folder, orig_xml_valid_folder, save_folder
):
    train_xmls = glob.glob(os.path.join(orig_xml_train_folder, '*.xml'))
    for train_xml in train_xmls:
        file_name = train_xml.split('.xml')[0].split(os.path.sep)[-1]
        shutil.copy(
            train_xml,
            os.path.join(save_folder, file_name+'.xml')
        )

    valid_xmls = glob.glob(os.path.join(orig_xml_valid_folder, '*.xml'))
    for valid_xml in valid_xmls:
        file_name = valid_xml.split('.xml')[0].split(os.path.sep)[-1]
        shutil.copy(
            valid_xml,
            os.path.join(save_folder, file_name+'.xml')
        )

# Copy the images to the `JPEGImages` folder.
def images_to_jpegimages_folder(
    orig_image_train_folder, orig_image_valid_folder, save_folder
):
    train_images = glob.glob(os.path.join(orig_image_train_folder, '*.jpg'))
    for train_image in train_images:
        file_name = train_image.split('.jpg')[0].split(os.path.sep)[-1]
        image = cv2.imread(os.path.join(train_image))
        cv2.imwrite(os.path.join(save_folder, file_name+'.jpg'), image)

    valid_images = glob.glob(os.path.join(orig_image_valid_folder, '*.jpg'))
    for valid_image in valid_images:
        file_name = valid_image.split('.jpg')[0].split(os.path.sep)[-1]
        image = cv2.imread(os.path.join(valid_image))
        cv2.imwrite(os.path.join(save_folder, file_name+'.jpg'), image)

# Create train image set text file.
create_text_file_sets(TRAIN_IMAGES_PATH, TEXT_FILE_IMAGE_SETS, 'train')

# Create validation image set text file.
create_text_file_sets(VALID_IMAGES_PATH, TEXT_FILE_IMAGE_SETS, 'val')

# Create XML files in annotations folder.
xml_to_annotations_folder(
    TRAIN_XML_PATH, VALID_XML_PATH, save_folder=ANNOT_FOLDER
)

# Create JPEG images in JPEGImages folder.
images_to_jpegimages_folder(
    TRAIN_IMAGES_PATH, VALID_IMAGES_PATH, IMAGE_FOLDER
)