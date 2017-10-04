"""."""
import json
import os
import pickle

from PIL import Image

import numpy as np
import skimage
import skimage.io
from no_imagination.utils import get_project_directory
from resizeimage import resizeimage


def read_json_file(json_path):
    """Function to read json file."""
    with open(json_path, 'r') as f:
        json_obj = json.loads(f.read())
    return json_obj


def get_ignore_ids(ignore_ids_path):
    """."""
    print("Get Ignore Ids", ignore_ids_path)
    if os.path.isfile(ignore_ids_path):
        # Read
        with open(ignore_ids_path, 'rb') as f:
            ignore_ids = pickle.load(f)

        # Corner case
        if not isinstance(ignore_ids, list):
            ignore_ids = []
    else:
        ignore_ids = []

    print("Ignore Ids: {}".format(len(ignore_ids)))
    return ignore_ids


def update_ignore_ids(ignore_ids_path, ignore_ids):
    """."""
    print("Updating Ignore Ids", ignore_ids_path)
    if os.path.isfile(ignore_ids_path):
        # Read
        with open(ignore_ids_path, 'rb') as f:
            orig_ignore_ids = pickle.load(f)
            print("Original Ignore Ids: {}".format(len(orig_ignore_ids)))

        # Update
        if isinstance(orig_ignore_ids, list):
            new_ignore_ids = orig_ignore_ids + ignore_ids
        else:
            # Corner case
            new_ignore_ids = ignore_ids
    else:
        # Update
        new_ignore_ids = ignore_ids

    # Write
    with open(ignore_ids_path, 'wb') as f:
        pickle.dump(new_ignore_ids, f)
        print("New Ignore Ids: {}".format(len(new_ignore_ids)))


def get_image_id_to_image_caption_category_dict_from_sub_dir(sub_dir):
    """Return dict object mapping image id to image details."""
    images_dir = os.path.join(sub_dir, 'images')
    captions_json_path = os.path.join(sub_dir, 'captions.json')
    stuff_json_path = os.path.join(sub_dir, 'stuff.json')
    ignore_ids_path = os.path.join(sub_dir, 'ignore_ids.pkl')

    captions_json = read_json_file(captions_json_path)
    stuff_json = read_json_file(stuff_json_path)

    # Get Image Dict from captions json
    images_dict = {image['id']: image for image in captions_json['images']}

    # Update Image Dict:
    for key in images_dict.keys():
        # Update with empty captions list
        images_dict[key]['captions'] = []
        # Update with empty category list
        images_dict[key]['categories'] = []
        # Full path to image
        images_dict[key]['path'] = os.path.join(images_dir, images_dict[key]['file_name'])
        images_dict[key]['path_128'] = os.path.join(images_dir, "128_" + images_dict[key]['file_name'])
        images_dict[key]['path_224'] = os.path.join(images_dir, "224_" + images_dict[key]['file_name'])

    # Update Image Dict:
    for anno in captions_json['annotations']:
        # Update the captions list from annotations
        images_dict[anno['image_id']]['captions'].append(anno['caption'])

    # Update Image Dict with categories
    # images_dict['categories'] = stuff_json['categories']

    # Update Image Dict:
    for stuff_anno in stuff_json['annotations']:
        # Update the categories list from stuff-annotations
        images_dict[stuff_anno['image_id']]['categories'].append(stuff_anno['category_id'])

    # Create Categories Dict:
    categories_dict = {stuff_cat['id']: stuff_cat for stuff_cat in stuff_json['categories']}

#     # Update Categories Dict:
#     sorted_categories_ids = sorted(categories_dict.keys())
#     sorted_categories_dict = {}
#     for sorted_id, category_id in enumerate(sorted_categories_ids):
#         sorted_categories_dict[sorted_id] = categories_dict[category_id]
#         sorted_categories_dict[sorted_id]['sorted_id'] = sorted_id

    # Remove ignore ids
    ignore_ids = get_ignore_ids(ignore_ids_path)
    for ignore_id in ignore_ids:
        images_dict.pop(ignore_id, None)

    return images_dict, categories_dict


def generate_square_images(subset, images_dict, pixels):
    """."""
    ignore_ids = np.empty(shape=[len(images_dict.keys())], dtype=int)
    ignore_index = 0
    for i, key in enumerate(images_dict.keys()):
        image_dict = images_dict[key]
        try:
            pass
            with open(image_dict['path'], 'r+b') as f:
                with Image.open(f) as image:
                    cover = resizeimage.resize_cover(image, [pixels, pixels, 3])
                    cover.save(image_dict['path_{}'.format(pixels)], image.format)
        except resizeimage.ImageSizeError:
            # print("{},".format(image_dict['id']))
            ignore_ids[ignore_index] = image_dict['id']
            ignore_index += 1
        except FileNotFoundError:
            # print("{},".format(image_dict['id']))
            ignore_ids[ignore_index] = image_dict['id']
            ignore_index += 1

        if i % 1000 == 0:
            print("Iter: {}, Ignored: {}".format(i, ignore_index))

    ignore_ids = list(ignore_ids[:ignore_index])

    if ignore_ids:
        subset_dir = get_project_directory('mscoco', 'dataset', subset)
        ignore_ids_path = os.path.join(subset_dir, 'ignore_ids.pkl')
        update_ignore_ids(ignore_ids_path, ignore_ids)


def get_images_and_categories_dict(subset):
    """.

    Args:
        subset (str): 'train' or 'test' or 'val'.
    """
    mscoco_sub_dir = get_project_directory('mscoco', 'dataset', subset)
    images_dict, categories_dict = get_image_id_to_image_caption_category_dict_from_sub_dir(mscoco_sub_dir)

    return images_dict, categories_dict


def get_n_random_categories(categories_dict, n_categories):
    """Return random ."""
    np.random.seed(seed=11111)
    random_slice = list(np.random.permutation(len(categories_dict.keys()))[:n_categories])
    random_n_categories = list(np.array(sorted(categories_dict.keys()))[random_slice])
    return random_n_categories


def get_filtered_image_dict_from_categories(images_dict, categories):
    """."""
    categories = sorted(categories)
    filtered_images_dict = {}

    for key in images_dict.keys():
        image_dict = images_dict[key]
        # Get one intersection
        intersection_category = sorted(set(categories).intersection(set(image_dict['categories'])))[:1]
        if intersection_category:
            intersection_category = intersection_category[0]
            filtered_images_dict[key] = image_dict
            filtered_images_dict[key]['chosen_category_id'] = intersection_category
            # Label is the index of the category in the list of categories
            chosen_category_label = categories.index(intersection_category)
            filtered_images_dict[key]['chosen_category_label'] = chosen_category_label

    return filtered_images_dict


def get_filtered_mage_dict_for_subset_for_n_categories(subset, n_categories=10, categories=[]):
    """."""
    images_dict, categories_dict = get_images_and_categories_dict(subset)
    if not categories:
        print("Filtering using n_categories: {}".format(n_categories))
        random_n_categories = get_n_random_categories(categories_dict, n_categories)
    else:
        print("Filtering using categories  : {}".format(len(categories)))
        random_n_categories = categories
    filtered_images_dict = get_filtered_image_dict_from_categories(images_dict, random_n_categories)
    return filtered_images_dict, categories_dict


def load_image(path):
    """."""
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    # short_edge = min(img.shape[:2])
    # yy = int((img.shape[0] - short_edge) / 2)
    # xx = int((img.shape[1] - short_edge) / 2)
    # crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # # resize to 224, 224
    # resized_img = skimage.transform.resize(crop_img, (224, 224))
    return img


def add_faulty_images_to_ignore(subset):
    """."""
    images_dict, categories_dict = get_images_and_categories_dict(subset)
    subset_dir = get_project_directory('mscoco', 'dataset', subset)
    ignore_ids_path = os.path.join(subset_dir, "ignore_ids.pkl")
    ignore_ids_ndim = []
    ignore_ids_exception = []
    for i, key in enumerate(images_dict.keys()):
        image_dict = images_dict[key]
        try:
            img = load_image(image_dict['path_224'])
            if not img.ndim == 3:
                ignore_ids_ndim.append(image_dict['id'])
        except Exception as ex:
            ignore_ids_exception.append(image_dict['id'])

        if i % 1000 == 0:
            print("Iter: {}, Ignored: {}".format(i, len(ignore_ids_ndim + ignore_ids_exception)))

    ignore_ids = ignore_ids_ndim + ignore_ids_exception
    if ignore_ids:
        update_ignore_ids(ignore_ids_path, ignore_ids)


def one_hot_encoding(arr, n_categories):
    """."""
    n_rows = len(arr)
    one_hot = np.zeros(shape=[n_rows, n_categories], dtype=int)
    # Magic.
    one_hot[np.arange(n_rows), arr.astype(int)] = 1
    return one_hot


class mscoco_generator(object):
    """."""

    def __init__(self, subset='val', n_categories=10, categories=[], batch_size=32):
        """."""
        self.subset = subset
        self.n_categories = n_categories
        self.batch_size = batch_size
        self.categories = sorted(categories)

        if self.categories:
            self.n_categories = len(self.categories)

        self.images_dict, self.categories_dict = get_filtered_mage_dict_for_subset_for_n_categories(self.subset,
                                                                                                    self.n_categories,
                                                                                                    self.categories)

        self.batch_generator = self.__generate_batches()

    def __generate_batches(self):
        """."""
        start_index = 0
        images_keys = np.array(sorted(self.images_dict.keys()))

        # shuffle deterministically
        if self.subset == 'val':
            np.random.seed(seed=11111)
        elif self.subset == 'train':
            np.random.seed(seed=22222)
        else:
            np.random.seed(seed=33333)
        np.random.shuffle(images_keys)

        while True:
            batch_indices = range(start_index, start_index + self.batch_size, 1)
            batch_keys = images_keys.take(batch_indices, mode='wrap')
            start_index += self.batch_size

            batch_x = np.empty([self.batch_size, 224, 224, 3])
            batch_y = np.empty([self.batch_size, 1])

            # Load batch
            for i, image_key in enumerate(batch_keys):
                batch_x[i] = load_image(self.images_dict[image_key]['path_224'])
                batch_y[i] = float(self.images_dict[image_key]['chosen_category_label'])

            batch_y_one_hot = one_hot_encoding(batch_y.ravel(), self.n_categories)

            # Yield batch
            yield batch_x, batch_y_one_hot

    def next_batch(self):
        """."""
        return next(self.batch_generator)

    def test_batch(self, batch_size):
        """."""
        temp_batch_size = self.batch_size
        self.batch_size = batch_size
        self.reset_generator()
        test_batch = self.next_batch()
        self.batch_size = temp_batch_size
        self.reset_generator()
        return test_batch

    def reset_generator(self):
        """."""
        self.batch_generator = self.__generate_batches()

    def get_category_mapping(self):
        """."""
        category_labels = list(range(self.n_categories))
        category_names = []
        for category_label in category_labels:
            for key in self.images_dict.keys():
                if category_label == self.images_dict[key]['chosen_category_label']:
                    category_id = self.images_dict[key]['chosen_category_id']
                    category_name = self.categories_dict[category_id]['name']
                    super_category_name = self.categories_dict[category_id]['supercategory']
                    category_names.append([category_label, category_id, category_name, super_category_name])
                    break
        return category_names

    def get_category_ids(self):
        """."""
        category_labels = list(range(self.n_categories))
        category_ids = []
        for category_label in category_labels:
            for key in self.images_dict.keys():
                if category_label == self.images_dict[key]['chosen_category_label']:
                    category_id = self.images_dict[key]['chosen_category_id']
                    category_ids.append(category_id)
                    break
        return sorted(category_ids)
