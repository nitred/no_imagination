"""Download Datasets.

Source: https://github.com/paarthneekhara/text-to-image

File has been edited from the original version.
"""
import errno
import os
import sys

from no_imagination.utils import get_project_directory

if sys.version_info >= (3,):
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve


# http://stackoverflow.com/questions/273192/how-to-check-if-a-directory-exists-and-create-it-if-necessary
def make_sure_path_exists(path):
    """."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


# adapted from http://stackoverflow.com/questions/51212/how-to-write-a-download-progress-indicator-in-python
def dl_progress_hook(count, blockSize, totalSize):
    """."""
    percent = int(count * blockSize * 100 / totalSize)
    sys.stdout.write("\r" + "...%d%%" % percent)
    sys.stdout.flush()


def download_dataset(data_name):
    """."""
    if data_name == 'skipthoughts':
        print('== Skipthoughts models ==')
        SKIPTHOUGHTS_DIR = get_project_directory('skipthoughts', 'models')
        SKIPTHOUGHTS_BASE_URL = 'http://www.cs.toronto.edu/~rkiros/models/'
        make_sure_path_exists(SKIPTHOUGHTS_DIR)

        # following https://github.com/ryankiros/skip-thoughts#getting-started
        skipthoughts_files = [
            'dictionary.txt', 'utable.npy', 'btable.npy', 'uni_skip.npz', 'uni_skip.npz.pkl', 'bi_skip.npz',
            'bi_skip.npz.pkl',
        ]
        for filename in skipthoughts_files:
            src_url = SKIPTHOUGHTS_BASE_URL + filename
            print('Downloading ' + src_url)
            urlretrieve(src_url, os.path.join(SKIPTHOUGHTS_DIR, filename),
                        reporthook=dl_progress_hook)
    else:
        raise ValueError('Unknown dataset name: ' + data_name)


def main():
    """."""
    # TODO: make configurable via command-line
    download_dataset('skipthoughts')
    print('Done')


if __name__ == '__main__':
    main()
