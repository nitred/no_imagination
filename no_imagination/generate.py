import sys

sys.path.append("/home/nitred/projects/no_imagination/examples/mnist/GAN-tensorflow/")
sys.path.append("/home/nitred/projects/no_imagination/examples/text2image/text-to-image/")

from gan import GanTest as GenerateMnist
from train_txt2im_test import Text2Image as GenerateFlowers
