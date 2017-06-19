# About
This README is for the sub section of the `no_imagination` project pertaining specifically to the MNIST dataset.

# Sources & Acknowledgements
Some parts of the model architecture and code used in this section were taken from the following repositories and websites.
* https://github.com/gtoubassi/mnist-gan
* https://github.com/Zackory/Keras-MNIST-GAN (License in LICENSE-3RD-PARTY.txt)
* https://www.tensorflow.org/get_started/mnist/pros
* https://github.com/yihui-he/GAN-MNIST


# Notes
## v1
- The discriminator scores which are supposed to be between 0 and 1, always show scores which are very close to zero and move more close to zero as training takes place. This leads to loss being negative infinity and eventual NaN. Therefore it might be a good idea to pre-train the discriminator.

## v2
- Pre-training the discriminator with seperate (non mixed) real and fake mini batches.


# To Experiment
* Dimensionality of noise
* Pre training discriminator
* Normalizing the input images
* Batch normalization
* Dropout for generator and discriminator
* Soft labels
