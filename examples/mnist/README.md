# About
This README is for the sub section of the `no_imagination` repository pertaining specifically to the code and experiments with the MNIST dataset.

# Sources & Acknowledgements

## Research Papers
* [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
* [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)
* [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

## Code
Some parts of the model architecture and code used in this section were taken from the following repositories and websites.
* https://github.com/gtoubassi/mnist-gan
* https://github.com/Zackory/Keras-MNIST-GAN (License in LICENSE-3RD-PARTY.txt)
* https://www.tensorflow.org/get_started/mnist/pros
* https://github.com/yihui-he/GAN-MNIST
* https://github.com/carpedm20/DCGAN-tensorflow (License in LICENSE-3RD-PARTY.txt)


# Notes
## v1
- The discriminator scores which are supposed to be between 0 and 1, always show scores which are very close to zero and move more close to zero as training takes place. This leads to loss being negative infinity and eventual NaN. Therefore it might be a good idea to pre-train the discriminator.

## v2
* pre-training (with fake vs real)
* Pre-training the discriminator with seperate (non mixed) real and fake mini batches.
* Separate optimizers for pretrain, generation, discrimination
* training of discriminator ONLY with labels i.e. fake vs real i.e. using opt_D_pre
* training of generator with the usual opt_G

# v3
* Conditional GAN are the best :)
* No normalization
* Different y for D and G

# v4
* Conditional GAN are the best :)
* Different y for D and G, probably similar to original GAN paper
* Basic Normalization and scaling of images between -1 and 1



# To Experiment
* Dimensionality of noise
* Pre training discriminator
* Normalizing the input images
* Batch normalization
* Dropout for generator and discriminator
* Soft labels
