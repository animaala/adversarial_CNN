# Adversarial machine learning

Exploring the counter intuitive features of deep learning known as adversarial examples. In this project we use a custom dataset of traffic signs which has an inference on self driving cars. The dataset consists of 2 classes "Stop" and "Go" ("Stop", "30mph").

### Prerequisites

This project uses TensorFlow version 1.1.0. Follow the specific instructions on the link below for your machine (GPU support available for specific NVIDIA chipsets, if using a GPU you'll need NVIDIA CuDNN 5.1). Python 3.5 is also required along with various other modules including matplotlib for some of the tutorials. You should be guided through any missing modules when trying to run the code. It's important you use the correct version of TF and CuDNN !

[Installation][tf_installation]

### Getting Started

```git clone git@gitlab.ncl.ac.uk:securitylab/adversarial_ML.git```

There are various example deep learning models in ```/tutorials/```. The various directories contain traditional .py and also .ipynb files for use with jupyter notebooks. These models use various datasets such as MNIST and cifar-10. They go from a very simple softmax classification with 92% accuracy on the MNIST data set, and later introduces ReLU activation, batch normalisation, dropout and convolutional neural networks. Eventually achieving an accuracy of 99.5% on MNIST with very good visualisations using the matplotlib module. The tutorials should work for the majority, however a couple may need tampering with to get them to work (due to different directory structures etc.)

### Recommendations

If you don't understand how Neural Networks and **especially** Convolutional Neural Networks work, then you need to learn that first to understand the code in this repo. Check out the wiki above which has a page explaining the basics and has links at the bottom to two very good YouTube videos which I highly recommend you watch.

### Check the wiki for more documentation and background resources!


[tf_installation]: https://www.tensorflow.org/install/
