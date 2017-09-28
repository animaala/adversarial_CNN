## Welcome to the Traffic Sign dataset

The data is split 4 to 1 between ```/train``` and ```/validation```. This is slightly lower than the 6 to 1 split of the MNIST dataset. However data augmentation is used in the model to expand the training and test sets. Note you can follow the below instructions to create your own custom dataset for use with Tensorflow Convolutional Neural Networks.

### To create a dataset for Tensorflow to use:

1. Set up some directories to contain training and validation data, for each of our two classes (go , stop)

```
mkdir -p traffic_sign/train/go
mkdir -p traffic_sign/train/stop
mkdir -p traffic_sign/validation/go
mkdir -p traffic_sign/validation/stop
```

2. In the ```/dataset/traffic_sign/``` directory create a file called mylabels.txt and write to it the names of our classes:

```
go
stop
```

3. To convert our images to TensorFlow TFRecord format, use the ```build_image_data.py``` script located in the ```/dataset/traffic_sign/``` directory. This script is bundled with the *Inception* TensorFlow model so is official Google.

We can just use this a “black box” to convert our data (but we get some insight as to what it is doing later 
when we read the data within TensorFlow). Run the following command

```
python build_image_data.py --train_directory=./train --output_directory=./  \
--validation_directory=./validation --labels_file=mylabels.txt   \
--train_shards=1 --validation_shards=1 --num_threads=1
```

We have told the script where to find the input files, and labels, and it will create a file containing 
all training images train-00000-of-00001 and another containing all validation images 
validation-00000-of-00001 in TensorFlow TFRecord format. We can now use these to train and validate our model.

