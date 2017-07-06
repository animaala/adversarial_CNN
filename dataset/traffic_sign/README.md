## Welcome to the Traffic Sign dataset

The data is split 9 to 1 between ```/train``` and ```/validation```. This is slightly higher than the 6 to 1 split of the MNIST dataset to reduce risk of imbalance. Later in the project I will be putting some data augmentation techniques in the model to expand the number of examples.


I set up some directories to contain training and validation data, for each of our two classes (go , stop)

```
mkdir -p traffic_sign/train/go
mkdir -p traffic_sign/train/stop
mkdir -p traffic_sign/validation/go
mkdir -p traffic_sign/validation/stop
```

### To convert images to a TensorFlow format:

In the ```/traffic_sign``` directory:

### Create a file called mylabels.txt and write to it the names of our classes:

```
go
stop
```

To convert our images to TensorFlow TFRecord format, we are going to just use the build_image_data.py 
script that is bundled with the Inception TensorFlow model.

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