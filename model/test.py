import tensorflow as tf
import trafficsign_input as input
import os, random


some_file = random.choice(os.listdir("../../dataset/traffic_sign/train/stop/"))


print(input.DATA_PATH + "train/stop/" + some_file)