## Welcome to the Traffic Sign model directory.

### Running the model:

1. Run ```trafficsign_adversarial.py``` either via the command line or in an IDE. 
2. The ```/inference_layers``` directory contains 3 files. Code for 1, 2 and 3 convolution layers. Just copy the code and paste it into ```trafficsign_model.py``` in the appropriate place within the ```inference``` function. You can then run ```trafficsign_adversarial.py``` as normal.
3. To use Tensorboard; on the command line navigate to the ```/model``` directory and then type:

```
tensorboard --logdir ./graph/
```

4. Then open a browser and type ```localhost:6006``` (there are a number of log files so it might take a minute to load them all).

5. The ```test.py``` was used for testing code (shock!!) so can be ignored.

### Good luck!
