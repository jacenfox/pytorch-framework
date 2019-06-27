# pytorch-framework
Train/Test pytorch network.

You may only need two scripts from this repo: **deepnet_training.py**, **deepnet_test.py**.
The rest are used as an example and hope to make this repo reusable. No special dependencies are required.


## Usage
- The train/test code are defined in *train.py* and *test.py*, they are project dependent. You would like to rewrite part of them to meet your requirements.

- Train loop is defined in *deepnet_training.py*, You can keep them as it is and just simply copy it to different projects.

- In order to load the pre-trained model, see the code in *deepnet_test.py*. Most of time, I simply copy it to the project folder.

## Details
- *net.py*, the network structure can be defined here, or you can create a new file to define different modules.
- *train.py* train the network. The datasets, network to use can be declared here. It will write the output to the *--output* folder.
- *test.py* loads the pre-trained network, do the forward pass, use some callbacks to make the code clean.
- *dataset.py* defined the pytorch dataset for Dataloader. You can define different datasets and combine them together.
- *callbacks_training.py*, after each minibatch/epoch the callback will be called. This is very practical that you can manipulate/visualize the network input/output without change other scripts. Do the same in a *callbacks_test.py*. Three functions are pre-defined:
	- plot loss curves, in the mean while save them as *.csv* file for further processing.
	- plot gradient curves, the gradient hook layer is defined in *net.py*.
	- visualize some samples during training.
- *[torchsummary.py](https://github.com/sksq96/pytorch-summary)*, updated to show the size of the feature map.
- *deepnet_train/test.py* is used to make the training loops or test forwards happen. Most boring code in many pytorch projects, just copy/paste them to the new project. Tested in pytorch 0.4&1.0.

## TODO
Clean the dirty code. Update GradientHook
Gradient Record won't work for pytroch>=1.0, please remove these parts. It's easy to be fixed, you can choose to use gradient_hooks.

## Acknowledgment
Thanks to MathGaron https://github.com/MathGaron/pytorch_toolbox
