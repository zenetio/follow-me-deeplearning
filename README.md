# Deep Learning Project - Follow me #
Author: Carlos R. Lacerda

1. [Introduction](#intro)
2. [Hardware](#hard)
3. [Software](#soft)
4. [Hyperparameters](#hyper)
5. [Model](#model)
6. [Layers](#layers)
7. [Putting all together](#all)
8. [Prediction](#pred)
9. [Future enhancements](#enhance)

## Introduction <a id='intro'></a>
The target of this project is to train a deep neural network to identify and track a target in simulation. So-called "follow me" applications, like this one, are key to many fields of robotics. The very same techniques applied here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

[image_0]: ./docs/misc/sim_screenshot.png
![alt text][image_0]

### Check the repository ### 
You can check the entire project located in the [git repository.](https://github.com/zenetio/follow-me-deeplearning)

### Check the video ###

[The video](https://youtu.be/FuXLh1z44dU)

## Hardware <a id='hard'></a>
Train a model for deep learning may take hours or even months due to heavy computational tasks. So, get access to a fast machine is very important in this scenario. You can make use of available services like AWS, Google Cloud and others. In this project I used the following hardware.

* Memory: 16 Gb
* CPU: Intel Core i5 7500
* Graphic: NVIDIA GeForce GTX 1080 (8Gb)

Here is an image of GPU while training the model.

[image_1]: ./docs/misc/gpu_use.png
![alt text][image_1]

## Software <a id='soft'></a>

For this project I used the following software.

* Windows 10 
* TensorFlow 1.3.0
* Keras 2.0.5
* Python 3.6.0
* CNTK 2.0
* NVIDIA CUDA 8.0
* cuDNN 6.0

## Hyperparameters <a id='hyper'></a>

Hyperparameters are the variables which determines the network structure and how the network is trained. In this project I worked with the following hyparameters:

* `learning rate.` Define how quickly a network updates its parameters.
* `number of epochs.` Number of times the whole training data is shown to the network while training.
* `steps per epoch.` Number of batches of training images that go through the network in 1 epoch.
* `batch size.` The number of training samples/images that get propagated through the network in a single pass.
* `number of hidden layers.` Hidden layer is a layer thas has no connection with external world.
* `choice of activation function.` Activation function of a node defines the output of that node given an input or set of inputs.
* `workers.` Maximum number of processes to spin up the training speed.

There is not a role of thumb to setup hyperparameters in neural network. So, to find the best parameters that fit the model to dataset I, randomly, choosed some values and for each run I just got the results, in this case the ploted loss values and the accuracy, and did check for overfitting, underfitting, noises and the general behavior of loss graphic.

[image_2]: ./docs/misc/param.png
![alt text][image_2]
Note that the best accuracy was achieved in run 36, showed as yellow line.
I tried two types of optimizers: Adam and Nadam. With Nadam I could improve the results.

Another parameter that caused impact in results was the `steps per epochs` and `validation steps.` Using aumumentation to change the number of samples, these values are also impacted as you can see in the following snippet code used in the notebook.

```
steps_per_epoch = np.int(((len(os.listdir(os.path.join('..', 'data', 'train', 'images')))/batch_size) + 1))
validation_steps = np.int((len(os.listdir(os.path.join('..', 'data', 'validation', 'images')))/batch_size)+ 1))
```
The code checks the number of samples in train/validation directories and calculates the two parameters used in the model. 

## The model architecture for Fully Convolutional Network (FCN) <a id='model'></a>

In this project our model will requires the downsampling of an image between convolutional and ReLU layers, and then upsample the output to match the input size. However, during this process the network performs the operations using non-linear filters optimized for our 3 classes.

## Model Layers <a id='layers'></a>

In this project we use Fully Convolutional Network (FCN) because we want to produce `semantic segmentation` that classifies every pixel in an image. In this case, we want to say where is the target in the image.

[image_3]: ./docs/misc/FCN.png
![alt text][image_3]
As we can see in the image, FCN is comprised of encoder and decoder. After some tests I concluded that 6 layers was enough to reach the expected prediction for this project.

So, to facilitate the understanding, I will call encoder a set of functions that will result in downsample the input and decoder as a set of functions that will upsample the input.

[image_4]: ./docs/misc/updn.PNG
![Fig. 4- Downsampling and Updampling][image_4]

So, why encode and decode the image? There is a good explanation [here](https://www.youtube.com/watch?v=nDPWywWRIRo) in this video from Stanford University, but lets try to summarize.

Suppose you have a model to process a high resolution input image and so, a bunch of convolutions that are all keeping the same spatial size of the input image. So, running those convolutions on this high resolution input image over a sequence of layers, would be extremelly computational expensive with the need of lots of memory. Instead, we use an architecture like the Fig. 4 above, where we use downsampling and upsampling. 

As showed in Fig.4, the idea is to work with low resolution instead of high resolution, which means, downsampling and upsampling. 

So, Fig. 4 shows what downsamplig does: it gets the high resolution image as input and ouputs a low resolution image, that is less expensive to work with.
The basic idea is to do a small number of convolutional layers of the original resolution and then downsample the feature maps using functions like `separable_conv2D`, described in more details later. Note in the Fig. 4 above, that we go from high-res to low-res, or, 3xHxW -> High-res: D1xH/2xW/2 -> Low-res: D3xH/4xW/4. In this operation, one drawback is that we lose some information, like spatial measure and pixel features. Managing the numbere of filters is one way to keep the spatial information.
Note that in downsampling, I am using stride equal 2. Now we have a small number of original input and the price of computational operation is very cheap. 

Now we need recover the original input image and Fig. 4 shows that we can do it using updampling. The upsampling is the reverse operation where we need to increase the spatial resolution of our predictions in the second half of the network. Now we decrease the network depth, decreasing the number of filter, until reach the original spatial input information. During upsample we are trying to recover the original input pixel features, but we may not be able to recover all the original information. So, to help on this point we add a *concatenate* operation that will add some collected information from input and add to upsampling layer. In this project I used 3 functions to compose the decoder: `bilinear_up_sampling2d`, `concatenate` and `separable_conv2D`. Note that the *bilinear_up_sampling2d* will take the input and increase the size by 2 or the reverse operation in downsample. Then *concatenate* gets same info from encoder layer to tries to improve the upsampling output. Note that for each layer, the number of filters is the halth when compared with the previous layer which decrease the network depth, to make the output size the same when compared with the relative layer in downsampling process.

The function `fcn_model`, in notebook, creates the model as follow.
```
def fcn_model(inputs, num_classes):
    strides = 2
    filters = 32
    filter_size = 2    # FilterSize defines the size of the local regions to which the neurons connect in the input.
    kernel_size = 3
    
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    x0 = encoder_block(inputs, filters, strides, kernel_size)
    x1 = encoder_block(x0, filters*2, strides, kernel_size)
    x2 = encoder_block(x1, filters*4, strides, kernel_size)
    
    # Add 1x1 Convolution layer using conv2d_batchnorm().
    x_mid = conv2d_batchnorm(x2, filters*4, kernel_size=1, strides=1)
    
    # Add the same number of Decoder Blocks as the number of Encoder Blocks
    x = decoder_block(x_mid, x1, filters*4)
    x = decoder_block(x, x0, filters*2)
    x = decoder_block(x, inputs, filters)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
``` 
Using the model `summary()` function, we can have a better idea about the model used in this project.
```
    model.summary()
```
So, here are the layers used in this project, that comprise encoder and decoder.
* `separable_conv2D.` Perform a depthwise spatial convolution (which acts on each input channel separately) followed by a pointwise convolution which mixes together the resulting output channels. Note that it is important to specify activation, chosed as `ReLU`. Otherwise, "linear" is performed. Note that, for each decode layer, the shape is half the previous one.
* `batchNormalization.` Here we normalize the activations of the previous layer to reduce internal covariate shift in neural networks.
* `conv2D 1x1.` At this point the output shape of convolutional layer is a 4D tensor. To avoid loss of spatial information, we use 1x1 convolutional layer. Note that to create a 1x1 convolutional layer, we must have the setup below. You can check this in `fcn_model` function above.

[image_5]: ./docs/misc/conv2D_1x1.png
![alt text][image_5]
We can check in the model where the output of conv2d, [20,20,20,128], is filled into **conv2d 1x1** and the shape is preserved, assuming output as [20,20,20,128].

    * 1x1 filter size
    * stride = 1
    * zero padding (same)

Now lets go over the decoders to upsample the encoders output until reach the same size of original image.

* `bilinear_up_sampling2d.` Will help resample the information via interpolation.
* `concatenate.` Here we concatenate two layers, the upsampled layer and a layer with more spatial information than the upsampled one, which will retain some of the finer details from the previous layers.
* `separable_conv2D.` Note that in the upsampling process we are using the *separable_conv2D* again but now we have a stride value of one that will prevent the depthwise behavior we got in encoder.

In upsampling we repeat the decoder layer in the same number of encoder.

* `conv2D.` Now we are ready to provide the output layer. The final layer is responsible for making pixel classifications. This final layer process an input that has the same spatial dimention as the input image. However, the number of channels([160,160,32]) is larger and is equal to number of filters in the last upsampled convolution layer. This third dimension needs to be squeezed down to the number of classes we wish to segment. This can be done using a 1-by-1 convolution layer whose number of filters is equal the number of classes, or 3. Now we use `softmax` as activation function.

## Putting all together <a id='all'></a>

At this point we have setup the hyperparameters and designed the model. As commented above, for optimizer I decided to use *Nadam* that got better results over *Adam*. Then we have the following.

```
# Define the Keras model and compile it for training
model = models.Model(inputs=inputs, outputs=output_layer)

model.compile(optimizer=keras.optimizers.Nadam(lr=learning_rate), loss='categorical_crossentropy')

# Data iterators for loading the training and validation data
train_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
    data_folder=os.path.join('..', 'data', 'train'),
    image_shape=image_shape, shift_aug=True)

val_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
    data_folder=os.path.join('..', 'data', 'validation'),
                                        image_shape=image_shape)
```
## Training the model <a id='training'></a>

To train the model I use the `fit_generator` function.
```
model.fit_generator()
```
The figure below show the final epoch iteration of training process.

[image_6]: ./docs/misc/loss36.PNG
![alt text][image_6]
Note that while training we can see some overfitting but the model is able to go back to low loss values. Even so it seems that we are getting a bit of noise with this model. 

## Compute the class scores <a id='pred'></a>

The final score for this project is to reach at least 40%.
Here we have two different scenarios that need be managed by the model with low prediction error.

[image_7]: ./docs/misc/score.PNG
![alt text][image_7]
1. A good prediction when the quadrotor is following the target. In this case, the model need detect the target (object detection) and then follow the target in different scenarios. Here we got a vlaue of `0.910`.
2. Detect the target when target is from far away. This scenario is difficult, mainly when we have croud and the pixel classification must be well trained to detect a low percentage of target in the scene. Here we got a value of `0.214`. The difficulty here can be reflected with the high true positive value of `128`.

So, the model got a final score of `0.422`

Can this model be used to follow a different object? For instance, a cat?

Well, semantic segmentatin has been used for image analysis tasks and so, it describes the process of associating each pixel of an image with a class label. In this project, we used 3 classes and one is the target. So, using the same model, we can replace the object associating the target to a cat and the drone will follow the cat.

But how about the data sample provided? Can this data be used to detect the cat? As I said, as the model is taking it pixel and associating to a class label, it seems intuitive that we need add some cats objects in the samples. Otherwise, the model will not be able to make this association, not due to a lack in the model but due to not enough information in the dataset sample. So, this sample dataset is not enough to predict a cat.

## Future enhancements <a id='enhance'></a>

There are many points that we can play with to enhance the model and reach a better score. For instence
1. Collect more data for train and validation, mainly with target far away the drone. That might improve the iou3 value (hyperparameter).
2. Check the collected data to remove unusefull samples like scenes where there is no target and other people.
3. Try others optimizers.
4. Try a different number of layers (model design).
5. Try others functions for encoder and decoder process (model design).

