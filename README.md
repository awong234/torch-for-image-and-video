# A foray into neural networks for image and video

I sought to understand basic principles and approaches of modeling images and videos using pytorch. This is all elementary learnings and products, but there is an increase in complexity across the implementations moving from a simplistic image to video.

## unet.ipynb

In this example, I conceived of a problem that is trivial for a neural network, but also where a statistical model may struggle -- completing a grid of binary values. I presented the neural network with training images such as below, with the performance evaluation being 
the quality of the predictions against the held-out portion of the image.

![image](https://github.com/user-attachments/assets/65d192cc-090f-4213-b744-b76a7778a472)

It was not difficult to obtain an architecture that could accomplish this task, even one as overwrought and facile as first attempted. I could conceive of ways to use statistical models to evaluate this but not with such generality; this is of course the 
benefit of neural networks.

I attempt to replicate a *proper* U-Net architecture at the end of this document.

## nn_video.ipynb

The objective of the `unet.ipynb` document was to generate missing portions of an image. This provided lessons in the mechanics of data management, training, evaluation, and prediction. 

The next objective -- that of `nn_video.ipynb` -- was to see how I could build a neural network that accepted *many* images as input, and produced a single image as output. These *many* images are temporally seequenced and dependent -- a video.

In this document, I simulate a "blob" that moves over space -- which is nothing more than a probability distribution in two dimensions -- and deposits "events" over the space conditional on the movement of the blob over time. 

![image](https://github.com/user-attachments/assets/7b9244c9-35d8-44c2-b24c-976502666265)

The task was to estimate the total amount of events given the blob's motion -- and devise data loaders and model architectures to handle this appropriately. 

This document displays a first-attempt at this model architecture, highlights a roadblock, and includes some of the debugging efforts to locate the problem in the architecture.

## nn_video_2.ipynb

This document -- largely without commentary -- takes the learnings from `nn_video.ipynb` and produces a model framework that I deemed acceptable in the context of the task. It generates a smoothly varying distribution of the "events" rather than noisy estimates of older
implementations. Its major improvements were the use of BatchNorm3d, and a loss function that evaluated prediction quality at various output resolutions (instead of a single final output resolution at the end).
