# CIC Meteor detection machine learning tutorial

This repository contains Juypter notebooks for the [Curtin Institute for Computation (CIC)](http://computation.curtin.edu.au) - *Meteor detection machine learning tutorial* run for the [7th International Conference on Smart Computing & Communications (ICSCC 2019)](http://icscc.online/) hosted at Curtin University in Miri, Sarawak, Malaysia on the 28-30 June 2019.

## Background
The [Desert Fireball Network](http://fireballsinthesky.com) (DFN) is a network of cameras in Australia. It is designed to track meteoroids entering the atmosphere, and recover meteorites. It currently operates 50 autonomous cameras, spread across Western Australia and South Australia: Nullarbor plain, WA wheatbelt, and South Australian desert, covering an area of 2.5 million km^2. Recovering meteorites observed through the DFN will help address some of the biggest questions in planetary science: how our planetary system came into being, and how dust and gas produced a planet capable of supporting life – our Earth.
 
The DFN produces ~1.5 petabytes of data per year, which mostly consists of high resolution all-sky images. The rate of data acquisition requires an automated digital pipeline for data reduction.
 
In this tutorial, we will look at the complete pipeline for a system that can automatically detect the presence of meteoroids, even faint ones, in images from the DFN. The first issue to be addressed is that meteoroids in images are a rare event and yet to train a machine learning model, a large number of images containing meteoroids are required. Thus methods for generating artificial training images are explored.
 
We will then go through training a convolutional neural network using [Tensorflow](https://www.tensorflow.org) and [Keras]( https://keras.io) to identify meteoroids in images and demonstrate how the completed workflow functions.
 