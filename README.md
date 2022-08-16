# Multi-class Dog Breed Classification or Dog Vision

![](https://github.com/samyakmohelay/Dog-Breed-Predictor/blob/main/readme_images/dog.png)

This notebook builds an end-to-end multi-class dog image classifier using TensorFlow 2.0 and TensorFlow Hub.

## Problem
Identifying the breed of a dog given an image of a dog.

## Data
The data we're using is from Kaggle's [dog breed identification competition](https://www.kaggle.com/c/dog-breed-identification/data).

We have a training set and a test set of images of dogs. 
Each image has a filename that is its unique id. 

The dataset comprises **120** breeds of dogs. 
The goal of the competition is to create a classifier capable of determining a dog's breed from a photo. 

### The list of breeds is as follows: 

![](https://github.com/samyakmohelay/Dog-Breed-Predictor/blob/main/readme_images/1.png)

## Features
Some information about the data:

- We're dealing with images (unstructured data) so it's probably best we use deep learning/transfer learning.
- There are `120` breeds of dogs (this means there of 120 different classes).
- There are around `10,000+` images in the training set.(have labels)
- There are around `10,000+` images in the test set. (don't have labels- we have to predict them)

## - The workflow will be as follows

## 1. Getting our workspace ready
- Import Tensorflow
- Import Tensorflow Hub
- Make sure we're using a GPU

## 2. Getting the Data ready (turning into Tensors)
First, we'll visualize the number of dogs per breed.
![](https://github.com/samyakmohelay/Dog-Breed-Predictor/blob/main/readme_images/2.png)

With all machine learning models, our data has to be in numerical format.

So that's what we'll be doing next. Turning our images into Tensors (numerical representations).

![](https://github.com/samyakmohelay/Dog-Breed-Predictor/blob/main/readme_images/3.png)

Now we've demonstrated what an image looks like as a tensor, let's make a function to preprocess them.

> The function will be as follows: 

- Take an image filepath as input.
- Use TensorFlow to read the file and save it to a variable, image.
- Turn ourimage (a jpg) into Tensors.
- Normalize our image (convert color channel values from 0-255 to 0-1)
- Resize the image to be a shape of (224, 224)
- Return the modified image.

## 3. Turning our data into batches

Why turn our data into batches?

Let's say we are trying to process 10,000+ images in one go... they all might not fit into memory.

That's why we do about 32 (batch size) images at a time (can manually adjust batch size if need be).

In order to use TenserFlow effictively, we need our data in the form of Tensor tuples which look like this: (image, label).

## 4. Visualizing the Data Batches

Our data is now in batches, however, these can be a little hard to understand, let's visialize them.

Creating a function for viewing images in a data batch which sisplays a plot of 25 images and their labels from a data batch.

![](https://github.com/samyakmohelay/Dog-Breed-Predictor/blob/main/readme_images/4.png)

## 5. Building a model

> Before we build a model, there are a few things we need to define:

- The input shape (our images shape, int the form of Tensors) of our model.
- The output shape (image labels, in the forms of Tensors) of our model.
- The URL of the model we will use from [TenserFlow Hub](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5).

After getting our inputs, outputs and model ready to go. Let's put them together into a Keras deep learning model.

> Knowing this, lets create a function which:

- Takes the input shape, output shape and the model we've chosen as parameters.
- Defines the layers in a Keras model in sequential fashion (do this first, then this, then that).
- Compiles the model (says how it should be evaluated and improved).
- Builds the model (tells the model the input shape it'll be getting).
- Returns the model.

All of these steps can be found [here](https://www.tensorflow.org/guide/keras/sequential_model).

## 6. Creating callbacks
Callbacks are helper functions a model can use during training to do such things as save its progress, check its progress or stop training early if a model stops improving.

We'll create two callbacks, one for **TensorBoard** which helps our models progress and another for **Early stopping** which prevents our model from training for too long.

- TensorBoard Callback
- Early Stopping Callback

## 7. Training a model (on subset of data)
Our first model is only going to train on 1000 images, to make sure everything is working.

## 8. Making and evaluating predictions using a trained model

![](https://github.com/samyakmohelay/Dog-Breed-Predictor/blob/main/readme_images/5.png)

## 9. A function to make these all a bit more visaulized.

We'll create a function which:

- Takes an array of prediction probabilities, an array of truth labels and an array of images and an integer. ✅
- Convert the prediction probabilities to a predicted label. ✅
- Plot the predicted label, its predicted probability, the truth label and the target image on a single plot. ✅

![](https://github.com/samyakmohelay/Dog-Breed-Predictor/blob/main/readme_images/6.png)

## 10. Another function to view our models top 10 predictions.

Now we've got one function to visualize our models top prediction, let's make another to view our models top 10 predictions.

> This function will:

- Take an input of prediction probabilities array and a ground truth array and an integer ✅
- Find the prediction using get_pred_label() ✅
- Find the top 10:
  - Prediction probabilities indexes ✅
  - Prediction probabilities values ✅
  - Prediction labels ✅
- Plot the top 10 prediction probability values and labels, coloring the true label green ✅

![](https://github.com/samyakmohelay/Dog-Breed-Predictor/blob/main/readme_images/7.png)

## 11. Training the model on FULL DATA! 🐶
- Created a data batch with the full data set
- Created a model for full model
- Created full model callbacks

Fitted the full model to the full data.
![Full Data](https://github.com/samyakmohelay/Dog-Breed-Predictor/blob/main/readme_images/8.png)

## 12. Making predictions on the test dataset
Since our model has been trained on images in the form of Tensor batches, to make predictions on the test data, we'll have to get it into the same format.

> To make predictions on the test data, we'll:

- Get the test image filenames.
- Convert the filenames into test data batches using create_data_batches() and setting the test_data parameter to True (since the test data doesn't have labels). 
- Make a predictions array by passing the test batches to the predict() method called on our model.

Now, we can make predictions on test data batch using the loaded full model.

Saving the predictions (NumPy array) to csv file (for access later).

## 13. Preparing test dataset predictions for Kaggle
I am also going to submit the model's prediction probability output to **Kaggle's dog breed submissions**:

Looking at the Kaggle sample submission, we find that it wants our models prediction probaiblity outputs in a DataFrame with an ID and a column for each different dog breed. [Link](https://www.kaggle.com/c/dog-breed-identification/overview/evaluation)

> To get the data in this format, we'll:

- Create a pandas DataFrame with an ID column as well as a column for each dog breed.
- Add data to the ID column by extracting the test image ID's from their filepaths.
- Add data (the prediction probabilites) to each of the dog breed columns.
- Export the DataFrame as a CSV to submit it to Kaggle. 

The DataFrame for Kaggle is ["full_model_predictions_submission_1_mobilenetV2.csv"](https://github.com/samyakmohelay/Dog-Breed-Predictor/blob/main/full_model_predictions_submission_1_mobilenetV2.csv).

## 14. Final: Making predictions on custom images
#### At last, our model is capable of predicting the breed of dog from any picture taken on the spot!

> To make predictions on custom images, we'll:

- Get the filepaths of our own images.
- Turn the filepaths into data batches using create_data_batches(). And since our custom images won't have labels, we set the test_data parameter to True.
- Pass the custom image data batch to our model's predict() method.
- Convert the prediction output probabilities to predictions labels.
- Compare the predicted labels to the custom images.

The below images of some dogs are taken from my camera to test the model.

![](https://github.com/samyakmohelay/Dog-Breed-Predictor/blob/main/readme_images/9.png)

Interesting,
- The dog on the upper row is actually a mix-breed of German Shepherd and Rottweiler both. That's why the model has predicted in such a way.
- And both dogs on the bottom are pure German Shepherd and Beagle respectively.

#### Great, the model is working very accurately!
