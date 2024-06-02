# Predict Human Character from a Facial Image : Project Overview
   ![Image](https://github.com/germeengehad/Predict-Human-Character-from-a-Facial-Image/blob/main/dataset-cover.jpg)
- This project aims to predict whether a person's character is savory or unsavory based on their images using deep learning techniques, specifically a Convolutional Neural Network (CNN) model.
- All data in this dataset was gathered from PUBLICLY accessible web sites or databases .This dataset consists of 2 classes, savory and unsavory. The unsavory class is populate with i facial mages of convicted felons. The savory class is populated with facial images of "ordinary" people. Granted some "ordinary" people may be convicted felon but I expect the percentage is very low.
-Initially, I planned to import the dataset from Kaggle. I then applied data augmentation and preprocessing techniques. After building and training a CNN model, I tested it by predicting the character (savory or unsavory) of two individuals using images. Note that only the training and validation datasets were used for model development. The model was tested with images sourced from the web, which I uploaded to this repository.

# The Data Set
This directory contains 11 facial images. 10 of the images are of Dr Fauci. The 11th image (11.jpg) is of a convicted felon.
This directory is used to demonstrate the predictor function. Specifically with parameter averaged=False the predictor function will produce and print out a class prediction and probability for EACH image. If averaged=True, the function averages the probabilities for each image and predicts the majority class for the images including the averaged probability. This function with averaged=True if you are using a trained model to make a prediction on an individual person. Rather than basing the prediction on a single image of the person it is far more accurate to include several images of the person and use the average class prediction and probability

#  Data Augmentation And Data Preprossing
The data augmentation and preprocessing steps in this project involve several techniques to enhance the robustness and performance of the CNN model. For the training dataset: 
- images are randomly rotated, sheared, and zoomed to create variations.
-  They are also rescaled to normalize pixel values, shifted horizontally and vertically, and filled with a constant value where necessary.
-  Additionally, brightness adjustments and horizontal flips are applied to further diversify the dataset.
-   The training images are resized to 64x64 pixels and processed in batches. For the validation dataset, images are simply rescaled to normalize pixel values and resized to 64x64 pixels
- ensuring that the evaluation data is consistent with the training data format. This approach helps the model generalize better by training on a varied set of augmented images and validating on properly scaled images.


# Model Building
- The model described is a Convolutional Neural Network (CNN) designed for binary classification, predicting whether an image corresponds to a savory or unsavory character. The input images are resized to 64x64 pixels with three color channels (RGB).

- Here's a brief overview of the model architecture:

  - Input Layer: Takes images of shape 64x64x3.
  - Convolutional Layers:
The first convolutional layer has 32 filters with a kernel size of 3x3, followed by ReLU activation and same padding.
This is followed by max pooling with a 2x2 pool size, batch normalization, and a 20% dropout.
The second convolutional layer also has 32 filters with a 3x3 kernel size, ReLU activation, and same padding.
This is followed by another 2x2 max pooling layer, batch normalization, and 20% dropout.
  - Flattening Layer: Converts the 2D matrix into a 1D vector.
  - Dense Layers:
The first dense layer has 512 neurons with ReLU activation, followed by batch normalization and a 20% dropout.
The second dense layer has 256 neurons with ReLU activation, followed by batch normalization and a 20% dropout.
  - Output Layer: A single neuron with sigmoid activation for binary classification.
  - Compilation: The model is compiled using the Adam optimizer, binary cross-entropy loss function, and accuracy as the performance metric.

# Model Performance
-    ![Image](https://github.com/germeengehad/Predict-Human-Character-from-a-Facial-Image/blob/main/download%20(1).png)
-    ![Image](https://github.com/germeengehad/Predict-Human-Character-from-a-Facial-Image/blob/main/download%20(2).png)

  
