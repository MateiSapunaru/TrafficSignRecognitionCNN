**Traffic Sign Classification Project**

This project is designed to classify traffic signs using a Convolutional Neural Network (CNN) built with Keras. The project consists of three main scripts: model training, data preparation, and model evaluation.
Project Structure
    
  PrepareData.py: Prepares the dataset by organizing training and validation data into respective folders.
    
  main.py: Contains the code for building, training, and saving the CNN model.
    
  ModelCheck.py: Loads the trained model to evaluate a test image and predicts its class.

**Requirements**

  Python 3.x
  
  Required Python packages: keras, numpy, matplotlib, os, shutil
  
  The dataset should be organized as follows:
  
        v_data/GTSRB/Files/Train/ (for training images)
        v_data/GTSRB/Files/Validation/ (for validation images)
        v_data/GTSRB/Files/Test/ (for test images)

**1. Training the Model

This script builds and trains a CNN model on traffic sign images. It uses data augmentation to enhance the training set and saves the trained model as model_saved.keras.


Key Features

    Uses a CNN architecture with Conv2D, MaxPooling, Dense, and Dropout layers.
  
    Applies data augmentation (shear, zoom, and horizontal flip) to the training data.
  
    Plots training and validation accuracy and loss.

**2. Preparing the Data**

This script organizes the dataset by creating validation data from the training set. It moves a subset of images from the training directory to the validation directory.
How to Run

Key Features

    Creates subdirectories for each class (0 to 42) in the validation folder.
  
    Moves up to 250 images per class from the training directory to the validation directory.
  
    Provides warnings if there are insufficient images in any class.

**3. Evaluating the Model**

This script loads the saved model (model_saved.keras) and evaluates it on a test image. It predicts the class and outputs the predicted class index along with the probability.
How to Run


Key Features

    Loads a pre-trained model and prints its summary.
  
    Preprocesses the test image (resizing and normalizing).
  
    Predicts the class of the image and displays the results.


**Modify batch sizes, epochs, or dataset paths in the script according to your dataset size and hardware capabilities.**
