# Real Time Creativity

The goal of current project is to track the real time creativity of user i.e. user draws a picture of object using his/her finger point and to recognize the object/emojis saved as a result of tracking using deep learning.

# Dependencies
Deep Learning architecture for recognizing the drawn objects/emojis uses [OpenCV](https://opencv.org/) (opencv==4.2.0) and Python (python==3.7). The model Convolution Neural Network(CNN) uses [Keras](https://keras.io/) (keras==2.3.1) on [Tensorflow](https://www.tensorflow.org/) (tensorflow>=1.15.2). Also, imutils==0.5.3, numpy==1.18.2, matplotlib==3.2.1, argparse==1.1, pandas==0.23.4 and scipy==1.1.0 are also used.

# Dataset
Quick, Draw! is an online game developed by Google that challenges players to draw a picture of an object or idea and then uses a neural network artificial intelligence to guess what the drawings represent. The AI learns from each drawing, increasing its ability to guess correctly in the future.The game is similar to Pictionary in that the player only has a limited time to draw (20 seconds).The concepts that it guesses can be simple, like 'foot', or more complicated, like 'animal migration'. This game is one of many simple games created by Google that are AI based as part of a project known as 'A.I. Experiments'. [Quick Draw](https://quickdraw.withgoogle.com/)

For the current project, 10 objects are used as mentioned below
1. apple <img src="https://github.com/Devashi-Choudhary/Real_Time_Creativity/blob/master/data/Apple.png" width="20">
2. cup <img src="https://github.com/Devashi-Choudhary/Real_Time_Creativity/blob/master/data/Cup.png" width="20">
3. laptop <img src="https://github.com/Devashi-Choudhary/Real_Time_Creativity/blob/master/data/Laptop.png" width="20">
4. leaf <img src="https://github.com/Devashi-Choudhary/Real_Time_Creativity/blob/master/data/Leaf.png" width="20">
5. penguin <img src="https://github.com/Devashi-Choudhary/Real_Time_Creativity/blob/master/data/Penguin.png" width="20">
6. pizza <img src="https://github.com/Devashi-Choudhary/Real_Time_Creativity/blob/master/data/Pizza.png" width="20">
7. shoe <img src="https://github.com/Devashi-Choudhary/Real_Time_Creativity/blob/master/data/Shoe.png" width="20">
8. The Eiffel Tower <img src="https://github.com/Devashi-Choudhary/Real_Time_Creativity/blob/master/data/The-Eiffel-Tower.png" width="15">
9. triangle <img src="https://github.com/Devashi-Choudhary/Real_Time_Creativity/blob/master/data/Triangle.png" width="20">
10. wine bottle <img src="https://github.com/Devashi-Choudhary/Real_Time_Creativity/blob/master/data/Wine-Bottle.png" width="20">

You can download `.npy` files from [google cloud](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap). 


# How to execute code:
1. You will first have to download the repository and then extract the contents into a folder.
2. Make sure you have the correct version of Python installed on your machine. This code runs on Python 3.6 above.
3. Now, run the following command in your Terminal/Command Prompt to install the libraries required
> `pip install requirements.txt`
4. Get the dataset as mentioned above  and create **Dataset** directory in the current directory and place the `.npy` files in `/Dataset` directory.
5. **Training of CNN Model :** Open terminal. Go into the project directory folder and type the following command:
> `python train.py --path Dataset` where --path is path to input dataset.

The above command will load the data from the `/Dataset` folder and store the features and labels, After that the training process begins.

**Note :** Apart from above mentioned objects, you can download other objects and train your model according to that.

6. Now you need to have the data, use the webcam to get what you have drawn and recognize it using the trained model. Open terminal type the following command:
> `python Finger_Tracking_Testing.py --model <your trained model>` 

The default value of model is `mods.h5` which is trained on 10 objects as mentined above. You can run above command for reference. When you are training your model for objects which are not mentioned above then you need to make following changes : 

1. Need to download the related object image and place it inside `/data` folder.
2. Make change in `class_dict` variable of `Finger_Tracking_Testing.py` according to object.
3. While running  `python Finger_Tracking_Testing.py --model <your trained model>` command, set path of your trained model.

# Results

1. Accuracy/Loss training curve plot.

![Accuracy](https://github.com/Devashi-Choudhary/Real_Time_Creativity/blob/master/Results/Accuracy.png)

![Loss](https://github.com/Devashi-Choudhary/Real_Time_Creativity/blob/master/Results/Loss.png)


2. Predicted Object

![predicted](https://github.com/Devashi-Choudhary/Real_Time_Creativity/blob/master/Results/output.JPG)

3. Real Time Creativity of User

# Contributors

[Meet Shah](https://github.com/mit41196)
[Neha Goyal](https://github.com/Neha-Goyal)
