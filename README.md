# Emotion_detection_using_cnn
## Project Description: Emotion Detection using CNN with FER 2013 Dataset

### Overview:
This project focuses on utilizing Convolutional Neural Networks (CNN) for emotion detection using the FER 2013 dataset. Emotion detection plays a crucial role in various fields such as human-computer interaction, healthcare, and marketing. By leveraging deep learning techniques like CNN, this project aims to accurately classify facial expressions into different emotional categories.

### Key Components:
1. **CNN Architecture**: Implementing a CNN architecture tailored for image-based emotion recognition. This may include convolutional layers, pooling layers, and fully connected layers.
   
2. **FER 2013 Dataset**: Utilizing the FER 2013 dataset, which contains facial images categorized into seven different emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral.

3. **Data Preprocessing**: Preprocessing the dataset by resizing images, normalizing pixel values, and augmenting data to enhance model generalization.

4. **Training and Evaluation**: Training the CNN model on the FER 2013 dataset and evaluating its performance using metrics like accuracy, precision, recall, and F1 score.

5. **Model Deployment**: Exploring options for deploying the trained model for real-time emotion detection applications.

### Goals:
- Develop a robust CNN model for accurate emotion detection.
- Enhance understanding of deep learning techniques in image classification tasks.
- Contribute to the field of emotion recognition and its practical applications.

### GitHub Repository:
The GitHub repository for this project will contain:
- Source code for data preprocessing, model training, and evaluation.
- Documentation detailing the project's architecture, methodology, and results.
- Instructions for replicating the experiment and utilizing the trained model.

### Benefits:
- Improved emotion detection accuracy.
- Potential applications in areas like mental health monitoring, customer sentiment analysis, and personalized user experiences.

- ### Packages need to be installed
- pip install numpy
- pip install opencv-python
- pip install keras
- pip3 install --upgrade tensorflow
- pip install pillow

### download FER2013 dataset
- from below link and put in data folder under your project directory
- https://www.kaggle.com/msambare/fer2013

### Train Emotion detector
- with all face expression images in the FER2013 Dataset
- command --> python TrainEmotionDetector.py

It will take several minutes depends on your processor.
after Training , you will find the trained model structure and weights are stored in your project directory.
emotion_model.json
emotion_model.h5

copy these two files create model folder in your project directory and paste it.

### run your emotion detection test file
python TestEmotionDetector.py

### Conclusion:
By combining CNN with the FER 2013 dataset, this project aims to advance emotion detection technology and contribute to the growing field of deep learning applications in computer vision.
