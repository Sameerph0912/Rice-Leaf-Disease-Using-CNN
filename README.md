**RiceLeaf - Domain analysis**

(1) Riceleaf is the leaf of the Rice producing crop. Rice is stape food for large parts of this world, mostly grown and savoured in the asiatic regions. The size and shape of rice and it's leaves will vary depending on the rice variety, environmental conditions, and stage of growth. As the rice plant matures, the leaves may change in color and exhibit different characteristics, especially if affected by diseases, pests, or environmental stressors. The riceleaves heath is essential to determine the health of the crop, so a regular monitering is essential to detect any diseases, problems caused by the insects. In this project we are motivated to handle the data to detect three imports issues to tackle the health of the rice leaves (i) Leaf smut (ii) Brown spot (iii) Bacterial leaf blight Data contain a total of 119 Images, eqaully distributed among these three diseases. Bacterial leaf blight, caused by the bacterium Xanthomonas oryzae pv. oryzae, is a serious disease that can cause extensive damage to rice plants. The symptoms of bacterial leaf blight include water-soaked lesions on the leaves, which later turn brown and dry up. In severe cases, the disease can cause wilting and death of the plant, leading to a significant reduction in crop yield.

(2) Brown spot, caused by the fungus Cochliobolus miyabeanus, is another major rice disease that can cause significant yield losses. The disease is characterized by small, oval to elliptical spots on the leaves, which turn brown with a yellow halo. In severe cases, the spots can coalesce and cause the leaves to wither and die. Brown spot can also affect the panicles, leading to a reduction in grain quality and yield.

(3) Leaf smut, caused by the fungus Entyloma oryzae, is a relatively less common rice disease. The symptoms of leaf smut include the formation of small, round, and reddish-brown spots on the leaves. These spots later turn black and produce powdery spores. Although leaf smut does not usually cause significant yield losses, it can affect the quality of rice grains by reducing their weight and size.

(4) Effective disease management strategies are crucial for controlling these diseases and reducing their impact on rice production. Some of the common methods for disease management include crop rotation, use of resistant cultivars, application of fungicides, and cultural practices such as proper crop spacing and irrigation management.

(5) In recent years, the use of machine learning algorithms for plant disease diagnosis and classification has gained significant attention. With the availability of large datasets of plant images and the advancements in deep learning algorithms, it has become possible to accurately classify plant diseases based on their visual symptoms. This has the potential to improve disease management strategies and reduce the impact of plant diseases on crop production.

**An approach to CNN implementation**

1. import the libaraies and the dataset
2. Image visualization
3. Rescaling the pixel image by 255
4. Dividing dataset into training, validation and testing
5. Model architecture
6. Model compilation
7. Model training
8. Load the Model with the Best Validation Accuracy
9. visualize some predictions
10. Model Camparision and Challenges Faced

**Normalisation**

Normalization of image data is an important step in the pre-processing of data before it is used to train a neural network. It involves transforming the pixel values of the input images so that they fall within a specific range, typically [0, 1]. Normalization ensures that the input features have similar scales, which can prevent some input features from dominating others during the training process.

By scaling the pixel values, normalization also makes it easier for the neural network to learn the underlying patterns in the data. This is because the weights in the neural network can be updated more easily and quickly when the input data has a similar scale. In addition, normalization can help to reduce the effects of lighting conditions, noise and other factors that can cause variation in the input data.

Overall, normalization is an important step that can improve the performance and accuracy of a neural network by ensuring that the input data is in a consistent and standardized format.


**we normalize the dataset of our dataset images. Images are essentially numerical data with pixel values representing color intensity. # Converting images into Numpy arrays ensures that they can be treated as numerical data by the model.**

**Data Augmentation**

Data augmentation is a technique used to increase the size and diversity of a dataset by applying various transformations to the existing data. This technique has become an essential tool in computer vision and image processing tasks, such as object recognition and classification, due to its ability to enhance the generalization ability of machine learning models and prevent overfitting.

In the context of our rice leaf disease image classification project, data augmentation can play a crucial role in improving the performance and robustness of our model. By generating new images with different variations such as rotations, flips, zooms, and other transformations, we can increase the diversity of our dataset and provide our model with more examples to learn from, which can lead to better classification accuracy and robustness to variations in the real-world data.

Therefore, data augmentation is an important aspect to consider in our project, and we will explore various techniques and approaches to implement it effectively.


**Keras Tuner**

Keras Tuner is a hyperparameter tuning library for Keras, which allows users to search for the best hyperparameters in an automated way. Hyperparameter tuning is a crucial step in building machine learning models, and it involves finding the best set of hyperparameters for a given model architecture and dataset. The optimal hyperparameters can help to achieve better performance in terms of accuracy, speed, and generalization ability of the model.

In this project, Keras Tuner has been used to search for the optimal hyperparameters of the convolutional neural network model, which includes the number of convolutional layers, the number of filters in each layer, the kernel size, the activation function, and the learning rate of the optimizer. The goal is to find the best combination of hyperparameters that can improve the accuracy of the model on the test dataset.

At the end, we will analyse whether the model performance improves or not.


**Model Comparision**

(1) Main CNN model provides a accuracy of 62.5%
(2) Learning of 0.0001 provides 75% accuracy and it deterioates when the learning rate increases
(3) Data Augmentation 83%
(4) MLP was unable to identify all the disease classifications.
(5) Random Search is also able to identify only one class



**Challenges Faced**
(1) The dataset is limited, train and testing is limited aswell
(2) Limited knowledge on Deep learning algorithms to implement on Image processing
(3) image shape, array and convertion error was hard to understand, and it took a certain amount of time to understand the code and implement the data
(4) when compiling the model, the optimizer, loss and metrics has to be fixed when chnaged there is an error in the output
(5) loss = sparse_categorical_entropy, when just categorical_entropy is used there is an error in output and distrupts the flow process as our labels are class encoded.
(6) In hyperparameter tuning, MLP and RandomSearch can only identify one class, when the inout are chnaged the class identification varies but still identifies only one class.
(7) With limited dataset various models and hyperparameters were not able to be implemented as testing and validation data is required more. 
