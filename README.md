#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1-1]: ./examples/datasetDistribution.png "Distribution"
[image1-2]: ./examples/randonCollection.png "Randon Data"

[image2-1]: ./internet-signs/sign1-full.png "Traffic Sign 1"
[image2-2]: ./internet-signs/sign2-full.png "Traffic Sign 2"
[image2-3]: ./internet-signs/sign3-full.png "Traffic Sign 3"
[image2-4]: ./internet-signs/sign4-full.jpg "Traffic Sign 4"
[image2-5]: ./internet-signs/sign5-full.png "Traffic Sign 5"

[image3-1]: ./predictions/predicton-1.png "Prediction Sign 1"
[image3-2]: ./predictions/predicton-2.png "Prediction Sign 2"
[image3-3]: ./predictions/predicton-3.png "Prediction Sign 3"
[image3-4]: ./predictions/predicton-4.png "Prediction Sign 4"
[image3-5]: ./predictions/predicton-5.png "Prediction Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python to find the number of examples and numpy to find the number of unique classes. The resuls were:

* The size of training set is 34799 examples
* The size of the validation set is 4410 examples
* The size of test set is 12630 examples
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the dataset is distributed in Training, Test and Validation.

![alt text][image1-1]

Also, to have a general idea of our dataset, I printed few randon images from it:

![alt text][image1-2]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


The first step taken was randomize the training set. 
As a last step, I normalized the image data, from (0,255) range to (-1,1) range, because small inputs provide better results.
No other preprocessing technique was used.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| tahn					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6                  |
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16  	|
| tahn                  |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Flatten               | Output 400                                    |
| Fully connected		| Output 120   									|
| tahn                  |                                               |
| Fully connected		| Output 84   									|
| tahn                  |                                               |
| Fully connected		| Output 43  									|
 
The change here in relation to LeNet archictecture presented in the course is the usage of Tahn as activation function.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam optimizer with a learning rate of 0.0009. The batch size was 128, and the number of epochs was definied as 20.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95,4%
* test set accuracy of 93.6%

In my case, an iterative approach was chosen:

The initial archictecture chose was LeNet. It was chose because it proved be an archictecture that can provide good results. However, with the given Traffic-Sign data set, the problem of this archictecture was the low initial accuracy. To improve the accuracy, I did a few iterative changes, and empirally found out that using tanh as activation function increased significantly the accuracy to about 0.92 ~ 0.935.
Then, looking for even more improvement, I tried to find the best value for learning rate mainly because this is the hyperparameter that can impact a lot in results, and with that I found the value of 0.0009, and the rate increase to about 0.95. Then I increase the number of epochs to improve even more the results, mainly on validation set.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2-1] ![alt text][image2-2] ![alt text][image2-3] 
![alt text][image2-4] ![alt text][image2-5]

The first image might be difficult to classify because it has three curved arrows, and includes a lot of different images in the background.
The second image might be difficult to classify because it includes a complex shape that represents the snow shape.
The same happens on the third image, that has the two kids draw.
The fourth image might be difficult to classify because the 80 shape is can be similar to other values shapes, like 30, 50 and 60.
The fifth image can be difficult to classify because it has similarity with other signs that also are arrows.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Roundabout mandatory 	| Roundabout mandatory                          | 
| Beware of ice/snow	| Bicycles crossing                             |
| Children crossing     | Children crossing								|
| 80 km/h	      		| 80 km/h                                       |
| Ahead only			| Ahead only                                    |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This value is smaller than the values saw in test set, but considering the small number of prediction cases and the fact that the sign 'Beware of ice/snow' has a small training set compared to other classes, I consider 80% a good result for this classification set. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 45th cell of the Ipython notebook.

For all images tested, the most probably class given is significantly bigger than the other probably classes, showing that the model was significantly certain about its predictions. Below the prediction percentages shown for each image:

  
| Probability         	|     Prediction for image 1                        | 
|:---------------------:|:-------------------------------------------------:| 
| .907        			| Roundabout mandatory                              | 
| .090     				| Speed limit (100km/h)								|
| .0009					| Beware of ice/snow								|
| .0007	      			| End of no passing by vehicles over 3.5 metric tons|
| .0002				    | General caution                                   |

![alt text][image3-1]


| Probability         	|     Prediction for image 2                        | 
|:---------------------:|:-------------------------------------------------:| 
| .9860        			| Bicycles crossing                                 | 
| .0035     			| Road work                                         |
| .0028					| Beware of ice/snow								|
| .0020	      			| Speed limit (60km/h)                              |
| .0019				    | Children crossing                                 |

![alt text][image3-2]

| Probability         	|     Prediction for image 3                        | 
|:---------------------:|:-------------------------------------------------:| 
| .8720        			| Children crossing                                 | 
| .0958     			| Beware of ice/snow                                         |
| .0297					| Speed limit (100km/h)								|
| .0005	      			| Speed limit (60km/h)                              |
| .0004				    | Right-of-way at the next intersection             |

![alt text][image3-3]

| Probability         	|     Prediction for image 4                        | 
|:---------------------:|:-------------------------------------------------:| 
| .9940        			| Speed limit (80km/h)                              | 
| .0046     			| No passing for vehicles over 3.5 metric tons      |
| .0005					| Speed limit (50km/h)								|
| .0003	      			| Speed limit (70km/h)                              |
| .0002				    | Stop                                              |

![alt text][image3-4]

| Probability         	|     Prediction for image 5                        | 
|:---------------------:|:-------------------------------------------------:| 
| .9988        			| Ahead only                                        | 
| .0004     			| Turn left ahead                                   |
| .0003					| End of no passing                                 |
| .0002	      			| Priority road                                     |
| .0001				    | Go straight or right                              |

![alt text][image3-5]

