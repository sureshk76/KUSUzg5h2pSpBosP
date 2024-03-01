Detecting page filps

The task is to help an app identify if a page is being flipped or not by looking at an image.
The requirement is to have a light-weight neural network model that makes accurate predictions of whether a page is being flipped or not by looking at an image with a high accuracy.  We need a model that can achieve as high accuracy as possible without exceeding a size of 40MB but a smaller size makes the user experience more satisfying.
The flip detection model 
fd_model_v5: is a promising model achieving a 98% accuracy and f1 score with a size as small as 2.5 MB.  It consists of below layers:
 
As can be seen, the output from a couple of dense layers with 32 neurons and ReLU activation and 2 neurons and Softmax activation when fed with the flattened output from a combination of convolutional layers with ReLU activation and pooling layers gives a model that achieves this result.
 
Adam optimizer with sparse categorical crossentropy loss function is used.
 
Pretrained Models:
The below pretrained models were used  simply by flattening the output received from these models and feeding that as input to the Dense layer 1 (32 neurons and ReLU activation) and Dense layer 2 (2 neurons and Softmax activation) and then training the resultant model with our training data.
1.	Mobilenet
2.	Efficientnet
3.	VGG
4.	Resnet
The above pretrained models used produced good results with Mobilenet and Efficient meeting both accuracy and memory requirements while Resnet and VGG did not meet the memory (size) requirements.
 
 
 
 

