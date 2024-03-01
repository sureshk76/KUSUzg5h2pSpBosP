Detecting page filps

The task is to help an app identify if a page is being flipped or not by looking at an image.
The requirement is to have a light-weight neural network model that makes accurate predictions of whether a page is being flipped or not by looking at an image with a high accuracy.  We need a model that can achieve as high accuracy as possible without exceeding a size of 40MB but a smaller size makes the user experience more satisfying.
The flip detection model 
fd_model_v5: is a promising model achieving a 98% accuracy and f1 score with a size as small as 2.5 MB.  It consists of below layers:
 ![image](https://github.com/sureshk76/KUSUzg5h2pSpBosP/assets/18317652/9d0cabb7-c073-49f2-848c-8415d213304b)

As can be seen, the output from a couple of dense layers with 32 neurons and ReLU activation and 2 neurons and Softmax activation when fed with the flattened output from a combination of convolutional layers with ReLU activation and pooling layers gives a model that achieves this result.
 ![image](https://github.com/sureshk76/KUSUzg5h2pSpBosP/assets/18317652/eda8951d-456b-402c-b4ad-cb852029a3f6)

Adam optimizer with sparse categorical crossentropy loss function is used.
 ![image](https://github.com/sureshk76/KUSUzg5h2pSpBosP/assets/18317652/5b837a2c-fc60-4c0e-bb19-4d5e491eb9d5)

Pretrained Models:
The below pretrained models were used  simply by flattening the output received from these models and feeding that as input to the Dense layer 1 (32 neurons and ReLU activation) and Dense layer 2 (2 neurons and Softmax activation) and then training the resultant model with our training data.
1.	Mobilenet
2.	Efficientnet
3.	VGG
4.	Resnet
The above pretrained models used produced good results with Mobilenet and Efficient meeting both accuracy and memory requirements while Resnet and VGG did not meet the memory (size) requirements.
 ![image](https://github.com/sureshk76/KUSUzg5h2pSpBosP/assets/18317652/1799e01f-add9-437b-a255-91992c992795)

 ![image](https://github.com/sureshk76/KUSUzg5h2pSpBosP/assets/18317652/50fb3706-32a1-4730-b86e-10677ed6bfc6)

![image](https://github.com/sureshk76/KUSUzg5h2pSpBosP/assets/18317652/d1b7f1ee-1068-48a9-af31-c88ac66960c0)
 
![image](https://github.com/sureshk76/KUSUzg5h2pSpBosP/assets/18317652/f975cdf1-f318-4f5a-b916-860e7a136508)
 

