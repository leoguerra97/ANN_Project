# Colab-Oration 
Artificial Neural Networks and Deep Learning - Challenge 1 - Image Classification on Mask Dataset

We approached the task by building our own convolutional neural network model following the outline provided during lectures. 
We created datasets generator through the flow_from_dataframe function to combine the image with the JSON label and then created our datasets by splitting our training data in two sets for training and validation. 
The model was setup with convolutional part 5 layers deep (each layer composed of conv. layer, activation layer and maxpooling layer), to which we fed an image of size (IMG_H, IMG_W) (at the beginning set to 407,407), followed by a classifier with one 512 unit dense layer and a softmax activation layer.
We then set the loss function the optimizer with its learning rate and the validation metric (accuracy). 
After this our model was ready to be trained and we set a training lenght of 30 epochs. After training, we created the test set and made predictions on it using the model. Finally using the provided create_csv function we created the csv file.
After completing a base model we introduced data augmentation on our training data to reduce overfitting and increase our performace on the test set.

Since our model did meet our expectations and scored about 50% accuracy on the test set, we decided to change strategy and use a transfer learining approach.
We imported the VGG16 keras model setup with fine-tuning starting from the 15th layer. We also decided to increase the number of layers in the fully connected classifier by adding a new dense layer with 128 units. 
This substantially imporved our score as we set a 77% test score on our first attempt with VGG.

To further improve and speedup the training process we decided to implement early stopping and dropout. By doing so, and also tuning all parameter values (such as learining rate, data augmentation parameters, number of units in a dense layer and early stopping patience) we managed to achieve a 91% test after some trial and error.

We then decided to experiment with different models other than VGG16: we tried VGG19, MobileNet, ResNet50 and InceptionV3. Between these, the Inception model yielded the best result and managed to boost our score to 94.44%. Then we decided to try the Xception model. Using this model, after some parameter tuning, we managed to slighlty improve our score to 94.67%, our best result to date.

Parallel to this we tried to improve our base model using the knowledge gained with the transfer learning approach. We implemented early stopping and dropout like the transfer learning approach and improved the convolutional phase by modifying the number of initial features and depth of the network to optimize feature learning. This changes greatly improved our accuracy bringing it from about 50% to 75%.


It is important to note that we set the seed to make experiments reproducible but, since using Colab's GPU introduces a randomic element, the experiments may not be repeatable with perfect accuracy.

We are submitting 4 notebooks (and their respective result in csv format) which represent our most critical attempts at the challenge: the first notebook "INSERT_NAME" shows our base model in its most updated form, the other three show our attempts using transfer learning. In particular, the second notebook "INSERT_NAME" is our highest scoring VGG model, the third notebook "INSERT_NAME" is our best result with InceptionV3 and the last "INSERT_NAME" is the Xception model which achieved our best result overall.

# Le API Maier
Artificial Neural Networks and Deep Learning - Challenge 2 - Image Segmentation

We tackled the task by splitting the given dataset in a validation and training split and associating the given training masks with their respective images. To do so we used the approach we saw in Professor Lattari's notebook with the Custom Dataset class and created two text files containing the file names for the training-validation split.

We decided to focus only on the BipBip plant species so we set the size parameters to those of BipBip images: img_h = 1536, img_w = 2048.

Then we set up the network model, at first we used a pre-loaded vgg model but, since we were not pleased with the results, we implemented our version on Unet and successively adapted a pre-made Unet model which gave us our best results.

The first submission using the Vgg model returned us a result of 0.49 IoU on BipBip.
The next submission using Unet substantially improved the score by 0.1 and after some fine tuning we achieved our best score of 0.6661, 0.7753, 0.7351 on BipBip Haricot, BipBip Mais and BipBip general respectively.

The Unet model that achieved this score was a model composed of 10 convolutional layers with filters starting from 32 up to 512 and there are 4 MaxPooling/UpSampling layers.

In parallel with this we tried to implement a tiling approach to improve our results, but we found that the improvement provided was not substantial on the training/validation sets and it created problems in the output creation so we chose to not pursue this path further.

One of the main challenges we faced was training the model since the training process became very long (to accurately train the model more than 50 epochs were needed and each epoch took about 8 minutes) and Colab set limits to our GPU time. 
We maanged this problem with the use of checkpoints and by dividing the whole training process in different segments composed of 20 epochs each.

For the output, as we decided to only focus on the BipBip variant, we set the masks of the other types of plants directly to all zeros. 
Then, for all BipBip images in the test set, we created a prediction of the mask and used the provided functions to encode it. Once all images were processed and all predictions were made, the submission json file was created.

The Development Dataset has to be located on drive (or in the same folder if running without drive) and contain the validation and training txt file splits.
Its structure is:

Development_Dataset: 
  > Images (contains all training images, so in our case only BipBip images)

  > Masks (contains all training masks, so in our case only BipBip images)

  > Splits (contains split txt files that can be found in this zip file)

  > Test_Dev (contains all test images)



# Challenge 3
We decided to separate the convolutional part of the network from the rest of the network to reduce the training time on the various epochs. So we imported VGG19 with parameter include_top = False to remove the FC layers and processed all images to create a dictionary of feature maps indexed by image id. Each image id now has a flattened vector of  512 3x5 feature maps associated to it. 

Then we focused on the answers and created a dictionary to map answers to a number and created functions to encode/decode the answers with our dictionary.
As for the questions we decided to encode them using the Space library to encode them as vectors of length equal to maximum question length (24) where each word is encoded as a 300 dim vector.

Then we created two functions, create_tuples and create_test_tuples, to create the tuples to feed to the network. These function take as input the training questions in the json files, extract the text question, answer (the answer part is only relative to the training part and is not present in the create_test_tuples function as it concerns the test set) and image id. Then the questions are encoded with spacy, the answers are encoded with our vocabulary and the features relative to the image id are fetched. Finally a tuple is returned for every entry in the json.  

After this was done we had our training set ready to go, so we created a model by merging two models: the image model, which is an empty model that just takes as input the flattened features and outputs them, and LSTM model, composed of 3 LSTM layers, which takes as input the encoded questions. 
Then these two models are concatenated and followed by a FC layer made of 256 units with ReLu activation functions with Dropout.
At the end there is a layer with 58 neurons (number of possible answers/classes) with softmax activation.

We trained the model by batches for about 50 epochs and, once training was done, we created the predictions. 
The final step was encoding the prediction using the vocabulary provided on Kaggle, so we decoded the predictions using our vocabulary and re-encoded them using the “correct” vocabulary.

