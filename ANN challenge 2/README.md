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






