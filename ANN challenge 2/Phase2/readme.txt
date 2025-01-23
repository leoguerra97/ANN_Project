We approached the task by splitting the dataset (now composed of the phase 1 training and the Test_Dev data) in a validation and training split and associating the given training masks with their respective images. To do so, we used the method presented in Professor Lattari's notebook with the Custom Dataset class and created two text files containing the file names for the training-validation split.

We decided to focus only on the BipBip plant species so we set the size parameters to those of BipBip images: img_h = 1536, img_w = 2048.

The Unet model that achieved this score was a model composed of 10 convolutional layers with filters starting from 32 up to 512 and there are 4 MaxPooling/UpSampling layers.

We set the learning rate equal to 1e-4 and we trained this model for 30 epochs.

Finally we created the prediction for the second phase test data.