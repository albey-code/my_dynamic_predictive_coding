# my_dynamic_predictive_coding

This repository holds my adaptation of the dynamic predictive coding (DPC) model by Jiang & Rao (2024). I first trained and tested the DPC model on the original natural forest movie data and then used the pre-trained weights to test the model on new, unseen data. The testing data I used is from a naturalistic movie stimulus called Birdman. Importantly, the Birdman movie was filmed with almost no cuts and therefore allows for continuous perception. Conversely to the natural forest movie, which shows a stroll through a forest, the Birdman movie features complex social interaction incl. movement and facial expression.

Cave: I didn't fine-tune the DPC model, nor did I change any parameters, architecture or hyperparameters with the mere exception of reducing the batch size from 512 to 500 (to be found in the params.json) to make it adaptable to the Birdman input data. In addition, I rewrote the testing script and the data_loader.py.

The same code can also be found in all_predictive_coding together with image and fMRI (pre-)processing steps.
