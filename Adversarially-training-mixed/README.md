Clone the project.
Use train.ipynb to train your model.

Use the params dictionary in the first cell to set the model parameters.
Set resume to False to create and train your model. Set it to true to load your model (with specified name).
Set a name for your model. Check points will be saved using this name.

In the Loading Data section change download to Ture to download the CIFAR-10 train and test datasets.

To start/resume training run Train the Model section.  Use train_normal function to train the model normally. Remember to (set advs_train to false)

For training using the mixing method you should use train_mixup function. Set advs_train to false to train a vanilla linear mixup model without adversarial noise in data.

The rest of the code is for Robustness Evaluation so you can simply discard it.

If you have any question about the code ask me. I would be happy to help.
