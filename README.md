# W-DCGAN-GP (in PyTorch)
 
 A GAN (generative adversarial network) using exclusively convolutional layers (DC), Wasserstein loss (W) and gradient penalty (GP).

 This GAN is currently optimized for 512x512 px grayscale images. It should also work with RGB images, however, I haven't tested that yet. 

 There are some program folders, which will be automatically created once the software is started:
 
 - checkpoints/: It is possible to regularly save generator checkpoints during training. The frequency of the savings can be adjusted in settings file under "generate_checkpoints_epochs".
 - data/: This folder is for the training images. The folder can contain several tarining sets. The path to the training set can be set in the settings file under "pth_data". Currently it is set to "data/fibroblasts/". Please note that this folder needs to contain another folder for the images! An example would be "data/fibroblasts/wt/".
 - plots/: In this folder training plots will be saved. The plots will track Generator loss, Discriminator loss, and gradient penalty. The frequency of the savings can be set in the settings file under "generate_plot_epochs".
 - samples/: Here, sample images are saved during training to keep track of the progress. The frequency, in which the images are saved, can be set in the settings file under "generate_samples_epochs". Currently a set of two images are generated. It is planned to make this adjustable.

 It is planned to implement different GAN versions, which can be choosen in the settings file. 

 Also a simple console menu should be implemented soon.
