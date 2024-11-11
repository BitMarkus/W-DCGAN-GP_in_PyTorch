#############################################################################
# Tips for GAN modelling:
# Source: https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/
#
# - Replace all pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator):
#      Use the stride in convolutional layers to perform downsampling in the discriminator model
#      Use ConvTranspose2D and stride for upsampling in the generator model
#
# - Use batchnorm in both the generator and the discriminator:
#      Batch norm layers are recommended in both the discriminator and generator models, except the output of the generator 
#      and input to the discriminator
#
# - Remove fully connected hidden layers for deeper architectures:
#      In the discriminator the convolutional layers are flattened and passed directly to the output layer
#      The random Gaussian input vector passed to the generator model is reshaped directly into a multi-dimensional 
#      tensor that can be passed to the first convolutional layer ready for upscaling
#
# - Use LeakyReLU activation in generator for all layers except the output layer
#
# - Normalize inputs to the range [-1, 1] and use Tanh activation in generator for output layer
#
# - Use LeakyReLU activation in the discriminator for all layers
#
# - Use Adam Optimization: 
#      recommended batch size: 128
#      learning rate: 0.0002
#      momentum (beta1): 0.5
#
# - Use dropout of 50 percent during train and generation
#
# - Use a kernel size that’s divisible by the stride size for strided Conv2DTranpose or Conv2D in both the generator and the discriminator
#############################################################################

import torch
from torch import nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from datetime import datetime
from tqdm import tqdm
# Own modules
from settings import setting
from generator import Generator
from discriminator import Discriminator

class Train():

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, device, dataloader):

        # Passed parameters
        self.dataloader = dataloader
        self.device = device

        # Settings parameters
        self.num_epochs = setting["num_epochs"]
        # Size of z latent vector (i.e. size of generator input)
        self.latent_vector_size = setting["latent_vector_size"]
        # Number of GPUs available. Use 0 for CPU mode.
        self.num_gpu = setting["num_gpu"]
        # Learning rate for generator
        self.gen_learning_rate = setting["gen_learning_rate"]
        # Learning rate for discriminator
        self.disc_learning_rate = setting["disc_learning_rate"]
        # Beta1 hyperparameter for Adam optimizers
        self.beta1 = setting["opt_beta_1"]
        # Boolian variable if samples suppose to be generated during training
        self.generate_samples = setting["generate_samples"]
        # Number of samples to visualize the result of the generator
        self.num_samples = setting["num_samples"]
        # Save sample images every x epochs in samples folder
        self.generate_samples_epochs = setting["generate_samples_epochs"]
        # Save generator every x epochs in checkpoints folder
        self.generate_checkpoints_epochs = setting["generate_checkpoints_epochs"]
        # Save loss plot every x epochs in plots folder
        self.generate_plot_epochs = setting["generate_plot_epochs"]
        # Path for samples
        self.pth_samples = setting["pth_samples"]
        # Path for saving plots
        self.pth_plots = setting["pth_plots"]
        # Path for saving checkpoints
        self.pth_checkpoints = setting["pth_checkpoints"]

        # Initialize the BCEWithLogitsLoss function
        # Using this loss function doesn't require to define a sigmoid function
        # in the discriminator network as it is included
        self.criterion = nn.BCEWithLogitsLoss()

    #############################################################################################################
    # METHODS:

    # custom weights initialization
    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # Function for naming plots and figures (datetime in filename)
    def _get_filename(self, name, extension):
        # Datetime for saved files
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M")
        # Generate filename
        filename = f'{current_datetime}_{name}{extension}'
        return filename

    # Prints plot with generator and discriminator losses after training
    def _plot_training_losses(
            self, 
            history,
            plot_path,
            epoch, 
            show_plot=True, 
            save_plot=True
            ):
        # losses versus training iterations
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(history["G_loss"], label="G loss")
        plt.plot(history["D_loss"], label="D loss")
        plt.plot(history["D_x"], label="D(x)")
        plt.plot(history["D_G_z1"], label="D(G(x))_1")
        plt.plot(history["D_G_z2"], label="D(G(x))_2")                 
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        # Save plot
        if(save_plot):
            filename = self._get_filename(f"Losses_epoch_{epoch}", ".png")
            plt.savefig(str(plot_path) + '/' + filename, bbox_inches='tight') 
        # Show and save plot
        if(show_plot):
            plt.show()  

    # Function saves weights of a given model
    def _save_weights(self, model, epoch, checkpoint_pth):
        filename = self._get_filename(f"gen_checkpoint_epoch_{epoch}", ".model")
        torch.save(model.state_dict(), str(checkpoint_pth) + '/' + filename)

    # Plots a grid of images from a given tensor.
    # The function first scales the image tensor to the range [0, 1]. It then detaches the tensor from the computation
    # graph and moves it to the CPU if it's not already there. After that, it creates a grid of images and plots the grid.
    # Args:
    # image_tensor (torch.Tensor): A 4D tensor containing the images. The tensor is expected to be in the shape (batch_size, channels, height, width).
    # num_images (int, optional): The number of images to include in the grid. Default is 25.
    # nrow (int, optional): Number of images displayed in each row of the grid. The final grid size is (num_images // nrow, nrow). Default is 5.
    # show (bool, optional): Determines if the plot should be shown. Default is True.
    # Returns: None. The function outputs a plot of a grid of images.
    def _plot_images_from_tensor(self, image_tensor, pth_samples, step, num_images=2, nrow=2, show=False):
        # Normalize the image tensor to [0, 1]
        image_tensor = (image_tensor + 1) / 2
        # Detach the tensor from its computation graph and move it to the CPU
        image_unflat = image_tensor.detach().cpu()
        # Create a grid of images using the make_grid function from torchvision.utils
        image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
        plt.figure(figsize=(16, 8))
        # Plot the grid of images
        # The permute() function is used to rearrange the dimensions of the grid for plotting
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        # Save images
        plt.savefig(f"{pth_samples}sample_img_epoch_{step}")
        # Show the plot if the 'show' parameter is True
        if show:
            plt.show()
        else:
        # Close plot
            plt.close()

    # Create noise vector(s) for the generator
    # Depending on the generator architecture, the vector needs to come in two different shapes
    def _create_noise(self, batch_size, latent_vector_size, shape="2D"):
        if(shape=="4D"):
            return torch.randn(batch_size, latent_vector_size, 1, 1, device=self.device)
        elif(shape=="2D"):
            return torch.randn(batch_size, latent_vector_size, device=self.device)
        else:
            return False

    #############################################################################################################
    # CALL:

    def __call__(self):

        #################
        # DISCRIMINATOR #
        #################

        # Create the discriminator network
        netD = Discriminator().to(self.device)
        # Handle multi-GPU if desired
        if(self.device.type == 'cuda') and (self.num_gpu > 1):
            netG = nn.DataParallel(netG, list(range(self.num_gpu)))
        # print(netD)
        # Apply the weights_init function to randomly initialize all weights
        netD.apply(self._weights_init)

        #############
        # GENERATOR #
        #############

        # Create the generator network
        netG = Generator().to(self.device)
        # Handle multi-GPU if desired
        if (self.device.type == 'cuda') and (self.num_gpu > 1):
            netG = nn.DataParallel(netG, list(range(self.num_gpu)))
        # print(netG)  
        # Apply the weights_init function to randomly initialize all weights
        netG.apply(self._weights_init)

        ####################
        # LOSS / OPTIMIZER #
        ####################

        # Create batch of latent vectors that we will use to visualize the progression of the generator
        fixed_noise = self._create_noise(self.num_samples, self.latent_vector_size, shape="2D")

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(netD.parameters(), lr=self.disc_learning_rate, betas=(self.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=self.gen_learning_rate, betas=(self.beta1, 0.999)) 

        #######################
        # START TRAINING LOOP #
        #######################

        # Based on: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

        # Metrics for plot history:
        # - Loss_G: Generator loss calculated as log(D(G(z)))log(D(G(z)))   
        # - Loss_D: Discriminator loss calculated as the sum of losses for the all real and all fake batches (log(D(x))+log(1−D(G(z)))log(D(x))+log(1−D(G(z))))
        # - D(x): The average output (across the batch) of the discriminator for the all real batch. This should start close to 1 then 
        #   theoretically converge to 0.5 when G gets better.
        # D(G(z)): average discriminator outputs for the all fake batch. The first number is before D is updated and the second number is after D is updated. 
        #   These numbers should start near 0 and converge to 0.5 as G gets better.
        history = {"G_loss": [], "D_loss": [], "D_x": [], "D_G_z1": [], "D_G_z2": []}

        # Lists to keep track of progress
        iters = 0 
        # img_list = []   

        print("\nStarting training loop...")

        # For each epoch
        for epoch in range(self.num_epochs):

            print(f'>> Epoch [{epoch+1}/{self.num_epochs}]:')

            # For each batch in the dataloader
            with tqdm(self.dataloader, unit="batch") as tepoch:
                for i, data in enumerate(tepoch, 0):

                    ###############################################################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) #
                    ###############################################################

                    ## Train with all-real batch
                    netD.zero_grad()
                    # Format batch
                    real_cpu = data[0].to(self.device)
                    batch_size = real_cpu.size(0)
                    label = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device)
                    # Forward pass real batch through D
                    output = netD(real_cpu).view(-1)
                    # Calculate loss on all-real batch
                    errD_real = self.criterion(output, label)
                    # Calculate gradients for D in backward pass
                    errD_real.backward()
                    D_x = output.mean().item()
                    # Add D(x) value to history
                    history["D_x"].append(D_x)

                    ## Train with all-fake batch
                    # Generate batch of latent vectors
                    # noise = torch.randn(batch_size, self.latent_vector_size, 1, 1, device=self.device)
                    noise = self._create_noise(batch_size, self.latent_vector_size, shape="2D")

                    # Generate fake image batch with G
                    fake = netG(noise)
                    label.fill_(fake_label)
                    # Classify all fake batch with D
                    output = netD(fake.detach()).view(-1)
                    # Calculate D's loss on the all-fake batch
                    errD_fake = self.criterion(output, label)
                    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                    errD_fake.backward()
                    D_G_z1 = output.mean().item()
                    # Add D(G(z)) value to history
                    history["D_G_z1"].append(D_G_z1)
                    # Compute error of D as sum over the fake and the real batches
                    errD = errD_real + errD_fake
                    # Update D
                    optimizerD.step()

                    ###############################################
                    # (2) Update G network: maximize log(D(G(z))) #
                    ###############################################

                    netG.zero_grad()
                    label.fill_(real_label)  # fake labels are real for generator cost
                    # Since we just updated D, perform another forward pass of all-fake batch through D
                    output = netD(fake).view(-1)
                    # Calculate G's loss based on this output
                    errG = self.criterion(output, label)
                    # Calculate gradients for G
                    errG.backward()
                    D_G_z2 = output.mean().item()
                    # Add D(G(z)) value to history
                    history["D_G_z2"].append(D_G_z2)
                    # Update G
                    optimizerG.step()

                    # Save Losses for plotting later
                    history["G_loss"].append(errG.item())
                    history["D_loss"].append(errD.item())

                    # Check how the generator is doing by saving G's output on fixed_noise
                    ################################################################################################################################
                    # if(iters % 500 == 0) or ((epoch == self.num_epochs-1) and (i == len(self.dataloader)-1)):
                    #     with torch.no_grad():
                    #         fake = netG(fixed_noise).detach().cpu()
                    #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    ################################################################################################################################

                    iters += 1

                ######################
                # SAVE TRAINING DATA #
                ######################

                # Generate fake samples
                if((epoch + 1) % self.generate_samples_epochs == 0 and self.generate_samples):
                    self._plot_images_from_tensor(fake, self.pth_samples, (epoch + 1))

                # Save checkpoints
                if((epoch + 1) % self.generate_checkpoints_epochs == 0):
                    self._save_weights(netG, (epoch + 1), self.pth_checkpoints)   

                # Prints plot with generator and discriminator losses
                if((epoch + 1) % self.generate_plot_epochs == 0):
                    self._plot_training_losses(
                        history,
                        self.pth_plots,
                        (epoch + 1),
                        show_plot=False, 
                        save_plot=True)
                    
            # Print losses and other parameters
            print(f'Loss_D: {errD.item():.4f}, Loss_G: {errG.item():.4f}, D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

        print("Training finished!")


        

"""
# Visualizes training progression as animated GIF
def _print_training_progress(self, img_list):
    # Visualization of G's progression
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml()) 

# Prints examples of real and fake images after training
def _print_training_examples(self, img_list):
    # Real Images vs. Fake Images
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(self.dataloader))
    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:self.num_samples], padding=5, normalize=True).cpu(),(1,2,0)))
    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.tight_layout()
    plt.show()  
"""