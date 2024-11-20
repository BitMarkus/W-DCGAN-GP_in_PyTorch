# Source for W-GAN: https://agustinus.kristia.de/blog/wasserstein-gan/   
# https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/

import torch
import torch.optim as optim
# https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
# Own modules
from train import Train
from settings import setting

class Train_WGAN(Train):

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, device, dataloader):
        super().__init__(device, dataloader)

        # W-GAN specific optimizers   
        self.optimizerD = optim.RMSprop(self.netD.parameters(), lr=self.disc_learning_rate)
        self.optimizerG = optim.RMSprop(self.netG.parameters(), lr=self.gen_learning_rate)

    #############################################################################################################
    # METHODS:

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
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        # Save plot
        if(save_plot):
            filename = self._get_filename(f"Losses_epoch_{epoch}", ".png")
            plt.savefig(str(plot_path) + '/' + filename, bbox_inches='tight') 
            plt.close()
        # Show plot
        if(show_plot):
            plt.show() 

    def _train_discriminator(self, real_images, fake_images):
        # Reset gradients
        self.netD.zero_grad()
        # Send real and fake batch through discriminator
        D_real = self.netD(real_images)
        D_fake = self.netD(fake_images.detach())
        # Calculate Wasserstein loss
        D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
        # Calculate gradients for D
        D_loss.backward()
        # Update G
        self.optimizerD.step()
        # Weight clipping -> WGAN specific
        for p in self.netD.parameters():
            p.data.clamp_(-0.01, 0.01)       

        return D_loss.item()

    def _train_generator(self, fake_images):
        # Reset gradients
        self.netG.zero_grad()
        # Send fake batch through D
        D_fake = self.netD(fake_images)
        # Calculate Wasserstein loss
        G_loss = -torch.mean(D_fake)
        # Calculate gradients for G
        G_loss.backward()
        # Update G
        self.optimizerG.step()

        return G_loss.item() 
    
    #############################################################################################################
    # TRAINING LOOP:

    def train(self):

        #######################
        # START TRAINING LOOP #
        #######################

        # Based on: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        # and on: https://neptune.ai/blog/gan-failure-modes
        # and on: https://agustinus.kristia.de/blog/wasserstein-gan/  

        # Metrics for plot history:
        history = {"G_loss": [], "D_loss": []}

        print("\nStarting training loop...")

        # For each epoch
        for epoch in range(self.num_epochs):

            print(f'>> Epoch [{epoch+1}/{self.num_epochs}]:')

            # For each batch in the dataloader
            with tqdm(self.dataloader, unit="batch") as tepoch:
                for step, data in enumerate(tepoch, 0):

                    ############################################
                    # Generate batches of real and fake images #
                    ############################################

                    # Get a batch of real images
                    real_images = data[0].to(self.device)
                    # Get batch size from actual batch (last batch can be smaller!)
                    batch_size = real_images.size(0)
                    # Generate fake image batch with G
                    fake_images = self.create_generator_samples(batch_size)

                    #######################
                    # Train Discriminator #
                    #######################

                    # Discriminator is supposed to be trained at least 5x more that the generator in WGAN
                    # This might not be the case here, as the generator network is about 9x bigger
                    # than the discriminator network and thus needs more training

                    D_loss = self._train_discriminator(real_images, fake_images)

                    ###################
                    # Train Generator #
                    ###################

                    G_loss = self._train_generator(fake_images)

                ######################
                # SAVE TRAINING DATA #
                ######################

                # Generate fake samples
                if((epoch + 1) % self.generate_samples_epochs == 0 and self.generate_samples):
                    self._plot_images_from_tensor(fake_images, self.pth_samples, (epoch + 1))

                # Save checkpoints
                if((epoch + 1) % self.generate_checkpoints_epochs == 0):
                    self._save_weights(self.netG, (epoch + 1), self.pth_checkpoints)   

                # Prints plot with generator and discriminator losses
                if((epoch + 1) % self.generate_plot_epochs == 0):
                    self._plot_training_losses(
                        history,
                        self.pth_plots,
                        (epoch + 1),
                        show_plot=False, 
                        save_plot=True)
                    
            ##################
            # UPDATE HISTORY #
            ##################

            history["G_loss"].append(G_loss)
            history["D_loss"].append(D_loss) 
            # Print history
            print(f'Loss_D: {D_loss:.4f}, Loss_G: {G_loss:.4f}')
            
        print("Training finished!")