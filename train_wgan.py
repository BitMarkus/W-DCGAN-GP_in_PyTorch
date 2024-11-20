# Source for W-GAN: https://agustinus.kristia.de/blog/wasserstein-gan/   
# https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import grad
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

        # Training discriminator more that generator
        self.num_disc_training = setting["num_disc_training"]

        # Gradient penalty -> to settings once it works!!!
        self.gp_weight = 10

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

    # Function for gradient clipping
    # Source: https://antixk.github.io/blog/lipschitz-wgan/
    def _gradient_clipping(self, netD, clip_val=0.01):
        for p in netD.parameters():
            p.data.clamp_(-clip_val, clip_val)

    # Function for gradient penalty 1
    # Source: https://towardsdatascience.com/demystified-wasserstein-gan-with-gradient-penalty-ba5e9b905ead
    def _compute_gradient_penalty_1(self, real_data, fake_data):
            batch_size = real_data.size(0)
            # Sample Epsilon from uniform distribution
            eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
            eps = eps.expand_as(real_data)
            # Interpolation between real data and fake data.
            interpolation = eps * real_data + (1 - eps) * fake_data
            # get logits for interpolated images
            interp_logits = self.netD(interpolation)
            grad_outputs = torch.ones_like(interp_logits)
            # Compute Gradients
            gradients = grad(
                outputs=interp_logits,
                inputs=interpolation,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            # Compute and return Gradient Norm
            gradients = gradients.view(batch_size, -1)
            grad_norm = gradients.norm(2, 1)
            grad_norm = torch.mean((grad_norm - 1) ** 2)

            return grad_norm
    
    # Function for gradient penalty 2
    # Source: https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
    def _compute_gradient_penalty_2(self, real_data, fake_data):
        batch_size = real_data.size()[0]
        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if(self.device.type == 'cuda'):
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * fake_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if(self.device.type == 'cuda'):
            interpolated = interpolated.cuda()
        # Calculate probability of interpolated examples
        prob_interpolated = self.netD(interpolated)
        # Calculate gradients of probabilities with respect to examples
        gradients = grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.device.type == 'cuda' else torch.ones(
                            prob_interpolated.size()),
                            create_graph=True, retain_graph=True)[0]
        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        # Calculate gradient penalty
        gradient_penalty = self.gp_weight * ((gradients_norm - 1) ** 2).mean()

        return gradient_penalty

    def _train_discriminator_grad_clip(self, real_images, fake_images):
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
        self._gradient_clipping(self.netD)   

        return D_loss.item()
    
    def _train_discriminator_grad_pen(self, real_images, fake_images):
        # Reset gradients
        self.netD.zero_grad()
        # Send real and fake batch through discriminator
        D_real = self.netD(real_images)
        D_fake = self.netD(fake_images.detach())
        # Claculate gradient penalty -> WGAN specific
        gradient_penalty = self._compute_gradient_penalty_2(real_images, fake_images)
        # Calculate Wasserstein loss
        D_loss = torch.mean(D_fake) - torch.mean(D_real) + gradient_penalty
        D_loss.backward()
        # Update G
        self.optimizerD.step()

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
                    # TEST:
                    for _ in range(self.num_disc_training):
                        D_loss = self._train_discriminator_grad_pen(real_images, fake_images)

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