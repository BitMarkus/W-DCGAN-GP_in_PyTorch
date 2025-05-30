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
        # Gradient penalty
        self.gp_weight = setting["gradient_penalty_weight"]
        # Parameters for optimizer
        self.adam_beta_1 = setting["adam_beta_1"]
        self.adam_beta_2 = setting["adam_beta_2"]

        # W-GAN specific optimizers  
        # RMSprop: 
        # self.optimizerD = optim.RMSprop(self.netD.parameters(), lr=self.disc_learning_rate)
        # self.optimizerG = optim.RMSprop(self.netG.parameters(), lr=self.gen_learning_rate)
        # ADAM:
        self.optimizerD = optim.Adam(
            self.netD.parameters(), 
            lr=self.disc_learning_rate, 
            betas=(self.adam_beta_1, self.adam_beta_2)
        )
        self.optimizerG = optim.Adam(
            self.netG.parameters(), 
            lr=self.gen_learning_rate,
            betas=(self.adam_beta_1, self.adam_beta_2)
        )

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
        plt.plot(history["Grad_pen"], label="Grad pen")  
        plt.plot(history["Grad_norm"], label="Grad norm")           
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.ylim(-400, 400)
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

    # Function for gradient penalty
    # Source: https://github.com/EmilienDupont/wgan-gp/blob/master/training.py    
    # Corrected and improved according to DeepSeek:
    def _compute_gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size(0)
        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data

        # Disable critic's BN/IN during GP calculation (if any)
        self.netD.eval()    
        # Compute critic scores for interpolated data
        prob_interpolated = self.netD(interpolated)
        self.netD.train()

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(prob_interpolated),
            create_graph=True,  # Needed for higher-order derivatives
            # retain_graph=False,  # No need to retain
            retain_graph=True,  # Optional (only if you reuse the graph later)
        )[0]
        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        # Calculate gradient norm
        # This value should be close to 1!
        gradients_norm = gradients.norm(2, dim=1)    
        # Calculate gradient penalty
        gradient_penalty = self.gp_weight * ((gradients_norm - 1) ** 2).mean()
        # assert not torch.isnan(gradient_penalty)
        
        return gradient_penalty, gradients_norm.mean().item()

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
        # self._gradient_clipping(self.netD)   
        for p in self.netD.parameters():
            p.data.clamp_(-0.01, 0.01)

        return D_loss.item()

    def _train_discriminator_grad_pen(self, real_images, fake_images):
        # Reset gradients
        self.netD.zero_grad()
        # Send real and fake batch through discriminator
        D_real = self.netD(real_images).mean()
        D_fake = self.netD(fake_images.detach()).mean()
        # Claculate gradient penalty
        gradient_penalty, grad_norm = self._compute_gradient_penalty(real_images, fake_images)
        # Calculate Wasserstein loss
        D_loss = D_fake - D_real + gradient_penalty
        # Backward pass with retained graph
        D_loss.backward(retain_graph=True)
        # Update G
        self.optimizerD.step()
        
        return D_loss.item(), gradient_penalty.item(), grad_norm

    def _train_generator(self, fake_images):
        # Reset gradients
        self.netG.zero_grad()
        # Send fake batch through D
        D_fake = self.netD(fake_images).mean()
        # Calculate Wasserstein loss: Generator tries to maximize D_fake
        G_loss = -D_fake
        # Calculate gradients for G
        G_loss.backward()
        # Update G
        self.optimizerG.step()

        return G_loss.item()
    
    def create_generator_samples(self, num_samples):
        # Generate batch of latent vectors
        noise = self._create_noise(num_samples, self.latent_vector_size, shape="2D")
        # Generate fake image batch with G
        samples_tensors = self.netG(noise)

        return samples_tensors
    
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
        history = {"G_loss": [], "D_loss": [], "Grad_pen": [], "Grad_norm": []}

        print("\nStarting training loop...")

        # For each epoch
        for epoch in range(self.num_epochs):

            print(f'>> Epoch [{epoch+1}/{self.num_epochs}]:')

            # For each batch in the dataloader
            with tqdm(self.dataloader, unit="batch") as tepoch:
                for step, data in enumerate(tepoch, 0):

                    real_images = data[0].to(self.device)
                    batch_size = real_images.size(0)

                    #######################
                    # Train Discriminator #
                    #######################

                    # Discriminator is supposed to be trained at least 5x more that the generator in WGAN
                    fake_images = self.create_generator_samples(batch_size)
                    for _ in range(self.num_disc_training):
                        D_loss, Grad_pen, Grad_norm = self._train_discriminator_grad_pen(real_images, fake_images)

                    ###################
                    # Train Generator #
                    ###################

                    # Generate fake image batch with G
                    fake_images = self.create_generator_samples(batch_size)
                    # Train Generator
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
            history["Grad_pen"].append(Grad_pen)
            history["Grad_norm"].append(Grad_norm)
            # Print history
            print(f'Loss_D: {D_loss:.4f}, Loss_G: {G_loss:.4f}, Grad_pen: {Grad_pen:.4f}, Grad_norm: {Grad_norm:.4f}')
            
        print("Training finished!")


"""
# Function for gradient penalty
# Source: https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
def _compute_gradient_penalty(self, real_data, fake_data):
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
"""
"""
def _train_discriminator_grad_pen(self, real_images, fake_images):
    # Reset gradients
    self.netD.zero_grad()

    # Send real and fake batch through discriminator
    D_real = self.netD(real_images)

    D_fake = self.netD(fake_images.detach())
    # Claculate gradient penalty -> WGAN specific
    gradient_penalty, log_gradient_norm = self._compute_gradient_penalty_2(real_images, fake_images)
    # Calculate Wasserstein loss
    D_loss = torch.mean(D_fake) - torch.mean(D_real) + gradient_penalty
    D_loss.backward()
    # Update G
    self.optimizerD.step()

    return D_loss.item(), gradient_penalty.item(), log_gradient_norm
"""
"""
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
"""
"""
def train(self):

    #######################
    # START TRAINING LOOP #
    #######################

    # Based on: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    # and on: https://neptune.ai/blog/gan-failure-modes
    # and on: https://agustinus.kristia.de/blog/wasserstein-gan/  

    # Metrics for plot history:
    history = {"G_loss": [], "D_loss": [], "Grad_pen": [], "Grad_norm": []}

    print("\nStarting training loop...")

    # For each epoch
    for epoch in range(self.num_epochs):

        print(f'>> Epoch [{epoch+1}/{self.num_epochs}]:')

        # For each batch in the dataloader
        with tqdm(self.dataloader, unit="batch") as tepoch:
            for step, data in enumerate(tepoch, 0):

                #######################
                # Train Discriminator #
                #######################

                # Discriminator is supposed to be trained at least 5x more that the generator in WGAN
                for _ in range(self.num_disc_training):
                    # Get a batch of real images
                    real_images = data[0].to(self.device)
                    # Get batch size from actual batch (last batch can be smaller!)
                    batch_size = real_images.size(0)
                    # Generate fake image batch with G
                    fake_images = self.create_generator_samples(batch_size)
                    # Train Discriminator
                    D_loss, Grad_pen, Grad_norm = self._train_discriminator_grad_pen(real_images, fake_images)

                ###################
                # Train Generator #
                ###################

                # Generate fake image batch with G
                fake_images = self.create_generator_samples(batch_size)
                # Train Generator
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
        history["Grad_pen"].append(Grad_pen)
        history["Grad_norm"].append(Grad_norm)
        # Print history
        print(f'Loss_D: {D_loss:.4f}, Loss_G: {G_loss:.4f}, Grad_pen: {Grad_pen:.4f}, Grad_norm: {Grad_norm:.4f}')
        
    print("Training finished!")
"""