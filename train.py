import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid
from datetime import datetime
from tqdm import tqdm
# https://stackoverflow.com/questions/52839758/matplotlib-and-runtimeerror-main-thread-is-not-in-main-loop
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Own modules
from settings import setting
from critic import Critic
from generator import Generator

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
        # Parameters for optimizer

        self.adam_beta_1 = setting["adam_beta_1"]
        self.adam_beta_2 = setting["adam_beta_2"]
        # Training critic more that generator
        self.num_crit_training = setting["num_crit_training"]
        # Gradient penalty
        self.gp_weight = setting["gradient_penalty_weight"]

        # Learning rate (scheduler) parameters
        self.use_lr_scheduler = setting["use_lr_scheduler"]
        # Select type of scheduler
        self.use_cosine_ann = setting["use_cosine_ann"]
        self.use_cosine_ann_wr = setting["use_cosine_ann_wr"]
        # Generator:
        self.gen_learning_rate = setting["gen_learning_rate"]
        self.gen_lrs_t_0 = setting["gen_lrs_t_0"]
        self.gen_lrs_eta_min = setting["gen_lrs_eta_min"]
        self.gen_lrs_t_mult = setting["gen_lrs_t_mult"]
        # Critic:
        self.crit_learning_rate = setting["crit_learning_rate"]
        self.crit_lrs_t_0 = setting["crit_lrs_t_0"]
        self.crit_lrs_eta_min = setting["crit_lrs_eta_min"]
        self.crit_lrs_t_mult = setting["crit_lrs_t_mult"]

        # Sample generation during training:
        self.generate_samples = setting["generate_samples"]
        self.num_sample_images = setting["num_sample_images"]
        self.num_rows_sample_images = setting["num_rows_sample_images"]
        self.generate_samples_epochs = setting["generate_samples_epochs"]
        # Checkpoint generation during training
        self.generate_checkpoints = setting["generate_checkpoints"]
        self.generate_checkpoints_epochs = setting["generate_checkpoints_epochs"]
        # Metrics plot generation during training
        self.generate_plots = setting["generate_plots"]
        self.generate_plot_epochs = setting["generate_plot_epochs"]

        # Path for samples
        self.pth_samples = setting["pth_samples"]
        # Path for saving plots
        self.pth_plots = setting["pth_plots"]
        # Path for saving checkpoints
        self.pth_checkpoints = setting["pth_checkpoints"]

        ##########
        # Critic #
        ##########

        self.netC = Critic().to(self.device)
        # Handle multi-GPU if desired
        if(self.device.type == 'cuda') and (self.num_gpu > 1):
            self.netG = nn.DataParallel(self.netG, list(range(self.num_gpu)))
        # Apply the weights_init function to randomly initialize all weights
        self.netC.apply(self._weights_init)
        # Setup Adam optimizer
        self.optimizerC = optim.Adam(
            self.netC.parameters(), 
            lr=self.crit_learning_rate, 
            betas=(self.adam_beta_1, self.adam_beta_2)
        )

        # Learning rate scheduler
        # CosineAnnealingWarmRestarts:
        if(self.use_cosine_ann_wr):
            self.schedulerC = CosineAnnealingWarmRestarts(
                self.optimizerC,
                T_0=self.crit_lrs_t_0,           
                T_mult=self.crit_lrs_t_mult,
                eta_min=self.crit_lrs_eta_min
            )
        # CosineAnnealing:
        elif(self.use_cosine_ann):
            self.schedulerC = CosineAnnealingLR(
                self.optimizerC, 
                T_max=self.num_epochs
            )
        # No LRS:
        else:
            self.use_lr_scheduler = False

        #############
        # Generator #
        #############       

        # Create the generator network
        self.netG = Generator().to(self.device)
        # Handle multi-GPU if desired
        if (self.device.type == 'cuda') and (self.num_gpu > 1):
            self.netG = nn.DataParallel(self.netG, list(range(self.num_gpu)))
        # Apply the weights_init function to randomly initialize all weights
        self.netG.apply(self._weights_init) 
        # Setup Adam optimizer       
        self.optimizerG = optim.Adam(
            self.netG.parameters(), 
            lr=self.gen_learning_rate,
            betas=(self.adam_beta_1, self.adam_beta_2)
        )

        # Learning rate scheduler
        # CosineAnnealingWarmRestarts:
        if(self.use_cosine_ann_wr):
            self.schedulerG = CosineAnnealingWarmRestarts(
                self.optimizerG,
                T_0=self.gen_lrs_t_0,                    
                T_mult=self.gen_lrs_t_mult,                   
                eta_min=self.gen_lrs_eta_min
            )
        # CosineAnnealing:
        elif(self.use_cosine_ann):
            self.schedulerG = CosineAnnealingLR(
                self.optimizerG, 
                T_max=self.num_epochs
            )
        # No LRS:
        else:
            self.use_lr_scheduler = False

    #############################################################################################################
    # METHODS:

    # Custom weights initialization
    # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#weight-initialization
    # https://github.com/martinarjovsky/WassersteinGAN/blob/f7a01e82007ea408647c451b9e1c8f1932a3db67/main.py#L108
    # https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
    # Initialization function according to DeepSeek:
    def _weights_init(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            # Xavier/Glorot initialization for Conv2d, ConvTranspose2d, and Linear layers
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                # Initialize biases to zero
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            # Initialize scale (weight) to 1 and shift (bias) to 0 for normalization layers
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)   

    # Function for naming plots and figures (datetime in filename)
    def _get_filename(self, name, extension):
        # Datetime for saved files
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M")
        # Generate filename
        filename = f'{current_datetime}_{name}{extension}'
        return filename

    # Plots accuracy, loss, and learning rate after training
    def _plot_metrics(self, 
                    history, 
                    epoch, 
                    show_plot=False, 
                    save_plot=True):
        # Number of epochs
        epochs_range = range(1, len(history["G_loss"]) + 1)
        # Draw plots (b, h)
        plt.figure(figsize=(10, 10))
        # INFO: plt.subplot(rows, colums, plot position)
        # Generator and Critic loss
        plt.subplot(2, 2, 1)
        # plt.ylim(-400, 400)
        plt.plot(epochs_range, history["G_loss"], label='G loss', color='green')
        plt.plot(epochs_range, history["C_loss"], label='C loss', color='red')
        plt.legend(loc='upper right')
        plt.title(f'Loss')
        # Learning rates
        plt.subplot(2, 2, 2)
        # convert y-axis to Logarithmic scale
        # plt.yscale("log")
        plt.plot(epochs_range, history["G_lr"], label='G lr', color='green')
        plt.plot(epochs_range, history["C_lr"], label='C lr', color='red')
        plt.legend(loc='upper right')
        plt.title('Learning Rate')  
        # Gradient penalty
        plt.subplot(2, 2, 3)
        plt.plot(epochs_range, history["Grad_pen"], label='Grad pen', color='blue')
        plt.title('Gradiant penalty')  
        # Gradient norm
        plt.subplot(2, 2, 4)
        plt.plot(epochs_range, history["Grad_norm"], label='Grad norm', color='blue')
        plt.title('Gradient norm')      
        plt.tight_layout()
        # Save plot
        if(save_plot):
            filename = self._get_filename(f"Metrics_epoch_{epoch}", ".png")
            plt.savefig(self.pth_plots + '/' + filename, bbox_inches='tight') 
            plt.close()
        # Show plot
        if(show_plot):
            plt.show() 
        print(f"Metrics plot for epoch {epoch} was succsessfully saved in {self.pth_plots}")
        return True    

    # Function saves weights of a given model
    def _save_weights(self, model, epoch):
        filename = self._get_filename(f"gen_checkpoint_epoch_{epoch}", ".model")
        torch.save(model.state_dict(), self.pth_checkpoints + '/' + filename)
        print(f"Checkpoint {filename} for epoch {epoch} was succsessfully saved in {self.pth_checkpoints}")
        return True  

    # Plots a grid of sample images during training
    def _plot_sample_images(self, 
                            image_tensor, 
                            pth_samples, 
                            epoch, 
                            show_plot=False, 
                            save_plot=True):
        # Normalize the image tensor to [0, 1]
        image_tensor = (image_tensor + 1) / 2
        # Detach the tensor from its computation graph and move it to the CPU
        image_unflat = image_tensor.detach().cpu()
        # Create a grid of images using the make_grid function from torchvision.utils
        image_grid = make_grid(image_unflat[:self.num_sample_images], nrow=self.num_rows_sample_images)
        plt.figure(figsize=(15, 15))
        # Plot the grid of images
        # The permute() function is used to rearrange the dimensions of the grid for plotting
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.tight_layout()
        # Save plot
        if(save_plot):
            plt.savefig(f"{pth_samples}sample_img_epoch_{epoch}")
            plt.close()
        # Show plot
        if(show_plot):
            plt.show() 
        print(f"Sample images for epoch {epoch} were succsessfully saved in {self.pth_samples}")
        return True  

    # Create noise vector(s) for the generator
    # Depending on the generator architecture, the vector needs to come in two different shapes
    def _create_noise(self, batch_size, latent_vector_size):
            return torch.randn(batch_size, latent_vector_size, device=self.device)
        
    # Function for gradient penalty
    # Source: https://github.com/EmilienDupont/wgan-gp/blob/master/training.py    
    # Corrected and improved according to DeepSeek:
    def _compute_gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size(0)
        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data

        # Disable critic's BN/IN during GP calculation (if any)
        self.netC.eval()    
        # Compute critic scores for interpolated data
        prob_interpolated = self.netC(interpolated)
        self.netC.train()

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(prob_interpolated),
            create_graph=True,  # Needed for higher-order derivatives
            retain_graph=True,  # Needed!
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
        
    def _train_critic(self, real_images, fake_images):
        # Reset gradients
        self.netC.zero_grad()
        # Send real and fake batch through critic
        C_real = self.netC(real_images).mean()
        C_fake = self.netC(fake_images.detach()).mean()
        # Claculate gradient penalty
        gradient_penalty, grad_norm = self._compute_gradient_penalty(real_images, fake_images)
        # Calculate Wasserstein loss
        C_loss = C_fake - C_real + gradient_penalty
        # Backward pass with retained graph
        C_loss.backward(retain_graph=True)
        # Update critic
        self.optimizerC.step()
        
        return C_loss.item(), gradient_penalty.item(), grad_norm
    
    def _train_critic_with_noise(self, real_images, fake_images, epoch):
        # Reset gradients
        self.netC.zero_grad()
        # Linear noise:
        # noise_std = 0.02
        # Add adaptive noise: Start with 2% noise (adjust based on your data scale)
        # Reduce noise over time (e.g., linear decay):
        noise_std = max(0.01, 0.05 * (1 - epoch / self.num_epochs))
        # Add Gaussian noise to real and fake images
        real_images_noisy = real_images + noise_std * torch.randn_like(real_images)
        fake_images_noisy = fake_images.detach() + noise_std * torch.randn_like(fake_images)
        # Send real and fake batch with noise through critic
        C_real = self.netC(real_images_noisy).mean()
        C_fake = self.netC(fake_images_noisy).mean()
        # Claculate gradient penalty
        # Note: Use original (non-noisy) images for GP to avoid noise interference
        gradient_penalty, grad_norm = self._compute_gradient_penalty(real_images, fake_images)
        # Calculate Wasserstein loss-
        C_loss = C_fake - C_real + gradient_penalty
        # Backward pass with retained graph
        C_loss.backward(retain_graph=True)
        # Update critic
        self.optimizerC.step()
        
        return C_loss.item(), gradient_penalty.item(), grad_norm
    
    def _train_generator(self, fake_images):
        # Reset gradients
        self.netG.zero_grad()
        # Send fake batch through Critic
        C_fake = self.netC(fake_images).mean()
        # Calculate Wasserstein loss: Generator tries to maximize C_fake
        G_loss = -C_fake
        # Calculate gradients for G
        G_loss.backward()
        # Update G
        self.optimizerG.step()

        return G_loss.item()
    
    def create_generator_samples(self, num_samples):
        # Generate batch of latent vectors
        noise = self._create_noise(num_samples, self.latent_vector_size)
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
        history = {"G_loss": [], "C_loss": [], "Grad_pen": [], "Grad_norm": [], "G_lr": [], "C_lr": []}

        print("\nStarting training loop...")

        # For each epoch
        for epoch in range(self.num_epochs):

            print(f'\n>> Epoch [{epoch+1}/{self.num_epochs}]:')

            # For each batch in the dataloader
            with tqdm(self.dataloader, unit="batch") as tepoch:
                for step, data in enumerate(tepoch, 0):

                    real_images = data[0].to(self.device)
                    batch_size = real_images.size(0)

                    ################
                    # Train Critic #
                    ################

                    # critic is supposed to be trained at least 5x more that the generator in WGAN
                    fake_images = self.create_generator_samples(batch_size)
                    for _ in range(self.num_crit_training):
                        # Critic training without noise:
                        # C_loss, Grad_pen, Grad_norm = self._train_critic(real_images, fake_images)
                        # Critic training with noise:
                        C_loss, Grad_pen, Grad_norm = self._train_critic_with_noise(real_images, fake_images, epoch)

                    ###################
                    # Train Generator #
                    ###################

                    # Generate fake image batch with G
                    fake_images = self.create_generator_samples(batch_size)
                    # Train Generator
                    G_loss = self._train_generator(fake_images)

                ##################################
                # Update learning rate scheduler #
                ##################################

                # Use lr scheduler
                if(self.use_lr_scheduler):
                    self.schedulerC.step()
                    self.schedulerG.step()

                # Get last learning rate value
                C_lr = self.schedulerC.get_last_lr()[0]
                G_lr = self.schedulerG.get_last_lr()[0]

                ######################
                # SAVE TRAINING DATA #
                ######################

                # Generate fake samples
                if((epoch + 1) % self.generate_samples_epochs == 0 and self.generate_samples):
                    self._plot_sample_images(fake_images, self.pth_samples, (epoch + 1))

                # Save checkpoints
                if((epoch + 1) % self.generate_checkpoints_epochs == 0 and self.generate_checkpoints):
                    self._save_weights(self.netG, (epoch + 1))   

                # Plot training metrics
                if((epoch + 1) % self.generate_plot_epochs == 0 and self.generate_plots):
                    self._plot_metrics(
                            history,
                            (epoch + 1),
                            show_plot=False, 
                            save_plot=True)
                    
            ##################
            # UPDATE HISTORY #
            ##################

            # TO DO:
            # Monitor and plot the Wasserstein Distance:
            # wasserstein_distance = history["C_loss"] - history["G_loss"]

            history["G_loss"].append(G_loss)
            history["C_loss"].append(C_loss) 
            history["Grad_pen"].append(Grad_pen)
            history["Grad_norm"].append(Grad_norm)
            history["G_lr"].append(G_lr)
            history["C_lr"].append(C_lr)
            # Print history
            print(f'C_Loss: {C_loss:.4f}, G_Loss: {G_loss:.4f}, Grad_pen: {Grad_pen:.4f}, Grad_norm: {Grad_norm:.4f}, C_LR: {C_lr:.6f}, G_LR: {G_lr:.6f}')
            
        print("Training finished!")

