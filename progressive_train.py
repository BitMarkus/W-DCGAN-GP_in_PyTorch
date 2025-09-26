
class ProgressiveTrainManager:

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, total_epochs, progressive_phases, progressive_params):

        self.total_epochs = total_epochs
        self.phases = progressive_phases
        self.params = progressive_params
        
        # Convert percentage phases to epoch numbers
        self.early_end = int(total_epochs * progressive_phases["early_phase_end"])
        self.mid_end = int(total_epochs * progressive_phases["mid_phase_end"])

    #############################################################################################################
    # METHODS:
        
    # Determine which phase it is
    def get_phase(self, current_epoch):
        if current_epoch < self.early_end:
            return "early"
        elif current_epoch < self.mid_end:
            return "mid" 
        else:
            return "late"
    
    # Get parameters for a specific type in the current phase
    def get_current_params(self, current_epoch, param_type):
        phase = self.get_phase(current_epoch)
        
        if param_type in self.params and phase in self.params[param_type]:
            return self.params[param_type][phase]
        else:
            return None
    
    # Apply progressive label smoothing
    def apply_label_smoothing_progression(self, train_instance, current_epoch):
        current_smoothing = self.get_current_params(current_epoch, "label_smoothing")
        
        if current_smoothing:
            # Update label smoothing parameters
            train_instance.smooth_real = current_smoothing["smooth_real"]
            train_instance.smooth_fake = current_smoothing["smooth_fake"]
            
            # Enable/disable based on progressive settings
            train_instance.use_label_smoothing = current_smoothing["enabled"]
                
            return current_smoothing
        return None
    
    # Apply progressive critic training ratio
    def apply_critic_training_ratio_progression(self, train_instance, current_epoch):
        current_ratio = self.get_current_params(current_epoch, "critic_training_ratio")
        
        if current_ratio:
            # Update critic training ratio
            train_instance.num_crit_training = current_ratio["num_crit_training"]
            return current_ratio
        return None
    
    # Apply progressive noise injection
    def apply_noise_injection_progression(self, train_instance, current_epoch):
        # Only apply if progressive training is enabled
        if not train_instance.use_progressive_params:
            return None
            
        current_noise = self.get_current_params(current_epoch, "noise_injection")
        
        if current_noise:
            # COMPLETE CONTROL: Override both enabled state AND provide noise_std
            train_instance.use_noise_injection = current_noise["enabled"]
            
            if current_noise["enabled"]:
                # When enabled: set the noise_std that _train_critic will use
                train_instance.current_noise_std = current_noise["noise_std"]
            else:
                # When disabled: ensure _train_critic sees use_noise_injection = False
                # No need to set current_noise_std since it won't be used
                pass
                
            return current_noise
        
        return None
    
    # Apply progressive gradient penalty weight (unchanged)
    def apply_gradient_penalty_progression(self, train_instance, current_epoch):
        current_gp = self.get_current_params(current_epoch, "gradient_penalty_weight")
        
        if current_gp:
            train_instance.gp_weight = current_gp["weight"]
            return current_gp
        return None
    
    # Apply all progressive parameters
    def apply_progressive_params(self, train_instance, current_epoch):
        if not hasattr(train_instance, 'use_progressive_params') or not train_instance.use_progressive_params:
            return
        
        applied_params = {}
        
        # Apply critic training ratio progression
        ratio_params = self.apply_critic_training_ratio_progression(train_instance, current_epoch)
        if ratio_params:
            applied_params["critic_training_ratio"] = ratio_params

        # Gradient penalty progression
        gp_params = self.apply_gradient_penalty_progression(train_instance, current_epoch)
        if gp_params:
            applied_params["gradient_penalty_weight"] = gp_params

        # Apply label smoothing progression
        smoothing_params = self.apply_label_smoothing_progression(train_instance, current_epoch)
        if smoothing_params:
            applied_params["label_smoothing"] = smoothing_params

        # Apply noise injection progression
        noise_params = self.apply_noise_injection_progression(train_instance, current_epoch)
        if noise_params:
            applied_params["noise_injection"] = noise_params
        
        # Print phase info on important transitions
        self._print_phase_info(train_instance, current_epoch, applied_params)
   
    # Print phase information
    def _print_phase_info(self, train_instance, current_epoch, applied_params):
        phase = self.get_phase(current_epoch)
        
        # Print on phase changes or every 25 epochs
        if (current_epoch == 0 or 
            current_epoch == self.early_end or 
            current_epoch == self.mid_end or 
            current_epoch % 25 == 0):
            
            print(f"> Progressive Training - {phase.upper()} Phase:")

            if "critic_training_ratio" in applied_params:
                ratio = applied_params["critic_training_ratio"]
                print(f"  - Critic Training Ratio: {ratio['num_crit_training']}x")

            if "gradient_penalty_weight" in applied_params:
                gp = applied_params["gradient_penalty_weight"]
                print(f"  - Gradient Penalty Weight: {gp['weight']}")
            
            if "label_smoothing" in applied_params:
                smoothing = applied_params["label_smoothing"]
                smoothing_status = "ON" if train_instance.use_label_smoothing else "OFF"
                if train_instance.use_label_smoothing:
                    print(f"  - Label Smoothing: {smoothing_status} (Real={smoothing['smooth_real']}, Fake={smoothing['smooth_fake']})")
                else:
                    print(f"  - Label Smoothing: {smoothing_status}")

            if "noise_injection" in applied_params:
                noise = applied_params["noise_injection"]
                noise_status = "ON" if train_instance.use_noise_injection else "OFF"
                if train_instance.use_noise_injection:
                    print(f"  - Noise Injection: {noise_status} (σ={noise['noise_std']:.3f})")
                else:
                    print(f"  - Noise Injection: {noise_status}")


