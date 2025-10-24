import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os
import random
from place_cell import PlaceCellModel  # Replace with your actual module name

class PlaceCellWithFinetuning(PlaceCellModel):
    """
    Extended PlaceCell model with embedding loading and finetuning capabilities.
    This class inherits from your existing PlaceCell implementation.
    """
    
    def load_embeddings(self, filepath, device=None):
        """
        Load previously learned embeddings (H vectors) from a checkpoint file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model file
        device : torch.device, optional
            Device to load the embeddings to. If None, uses the current device
        
        Returns:
        --------
        dict
            Dictionary of loaded embeddings mapped by time scale
        """
        if device is None:
            device = self.device
        
        try:
            checkpoint = torch.load(filepath, map_location=device)
            
            # Load embeddings
            loaded_embeddings = {}
            for t in checkpoint['H']:
                # Convert numpy arrays back to torch tensors
                loaded_embeddings[t] = torch.tensor(checkpoint['H'][t], device=device)
            
            print(f"Embeddings successfully loaded from {filepath}")
            print(f"Loaded embeddings for time scales: {list(loaded_embeddings.keys())}")
            return loaded_embeddings
        
        except Exception as e:
            print(f"Error loading embeddings from {filepath}: {e}")
            return {}
    
    def finetune_embeddings(self, obstacles, base_checkpoint_path, n_iter=500, learning_rate=0.01, decay_factor=0.1):
        """
        Finetune embeddings for a new environment with obstacles by:
        1. Loading existing embeddings from a checkpoint
        2. Computing transition matrices for the new environment
        3. Learning new embeddings starting from the loaded ones
        
        Parameters:
        -----------
        obstacles : list of tuples
            List of (i, j) coordinates representing obstacle locations in the new environment
        base_checkpoint_path : str
            Path to the checkpoint containing base embeddings
        n_iter : int
            Number of iterations for learning
        learning_rate : float
            Learning rate for optimization
        decay_factor : float
            Factor to decay learning rate
        
        Returns:
        --------
        dict
            Dictionary of finetuned embeddings mapped by time scale
        """
        # Step 1: Load base embeddings
        base_embeddings = self.load_embeddings(base_checkpoint_path)
        if not base_embeddings:
            print("Failed to load base embeddings. Will initialize randomly.")
        
        # Step 2: Compute transition matrices for the new environment with obstacles
        print("Building transition matrices for the new environment...")
        self.build_transition_matrix(obstacles=obstacles)
        self.compute_multi_time_transition()
        self.normalize_transition_matrices()
        
        # Step 3: Set up the obstacle handling
        obstacle_indices = set()
        if obstacles is not None:
            for obs in obstacles:
                idx = self.coords_to_idx.get(obs)
                if idx is not None:
                    obstacle_indices.add(idx)
            print(f"Identified {len(obstacle_indices)} obstacle indices to exclude")
        
        # Create a mask for non-obstacle states
        non_obstacle_mask = torch.ones(self.n_states, dtype=torch.bool, device=self.device)
        for idx in obstacle_indices:
            non_obstacle_mask[idx] = False
        
        # Initialize embeddings dictionary
        self.H = {}
        
        # Loop through each time scale and learn embeddings
        print("Learning embeddings for each time scale...")
        for t in self.t_values:
            if t == 1:  # Skip t=1 as specified
                continue
            
            print(f"Learning embeddings for t={t}...")
            Q_t = self.Qt[t]
            
            # Initialize embeddings
            # If we have base embeddings for this scale, use them as initialization
            if t in base_embeddings:
                print(f"  Initializing from base embeddings for t={t}")
                initial_values = base_embeddings[t].clone()
                
                # Ensure obstacle embeddings are zero
                for idx in obstacle_indices:
                    initial_values[idx] = 0.0
            else:
                print(f"  Initializing randomly for t={t}")
                # Initialize with small random values, zero for obstacles
                initial_values = torch.zeros((self.n_states, self.embedding_dim), device=self.device)
                # Set non-obstacle values to small random numbers
                initial_values[non_obstacle_mask] = torch.normal(
                    0, 0.001, 
                    size=(torch.sum(non_obstacle_mask).item(), self.embedding_dim), 
                    device=self.device
                )
            
            # Create parameter for optimization
            H_t = torch.nn.Parameter(initial_values)
            
            # Define optimizer
            optimizer = torch.optim.AdamW([H_t], lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=decay_factor,
                total_iters=n_iter
            )
            
            # Training loop
            pbar = tqdm(range(n_iter))
            loss_history = []
            
            for i in pbar:
                # Step 1: Create normalized embeddings for forward pass (zeros for obstacles)
                # H_normalized = torch.zeros_like(H_t)
                # H_normalized[non_obstacle_mask] = torch.nn.functional.normalize(
                #     torch.clamp(H_t[non_obstacle_mask], min=0.0), dim=1
                # )
                
                # Step 2: Compute inner products
                Q_approx = torch.matmul(H_t, H_t.t())
                
                # Step 3: Compute loss for non-obstacle transitions
                # valid_transitions = torch.outer(non_obstacle_mask, non_obstacle_mask)
                error_matrix = (Q_approx - Q_t) ** 2
                # masked_error = error_matrix[valid_transitions].reshape(-1)
                loss = torch.mean(error_matrix)
                loss_history.append(loss.item())
                
                # Step 4: Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                with torch.no_grad():
                    # Create a normalized copy for the forward pass
                    H_normalized = torch.nn.functional.normalize(
                        torch.clamp(H_t.detach(), min=0.0), dim=1)
                    
                    # Update the parameter with the normalized values
                    # This breaks the computation graph but avoids double backward
                    H_t.data.copy_(H_normalized)
                
                # Update progress bar
                if i % 10 == 0:
                    pbar.set_description(f"Loss: {loss.item():.6f}")
            
            # Store final embeddings
            with torch.no_grad():
                final_H = torch.zeros_like(H_t)
                final_H[non_obstacle_mask] = torch.nn.functional.normalize(
                    torch.clamp(H_t[non_obstacle_mask], min=0.0), dim=1
                )
                self.H[t] = final_H
            
            # Compute final loss and correlation
            Q_approx = torch.matmul(self.H[t], self.H[t].t())
            valid_transitions = torch.outer(non_obstacle_mask, non_obstacle_mask)
            final_error = ((Q_approx - Q_t) ** 2)[valid_transitions].reshape(-1)
            final_loss = torch.mean(final_error).item()
            print(f"Final loss for t={t}: {final_loss:.6f}")
            
            # Plot loss history
            dir = "loss"
            if not os.path.exists(dir):
                os.makedirs(dir)
            try:
                plt.figure(figsize=(10, 5))
                plt.plot(loss_history)
                plt.title(f"Loss History for t={t}")
                plt.xlabel("Iteration")
                plt.ylabel("MSE Loss")
                plt.yscale('log')
                plt.grid(True)
                plt.savefig(f"{dir}/finetuning_loss_history_t{t}.png")
                plt.close()
            except ImportError:
                print("Matplotlib not available for plotting loss history.")
        
        print("Finetuning complete!")
        return self.H


# Example usage:
if __name__ == "__main__":
    import torch
    import numpy as np
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import os
    
    # Parameters for model initialization
    grid_size = 40
    embedding_dim = 500
    t_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]#, 4096, 8192]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model instance
    model = PlaceCellWithFinetuning(grid_size=grid_size, embedding_dim=embedding_dim, 
                                   t_values=t_values, device=device)
    
    # Define obstacles for the environment
    # maze
    obstacles_maze = []
    # bound
    for i in [0,1,38,39]: 
        for j in range(0, 40): 
            obstacles_maze.append((i, j))
    
    for i in range(2, 38): 
        for j in [0,1,38,39]: 
            obstacles_maze.append((i, j))
    # wall
    for i in [2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37]: 
        for j in range(17, 23): 
            obstacles_maze.append((i, j))
            
    for i in range(17, 23): 
        for j in [2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37]: 
            obstacles_maze.append((i, j))
    
    # block single
    for i in range(5,8): 
        for j in [5,6,7,11,12,13,29,30,31]: 
            obstacles_maze.append((i, j))
    for i in range(8,11): 
        for j in range(26,35): 
            obstacles_maze.append((i, j))
    for i in range(11,14): 
        for j in [5,6,7,29,30,31]: 
            obstacles_maze.append((i, j))
    for i in range(23,26): 
        for j in range(26,29): 
            obstacles_maze.append((i, j))
    for i in range(26,29): 
        for j in [2,3,4,11,12,13]: 
            obstacles_maze.append((i, j))
        for j in range(26,35): 
            obstacles_maze.append((i, j))
    for i in range(32,35): 
        for j in range(26,35): 
            obstacles_maze.append((i, j))
    for i in range(35, 38): 
        for j in [11,12,13,32,33,34]: 
            obstacles_maze.append((i, j))
            
    # center block
    for i in range(11,17): 
        for j in range(11,23): 
            obstacles_maze.append((i, j))
    for i in range(17,23): 
        for j in range(11,29): 
            obstacles_maze.append((i, j))
    for i in range(23,29): 
        for j in range(17,23): 
            obstacles_maze.append((i, j))

    for i in range(17,23): 
        for j in range(8,11): 
            obstacles_maze.append((i, j))
        
    
    # Finetune the model
    obstacles = obstacles_maze
    # model.finetune_embeddings(
    #     obstacles=obstacles,
    #     base_checkpoint_path="maze/place_cell_model.pt",
    #     n_iter=100,
    #     learning_rate=0.01,
    #     decay_factor=0.1
    # )
    
    # # # Save the finetuned model
    # model.save_model("finetuned_model.pt")

    model.load_model("finetuned_model.pt")
    model.load_embeddings("finetuned_model.pt")

    # for t in model.t_values:
    #     model.plot_place_fields(t, foldername="finetuned")
    #     model.plot_place_cell_centers(t, foldername="finetuned")
        
    
    # Test path planning with the finetuned model
    start_pos = (8, 15)
    target_pos = (36, 4)
    path, selected_t_values, reached_target, dist_to_target = model.plan_path(
        start_pos=start_pos,
        target_pos=target_pos,
        obstacles=obstacles,
        visualize=True
    )    

    start_pos = (8, 15)
    target_pos = (36, 4)
    path, selected_t_values, reached_target, dist_to_target = model.plan_path(
        start_pos=start_pos,
        target_pos=target_pos,
        obstacles=obstacles,
        visualize=True
    )    

    start_pos = (14, 6)
    target_pos = (36, 4)
    path, selected_t_values, reached_target, dist_to_target = model.plan_path(
        start_pos=start_pos,
        target_pos=target_pos,
        obstacles=obstacles,
        visualize=True
    )    

    start_pos = (14, 5)
    target_pos = (36, 4)
    path, selected_t_values, reached_target, dist_to_target = model.plan_path(
        start_pos=start_pos,
        target_pos=target_pos,
        obstacles=obstacles,
        visualize=True
    )    

    start_pos = (13, 10)
    target_pos = (36, 4)
    path, selected_t_values, reached_target, dist_to_target = model.plan_path(
        start_pos=start_pos,
        target_pos=target_pos,
        obstacles=obstacles,
        visualize=True
    )   

    

    start_pos = (14.726696062173405, 6.7865129021149615)
    target_pos = (36, 4)    
    path, selected_t_values, reached_target, dist_to_target = model.plan_path(
        start_pos=start_pos,
        target_pos=target_pos,
        obstacles=obstacles,
        visualize=True
    )

    # model.load_model("finetuned_model.pt")
    start_pos = (15.492740505292383,7.4293005118015)
    target_pos = (36, 4)    
    path, selected_t_values, reached_target, dist_to_target = model.plan_path(
        start_pos=start_pos,
        target_pos=target_pos,
        obstacles=obstacles,
        visualize=True
    )

    start_pos = (16.25878494841136, 8.07208812148804)
    target_pos = (36, 4)    
    path, selected_t_values, reached_target, dist_to_target = model.plan_path(
        start_pos=start_pos,
        target_pos=target_pos,
        obstacles=obstacles,
        visualize=True
    )    