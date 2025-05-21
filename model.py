"""
Place Cell Navigation Model
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os
import random
import json
from environment import get_environment

class PlaceCellModel:
    def __init__(self, grid_size=40, embedding_dim=50, t_values=None, device=None):
        """
        Initialize the place cell model using PyTorch.
        
        Parameters:
        -----------
        grid_size : int
            Size of the grid environment (grid_size x grid_size)
        embedding_dim : int
            Dimensionality of place cell population embeddings
        t_values : list or None
            List of time scales to compute. If None, uses default scales.
        device : str or None
            PyTorch device to use ('cuda' or 'cpu'). If None, selects automatically.
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.embedding_dim = embedding_dim
        
        if t_values is None:
            # Generate powers of 2 for time scales
            self.t_values = [1]
            i = 1
            while True:
                t_raw = np.power(2, i)
                t = int(np.round(t_raw))
                
                # Make values > 1 even
                if t > 1 and t % 2 == 1:
                    t += 1
                
                # Stop if we exceed a threshold
                if t >= 4000:
                    break
                    
                # Add to list if not a duplicate
                if t > self.t_values[-1]:
                    self.t_values.append(t)
                
                i += 1
        else:
            self.t_values = t_values
            
        print(f"Time scales: {self.t_values}")
        
        # Initialize matrices and tensors
        self.P1 = None  # One-step transition matrix
        self.Pt = {}    # Multi-step transition matrices for each t
        self.Qt = {}    # Normalized transition matrices for each t
        self.H = {}     # Place cell population embeddings for each t
        
        # Environment representation
        self.environment = torch.ones(grid_size, grid_size, dtype=torch.float32, device=self.device)
        self.coords_to_idx = {}  # Map from (i,j) to state index
        self.idx_to_coords = {}  # Map from state index to (i,j)
        
        # Initialize the coordinate mappings
        self._init_coordinate_mappings()
        
    def _init_coordinate_mappings(self):
        """Initialize coordinate-to-index and index-to-coordinate mappings."""
        idx = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.coords_to_idx[(i, j)] = idx
                self.idx_to_coords[idx] = (i, j)
                idx += 1

    def build_transition_matrix(self, obstacles=None):
        """
        Build the one-step transition matrix P1 based on a symmetric random walk.
        
        Parameters:
        -----------
        obstacles : list of tuples, optional
            List of (i, j) coordinates representing obstacle locations
            
        Returns:
        --------
        P1 : torch.Tensor
            One-step transition matrix
        """
        print("Building one-step transition matrix P1...")
        
        # Initialize P1 with zeros
        self.P1 = torch.zeros(self.n_states, self.n_states, dtype=torch.float32, device=self.device)
        
        # Convert obstacle list to a set for faster lookups
        obstacle_set = set()
        obstacle_indices = []
        if obstacles is not None:
            for obs in obstacles:
                # Make sure obstacles are within grid bounds
                if 0 <= obs[0] < self.grid_size and 0 <= obs[1] < self.grid_size:
                    obstacle_set.add(obs)
                    # Get the index for this obstacle if it exists in coords_to_idx
                    idx = self.coords_to_idx.get(obs)
                    if idx is not None:
                        obstacle_indices.append(idx)
        
        # Define the 3x3 neighborhood directions (including staying in place)
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        base_prob = 1.0 / 9.0  # Base probability for each direction
        
        # First pass: Set transitions for non-obstacle states
        for idx in range(self.n_states):
            i, j = self.idx_to_coords[idx]
            
            # Skip if current position is an obstacle (handle obstacles separately)
            if (i, j) in obstacle_set:
                continue
            
            # Initialize self-transition probability
            self_prob = base_prob  # Start with base prob for staying in place
            
            # Process each neighbor in the 3x3 neighborhood
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                # Skip self (already handled as self_prob)
                if di == 0 and dj == 0:
                    continue
                    
                # Check if neighbor is within bounds and not an obstacle
                if (0 <= ni < self.grid_size and 
                    0 <= nj < self.grid_size and 
                    (ni, nj) not in obstacle_set):
                    
                    # Valid transition - set probability to base value
                    neighbor_idx = self.coords_to_idx.get((ni, nj))
                    if neighbor_idx is not None:
                        self.P1[idx, neighbor_idx] = base_prob
                else:
                    # Forbidden transition - add probability to self-transition
                    self_prob += base_prob
            
            # Set self-transition probability (including redirected probabilities)
            self.P1[idx, idx] = self_prob
        
        # Second pass: Set p(x|x,t)=1 for obstacles (absorbing states)
        for obs_idx in obstacle_indices:
            # Clear any existing transitions
            self.P1[obs_idx, :] = 0.0
            # Set self-transition to 1
            self.P1[obs_idx, obs_idx] = 1.0
        
        # Verify row sums to ensure stochasticity
        row_sums = torch.sum(self.P1, dim=1)
        is_stochastic = torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-5)
        if not is_stochastic:
            print("WARNING: Not all rows sum to 1. Maximum deviation:", torch.max(torch.abs(row_sums - 1.0)).item())
        
        # Check symmetry before enforcing it (excluding obstacle states)
        # Create a mask for non-obstacle states
        non_obstacle_mask = torch.ones((self.n_states, self.n_states), dtype=torch.bool, device=self.device)
        for idx in obstacle_indices:
            non_obstacle_mask[idx, :] = False
            non_obstacle_mask[:, idx] = False
        
        P1_non_obstacles = self.P1[non_obstacle_mask].reshape(-1)
        P1_transpose_non_obstacles = self.P1.t()[non_obstacle_mask].reshape(-1)
        is_symmetric_before = torch.allclose(P1_non_obstacles, P1_transpose_non_obstacles, rtol=1e-5, atol=1e-5)
        print(f"P1 is symmetric before enforcement (excluding obstacles): {is_symmetric_before}")
        
        # Save original obstacle transitions
        obstacle_transitions = self.P1.clone()
        if obstacle_indices:
            for obs_idx in obstacle_indices:
                obstacle_transitions[obs_idx, :] = self.P1[obs_idx, :]
                obstacle_transitions[:, obs_idx] = self.P1[:, obs_idx]
        
        # Enforce symmetry by averaging with transpose (only for non-obstacle states)
        self.P1 = 0.5 * (self.P1 + self.P1.t())
        
        # Restore original obstacle transitions
        if obstacle_indices:
            for obs_idx in obstacle_indices:
                self.P1[obs_idx, :] = obstacle_transitions[obs_idx, :]
                self.P1[:, obs_idx] = obstacle_transitions[:, obs_idx]
        
        # Renormalize rows to ensure stochasticity after symmetrization (skip obstacle states)
        row_sums = torch.sum(self.P1, dim=1, keepdim=True)
        non_obstacle_rows = torch.ones(self.n_states, dtype=torch.bool, device=self.device)
        non_obstacle_rows[obstacle_indices] = False
        
        # Only renormalize non-obstacle rows
        self.P1[non_obstacle_rows] = self.P1[non_obstacle_rows] / row_sums[non_obstacle_rows]
        
        # Verify symmetry and stochasticity after enforcement
        P1_non_obstacles = self.P1[non_obstacle_mask].reshape(-1)
        P1_transpose_non_obstacles = self.P1.t()[non_obstacle_mask].reshape(-1)
        is_symmetric = torch.allclose(P1_non_obstacles, P1_transpose_non_obstacles, rtol=1e-5, atol=1e-5)
        print(f"P1 is symmetric after enforcement (excluding obstacles): {is_symmetric}")
        
        row_sums = torch.sum(self.P1, dim=1)
        is_stochastic = torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-5)
        print(f"P1 is row-stochastic: {is_stochastic}")
        
        # Count non-zero entries to show sparsity
        n_nonzero = torch.count_nonzero(self.P1).item()
        print(f"P1 has {n_nonzero} non-zero entries out of {self.n_states**2} possible entries")
        
        return self.P1

    def compute_multi_time_transition(self):
        """
        Compute transition matrices for multiple time scales by building on
        previous calculations. Optimized for power-of-2 time scales (1,2,4,8,16).
        
        For each time scale t, we compute:
        P_{t=2} = P_{t=1} * P_{t=1}
        P_{t=4} = P_{t=2} * P_{t=2}
        P_{t=8} = P_{t=4} * P_{t=4}
        
        This way, we only need log(max_t) matrix multiplications.
        """
        print("Computing multi-time transition matrices...")
        max_t = max(self.t_values)
        
        # Store P1 in the dictionary
        self.Pt[1] = self.P1
        
        # For each power-of-2 time scale, square the previous result
        prev_t = 1
        for t in self.t_values[1:]:
            if t == prev_t * 2:  # Verify it's exactly double the previous t
                # Square the previous matrix: P_{t} = P_{t/2} * P_{t/2}
                self.Pt[t] = torch.matmul(self.Pt[prev_t], self.Pt[prev_t])
                print(f"Computed P{t} by squaring P{prev_t}")
                prev_t = t
            else:
                # Fallback for non-power-of-2 values (shouldn't happen with [1,2,4,8,16])
                print(f"Warning: t={t} is not a power of 2 or not in sequence")
                if t <= 8:
                    current_P = self.P1.clone()
                    for _ in range(1, t):
                        current_P = torch.matmul(current_P, self.P1)
                    self.Pt[t] = current_P
                    print(f"Computed P{t} recursively")
                else:
                    # Use binary exponentiation for larger non-sequential values
                    binary_t = bin(t)[2:]  # Remove '0b' prefix
                    result = torch.eye(self.n_states, device=self.device)
                    temp = self.P1.clone()
                    
                    for bit in binary_t:
                        if bit == '1':
                            result = torch.matmul(result, temp)
                        temp = torch.matmul(temp, temp)
                    
                    self.Pt[t] = result
                    print(f"Computed P{t} using binary exponentiation")
    
    def normalize_transition_matrices(self):
        """
        Normalize transition matrices to create q(y|x,t).
        q(y|x,t) = p(y|x,t) / sqrt(p(x|x,t) * p(y|y,t))
        """
        print("Normalizing transition matrices...")
        for t in tqdm(self.t_values):
            if t == 1:  # Skip t=1 as specified in the paper
                continue
                
            P_t = self.Pt[t]
            
            # Extract diagonal entries (self-transitions)
            diag = torch.diagonal(P_t)
            
            # Create normalization factors
            diag_sqrt = torch.sqrt(diag)
            norm_factor = torch.outer(1.0 / diag_sqrt, 1.0 / diag_sqrt)
            
            # Normalize
            Q_t = P_t * norm_factor
            
            # Verify diagonal elements are 1
            diag_Q = torch.diagonal(Q_t)
            is_diag_one = torch.allclose(diag_Q, torch.ones_like(diag_Q))
            print(f"Q{t} diagonal elements are 1: {is_diag_one}")
            
            self.Qt[t] = Q_t
    
    def learn_embeddings(self, obstacles=None, n_iter=500, learning_rate=0.01, decay_factor=0.1, foldername=None):
        """
        Learn place cell population embeddings using matrix factorization with PyTorch.
        We want to learn embeddings h(x,t) such that:
        <h(x,t), h(y,t)> ≈ q(y|x,t)
        
        For obstacle locations x, we fix h(x,t)=0 and exclude them from the learning process.
        
        Parameters:
        -----------
        obstacles : list of tuples, optional
            List of (i, j) coordinates representing obstacle locations
        n_iter : int
            Number of iterations for optimization
        learning_rate : float
            Initial learning rate
        decay_factor : float
            Factor to decay learning rate (final_lr = learning_rate * decay_factor)
        """
        print("Learning place cell population embeddings...")
        
        # Convert obstacle list to indices and create a mask for non-obstacle states
        obstacle_indices = set()
        if obstacles is not None:
            for obs in obstacles:
                if 0 <= obs[0] < self.grid_size and 0 <= obs[1] < self.grid_size:
                    idx = self.coords_to_idx.get(obs)
                    if idx is not None:
                        obstacle_indices.add(idx)
            print(f"Identified {len(obstacle_indices)} obstacles to exclude")
        
        # Create full embeddings container
        self.H = {}
        
        for t in self.t_values:
            if t == 1:  # Skip t=1 as specified in the paper
                continue
                
            print(f"Learning embeddings for t={t}...")
            Q_t = self.Qt[t]
            
            # Step 1: Create a mapping between original indices and reduced indices
            # (excluding obstacles)
            orig_to_reduced = {}
            reduced_to_orig = {}
            reduced_idx = 0
            
            for orig_idx in range(self.n_states):
                if orig_idx not in obstacle_indices:
                    orig_to_reduced[orig_idx] = reduced_idx
                    reduced_to_orig[reduced_idx] = orig_idx
                    reduced_idx += 1
            
            n_reduced = reduced_idx  # Number of non-obstacle states
            print(f"Learning embeddings for {n_reduced} non-obstacle states")
            
            # Step 2: Create reduced Q_t matrix (excluding obstacle states)
            reduced_Q_t = torch.zeros((n_reduced, n_reduced), device=self.device)
            
            for i, orig_i in reduced_to_orig.items():
                for j, orig_j in reduced_to_orig.items():
                    reduced_Q_t[i, j] = Q_t[orig_i, orig_j]
            
            # Step 3: Initialize reduced embeddings with small random values
            reduced_H_t = torch.nn.Parameter(
                torch.normal(0, 0.001, size=(n_reduced, self.embedding_dim), device=self.device)
            )
            
            # Step 4: Optimize only the reduced embeddings
            optimizer = torch.optim.AdamW([reduced_H_t], lr=learning_rate)
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
                reduced_Q_approx = torch.matmul(reduced_H_t, reduced_H_t.t())
                
                # Compute loss
                error_matrix = (reduced_Q_approx - reduced_Q_t) ** 2
                loss = torch.mean(error_matrix)
                loss_history.append(loss.item())
                
                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                with torch.no_grad():
                    # Create a normalized copy for the forward pass
                    reduced_H_t_normalized = torch.nn.functional.normalize(
                        torch.clamp(reduced_H_t.detach(), min=0.0), dim=1)
                    
                    # Update the parameter with the normalized values
                    # This breaks the computation graph but avoids double backward
                    reduced_H_t.data.copy_(reduced_H_t_normalized)
                
                # Update progress bar
                if i % 10 == 0:
                    pbar.set_description(f"Loss: {loss.item():.6f}")
            
                # Final normalization
            with torch.no_grad():
                reduced_H_final = torch.nn.functional.normalize(
                    torch.clamp(reduced_H_t, min=0.0), dim=1
                )
            
            # Step 5: Map reduced embeddings back to full embeddings (with zeros for obstacles)
            full_H_t = torch.zeros((self.n_states, self.embedding_dim), device=self.device)
            
            for reduced_idx, orig_idx in reduced_to_orig.items():
                full_H_t[orig_idx] = reduced_H_final[reduced_idx]
            
            # Store final embeddings
            self.H[t] = full_H_t
            
            # Compute final loss and correlation
            Q_approx = torch.matmul(self.H[t], self.H[t].t())
            
            # Create mask for non-obstacle transitions
            non_obstacle_mask = torch.ones(self.n_states, dtype=torch.bool, device=self.device)
            for idx in obstacle_indices:
                non_obstacle_mask[idx] = False
            
            valid_transitions = torch.outer(non_obstacle_mask, non_obstacle_mask)
            
            # Compute final loss (only for non-obstacle transitions)
            final_error = ((Q_approx - Q_t) ** 2)[valid_transitions].reshape(-1)
            final_loss = torch.mean(final_error).item()
            print(f"Final loss for t={t}: {final_loss:.6f}")
            
            # Compute correlation between Q_t and inner products (only for non-obstacle transitions)
            Q_t_valid = Q_t[valid_transitions].reshape(-1).cpu().numpy()
            Q_approx_valid = Q_approx[valid_transitions].reshape(-1).detach().cpu().numpy()
            correlation = np.corrcoef(Q_t_valid, Q_approx_valid)[0, 1]
            print(f"Correlation for t={t}: {correlation:.4f}")
            
            # Plot loss history
            dir = f"{foldername}/loss" if foldername is not None else "loss"
            if not os.path.exists(dir):
                os.makedirs(dir)

            plt.figure(figsize=(10, 5))
            plt.plot(loss_history)
            plt.title(f"Loss History for t={t}")
            plt.xlabel("Iteration")
            plt.ylabel("MSE Loss")
            plt.yscale('log')
            plt.grid(True)
            plt.savefig(f"{dir}/loss_history_t{t}.png")
            plt.close()


        print("Embedding learning complete!")
        return self.H

    def plan_path(self, start_pos, target_pos, obstacles=None, num_directions=16, max_steps=100, step_size=1.0, visualize=True, foldername=None):
        """
        Plan a path from start_pos to target_pos using place cell embeddings.
        Uses a fixed step size and selects optimal (direction, time scale) pair
        at each step based on the gradient of inner product with target embedding.

        Parameters:
        -----------
        start_pos : tuple of (float, float)
            Starting position (x, y) coordinates
        target_pos : tuple of (float, float)
            Target position (x, y) coordinates
        obstacles : list of tuples, optional
            List of (i, j) coordinates representing obstacle locations
        num_directions : int
            Number of directions to sample for each step
        max_steps : int
            Maximum number of steps to take
        step_size : float
            Fixed step size for movement (dr)
        visualize : bool
            Whether to visualize the path

        Returns:
        --------
        path : list of tuples
            List of (x, y) coordinates representing the path
        selected_t_values : list
            List of time scales selected at each step
        """
        reached_target = False
        # Convert obstacle list to a set for faster lookups
        obstacle_set = set()
        if obstacles is not None:
            for obs in obstacles:
                # Make sure obstacles are within grid bounds
                if 0 <= obs[0] < self.grid_size and 0 <= obs[1] < self.grid_size:
                    obstacle_set.add(obs)
        thetas = np.linspace(0, 2*np.pi, num_directions, endpoint=False)
        
        # Initialize path with start position
        path = [start_pos]
        current_pos = start_pos
        
        # Keep track of selected time scales for analysis
        selected_t_values = []
        
        # For visualization
        if visualize:
            plt.figure(figsize=(10, 10))
            grid = np.ones((self.grid_size, self.grid_size))
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Check if position is in coords_to_idx (valid) and not an obstacle
                    idx = self.coords_to_idx.get((i, j))
                    if idx is None or (i, j) in obstacle_set:
                        grid[i, j] = 0  # Mark as obstacle
            
            obstacle_mask = np.zeros((self.grid_size, self.grid_size))
            for obs in obstacle_set:
                obstacle_mask[obs[0], obs[1]] = 1
            
            # Plot the environment
            plt.imshow(grid, cmap='Blues_r', origin='lower', 
                    extent=[0, self.grid_size, 0, self.grid_size], alpha=0.5)
            plt.imshow(obstacle_mask, cmap='Reds', origin='lower',
                    extent=[0, self.grid_size, 0, self.grid_size], alpha=0.7)

            plt.plot(start_pos[1], start_pos[0], 'go', markersize=15, label='Start')
            plt.plot(target_pos[1], target_pos[0], 'ro', markersize=15, label='Target')

        # Get target embeddings for each time scale (skip t=1 as specified)
        target_embeddings = {}
        for t in self.t_values:
            if t > 1:  # Skip t=1 as it doesn't have place cell embeddings
                target_embeddings[t] = self.get_embedding_at_position(target_pos, t)

        # Main path planning loop
        for step in range(max_steps):
            # Check if we've reached the target (within a small threshold)
            dist_to_target = np.sqrt((current_pos[0] - target_pos[0])**2 + 
                                    (current_pos[1] - target_pos[1])**2)
            if dist_to_target < 1:  # Arrived at destination
                # print(f"Reached target in {step} steps")
                reached_target = True
                break
            
            # Get current position embedding for each time scale
            current_embeddings = {}
            for t in self.t_values:
                if t > 1:
                    current_embeddings[t] = self.get_embedding_at_position(current_pos, t)
            
            # Store the best direction, time scale, and gradient
            best_pos = None
            best_t = None
            best_gradient = -float('inf')
            
            # Generate all possible next positions using fixed step size
            next_positions = []
            for theta in thetas:
                # Calculate next position with fixed step size
                dx = step_size * np.cos(theta)
                dy = step_size * np.sin(theta)
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                
                # Round to nearest grid cell to check for obstacles
                nearest_cell = (round(next_pos[0]), round(next_pos[1]))
                
                # Check if next position is within bounds and not an obstacle
                if (0 <= next_pos[0] < self.grid_size and 
                    0 <= next_pos[1] < self.grid_size and
                    nearest_cell not in obstacle_set):
                    next_positions.append(next_pos)
            
            # For each position and time scale, compute gradient of similarity with target
            for next_pos in next_positions:
                for t in self.t_values:
                    if t <= 1:  # Skip t=1
                        continue
                    
                    # Get embeddings at next position
                    next_embedding = self.get_embedding_at_position(next_pos, t)
                    
                    # Calculate current similarity with target
                    current_similarity = np.dot(current_embeddings[t], target_embeddings[t])
                    
                    # Calculate next position similarity with target
                    next_similarity = np.dot(next_embedding, target_embeddings[t])
                    
                    # Calculate gradient (improvement in similarity)
                    gradient = next_similarity - current_similarity
                    
                    # Update best position and time scale if this gives better gradient
                    if gradient > best_gradient:
                        best_gradient = gradient
                        best_pos = next_pos
                        best_t = t
            
            if best_pos is None:
                print("No valid move found. Stopping.")
                break
            
            # Move to the best position
            current_pos = best_pos
            path.append(current_pos)
            selected_t_values.append(best_t)

        # ──────────────────────────  Visualise the path  ──────────────────────────
        if visualize and path:
            import matplotlib.cm as cm
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize

            fig, ax = plt.subplots(figsize=(10, 10))
            grid = np.ones((self.grid_size, self.grid_size))
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    idx = self.coords_to_idx.get((i, j))
                    if obstacles is not None and (i, j) in obstacle_set:
                        grid[i, j] = 0
            obstacle_mask = np.zeros((self.grid_size, self.grid_size))
            for obs in obstacle_set:
                obstacle_mask[obs[0], obs[1]] = 1

            from matplotlib.colors import ListedColormap
            white_grey_cmap = ListedColormap(["darkgrey", "lightgrey"])
            ax.imshow(grid, cmap=white_grey_cmap, origin='lower',
                    extent=[0, self.grid_size, 0, self.grid_size], alpha=1.0)

            # colour mapping in log₂ space
            unique_t   = [2,4,8,16,32,64,128,256,512,1024,2048] 
            log_unique = np.log2(unique_t)                     
            norm = Normalize(vmin=log_unique[0], vmax=log_unique[-1])
            cmap = cm.turbo

            for i in range(len(path) - 1):
                t_val   = selected_t_values[i]
                colour  = cmap(norm(np.log2(t_val)))
                x0, y0  = path[i][1],  path[i][0]
                x1, y1  = path[i+1][1], path[i+1][0]
                ax.plot([x0, x1], [y0, y1],
                        color=colour, linewidth=6, solid_capstyle='round')


            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.8)

            cbar.set_label("t value (time scale)", fontsize=14)
            cbar.set_ticks(log_unique)                          # positions in log₂
            cbar.set_ticklabels([str(t) for t in unique_t])     # show 2,4,8,...
            cbar.ax.tick_params(labelsize=12)

            # ── 6. start / target markers & legend ────────────────────────────
            # ax.plot(start_pos[1], start_pos[0], 'go', markersize=10, label='Start')
            # ax.plot(target_pos[1], target_pos[0], 'ro', markersize=10, label='Target')
            markersize = 20
            ax.plot(start_pos[1], start_pos[0], marker="o", color="green", markersize=markersize)
            ax.plot(target_pos[1], target_pos[0], marker="X", color="red", markersize=markersize)

            # ── 7. layout & save ───────────────────────────────────────────────
            ax.set_title(f"Path from {start_pos} to {target_pos}", fontsize=16)
            ax.set_axis_off() 

            dir_name = f"{foldername}/path_planning" if foldername else "path_planning"
            os.makedirs(dir_name, exist_ok=True)
            filename = f"{dir_name}/planned_path_{start_pos}_{target_pos}.png"
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            print(f"Saved path visualisation to {filename}")

        # Print summary statistics about time scale selection
        if selected_t_values:
            t_counts = {}
            for t in selected_t_values:
                t_counts[t] = t_counts.get(t, 0) + 1

        if not reached_target:
            print(f"Failed: start={start_pos}, target={target_pos} ")

        return path, selected_t_values, reached_target, dist_to_target

    def get_embedding_at_position(self, position, t):
        """
        Get the place cell embedding at a continuous position using bilinear interpolation.

        Parameters:
        -----------
        position : tuple of (float, float)
            Position (x, y) coordinates
        t : int
            Time scale

        Returns:
        --------
        embedding : numpy.ndarray
            Place cell embedding at the specified position
        """
        x, y = position

        # Get integer coordinates of the four nearest grid points
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = x0 + 1, y0 + 1

        # Ensure coordinates are within the grid
        x0 = max(0, min(x0, self.grid_size - 1))
        y0 = max(0, min(y0, self.grid_size - 1))
        x1 = max(0, min(x1, self.grid_size - 1))
        y1 = max(0, min(y1, self.grid_size - 1))

        # Calculate interpolation weights
        wx = x - x0
        wy = y - y0

        # Get embeddings for the four nearest points
        embeddings = {}
        weights = {}
        total_weight = 0

        for xi, yi, wi in [(x0, y0, (1-wx)*(1-wy)), 
                            (x1, y0, wx*(1-wy)), 
                            (x0, y1, (1-wx)*wy), 
                            (x1, y1, wx*wy)]:
            if (xi, yi) in self.coords_to_idx:
                idx = self.coords_to_idx[(xi, yi)]
                embeddings[(xi, yi)] = self.H[t][idx].cpu().numpy()
                weights[(xi, yi)] = wi
                total_weight += wi

        # If no valid grid points, return zeros
        if total_weight == 0:
            return np.zeros(self.embedding_dim)

        # Normalize weights to account for possible missing grid points
        for pos in weights:
            weights[pos] /= total_weight

        # Compute weighted average of embeddings
        embedding = np.zeros(self.embedding_dim)
        for pos, weight in weights.items():
            embedding += weight * embeddings[pos]

        return embedding

    def verify_inner_products(self, t, num_samples=10):
        """
        Verify that inner products between embedding vectors approximate
        the normalized transition probabilities.
        """
        if t not in self.H:
            print(f"No embeddings for t={t}")
            return
            
        H_t = self.H[t]
        Q_t = self.Qt[t]
        
        # Randomly sample state pairs
        x_indices = torch.randint(0, self.n_states, (num_samples,), device=self.device)
        y_indices = torch.randint(0, self.n_states, (num_samples,), device=self.device)
        
        print(f"\nVerifying inner products for t={t}:")
        print("x\ty\tq(y|x,t)\t<h(x,t),h(y,t)>")
        for i in range(num_samples):
            x = x_indices[i].item()
            y = y_indices[i].item()
            
            q_true = Q_t[x, y].item()
            q_approx = torch.dot(H_t[x], H_t[y]).item()
            
            print(f"{x}\t{y}\t{q_true:.4f}\t{q_approx:.4f}")
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        """
        # Convert tensors to numpy arrays for easier saving
        save_dict = {
            'grid_size': self.grid_size,
            'embedding_dim': self.embedding_dim,
            't_values': self.t_values,
            'H': {t: self.H[t].detach().cpu().numpy() for t in self.H}
        }
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from a file.
        """
        save_dict = torch.load(filepath, map_location=self.device)
        
        self.grid_size = save_dict['grid_size']
        self.embedding_dim = save_dict['embedding_dim']
        self.t_values = save_dict['t_values']
        
        # Convert numpy arrays back to tensors
        self.H = {t: torch.tensor(save_dict['H'][t], device=self.device) 
                 for t in save_dict['H']}
        
        print(f"Model loaded from {filepath}")

    def run_pipeline(self, obstacles=None, foldername = None):
        start_time = time.time()
        
        self.build_transition_matrix(obstacles)        
        self.compute_multi_time_transition()        
        self.normalize_transition_matrices()
        self.learn_embeddings(obstacles=obstacles, n_iter=2000, learning_rate=0.01, decay_factor=0.1, foldername=foldername)
        
        # Verify inner products for all time scales
        for t in self.t_values:
            self.verify_inner_products(t)

        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        print("\nPerforming path planning experiments...")

        # Example 1: Path from top-left to bottom-right
        plan_step_size = 1.5
        start_pos1 = (self.grid_size * 0.1 + 0.5, self.grid_size * 0.1 + 0.5)  # Near top-left
        target_pos1 = (self.grid_size * 0.9 + 0.5, self.grid_size * 0.9 + 0.5)  # Near bottom-right
        
        print(f"\nPlanning path from {start_pos1} to {target_pos1}")
        path1 = self.plan_path(
            start_pos=start_pos1,
            target_pos=target_pos1,
            num_directions=36,
            max_steps=100,
            step_size=plan_step_size,
            obstacles=obstacles,
            visualize=True,
            foldername = foldername
        )
        
        # Example 2: Path from bottom-left to top-right
        start_pos2 = (self.grid_size * 0.1 + 0.5, self.grid_size * 0.9 + 0.5)  # Near bottom-left
        target_pos2 = (self.grid_size * 0.9 + 0.5, self.grid_size * 0.1 + 0.5)  # Near top-right
        
        print(f"\nPlanning path from {start_pos2} to {target_pos2}")
        path2 = self.plan_path(
            start_pos=start_pos2,
            target_pos=target_pos2,
            num_directions=36,
            max_steps=50,
            step_size=plan_step_size,
            obstacles=obstacles,
            visualize=True,
            foldername = foldername
        )
        
        # Example 3: Path from left to right
        start_pos3 = (15, 5) # Near bottom-left
        target_pos3 = (15, 35)  # Near top-right
        
        print(f"\nPlanning path from {start_pos2} to {target_pos2}")
        path3 = self.plan_path(
            start_pos=start_pos3,
            target_pos=target_pos3,
            num_directions=36,
            max_steps=100,
            step_size=plan_step_size,
            obstacles=obstacles,
            visualize=True,
            foldername = foldername
        )
        
        print("\nPlanning paths for random start-target pairs...")
        for i in range(3):
            # Generate random start and target positions
            valid_positions = False
            while not valid_positions:
                # Generate random positions within grid boundaries
                start_x, start_y = np.random.uniform(1, self.grid_size-1, 2)
                target_x, target_y = np.random.uniform(1, self.grid_size-1, 2)
                
                # Ensure minimum distance between start and target
                min_distance = self.grid_size * 0.3  # At least 30% of grid size apart
                distance = np.sqrt((start_x - target_x)**2 + (start_y - target_y)**2)
                
                if distance >= min_distance:
                    valid_positions = True
            
            start_pos = (start_x, start_y)
            target_pos = (target_x, target_y)
            
            print(f"\nRandom path {i+1}: From {start_pos} to {target_pos}")
            path = self.plan_path(
                start_pos=start_pos,
                target_pos=target_pos,
                num_directions=36,
                max_steps=40,
                step_size=plan_step_size,
                obstacles=obstacles,
                visualize=True,
                foldername = foldername
            )
        
        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")



