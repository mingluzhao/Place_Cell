"""
Main script for Place Cell Navigation Model

This script demonstrates how to use the Place Cell Model for 
hierarchical navigation planning in different environments.
"""

import os
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import PlaceCellModel
from environment import get_environment


def parse_arguments():
    parser = argparse.ArgumentParser(description='Place Cell Navigation Model')
    
    parser.add_argument('--env', type=str, default='maze',
                        choices=['maze', 'u_shape', 's_shape', 'open_field'],
                        help='Environment type')
    
    parser.add_argument('--grid_size', type=int, default=40,
                        help='Size of the grid environment')
    
    parser.add_argument('--embedding_dim', type=int, default=500,
                        help='Dimensionality of place cell population embeddings')
    
    parser.add_argument('--n_iter', type=int, default=2000,
                        help='Number of iterations for embedding learning')
    
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for embedding optimization')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save output files')

    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Set output directory to environment name if not specified
    if args.output_dir is None:
        args.output_dir = args.env
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Get obstacles for the specified environment
    obstacles = get_environment(args.env, args.grid_size)
    print(f"Created {args.env} environment with {len(obstacles)} obstacles")
    
    model = PlaceCellModel(grid_size=args.grid_size, embedding_dim=args.embedding_dim)
    model.run_pipeline(obstacles = obstacles, foldername = args.output_dir)



if __name__ == "__main__":
    main()

