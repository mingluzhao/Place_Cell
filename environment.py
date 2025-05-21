"""
Environment Module for Place Cell Navigation

This module contains environment generators for different navigation scenarios,
including mazes, corridors, u-shaped environments, and open fields.

Each environment is defined as a set of obstacle coordinates that can be
passed to the PlaceCellModel for path planning and visualization.
"""

from typing import List, Tuple, Set, Dict, Optional


def create_open_field(grid_size: int = 40) -> List[Tuple[int, int]]:
    """
    Create an open field environment with no obstacles.
    
    Parameters:
    -----------
    grid_size : int
        Size of the grid environment
        
    Returns:
    --------
    obstacles : list
        Empty list (no obstacles)
    """
    return []


def create_u_shape(grid_size: int = 40, 
                  obstacle_lowerleft: Tuple[int, int] = (10, 10),
                  width: int = 20, 
                  height: int = 30) -> List[Tuple[int, int]]:
    """
    Create a U-shaped environment with a central obstacle.
    
    Parameters:
    -----------
    grid_size : int
        Size of the grid environment
    obstacle_lowerleft : tuple
        Lower-left corner coordinates of the U-shape
    width : int
        Width of the U-shape
    height : int
        Height of the U-shape
        
    Returns:
    --------
    obstacles : list
        List of obstacle coordinates
    """
    u_shape = []
    for i in range(obstacle_lowerleft[0], obstacle_lowerleft[0] + height):
        for j in range(obstacle_lowerleft[1], obstacle_lowerleft[1] + width):
            u_shape.append((i, j))
    return u_shape


def create_s_tunnel(grid_size: int = 40) -> List[Tuple[int, int]]:
    """
    Create an S-shaped tunnel environment with two major obstacles.
    
    Parameters:
    -----------
    grid_size : int
        Size of the grid environment
        
    Returns:
    --------
    obstacles : list
        List of obstacle coordinates
    """
    obstacles = []
    
    # Upper obstacle
    for i in range(5, 15): 
        for j in range(0, 30): 
            obstacles.append((i, j))
    
    # Lower obstacle
    for i in range(25, 35): 
        for j in range(10, 40): 
            obstacles.append((i, j))
            
    return obstacles


def create_maze(grid_size: int = 40) -> List[Tuple[int, int]]:
    """
    Create a complex maze environment with multiple paths and obstacles.
    
    Parameters:
    -----------
    grid_size : int
        Size of the grid environment
        
    Returns:
    --------
    obstacles : list
        List of obstacle coordinates
    """
    obstacles = []
    
    # Boundary walls
    for i in [0, 1, 38, 39]:  # Horizontal boundaries
        for j in range(0, grid_size): 
            obstacles.append((i, j))
    
    for i in range(2, 38):  # Vertical boundaries
        for j in [0, 1, 38, 39]: 
            obstacles.append((i, j))
    
    # Inner walls
    for i in [2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37]:  # Horizontal inner walls
        for j in range(17, 23): 
            obstacles.append((i, j))
            
    for i in range(17, 23):  # Vertical inner walls
        for j in [2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37]: 
            obstacles.append((i, j))
    
    # Block obstacles
    # Top-left small blocks
    for i in range(5, 8): 
        for j in [5, 6, 7, 11, 12, 13, 29, 30, 31]: 
            obstacles.append((i, j))
    
    # Top-mid blocks
    for i in range(8, 11): 
        for j in range(26, 35): 
            obstacles.append((i, j))
    
    # Mid-left blocks
    for i in range(11, 14): 
        for j in [5, 6, 7, 29, 30, 31]: 
            obstacles.append((i, j))
    
    # Mid-right small block
    for i in range(23, 26): 
        for j in range(26, 29): 
            obstacles.append((i, j))
    
    # Bottom-left blocks
    for i in range(26, 29): 
        for j in [2, 3, 4, 11, 12, 13]:  # Small blocks
            obstacles.append((i, j))
        for j in range(26, 35):  # Larger block
            obstacles.append((i, j))
    
    # Bottom-right blocks
    for i in range(32, 35): 
        for j in range(26, 35): 
            obstacles.append((i, j))
    
    # Bottom-most small blocks
    for i in range(35, 38): 
        for j in [11, 12, 13, 32, 33, 34]: 
            obstacles.append((i, j))
    
    # Center chamber
    for i in range(11, 17):  # Top section
        for j in range(11, 23): 
            obstacles.append((i, j))
    
    for i in range(17, 23):  # Middle section
        for j in range(11, 29): 
            obstacles.append((i, j))
    
    for i in range(23, 29):  # Bottom section
        for j in range(17, 23): 
            obstacles.append((i, j))
    
    return obstacles


def get_environment(env_type: str, grid_size: int = 40) -> List[Tuple[int, int]]:
    """
    Factory function to create environments based on type.
    
    Parameters:
    -----------
    env_type : str
        Type of environment ('maze', 'u_shape', 's_shape', 'open_field', 'grid_maze', 'corridor')
    grid_size : int
        Size of the grid environment
        
    Returns:
    --------
    obstacles : list
        List of obstacle coordinates for the specified environment
        
    Raises:
    -------
    ValueError
        If env_type is not recognized
    """
    if env_type == "maze":
        return create_maze(grid_size)
    elif env_type == "u_shape":
        return create_u_shape(grid_size)
    elif env_type == "s_shape":
        return create_s_tunnel(grid_size)
    elif env_type == "open_field":
        return create_open_field(grid_size)
    else:
        raise ValueError(f"Unknown environment type: {env_type}. Choose from 'maze', 'u_shape', 's_shape', 'open_field'.")