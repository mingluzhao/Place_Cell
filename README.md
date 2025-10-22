# Place Cells as Position Embeddings of Multi-Step Random Walk Transition Kernels: Euclideanized and Sparsified Cognitive Maps for Path Planning
Minglu Zhao*, Dehong Xu*, Deqian Kong*, Wen-Hao Zhang Ying Nian Wu

*Equal contribution

[[Paper link](https://arxiv.org/pdf/2505.14806)], [[Project Page](https://sites.google.com/view/place-cells)]

## Repository Structure

```
.
├── place_cell_model.py   # Core model implementation
├── environment.py        # Environment generators
├── main.py               # Command-line interface
└── README.md             # Project documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/mingluzhao/Place_Cell
cd Place_Cell

# Install dependencies
pip install torch numpy matplotlib tqdm
```

## Usage

### Command-Line Interface

The model can be run from the command line with various configuration options:

```bash
# Train on maze environment
python main.py --env maze --grid_size 40 --embedding_dim 500 --output_dir maze_results

# Try different environment types
python main.py --env u_shape 
python main.py --env s_shape
python main.py --env open_field
```
