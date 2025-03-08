# FaceDB

A face indexing, storage, and similarity search system built on InsightFace embeddings and Annoy approximate nearest neighbors.

## Overview

FaceDB extracts facial embeddings from image collections and enables efficient similarity searches. The system consists of two main components:

1. **Indexer** - Processes images to detect faces, extract embeddings, and build a searchable index
2. **Searcher** - Queries the index to find similar faces across your collection

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for performance)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/facedb.git
cd facedb

# Create a virtual environment
python -m venv env
source env/bin/activate  # Linux/macOS
# or
# env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Building the Index

Process a directory of images and build a searchable index:

```bash
python facedb.py /path/to/images --annoy index.ann --visualize vis_dir
```

Options:
- `--gpu` - Enable GPU acceleration
- `--det-size` - Detection size (default: 640)
- `--n-trees` - Number of trees for Annoy index (default: 10)
- `--distance` - Distance metric (angular, euclidean, manhattan, hamming, dot)

### Searching for Similar Faces

Find faces similar to a specific face in your collection:

```bash
python search.py /path/to/image.jpg 1 --annoy index.ann --top 10 --visualize results.jpg
```

Arguments:
- First argument: Path to the query image
- Second argument: Face number in the image (1-based indexing)
- `--annoy` - Path to the Annoy index
- `--top` - Number of similar faces to return
- `--visualize` - Path to save visualization image

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for face detection and recognition
- [Annoy](https://github.com/spotify/annoy) for efficient similarity search
