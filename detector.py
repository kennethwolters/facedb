import argparse
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
import json
from pathlib import Path
from annoy import AnnoyIndex
import shutil
import time
import hashlib
import signal
import sys

class FaceProcessor:
    def __init__(self, args):
        if args.gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self.app = FaceAnalysis(providers=providers)
        self.app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))
        
        self.visualization_dir = args.visualize
        self.annoy_path = args.annoy
        self.embeddings_dir = os.path.splitext(args.annoy)[0] + '_embeddings'
        self.metadata_path = args.metadata or os.path.splitext(args.annoy)[0] + '_metadata.json'
        self.checkpoint_interval = args.checkpoint_interval
        self.n_trees = args.n_trees
        self.distance_metric = args.distance
        
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        self.metadata_list = self._load_metadata()
        
        self.processed_images = set()
        for face_data in self.metadata_list:
            self.processed_images.add(face_data['source_image'])
        
        self.embedding_dim = self._determine_embedding_dim(args.input)
        print(f"Using embedding dimension: {self.embedding_dim}")
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.last_checkpoint_time = time.time()
        self.faces_since_checkpoint = 0
        
        if self.visualization_dir:
            os.makedirs(self.visualization_dir, exist_ok=True)
    
    def _determine_embedding_dim(self, input_path):
        try:
            sample_img = None
            if os.path.isdir(input_path):
                for ext in ['.jpg', '.jpeg', '.png']:
                    image_files = list(Path(input_path).glob(f'*{ext}'))
                    if image_files:
                        sample_img = cv2.imread(str(image_files[0]))
                        break
            else:
                sample_img = cv2.imread(input_path)
            
            if sample_img is not None:
                faces = self.app.get(sample_img)
                if faces:
                    return len(faces[0].embedding)
            
            print("Could not find a face to determine embedding dimension")
            print("Using default dimension of 512")
            return 512
        except Exception as e:
            print(f"Error determining embedding dimension: {e}")
            print("Using default dimension of 512")
            return 512
    
    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {e}")
                return []
        return []
    
    def _save_metadata(self):
        temp_path = f"{self.metadata_path}.tmp"
        with open(temp_path, 'w') as f:
            json.dump(self.metadata_list, f, indent=2)
        
        shutil.move(temp_path, self.metadata_path)
    
    def _build_annoy_index(self):
        print(f"Building Annoy index with {self.n_trees} trees...")
        
        index = AnnoyIndex(self.embedding_dim, self.distance_metric)
        
        for idx, face_data in enumerate(self.metadata_list):
            embedding_path = face_data.get('embedding_path')
            if embedding_path and os.path.exists(embedding_path):
                try:
                    embedding = np.load(embedding_path)
                    index.add_item(idx, embedding)
                except Exception as e:
                    print(f"Error loading embedding {embedding_path}: {e}")
        
        index.build(self.n_trees)
        
        temp_path = f"{self.annoy_path}.tmp"
        index.save(temp_path)
        
        shutil.move(temp_path, self.annoy_path)
        print(f"Saved Annoy index to {self.annoy_path}")
    
    def _checkpoint(self, force=False):
        current_time = time.time()
        time_elapsed = current_time - self.last_checkpoint_time
        
        if force or (time_elapsed >= self.checkpoint_interval or self.faces_since_checkpoint >= 100):
            print("Creating checkpoint...")
            
            self._save_metadata()
            
            self._build_annoy_index()
            
            self.last_checkpoint_time = current_time
            self.faces_since_checkpoint = 0
            
            print(f"Checkpoint created with {len(self.metadata_list)} faces total")
    
    def _signal_handler(self, sig, frame):
        print("\nReceived termination signal. Cleaning up...")
        self._checkpoint(force=True)
        sys.exit(0)
    
    def process_image(self, image_path):
        if str(image_path) in self.processed_images:
            print(f"Skipping already processed image: {image_path}")
            return []
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not read {image_path}")
            return []
        
        faces = self.app.get(img)
        if not faces:
            print(f"No faces detected in {image_path}")
            self.processed_images.add(str(image_path))
            return []
        
        vis_img = img.copy()
        
        face_data = []
        for i, face in enumerate(faces):
            embedding = face.embedding
            
            bbox = face.bbox.astype(int)
            
            face_id = f"{base_name}_{i+1}"
            
            embedding_hash = hashlib.md5(embedding.tobytes()).hexdigest()
            
            embedding_filename = f"{embedding_hash}.npy"
            embedding_path = os.path.join(self.embeddings_dir, embedding_filename)
            np.save(embedding_path, embedding)
            
            face_metadata = {
                "source_image": str(image_path),
                "face_id": face_id,
                "bbox": bbox.tolist(),
                "embedding_hash": embedding_hash,
                "embedding_path": embedding_path
            }
            self.metadata_list.append(face_metadata)
            
            cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            cv2.putText(vis_img, str(i+1), (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(vis_img, str(i+1), (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            print(f"Processed face #{i+1} in {image_path}")
            
            self.faces_since_checkpoint += 1
        
        if self.visualization_dir:
            vis_path = os.path.join(self.visualization_dir, f"{base_name}_faces.jpg")
            cv2.imwrite(vis_path, vis_img)
        
        self.processed_images.add(str(image_path))
        
        self._checkpoint()
        
        return face_data
    
    def process_path(self, input_path):
        path = Path(input_path)
        
        if path.is_file():
            if path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                self.process_image(path)
        elif path.is_dir():
            image_count = 0
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in path.glob(ext):
                    image_count += 1
                    self.process_image(img_path)
                    print(f"Progress: {len(self.processed_images)} images processed")
            
            if image_count == 0:
                print(f"No image files found in {input_path}")
        else:
            print(f"Error: {input_path} is neither a file nor a directory")
        
        self._checkpoint(force=True)


def main():
    parser = argparse.ArgumentParser(description='Extract face embeddings from images')
    parser.add_argument('input', help='Path to image file or directory')
    parser.add_argument('--annoy', '-a', required=True, help='Path to save Annoy index file')
    parser.add_argument('--metadata', '-m', help='Path to save face metadata JSON file')
    parser.add_argument('--visualize', '-v', help='Directory to save visualization images')
    parser.add_argument('--det-size', type=int, default=640, help='Detection size (default: 640)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--n-trees', type=int, default=10, help='Number of trees for Annoy index (default: 10)')
    parser.add_argument('--distance', type=str, default='angular', 
                       choices=['angular', 'euclidean', 'manhattan', 'hamming', 'dot'],
                       help='Distance metric for Annoy index (default: angular)')
    parser.add_argument('--checkpoint-interval', type=int, default=300, 
                       help='Time in seconds between checkpoints (default: 300)')
    args = parser.parse_args()
    
    processor = FaceProcessor(args)
    try:
        processor.process_path(args.input)
        print(f"Processed {len(processor.metadata_list)} faces total")
    except Exception as e:
        print(f"Error during processing: {e}")
        processor._checkpoint(force=True)
        raise

if __name__ == "__main__":
    main()