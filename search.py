import argparse
import json
import numpy as np
import os
from annoy import AnnoyIndex
import cv2

def load_resources(annoy_path, metadata_path, distance_metric):
    if not os.path.exists(annoy_path):
        raise FileNotFoundError(f"Annoy index file not found: {annoy_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if not metadata:
        raise ValueError("Metadata file contains no face information")
    
    first_embedding_path = metadata[0].get('embedding_path')
    if not first_embedding_path or not os.path.exists(first_embedding_path):
        raise ValueError(f"First embedding file not found: {first_embedding_path}")
    
    first_embedding = np.load(first_embedding_path)
    embedding_dim = len(first_embedding)
    
    index = AnnoyIndex(embedding_dim, distance_metric)
    index.load(annoy_path)
    
    print(f"Loaded Annoy index with {index.get_n_items()} items")
    print(f"Loaded metadata with {len(metadata)} faces")
    
    return index, metadata

def find_face_in_metadata(metadata, image_path, face_number):
    norm_image_path = os.path.normpath(image_path)
    
    image_faces = [face for face in metadata 
                  if os.path.normpath(face['source_image']) == norm_image_path]
    
    if not image_faces:
        raise ValueError(f"No faces found for image: {image_path}")
    
    image_faces.sort(key=lambda x: x['face_id'])
    
    if face_number < 1 or face_number > len(image_faces):
        raise ValueError(f"Face number {face_number} is out of range. Image has {len(image_faces)} faces.")
    
    target_face = image_faces[face_number - 1]
    
    return target_face

def find_similar_faces(annoy_index, metadata, target_face, num_results=10):
    embedding_path = target_face.get('embedding_path')
    if not embedding_path or not os.path.exists(embedding_path):
        raise ValueError(f"Embedding file not found: {embedding_path}")
    
    target_embedding = np.load(embedding_path)
    
    similar_indices = annoy_index.get_nns_by_vector(
        target_embedding, 
        num_results + 1,
        include_distances=True
    )
    
    similar_faces = []
    for idx, distance in zip(similar_indices[0], similar_indices[1]):
        if idx < len(metadata):
            face_data = metadata[idx].copy()
            face_data['similarity_distance'] = float(distance)
            similar_faces.append(face_data)
    
    return similar_faces

def display_face_info(face, include_bbox=False):
    face_id = face.get('face_id', 'Unknown')
    source_image = face.get('source_image', 'Unknown')
    distance = face.get('similarity_distance', 0.0)
    
    info = f"Face: {face_id}\n"
    info += f"Image: {source_image}\n"
    info += f"Similarity score: {1.0 - distance:.4f}\n"
    
    if include_bbox and 'bbox' in face:
        bbox = face['bbox']
        info += f"Bounding box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]\n"
    
    return info

def visualize_matches(target_face, similar_faces, output_path=None):
    if not output_path:
        return
    
    target_img_path = target_face['source_image']
    target_img = cv2.imread(target_img_path)
    if target_img is None:
        print(f"Warning: Could not load target image: {target_img_path}")
        return
    
    bbox = target_face['bbox']
    
    margin = 20
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(target_img.shape[1], x2 + margin)
    y2 = min(target_img.shape[0], y2 + margin)
    
    target_face_img = target_img[y1:y2, x1:x2]
    
    target_face_img = cv2.resize(target_face_img, (200, 200))
    
    num_matches = min(5, len(similar_faces) - 1)
    output_img = np.zeros((250, 200 * (num_matches + 1), 3), dtype=np.uint8)
    
    output_img[0:200, 0:200] = target_face_img
    cv2.putText(output_img, "Target", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    score_text = "Score: 1.0000"
    cv2.putText(output_img, score_text, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    for i in range(num_matches):
        match_face = similar_faces[i + 1]
        match_img_path = match_face['source_image']
        match_img = cv2.imread(match_img_path)
        
        if match_img is not None:
            match_bbox = match_face['bbox']
            
            mx1, my1, mx2, my2 = match_bbox
            mx1 = max(0, mx1 - margin)
            my1 = max(0, my1 - margin)
            mx2 = min(match_img.shape[1], mx2 + margin)
            my2 = min(match_img.shape[0], my2 + margin)
            
            match_face_img = match_img[my1:my2, mx1:mx2]
            match_face_img = cv2.resize(match_face_img, (200, 200))
            
            output_img[0:200, 200*(i+1):200*(i+2)] = match_face_img
            
            face_id = match_face['face_id'].split('_')[-1]
            cv2.putText(output_img, f"Match {i+1}", (200*(i+1) + 10, 220), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            score = 1.0 - match_face['similarity_distance']
            score_text = f"Score: {score:.4f}"
            cv2.putText(output_img, score_text, (200*(i+1) + 10, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, output_img)
    print(f"Visualization saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Find similar faces using Annoy index')
    parser.add_argument('image_path', help='Path to the image containing the face to match')
    parser.add_argument('face_number', type=int, help='Face number in the image (1-based indexing)')
    parser.add_argument('--annoy', '-a', required=True, help='Path to the Annoy index file')
    parser.add_argument('--metadata', '-m', help='Path to the metadata JSON file')
    parser.add_argument('--top', '-t', type=int, default=10, help='Number of similar faces to return')
    parser.add_argument('--distance', type=str, default='angular', 
                       choices=['angular', 'euclidean', 'manhattan', 'hamming', 'dot'],
                       help='Distance metric for Annoy index (default: angular)')
    parser.add_argument('--visualize', '-v', help='Path to save visualization image')
    args = parser.parse_args()
    
    metadata_path = args.metadata or os.path.splitext(args.annoy)[0] + '_metadata.json'
    
    try:
        annoy_index, metadata = load_resources(args.annoy, metadata_path, args.distance)
        
        print(f"Searching for face #{args.face_number} in image: {args.image_path}")
        target_face = find_face_in_metadata(metadata, args.image_path, args.face_number)
        print(f"Found target face: {target_face['face_id']}")
        
        print(f"Finding the top {args.top} similar faces...")
        similar_faces = find_similar_faces(annoy_index, metadata, target_face, args.top)
        
        print("\nTarget Face:")
        print(display_face_info(target_face, include_bbox=True))
        
        print(f"Top {len(similar_faces) - 1} Similar Faces:")
        for i, face in enumerate(similar_faces[1:], 1):
            print(f"Match #{i}:")
            print(display_face_info(face))
            print()
        
        if args.visualize:
            visualize_matches(target_face, similar_faces, args.visualize)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
