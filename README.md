# Create and Query a Facial Similarity Database

This Python script is an attempt to implement a simple API for storing faces and searching for a particular face based on stored faces.

Dependencies are:
- [https://github.com/ageitgey/face_recognition](ageitgey/face_recognition) (which dependes heavily on [https://github.com/davisking/dlib](dlib)) to provide a Convolutional Neural Network to get facial encodings (128-dimensional vectors) from images
- [https://github.com/facebookresearch/faiss](faiss) as a library for creating a simple vector database