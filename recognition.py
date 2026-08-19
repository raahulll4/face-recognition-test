from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from deepface import DeepFace

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"
SUPPORTED_EXTENSIONS = {".png"}

def load_reference_embeddings(reference_dir: str) -> dict[str, list[np.ndarray]]:
    """
    Loads all reference images from the given directory and generates FaceNet512 embeddings for each character.
    """

    reference_embeddings = {}

    reference_path = Path(reference_dir)

    for character_dir in sorted(reference_path.iterdir()):
        if not character_dir.is_dir():
            continue

        character = character_dir.name
        embeddings = []

        image_paths = sorted(
            path
            for path in character_dir.iterdir()
            if path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        print(f"Loading references for {character}...")

        for image_path in image_paths:
            print(f"  {image_path.name}")

            result = DeepFace.represent(
                img_path=str(image_path),
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
            )

            embedding = np.array(result[0]["embedding"])
            embeddings.append(embedding)

        reference_embeddings[character] = embeddings

    return reference_embeddings

def identify_face(face_embedding: np.ndarray, reference_embeddings: dict[str, list[np.ndarray]],
    similarity_threshold: float = 0.70,) -> tuple[str, float]:
    """
    Compares a detected face embedding against the reference embeddings and returns the best matching character if the
    similarity is high enough.
    """

    best_character = "Unknown"
    best_similarity = -1.0

    for character, embeddings in reference_embeddings.items():
        for reference_embedding in embeddings:
            similarity = cosine_similarity(
                [face_embedding],
                [reference_embedding],
            )[0][0]

            if similarity > best_similarity:
                best_similarity = similarity
                best_character = character

    if best_similarity < similarity_threshold:
        return "Unknown", best_similarity

    return best_character, best_similarity