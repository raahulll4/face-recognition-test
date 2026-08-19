
# White Swan Data - ML Engineer Coding Assessment

The goal is to detect faces in the video, draw bounding boxes around them, and label recognised characters where possible.

The solution uses **DeepFace** with:

- **RetinaFace** for face detection
- **Facenet512** for face embeddings and recognition

## Characters

The following characters are included in the recognition reference set:

- Harry Potter
- Ron Weasley
- Hermione Granger
- Prof. McGonagall
- Prof. Severus Snape

## Project Structure

```text
.
├── process_video.py
├── recognition.py
├── requirements.txt
├── README.md
│
├── input/
│   └── input.mp4
│
├── output/
│   └── output.mp4
│
└── references/
    ├── harry_potter/
    ├── ron_weasley/
    ├── hermione_granger/
    ├── prof_mcgonagall/
    └── prof_snape/
```

Each character folder in `references/` contains 7 reference images used to generate face embeddings.

## Approach

The pipeline follows these main steps:

1. Load reference images for each character.
2. Generate **Facenet512** embeddings for each reference image.
3. Open the input video using OpenCV.
4. Read the video frame by frame.
5. Detect faces using **RetinaFace**.
6. Generate an embedding for each detected face.
7. Compare detected face embeddings against the reference embeddings using cosine similarity.
8. Label the face with the closest matching character if the similarity is above the threshold.
9. Label unmatched detected faces as `Unknown`.
10. Draw bounding boxes and labels on the frame.
11. Save the processed frames as an output video.

## Performance Optimisation

Processing every frame with RetinaFace and Facenet512 can be slow, so the script performs detection and recognition every few frames instead of every single frame.

The current setting is:

```python
PROCESS_EVERY_N_FRAMES = 5
```

For intermediate frames, the most recent detections are reused.

This significantly reduces runtime while maintaining acceptable visual quality for the scene.

## Similarity Threshold

The recognition threshold is configured in `process_video.py`:

```python
SIMILARITY_THRESHOLD = 0.7
```

A higher threshold makes recognition stricter and reduces incorrect labels, but may cause more faces to be labelled as `Unknown`.

A lower threshold allows more matches, but may increase incorrect labels.

## Setup

Create and activate a virtual environment if desired:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

The main dependencies are:

```text
deepface
retina-face
opencv-python
tf-keras
numpy
scikit-learn
tqdm
```

## Input Video

Place the input video in the following location with the file name `input.mp4`:

```text
input/input.mp4
```

## Run

Run the video processing script:

```bash
python process_video.py
```

## Output

The labelled output video will be saved to:

```text
output/output.mp4
```

## Validation

The following checks were performed:

- Confirmed that reference images are loaded successfully.
- Confirmed that embeddings are generated for each target character.
- Confirmed that the input video opens correctly with OpenCV.
- Confirmed that the video is processed end to end.
- Confirmed that bounding boxes are drawn around detected faces.
- Confirmed that recognised characters are labelled where possible.
- Confirmed that unmatched detected faces are labelled as `Unknown`.
- Added filtering to avoid large false `Unknown` boxes when no face is visible.

## Limitations

- Recognition accuracy depends on the quality and variety of the reference images.
- Side-profile, blurred, partially hidden, or low-resolution faces may be labelled incorrectly or as `Unknown`.
- Detection and recognition are not performed on every frame for performance reasons.
- Bounding boxes may slightly lag between processed frames because detections are reused.
- Audio is not included in the OpenCV-generated output unless merged separately.

## Implementation Notes

- RetinaFace and Facenet512 were selected because they were recommended in the task brief.
- Multiple reference images are used for each character to improve recognition reliability.
- Cosine similarity is used to compare detected face embeddings against reference embeddings.
- A similarity threshold is used to decide whether a detected face should be labelled with a known character name.
- Unknown faces are shown with a red bounding box.
- Recognised faces are shown with a green bounding box.
