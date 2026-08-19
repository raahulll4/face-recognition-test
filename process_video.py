import os
import cv2
import numpy as np

from deepface import DeepFace
from tqdm import tqdm

from recognition import (
    load_reference_embeddings,
    identify_face,
    MODEL_NAME,
    DETECTOR_BACKEND,
)

REFERENCE_DIR = "./references"
INPUT_VIDEO_PATH = "./input/input.mp4"
OUTPUT_VIDEO_PATH = "./output/output.mp4"
SIMILARITY_THRESHOLD = 0.7
PROCESS_EVERY_N_FRAMES = 3

DISPLAY_NAMES = {
    "harry_potter": "Harry Potter",
    "ron_weasley": "Ron Weasley",
    "hermione_granger": "Hermione Granger",
    "prof_mcgonagall": "Prof. McGonagall",
    "prof_snape": "Prof. Severus Snape",
}

def draw_face_box(frame, x, y, w, h, label):
    """
    Draws a bounding box and label around a detected face on the frame.
    """

    if label == "Unknown":
        color = (0, 0, 255)
    else:
        color = (0, 180, 0)

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        color,
        2,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_width, text_height = text_size

    label_y = max(y - 10, text_height + 10)

    cv2.rectangle(
        frame,
        (x, label_y - text_height - 8),
        (x + text_width + 8, label_y + 4),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (x + 4, label_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

def process_frame(frame, reference_embeddings):
    """
    Processes one video frame by detecting faces, comparing them with reference embeddings, and returning
    labelled detections.
    """

    detections = []

    try:
        results = DeepFace.represent(
            img_path=frame,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
        )
    except Exception:
        return detections

    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height

    for result in results:
        embedding = np.array(result["embedding"])
        facial_area = result["facial_area"]

        x = int(facial_area["x"])
        y = int(facial_area["y"])
        w = int(facial_area["w"])
        h = int(facial_area["h"])

        if w <= 0 or h <= 0:
            continue

        x = max(0, x)
        y = max(0, y)
        w = min(w, frame_width - x)
        h = min(h, frame_height - y)

        if w <= 0 or h <= 0:
            continue

        ## Ignore large boxes (full screen Unknown)
        box_area = w * h

        if box_area > 0.60 * frame_area:
            continue

        character, similarity = identify_face(
            embedding,
            reference_embeddings,
            similarity_threshold=SIMILARITY_THRESHOLD,
        )

        display_name = DISPLAY_NAMES.get(character, character)

        if character == "Unknown":
            label = "Unknown"
        else:
            label = f"{display_name} ({similarity:.2f})"

        detections.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "label": label,
            }
        )

    return detections

def process_video():
    """
    Loads the reference embeddings, reads the input video frame by frame, applies face detection and recognition,
    draws labels, and saves the processed output video.
    """

    os.makedirs("./output", exist_ok=True)

    print("Loading reference embeddings...")
    reference_embeddings = load_reference_embeddings(REFERENCE_DIR)

    print("\nLoaded references:")
    for character, embeddings in reference_embeddings.items():
        print(f"{character}: {len(embeddings)} embeddings")

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {INPUT_VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\nVideo info:")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    print(f"Total frames: {total_frames}")
    print(f"Processing every {PROCESS_EVERY_N_FRAMES} frames")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        fourcc,
        fps,
        (width, height),
    )

    last_detections = []

    print("\nProcessing video...")

    for frame_index in tqdm(range(total_frames)):
        ret, frame = cap.read()

        if not ret:
            break

        should_process_frame = frame_index % PROCESS_EVERY_N_FRAMES == 0

        if should_process_frame:
            last_detections = process_frame(frame, reference_embeddings)

        for detection in last_detections:
            draw_face_box(
                frame,
                detection["x"],
                detection["y"],
                detection["w"],
                detection["h"],
                detection["label"],
            )

        out.write(frame)

    cap.release()
    out.release()

    print(f"\nDone. Output video saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    process_video()