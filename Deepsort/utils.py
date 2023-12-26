import cv2
import numpy as np

# Define a function to convert detections to SORT format.
def convert_detections(detections, threshold, classes):
    # Get the bounding boxes, labels and scores from the detections dictionary.
    boxes = detections["boxes"].cpu().numpy()
    labels = detections["labels"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()
    lbl_mask = np.isin(labels, classes)
    scores = scores[lbl_mask]
    # Filter out low confidence scores and non-person classes.
    mask = scores > threshold
    boxes = boxes[lbl_mask][mask]
    scores = scores[mask]
    labels = labels[lbl_mask][mask]

    # Convert boxes to [x1, y1, w, h, score] format.
    final_boxes = []
    for i, box in enumerate(boxes):
        # Append ([x, y, w, h], score, label_string).
        final_boxes.append(
            (
                [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                scores[i],
                str(labels[i])
            )
        )

    return final_boxes

# Function for bounding box and ID annotation.
def annotate(tracks, frame, resized_frame, frame_width, frame_height, class_names, colors, output_file=None):
    detected_classes_for_frame = set()
    classes_written_to_file = set()

    try:
        # Load classes that have already been written to the file
        with open(output_file, 'r') as file:
            classes_written_to_file = set(line.strip() for line in file)
    except FileNotFoundError:
        pass  # Ignore if the file doesn't exist yet

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        track_class = track.det_class

        if track_class in detected_classes_for_frame or track_class in classes_written_to_file:
            continue  # Skip if the class has already been detected in this frame or written to the file

        x1, y1, x2, y2 = track.to_ltrb()
        p1 = (int(x1/resized_frame.shape[1]*frame_width), int(y1/resized_frame.shape[0]*frame_height))
        p2 = (int(x2/resized_frame.shape[1]*frame_width), int(y2/resized_frame.shape[0]*frame_height))

        # Annotate boxes.
        color = colors[int(track_class)]
        cv2.rectangle(
            frame,
            p1,
            p2,
            color=(int(color[0]), int(color[1]), int(color[2])),
            thickness=2
        )

        # Annotate class name instead of class track.
        class_name = class_names[int(track_class)]
        cv2.putText(
            frame, f'class: {class_name}',
            (p1[0], p1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA
        )

        # Append class name to the output file if provided and not already detected or written to the file
        if output_file and class_name not in detected_classes_for_frame and class_name not in classes_written_to_file:
            with open(output_file, 'a') as file:
                file.write(f'{class_name}\n')

        detected_classes_for_frame.add(track_class)

    return frame
