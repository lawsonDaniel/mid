from collections import Counter
import cv2
import torch
import time
import numpy as np

import pyaudio
from openal import *
from PIL import Image


SEMI_TONE = 880
SAMPLE_RATE = 44100


def create_segments(matrix, num_splits):
    matrix_rows, matrix_cols = matrix.shape
    split_size = matrix_rows // num_splits

    segments = []

    for i in range(num_splits):
        for j in range(num_splits):
            segment = matrix[
                i * split_size: (i + 1) * split_size,
                j * split_size: (j + 1) * split_size,
            ]
            segments.append(segment)

    return segments


def get_image_super_pixels(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

    slic = cv2.ximgproc.createSuperpixelSLIC(
        image, algorithm=cv2.ximgproc.SLIC, region_size=60, ruler=60.0
    )  # Create a SLIC superpixel object

    slic.iterate()  # Perform superpixel segmentation

    slic.enforceLabelConnectivity()

    num_superpixels = slic.getNumberOfSuperpixels()  # Get the number of superpixels

    labels = slic.getLabels()  # Get the labels map (superpixel assignments)

    return labels, num_superpixels


def apply_superpixel_mean(image, labels, num_pixels):
    for superpixel_label in range(num_pixels):
        superpixel_mask = (
            labels == superpixel_label
        )  # Find all pixels with the current superpixel label

        superpixel_pixels = image[superpixel_mask]

        mean_value = np.mean(superpixel_pixels, axis=0)

        image[
            superpixel_mask
        ] = mean_value  # Set all the pixels within the superpixel to the mean value

    return image


def get_superpixel_mean(image, labels, num_pixels):
    labels_mean = {}

    for superpixel_label in range(num_pixels):
        superpixel_mask = (
            labels == superpixel_label
        )  # Find all pixels with the current superpixel label

        superpixel_pixels = image[superpixel_mask]

        mean_value = np.mean(superpixel_pixels, axis=0)
        labels_mean[superpixel_label] = mean_value

    return labels_mean


def get_pixel_count_per_label(labels, num_labels):
    res = {}
    for i in range(num_labels):
        mask = labels[labels == i]
        res[i] = len(mask)
    return res


def get_horizontal_slices(labels, k):
    slices = np.linspace(0, labels.shape[1], k)
    res = []

    for i in range(len(slices) - 1):
        start = int(slices[i])
        end = int(slices[i + 1])
        mask = np.zeros(labels.shape[:2])
        mask[:, start:end] = 1
        col = labels[mask == 1]
        res.append(col)

    return res


def get_vertical_slices(labels, k):
    slices = np.linspace(0, labels.shape[0], k)
    res = []

    for i in range(len(slices) - 1):
        start = int(slices[i])
        end = int(slices[i + 1])
        mask = np.zeros(labels.shape[:2])
        mask[start:end, :] = 1
        row = labels[mask == 1]
        res.append(row)
    return res


def generate_stereo_audio(freq, amplitude, azimuth_angle, sample_rate, duration=3):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    sound_wave = np.sin(2 * np.pi * freq * t)

    # Function to apply panning based on angle
    def apply_panning(sound_wave, angle):
        # Calculate left and right channel volume based on the angle
        left_volume = np.cos(np.deg2rad(angle)) * amplitude
        right_volume = np.sin(np.deg2rad(angle)) * amplitude

        # Create stereo audio
        stereo_audio = np.column_stack(
            (sound_wave * left_volume, sound_wave * right_volume)
        )

        return stereo_audio

    # Apply panning based on the angle
    panned_audio = apply_panning(sound_wave, azimuth_angle)

    return panned_audio


def sonify_image(labels_mean_depth, labels, num_labels):
    horizontal_segment_count = len(np.unique(labels[0]))
    vertical_segment_count = len(np.unique(labels[:, 0]))
    pix_count = get_pixel_count_per_label(labels, num_labels)

    azimuth_angles = np.linspace(0, 90, horizontal_segment_count - 1)

    horizontal_slice = get_horizontal_slices(labels, horizontal_segment_count)
    vertical_slices = get_vertical_slices(labels, vertical_segment_count)

    params = np.zeros(
        (
            num_labels,
            3,
        )
    )

    for idx, _slice in enumerate(horizontal_slice):
        label_propotion = {
            k: v / pix_count.get(k) for k, v in Counter(_slice).items()}
        for label in label_propotion:
            if label_propotion[label] >= 0.5:
                params[label, 0] = azimuth_angles[idx]

    for idx, _slice in enumerate(vertical_slices):
        label_propotion = {
            k: v / pix_count.get(k) for k, v in Counter(_slice).items()}
        for label in label_propotion:
            if label_propotion[label] >= 0.5:
                params[label, 1] = SEMI_TONE * (
                    2 ** ((idx - 6) / 12)
                )  # set to -6 to +6 st

    loud = 0
    for label in range(num_labels):
        depth = labels_mean_depth[label]
        if depth > 1.5:
            amp = 0
        else:
            amp = 1 / (depth + 1)
            loud += 1

        params[label, 2] = amp

    samples = [
        generate_stereo_audio(
            params[i, 1], params[i, 2], params[i, 0], SAMPLE_RATE, 0.2
        )
        for i in range(num_labels)
    ]

    audio = np.sum(samples, axis=0, dtype=np.float32)

    audio = audio / (np.max(audio) + 1)

    return audio


if __name__ == "__main__":
    # from PIL import Image
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Couldn't open the camera.")
        exit()

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,  # Format for floating-point data
        channels=2,  # Mono audio (change to 2 for stereo)
        rate=SAMPLE_RATE,  # Sample rate (adjust as needed)
        output=True,  # Set as output stream
    )

    # Open the camera
    try:
        model_type = "ZoeD_N"  # Trained on NewYork Dataset
        # model_type = "ZoeD_K"   #Trained on KITTI Dataset
        # model_type = "ZoeD_NK" #Trained on Both KITTI and NewYork Dataset

        model_zoe_nk = torch.hub.load(
            "./ZoeDepth", model_type, source="local", pretrained=True
        )

        """Move model to GPU if available"""

        device = (
            torch.device("cuda") if torch.cuda.is_available(
            ) else torch.device("cpu")
        )

        print(device)
        model_zoe_nk = model_zoe_nk.to(device)

        should_display = (
            len(sys.argv) > 1 and sys.argv[1] == "1"
        )  # controls if the depth map should be displayed back to the user
        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(rgb_frame)
            img = img.resize((int(img.width * 0.8), int(img.height * 0.8)))

            frame = np.array(img)

            labels, num_pix = get_image_super_pixels(frame)

            _s = time.time()

            output = model_zoe_nk.infer_pil(img)

            print(f"it took {time.time() - _s} to infer")

            h, w = output.shape

            # Rescale the depth map for visualization
            depth_colormap = cv2.normalize(output, None, 0, 1, cv2.NORM_MINMAX)

            depth_colormap = 1 - depth_colormap

            depth_colormap = (depth_colormap * 255).astype(np.uint8)
            depth_colormap = cv2.applyColorMap(
                depth_colormap, cv2.COLORMAP_MAGMA)

            depth_w_superpixel_norm = apply_superpixel_mean(
                depth_colormap, labels, num_pix
            )

            label_values = get_superpixel_mean(output, labels, num_pix)

            audio = sonify_image(label_values, labels, num_pix)

            stream.write(audio.tobytes())

            if should_display:
                cv2.putText(
                    depth_w_superpixel_norm,
                    f"FPS: {1 / (time.time() - start)}",
                    (7, 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (100, 255, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.imshow("Depth Map", depth_w_superpixel_norm)

            if cv2.waitKey(1) == 27:  # ESC key
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

        stream.stop_stream()
        stream.close()
        p.terminate()
