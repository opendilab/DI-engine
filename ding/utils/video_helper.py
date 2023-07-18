import cv2
import numpy as np

def numpy_array_to_video(numpy_array, output_file, fps=30.0, codec='mp4v'):
    height, width, channels = numpy_array.shape[1:]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for frame in numpy_array:
        out.write(frame)

    out.release()

# # Example usage
# # Assuming you have a numpy array called 'frames' with shape (num_frames, height, width, channels)
# numpy_array_to_video(frames, 'output_video.mp4', fps=30.0)

# pytest for function numpy_array_to_video
# use virtual directory
