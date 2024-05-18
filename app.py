import cv2
import numpy as np

# Video path
input_video = 'input.mp4'
output_video = 'output.mp4'

# Analysis time interval
start_time = 0
end_time = 7

# Arrow color
color = (255, 0, 0)

# Read the video
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Video attributes
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set output video codec and properties
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

# Get first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()

# Turn first frame into gray scale image
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# For each frame
count = 0
while cap.isOpened() and count >= start_time * fps and count < end_time * fps:
    ret, frame = cap.read()
    if not ret:
        break

    # Turn into gray scale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # A copy frame for visualization
    vis = frame.copy()

    # Plot
    s = 1
    step = 10
    h, w = gray.shape

    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            # Remove the points that don't change too much (static background)
            if np.sqrt(fx**2 + fy**2) > 1.0:
                cv2.line(vis, (x, y), (int(x + s*fx), int(y + s*fy)), color, 1)
                cv2.circle(vis, (x, y), 1, color, -1)

    # Visualization
    out.write(vis)

    # Update info for the next frame
    prev_gray = gray.copy()
    count += 1

# Release the video assets
cap.release()
out.release()
cv2.destroyAllWindows()
