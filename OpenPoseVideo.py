import cv2
import time
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="cpu", help="Device to inference on")
parser.add_argument("--video_file", default="sample_video.mp4", help="Input Video")

args = parser.parse_args()

MODE = "COCO"
INPUT_FILE = "video/2.avi"
output_file = f"video/{os.path.basename(INPUT_FILE)[:-4]}-output.avi"

if MODE == "COCO":
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
                  [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]
    KEY_POINTS = {"Nose": 0, "Neck": 1, "Right Shoulder": 2, "Right Elbow": 3, "Right Wrist": 4, "Left Shoulder": 5,
                  "Left Elbow": 6, "Left Wrist": 7, "Right Hip": 8, "Right Knee": 9, "Right Ankle": 10,
                  "Left Hip": 11, "Left Knee": 12, "LAnkle": 13, "Right Eye": 14, "Left Eye": 15, "Right Ear": 16,
                  "Left Ear": 17, "Background": 18}

elif MODE == "MPI":
    protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
                  [11, 12], [12, 13]]
    KEY_POINTS = {"Head": 0, "Neck": 1, "Right Shoulder": 2, "Right Elbow": 3, "Right Wrist": 4,
                  "Left Shoulder": 5, "Left Elbow": 6, "Left Wrist": 7, "Right Hip": 8, "Right Knee": 9,
                  "Right Ankle": 10, "Left Hip": 11, "Left Knee": 12, "Left Ankle": 13, "Chest": 14,
                  "Background": 15}

ARM_FOLD = ["Neck", "Right Shoulder", "Left Shoulder"]
# ARM_FOLD = ["Right Elbow", "Right Wrist", "Left Elbow", "Left Wrist"]
# ARM_FOLD = ["Neck", "Right Shoulder", "Right Elbow", "Right Wrist", "Left Shoulder", "Left Elbow", "Left Wrist"]
ARM_FOLD_IDS = [KEY_POINTS[kp] for kp in KEY_POINTS if kp in ARM_FOLD]
print(f"{ARM_FOLD_IDS=}")

inWidth = 368
inHeight = 368
threshold = 0.1

input_source = INPUT_FILE  # args.video_file
cap = cv2.VideoCapture(input_source)
hasFrame, frame = cap.read()

vid_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                             (frame.shape[1], frame.shape[0]))

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args.device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

while cv2.waitKey(1) < 0:
    t = time.time()
    hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)
    if not hasFrame:
        cv2.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold:
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else:
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if not (partA in ARM_FOLD_IDS and partB in ARM_FOLD_IDS):
            continue  # skip key points which are not in arm_fold

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8,
                (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.imshow('Output-Keypoints', frameCopy)
    cv2.imshow('Output-Skeleton', frame)

    vid_writer.write(frame)

vid_writer.release()
