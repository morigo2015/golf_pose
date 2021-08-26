import cv2
import time
import numpy as np
import argparse
import os

from my_util import Colours

parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="cpu", help="Device to inference on")
parser.add_argument("--video_file", default=None, help="Input Video")  # default="sample_video.mp4",
parser.add_argument("--output_file", default=None, help="Input Video")  # default="sample_video.mp4",

args = parser.parse_args()

MODE = "COCO"
INPUT_FILE = "video/3.avi"
output_file = args.output_file if args.output_file else f"video/{os.path.basename(INPUT_FILE)[:-4]}-output.avi"

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

KP_GROUPS = [("arms", Colours.BGR_RED, ("Right Elbow", "Right Wrist", "Left Elbow", "Left Wrist")),
             ("shoulders", Colours.BGR_BLUE, ("Neck", "Right Shoulder", "Left Shoulder",))]

ARM_FOLD = ["Neck", "Right Shoulder", "Left Shoulder", ]
# ARM_FOLD = ["Right Elbow", "Right Wrist", "Left Elbow", "Left Wrist"]
# ARM_FOLD = ["Neck", "Right Shoulder", "Right Elbow", "Right Wrist", "Left Shoulder", "Left Elbow", "Left Wrist"]
ARM_FOLD_IDS = [KEY_POINTS[kp] for kp in KEY_POINTS if kp in ARM_FOLD]
print(f"{ARM_FOLD_IDS=}")


def _get_colour(key_point_id):
    for grp in KP_GROUPS:
        if key_point_id in [KEY_POINTS[kp_name] for kp_name in grp[2]]:
            return grp[1]
    return None


def _draw_kp(kp_id, kp_colour):
    if kp_id is None:
        return
    if kp_colour:
        cv2.circle(frame, points[kp_id], 8, kp_colour, thickness=-1, lineType=cv2.FILLED)
    else:
        cv2.circle(frame, points[kp_id], 3, Colours.BGR_GRAY, thickness=-1, lineType=cv2.FILLED)


inWidth = 368
inHeight = 368
threshold = 0.1

# if args.video_file is None:
#     input_source = INPUT_FILE  # args.video_file
input_source = args.video_file if args.video_file else INPUT_FILE
print(f"{input_source=} {output_file=}")

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
            # cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            # cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
            #             lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else:
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        colour_A = _get_colour(partA)
        colour_B = _get_colour(partB)

        _draw_kp(partA, colour_A)
        _draw_kp(partB, colour_B)

        if points[partA] and points[partB]:
            if colour_A and colour_B and colour_A == colour_B:
                cv2.line(frame, points[partA], points[partB], colour_A, thickness=3, lineType=cv2.LINE_AA)
            else:
                cv2.line(frame, points[partA], points[partB], Colours.BGR_GRAY, thickness=1, lineType=cv2.LINE_AA)

    cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8,
                (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.imshow('Output-Keypoints', frameCopy)
    cv2.imshow('Output-Skeleton', frame)

    vid_writer.write(frame)

vid_writer.release()
