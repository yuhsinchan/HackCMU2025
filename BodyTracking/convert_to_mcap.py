import json
from body_tracking_interface.body_tracking_pb2 import (
    BodyTracking34,
    Keypoint2D,
    Keypoint3D,
    Counter,
)
from mcap_protobuf.writer import Writer
import numpy as np
import argparse


def get_proto_message(data) -> BodyTracking34:
    frame_number = data["frame_number"]
    confidence = data["bodies"][0]["confidence"]
    keypoints_2d = data["bodies"][0]["keypoints_2d"]
    keypoints_3d = data["bodies"][0]["keypoints_3d"]

    tags = [
        "pelvis",
        "naval_spine",
        "chest_spine",
        "neck",
        "left_clavicle",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "left_hand",
        "left_handtip",
        "left_thumb",
        "right_clavicle",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "right_hand",
        "right_handtip",
        "right_thumb",
        "left_hip",
        "left_knee",
        "left_ankle",
        "left_foot",
        "right_hip",
        "right_knee",
        "right_ankle",
        "right_foot",
        "head",
        "nose",
        "left_eye",
        "left_ear",
        "right_eye",
        "right_ear",
        "left_heel",
        "right_heel",
    ]

    body_tracking_dict = {
        "frame_number": frame_number,
        "confidence": confidence,
    }

    for tag, kp2d, kp3d in zip(tags, keypoints_2d, keypoints_3d):
        body_tracking_dict[f"{tag}_2d"] = Keypoint2D(x=kp2d[0], y=kp2d[1])
        body_tracking_dict[f"{tag}_3d"] = Keypoint3D(x=kp3d[0], y=kp3d[1], z=kp3d[2])

    body_tracking_msg = BodyTracking34(**body_tracking_dict)

    return body_tracking_msg


def split_squat(head_y_list: list):
    # pass the list to a low pass filter and find the peaks
    lpf_head_y = []
    for i in range(len(head_y_list)):
        if i == 0:
            lpf_head_y.append(head_y_list[i])
        else:
            lpf_head_y.append(0.9 * lpf_head_y[i - 1] + 0.1 * head_y_list[i])

    lpf_head_y_copy = lpf_head_y.copy()
    # throw away the first 10 values
    lpf_head_y = lpf_head_y[10:]

    # find the peaks
    h_peaks = []
    l_peaks = []

    for i in range(1, len(lpf_head_y) - 1):
        if lpf_head_y[i] > lpf_head_y[i - 1] and lpf_head_y[i] > lpf_head_y[i + 1]:
            h_peaks.append(i + 10)  # add 10 to account for the thrown away values
        if lpf_head_y[i] < lpf_head_y[i - 1] and lpf_head_y[i] < lpf_head_y[i + 1]:
            l_peaks.append(i + 10)  # add 10 to account for the thrown away values

    print(f"Found {len(h_peaks)} peaks")
    return h_peaks, l_peaks, lpf_head_y_copy

NUM_SQUAT_FAILURE_CLASS = 6

def parse_labels(label_str: str):
    y = np.zeros(NUM_SQUAT_FAILURE_CLASS, dtype=np.int8)
    if label_str.strip() == "":
        return y
    for idx in label_str.split(","):
        i = int(idx)
        if 0 <= i < NUM_SQUAT_FAILURE_CLASS:
            y[i] = 1
        else:
            raise ValueError(f"Invalid label index {i}, must be 0–{NUM_SQUAT_FAILURE_CLASS-1}")
    return y

def main():
    parser = argparse.ArgumentParser(description="Convert body tracking JSON to MCAP")
    parser.add_argument(
        "--input_json", type=str, help="Path to input JSON file", required=True
    )
    parser.add_argument(
        "--output_mcap", type=str, help="Path to output MCAP file", default=None
    )
    parser.add_argument(
        "--label",
        type=str,
        help="comma-separated list of error indices (0–5). Empty means perfect rep.",
        default=""
    )
    args = parser.parse_args()

    if args.output_mcap is None:
        args.output_mcap = args.input_json.replace(".json", ".mcap")

    with open(args.input_json, "r") as f:
        data = json.load(f)

    start_idx = -1
    ts = data["data"][start_idx]["timestamp"]

    while ts - data["data"][start_idx]["timestamp"] < 1:
        if start_idx * -1 == len(data["data"]):
            break
        ts = data["data"][start_idx]["timestamp"]
        start_idx -= 1

    start_idx += len(data["data"]) + 1  # convert to positive index
    head_y_list = [
        i["bodies"][0]["keypoints_3d"][26][1] for i in data["data"][start_idx:]
    ]
    peaks, l_peaks, lpf_head_y = split_squat(head_y_list)

    per_squat_data = []

    with open(args.output_mcap, "wb") as mcap_file:
        writer = Writer(mcap_file)

        for entry in data["data"][start_idx:]:
            body_tracking_msg = get_proto_message(entry)
            writer.write_message(
                topic="/body_tracking",
                message=body_tracking_msg,
                publish_time=int(entry["timestamp"] * 1e9),
                log_time=int(entry["timestamp"] * 1e9),
            )

        state = 0  # 0: idle, 1: down
        count = 0

        h = 0
        l = 0

        sq = []
        multi_label = parse_labels(args.label)

        for i in range(start_idx, len(data["data"])):
            if l < len(l_peaks) and i - start_idx == l_peaks[l]:
                if state == 1:  # was down
                    state = 0  # now idle
                    count += 1
                    per_squat_data.append(
                        {
                            "label": multi_label,
                            "data": sq,
                        }
                    )

                    sq = []
                    print(f"Count: {count} at frame {i - start_idx}")
                l += 1

            if h < len(peaks) and i - start_idx == peaks[h]:
                if state == 0:  # was idle
                    state = 1  # now down
                h += 1

            if state == 1:
                sq.append(
                    {
                        "ts": data["data"][i]["timestamp"],
                        "keypoints_3d": data["data"][i]["bodies"][0]["keypoints_3d"],
                    }
                )

            counter_msg = Counter(count=count, state=state)
            writer.write_message(
                topic="/squat_counter",
                message=counter_msg,
                publish_time=int(data["data"][i]["timestamp"] * 1e9),
                log_time=int(data["data"][i]["timestamp"] * 1e9),
            )

        writer.finish()

    with open(args.output_mcap.replace(".mcap", "_per_squat.json"), "w") as f:
        json.dump(per_squat_data, f)


if __name__ == "__main__":
    main()
