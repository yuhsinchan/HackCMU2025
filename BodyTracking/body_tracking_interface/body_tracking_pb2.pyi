from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Keypoint2D(_message.Message):
    __slots__ = ("x", "y")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ...) -> None: ...

class Keypoint3D(_message.Message):
    __slots__ = ("x", "y", "z")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ...) -> None: ...

class BodyTracking34(_message.Message):
    __slots__ = ("frame_number", "confidence", "pelvis_2d", "naval_spine_2d", "chest_spine_2d", "neck_2d", "left_clavicle_2d", "left_shoulder_2d", "left_elbow_2d", "left_wrist_2d", "left_hand_2d", "left_handtip_2d", "left_thumb_2d", "right_clavicle_2d", "right_shoulder_2d", "right_elbow_2d", "right_wrist_2d", "right_hand_2d", "right_handtip_2d", "right_thumb_2d", "left_hip_2d", "left_knee_2d", "left_ankle_2d", "left_foot_2d", "left_heel_2d", "right_hip_2d", "right_knee_2d", "right_ankle_2d", "right_foot_2d", "right_heel_2d", "nose_2d", "left_eye_2d", "left_ear_2d", "right_eye_2d", "right_ear_2d", "head_2d", "pelvis_3d", "naval_spine_3d", "chest_spine_3d", "neck_3d", "left_clavicle_3d", "left_shoulder_3d", "left_elbow_3d", "left_wrist_3d", "left_hand_3d", "left_handtip_3d", "left_thumb_3d", "right_clavicle_3d", "right_shoulder_3d", "right_elbow_3d", "right_wrist_3d", "right_hand_3d", "right_handtip_3d", "right_thumb_3d", "left_hip_3d", "left_knee_3d", "left_ankle_3d", "left_foot_3d", "left_heel_3d", "right_hip_3d", "right_knee_3d", "right_ankle_3d", "right_foot_3d", "right_heel_3d", "nose_3d", "left_eye_3d", "left_ear_3d", "right_eye_3d", "right_ear_3d", "head_3d")
    FRAME_NUMBER_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    PELVIS_2D_FIELD_NUMBER: _ClassVar[int]
    NAVAL_SPINE_2D_FIELD_NUMBER: _ClassVar[int]
    CHEST_SPINE_2D_FIELD_NUMBER: _ClassVar[int]
    NECK_2D_FIELD_NUMBER: _ClassVar[int]
    LEFT_CLAVICLE_2D_FIELD_NUMBER: _ClassVar[int]
    LEFT_SHOULDER_2D_FIELD_NUMBER: _ClassVar[int]
    LEFT_ELBOW_2D_FIELD_NUMBER: _ClassVar[int]
    LEFT_WRIST_2D_FIELD_NUMBER: _ClassVar[int]
    LEFT_HAND_2D_FIELD_NUMBER: _ClassVar[int]
    LEFT_HANDTIP_2D_FIELD_NUMBER: _ClassVar[int]
    LEFT_THUMB_2D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_CLAVICLE_2D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_SHOULDER_2D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_ELBOW_2D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_WRIST_2D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_HAND_2D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_HANDTIP_2D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_THUMB_2D_FIELD_NUMBER: _ClassVar[int]
    LEFT_HIP_2D_FIELD_NUMBER: _ClassVar[int]
    LEFT_KNEE_2D_FIELD_NUMBER: _ClassVar[int]
    LEFT_ANKLE_2D_FIELD_NUMBER: _ClassVar[int]
    LEFT_FOOT_2D_FIELD_NUMBER: _ClassVar[int]
    LEFT_HEEL_2D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_HIP_2D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_KNEE_2D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_ANKLE_2D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FOOT_2D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_HEEL_2D_FIELD_NUMBER: _ClassVar[int]
    NOSE_2D_FIELD_NUMBER: _ClassVar[int]
    LEFT_EYE_2D_FIELD_NUMBER: _ClassVar[int]
    LEFT_EAR_2D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_EYE_2D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_EAR_2D_FIELD_NUMBER: _ClassVar[int]
    HEAD_2D_FIELD_NUMBER: _ClassVar[int]
    PELVIS_3D_FIELD_NUMBER: _ClassVar[int]
    NAVAL_SPINE_3D_FIELD_NUMBER: _ClassVar[int]
    CHEST_SPINE_3D_FIELD_NUMBER: _ClassVar[int]
    NECK_3D_FIELD_NUMBER: _ClassVar[int]
    LEFT_CLAVICLE_3D_FIELD_NUMBER: _ClassVar[int]
    LEFT_SHOULDER_3D_FIELD_NUMBER: _ClassVar[int]
    LEFT_ELBOW_3D_FIELD_NUMBER: _ClassVar[int]
    LEFT_WRIST_3D_FIELD_NUMBER: _ClassVar[int]
    LEFT_HAND_3D_FIELD_NUMBER: _ClassVar[int]
    LEFT_HANDTIP_3D_FIELD_NUMBER: _ClassVar[int]
    LEFT_THUMB_3D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_CLAVICLE_3D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_SHOULDER_3D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_ELBOW_3D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_WRIST_3D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_HAND_3D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_HANDTIP_3D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_THUMB_3D_FIELD_NUMBER: _ClassVar[int]
    LEFT_HIP_3D_FIELD_NUMBER: _ClassVar[int]
    LEFT_KNEE_3D_FIELD_NUMBER: _ClassVar[int]
    LEFT_ANKLE_3D_FIELD_NUMBER: _ClassVar[int]
    LEFT_FOOT_3D_FIELD_NUMBER: _ClassVar[int]
    LEFT_HEEL_3D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_HIP_3D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_KNEE_3D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_ANKLE_3D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FOOT_3D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_HEEL_3D_FIELD_NUMBER: _ClassVar[int]
    NOSE_3D_FIELD_NUMBER: _ClassVar[int]
    LEFT_EYE_3D_FIELD_NUMBER: _ClassVar[int]
    LEFT_EAR_3D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_EYE_3D_FIELD_NUMBER: _ClassVar[int]
    RIGHT_EAR_3D_FIELD_NUMBER: _ClassVar[int]
    HEAD_3D_FIELD_NUMBER: _ClassVar[int]
    frame_number: int
    confidence: float
    pelvis_2d: Keypoint2D
    naval_spine_2d: Keypoint2D
    chest_spine_2d: Keypoint2D
    neck_2d: Keypoint2D
    left_clavicle_2d: Keypoint2D
    left_shoulder_2d: Keypoint2D
    left_elbow_2d: Keypoint2D
    left_wrist_2d: Keypoint2D
    left_hand_2d: Keypoint2D
    left_handtip_2d: Keypoint2D
    left_thumb_2d: Keypoint2D
    right_clavicle_2d: Keypoint2D
    right_shoulder_2d: Keypoint2D
    right_elbow_2d: Keypoint2D
    right_wrist_2d: Keypoint2D
    right_hand_2d: Keypoint2D
    right_handtip_2d: Keypoint2D
    right_thumb_2d: Keypoint2D
    left_hip_2d: Keypoint2D
    left_knee_2d: Keypoint2D
    left_ankle_2d: Keypoint2D
    left_foot_2d: Keypoint2D
    left_heel_2d: Keypoint2D
    right_hip_2d: Keypoint2D
    right_knee_2d: Keypoint2D
    right_ankle_2d: Keypoint2D
    right_foot_2d: Keypoint2D
    right_heel_2d: Keypoint2D
    nose_2d: Keypoint2D
    left_eye_2d: Keypoint2D
    left_ear_2d: Keypoint2D
    right_eye_2d: Keypoint2D
    right_ear_2d: Keypoint2D
    head_2d: Keypoint2D
    pelvis_3d: Keypoint3D
    naval_spine_3d: Keypoint3D
    chest_spine_3d: Keypoint3D
    neck_3d: Keypoint3D
    left_clavicle_3d: Keypoint3D
    left_shoulder_3d: Keypoint3D
    left_elbow_3d: Keypoint3D
    left_wrist_3d: Keypoint3D
    left_hand_3d: Keypoint3D
    left_handtip_3d: Keypoint3D
    left_thumb_3d: Keypoint3D
    right_clavicle_3d: Keypoint3D
    right_shoulder_3d: Keypoint3D
    right_elbow_3d: Keypoint3D
    right_wrist_3d: Keypoint3D
    right_hand_3d: Keypoint3D
    right_handtip_3d: Keypoint3D
    right_thumb_3d: Keypoint3D
    left_hip_3d: Keypoint3D
    left_knee_3d: Keypoint3D
    left_ankle_3d: Keypoint3D
    left_foot_3d: Keypoint3D
    left_heel_3d: Keypoint3D
    right_hip_3d: Keypoint3D
    right_knee_3d: Keypoint3D
    right_ankle_3d: Keypoint3D
    right_foot_3d: Keypoint3D
    right_heel_3d: Keypoint3D
    nose_3d: Keypoint3D
    left_eye_3d: Keypoint3D
    left_ear_3d: Keypoint3D
    right_eye_3d: Keypoint3D
    right_ear_3d: Keypoint3D
    head_3d: Keypoint3D
    def __init__(self, frame_number: _Optional[int] = ..., confidence: _Optional[float] = ..., pelvis_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., naval_spine_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., chest_spine_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., neck_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., left_clavicle_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., left_shoulder_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., left_elbow_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., left_wrist_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., left_hand_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., left_handtip_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., left_thumb_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., right_clavicle_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., right_shoulder_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., right_elbow_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., right_wrist_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., right_hand_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., right_handtip_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., right_thumb_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., left_hip_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., left_knee_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., left_ankle_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., left_foot_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., left_heel_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., right_hip_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., right_knee_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., right_ankle_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., right_foot_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., right_heel_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., nose_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., left_eye_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., left_ear_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., right_eye_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., right_ear_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., head_2d: _Optional[_Union[Keypoint2D, _Mapping]] = ..., pelvis_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., naval_spine_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., chest_spine_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., neck_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., left_clavicle_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., left_shoulder_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., left_elbow_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., left_wrist_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., left_hand_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., left_handtip_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., left_thumb_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., right_clavicle_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., right_shoulder_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., right_elbow_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., right_wrist_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., right_hand_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., right_handtip_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., right_thumb_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., left_hip_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., left_knee_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., left_ankle_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., left_foot_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., left_heel_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., right_hip_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., right_knee_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., right_ankle_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., right_foot_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., right_heel_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., nose_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., left_eye_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., left_ear_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., right_eye_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., right_ear_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ..., head_3d: _Optional[_Union[Keypoint3D, _Mapping]] = ...) -> None: ...

class Counter(_message.Message):
    __slots__ = ("count", "state")
    COUNT_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    count: int
    state: int
    def __init__(self, count: _Optional[int] = ..., state: _Optional[int] = ...) -> None: ...
