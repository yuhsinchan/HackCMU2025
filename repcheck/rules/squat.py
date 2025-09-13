import numpy as np
import argparse
import json
import pdb
from repcheck.canonicalize import canonicalize_body34

BODY34_IDX = {
    # Core / spine
    "PELVIS": 0,
    "L_BACK": 1,
    "U_BACK": 2,
    "NECK": 3,
    "R_CLAVICLE": 4,
    "L_CLAVICLE": 5,
    # Head
    "HEAD_BASE": 26,
    "HEAD": 27,   
    "HEAD_R": 28,
    "R_EAR": 29,
    "HEAD_L": 30,
    "L_EAR": 31,
    # Right arm
    "R_SHO": 6,
    "R_ELBOW": 7,
    "R_WRIST": 8,
    "R_HAND": 9,
    "R_HAND_TIP": 10,
    # Left arm
    "L_SHO": 11,
    "L_ELBOW": 12,
    "L_WRIST": 13,
    "L_HAND": 14,
    "L_HAND_TIP": 17,
    # Right leg
    "R_HIP": 18,
    "R_KNEE": 19,
    "R_ANK": 20,
    "R_TOE": 21,
    "R_HEEL": 32,
    # Left leg
    "L_HIP": 22,
    "L_KNEE": 23,
    "L_ANK": 24,
    "L_TOE": 25,
    "L_HEEL": 33,
}


class SquatMistakeDetector:
    MISTAKES = [
        "feet_too_wide",          # stance wider than hip width tolerance
        "feet_too_narrow",        # stance narrower than hip width tolerance
        "knee_valgus",            # knees collapsing inward
        "butt_wink",              # posterior pelvic tilt at the bottom
        "torso_lean",             # excessive forward torso collapse
        "spinal_rounding",        # lumbar or upper back rounding
        "squat_shallow",          # squat too shallow
        "heel_lift"               # heels coming off the ground
    ]

    # BODY_34 indices (edit if your mapping differs)
    IDX = dict(
        L_HIP=22, R_HIP=18,
        L_KNEE=23, R_KNEE=19,
        L_ANK=24, R_ANK=20,
        L_TOE=25, R_TOE=21,
        L_HEEL=33, R_HEEL=32,
        L_SHO=11, R_SHO=4,    
        U_BACK=2, L_BACK=1, HEAD=27
    )


    def __init__(self, cfg: dict = None):
        self.cfg = dict(
            wide_k=2.5,          # foot_width > 2.5 * hip_width
            narrow_k=0.9,        # foot_width < 0.9 * hip_width
            valgus_k=1.2,        # foot_width > 1.2 * knee_width
            torso_lean_deg=30.0, # torso angle from vertical
            spine_round_deg=30.0,# head–shoulder–hip curvature proxy
            heel_lift_eps=0.02,  # heel above toe by > 2 cm (unit-dependent)
        )
        if cfg:
            self.cfg.update(cfg)

    @staticmethod
    def _mid(a, b): return (a + b) * 0.5

    @staticmethod
    def _dist(a, b): return float(np.linalg.norm(a - b))

    @staticmethod
    def _angle_deg(u, v):
        den = (np.linalg.norm(u) * np.linalg.norm(v)) + 1e-9
        c = np.clip(np.dot(u, v) / den, -1.0, 1.0)
        return float(np.degrees(np.arccos(c)))
    
    def feet_to_hip_width_ratio(self, kps: np.ndarray) -> float:
        """
        Returns the ratio of foot width to hip width.
        """
        LHIP, RHIP = kps[BODY34_IDX["L_HIP"]], kps[BODY34_IDX["R_HIP"]]
        LANK, RANK = kps[BODY34_IDX["L_ANK"]], kps[BODY34_IDX["R_ANK"]]
        hip_width = self._dist(LHIP, RHIP)
        foot_width = self._dist(LANK, RANK)
        if hip_width < 1e-6:
            return 0.0
        print(f"Hip to feet width ratio: {foot_width / hip_width:.2f}")
        return foot_width / hip_width

    def feet_to_knee_width_ratio(self, kps: np.ndarray) -> float:
        """
        Returns the ratio of knee lateral width to foot lateral width.
        """
        LKNE, RKNE = kps[BODY34_IDX["L_KNEE"]], kps[BODY34_IDX["R_KNEE"]]
        LANK, RANK = kps[BODY34_IDX["L_ANK"]], kps[BODY34_IDX["R_ANK"]]
        knee_width = self._dist(LKNE, RKNE)
        foot_width = self._dist(LANK, RANK)
        if foot_width < 1e-6:
            return 0.0
        print(f"Foot to knee width ratio: {foot_width / knee_width:.2f}")
        return foot_width / knee_width

    def hip_below_knee(self, kps: np.ndarray, up_axis: int = 1) -> bool:
        """
        Returns True if hip center is below average knee height (in up_axis).
        """
        LHIP, RHIP = kps[BODY34_IDX["L_HIP"]], kps[BODY34_IDX["R_HIP"]]
        LKNE, RKNE = kps[BODY34_IDX["L_KNEE"]], kps[BODY34_IDX["R_KNEE"]]
        hip_center = self._mid(LHIP, RHIP)
        knees_up = 0.5 * (LKNE[up_axis] + RKNE[up_axis])
        print(f"Hip above knee {(knees_up - hip_center[up_axis]):.2f}")

        return hip_center[up_axis] < knees_up

    def torso_angle_from_vertical(self, kps: np.ndarray, up_axis: int = 1) -> float:
        """
        Returns the angle (degrees) between torso vector and vertical axis.
        """
        if "L_SHO" in BODY34_IDX and "R_SHO" in BODY34_IDX:
            LHIP, RHIP = kps[BODY34_IDX["L_HIP"]], kps[BODY34_IDX["R_HIP"]]
            LSHO, RSHO = kps[BODY34_IDX["L_SHO"]], kps[BODY34_IDX["R_SHO"]]
            hip_center = self._mid(LHIP, RHIP)
            sho_center = self._mid(LSHO, RSHO)
            torso = sho_center - hip_center
            up_vec = np.zeros(3); up_vec[up_axis] = 1.0
            print(f"Torso angle from vertical: {self._angle_deg(torso, up_vec):.2f} degrees")
            return self._angle_deg(torso, up_vec)
        return 0.0

    def spine_curvature_angle(self, kps: np.ndarray) -> float:
        """
        Returns the angle (degrees) between shoulder-hip and head-shoulder vectors.
        """
        if "L_SHO" in BODY34_IDX and "R_SHO" in BODY34_IDX and "HEAD" in BODY34_IDX:
            LHIP, RHIP = kps[BODY34_IDX["L_HIP"]], kps[BODY34_IDX["R_HIP"]]
            LSHO, RSHO = kps[BODY34_IDX["L_SHO"]], kps[BODY34_IDX["R_SHO"]]
            HEAD = kps[BODY34_IDX["HEAD"]]
            hip_center = self._mid(LHIP, RHIP)
            sho_center = self._mid(LSHO, RSHO)
            v1 = sho_center - hip_center
            v2 = HEAD - sho_center
            return self._angle_deg(v1, v2)
        return 0.0

    def heel_lift_amount(self, kps: np.ndarray, up_axis: int = 1) -> tuple[float, float]:
        """
        Returns the vertical difference between heel and toe for left and right foot.
        """
        LHEL, RHEL = kps[BODY34_IDX["L_HEEL"]], kps[BODY34_IDX["R_HEEL"]]
        LTOE, RTOE = kps[BODY34_IDX["L_TOE"]], kps[BODY34_IDX["R_TOE"]]
        l_heel_lift = LHEL[up_axis] - LTOE[up_axis]
        r_heel_lift = RHEL[up_axis] - RTOE[up_axis]
        return l_heel_lift, r_heel_lift

    def detect_mistakes(
        self,
        kps: np.ndarray, # keypoints
        up_axis: int = 1,     # 0:x, 1:y, 2:z  (y-up default)
        x_axis: int = 0
    ) -> list[bool]:
        """
        kps: (34,3) BODY_34 3D array. Returns [bool]*8 in MISTAKES order.
        Rule-based, single-frame.
        """

        # Width-based mistakes
        width_ratio = self.feet_to_hip_width_ratio(kps)
        feet_too_wide   = width_ratio > self.cfg["wide_k"]
        feet_too_narrow = width_ratio < self.cfg["narrow_k"]

        # Knee valgus
        knee_valgus_ratio = self.feet_to_knee_width_ratio(kps)
        knee_valgus = (knee_valgus_ratio > self.cfg["valgus_k"]) and (not feet_too_wide)

        # Depth error
        squat_shallow = not self.hip_below_knee(kps, up_axis)

        # Torso lean
        torso_angle = self.torso_angle_from_vertical(kps, up_axis)
        torso_lean = torso_angle > self.cfg["torso_lean_deg"]

        # Spinal rounding
        spine_angle = self.spine_curvature_angle(kps)
        spine_rounding = spine_angle > self.cfg["spine_round_deg"]

        # Heel lift
        l_heel_lift, r_heel_lift = self.heel_lift_amount(kps, up_axis)
        heel_lift = (l_heel_lift > self.cfg["heel_lift_eps"]) or (r_heel_lift > self.cfg["heel_lift_eps"])

        # Butt wink not robust single-frame; leave False
        butt_wink = False

        return [
            bool(feet_too_wide),
            bool(feet_too_narrow),
            bool(knee_valgus),
            bool(butt_wink),
            bool(torso_lean),
            bool(spine_rounding),
            bool(squat_shallow),
            bool(heel_lift),
        ]
    # User-friendly mistake descriptions as a dict
    MISTAKE_DESCRIPTIONS = {
        "feet_too_wide": "Your feet are too wide. Try narrowing your stance.",
        "feet_too_narrow": "Your feet are too close together. Widen your stance.",
        "knee_valgus": "Your knees are collapsing inward. Focus on keeping them out.",
        "butt_wink": "Push your hips back and keep your back straight as you go down. Stop before your lower back starts to round.",
        "torso_lean": "You're leaning your torso forward too much. Keep your chest up.",
        "spinal_rounding": "Your back is rounding. Maintain a straight spine.",
        "squat_shallow": "You are not squatting deep enough. Lower your hips below your knees.",
        "heel_lift": "Your heels are lifting off the ground. Keep your heels planted."
    }

    def get_mistake_messages(self, mistakes: list[bool]) -> list[str]:
        """
        Returns a list of user-friendly messages for each detected mistake.
        """
        return [
        self.MISTAKE_DESCRIPTIONS[name]
        for name, err in zip(self.MISTAKES, mistakes) if err
        ]
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Squat mistake detection from keypoints JSON.")
    parser.add_argument("json_file", type=str, help="Path to keypoints JSON file (shape: [34,3])")
    args = parser.parse_args()

    ts = []
    keypoints_3d = []

    with open(args.json_file, "r") as f:
        data = json.load(f)
        # Assume you want the first frame's keypoints_3d
        ts = [d["ts"] for d in data]
        keypoints_3d = np.array([np.array(d["keypoints_3d"], dtype=np.float32) for d in data])

    print(len(ts))
    print(keypoints_3d[0].shape)

    norm_keypoints, meta_data = canonicalize_body34(keypoints_3d)
    for name, idx in BODY34_IDX.items():
        pt = norm_keypoints[10][idx]
        print(f"(" + ",".join(map(str, pt)) + ")")

    detector = SquatMistakeDetector()
    errors = detector.detect_mistakes(keypoints_3d[10])
    for name, err in zip(detector.MISTAKES, errors):
        print(f"{name}: {err}")