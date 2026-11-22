"""Visual cortex for the real table (STUB, untested on hardware): overhead webcam + ArUco.

Four corner markers at known table positions (robot frame, metres) give a homography from
image pixels to the table. Each object carries its own marker; its ID tells the object
type and its corners give position and yaw. Requires: pip install opencv-contrib-python
"""
from __future__ import annotations

import numpy as np


class ArucoTable:
    def __init__(self, cal: dict):
        try:
            import cv2
            self.cv2 = cv2
            self.aruco = cv2.aruco
        except (ImportError, AttributeError) as e:
            raise ImportError("ArUco perception needs  pip install opencv-contrib-python") from e
        self.cal = cal
        self.cap = cv2.VideoCapture(int(cal.get("camera_index", 0)))
        dic = self.aruco.getPredefinedDictionary(getattr(self.aruco, cal.get("aruco_dict", "DICT_4X4_50")))
        self.detector = self.aruco.ArucoDetector(dic, self.aruco.DetectorParameters())
        self.corners_world = {int(k): v for k, v in cal["corner_markers"].items()}
        self.obj_ids = {int(k): v for k, v in cal["object_markers"].items()}
        self._last = None

    def _detect(self, tries=10):
        for _ in range(tries):
            ok, frame = self.cap.read()
            if not ok:
                continue
            corners, ids, _ = self.detector.detectMarkers(frame)
            if ids is None:
                continue
            found = {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners)}
            if all(k in found for k in self.corners_world) and any(k in found for k in self.obj_ids):
                return found
        raise RuntimeError("camera: could not see all 4 corner markers and an object marker")

    def _homography(self, found):
        src = np.array([found[k].mean(0) for k in sorted(self.corners_world)], np.float32)
        dst = np.array([self.corners_world[k] for k in sorted(self.corners_world)], np.float32)
        H, _ = self.cv2.findHomography(src, dst)
        return H

    def _to_table(self, H, pts):
        p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
        return p[:, :2] / p[:, 2:3]

    def object_pose(self):
        found = self._detect()
        H = self._homography(found)
        oid = next(k for k in self.obj_ids if k in found)
        c = self._to_table(H, found[oid])
        center = c.mean(0)
        edge = c[1] - c[0]                  # marker's top edge gives the yaw
        yaw = float(np.arctan2(edge[1], edge[0]))
        self._last = oid
        return float(center[0]), float(center[1]), yaw, 1.0   # upright assumed (a fallen marker isn't visible)

    def object_type(self) -> int:
        if self._last is None:
            self.object_pose()
        return int(self.obj_ids[self._last])

    def close(self):
        self.cap.release()
