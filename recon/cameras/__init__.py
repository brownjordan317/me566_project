"""Component 3, calibration and poses.

matching finds keypoints and matches frame pairs (SuperPoint + LightGlue);
sfm wraps COLMAP (database, verification, mapping, per-frame focal); solve is
the stage entry point; export writes trajectories and landmarks; reference
substitutes a dataset's reference poses for diagnosis.
"""
