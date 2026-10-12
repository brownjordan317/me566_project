"""Video sources must become an ordered, resized frame sequence."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from recon.inputs.capture.prepare import prepare


class PrepareVideoTests(unittest.TestCase):
    def test_video_keeps_sharpest_frame_per_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            video = folder / 'scan.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'),
                                     30, (320, 240))
            pattern = (np.indices((240, 320)).sum(0) // 8 % 2 * 255).astype(np.uint8)
            for number in range(30):
                frame = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
                if number % 5 != 2:
                    frame = cv2.GaussianBlur(frame, (21, 21), 8)
                writer.write(frame)
            writer.release()
            report = prepare(video, folder / 'prepared', max_edge=160, sample_fps=6)
            self.assertEqual(report['frames'], 6)
            self.assertEqual(report['video_step'], 5)
            self.assertEqual(report['size'], [160, 120])
            mapping = json.loads((folder / 'prepared' / 'sequence.json').read_text())['mapping']
            self.assertEqual([item['source'] for item in mapping],
                             [f'scan.mp4@{number}' for number in range(2, 30, 5)])
            self.assertTrue((folder / 'prepared' / 'masks' / '00000005.png').is_file())
            full = cv2.imread(str(folder / 'prepared' / 'full' / '00000005.png'))
            self.assertEqual(full.shape[:2], (240, 320))              # unresized copy kept


if __name__ == '__main__':
    unittest.main()
