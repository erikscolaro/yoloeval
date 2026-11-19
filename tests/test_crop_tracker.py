import cv2
import numpy as np

from src.crop_tracker import CropTracker
from src.tracking import Params

W, H = 1280, 720


def frame_con(box):
    f = np.zeros((H, W, 3), np.uint8)
    if box is not None:
        x1, y1, x2, y2 = map(int, box)
        f[y1:y2, x1:x2] = 255
    return f


def detect_finto(chiamate):
    """Trova il rettangolo bianco in ogni immagine (fotogramma o crop), nelle coordinate di quell'immagine."""
    def detect(images, imgsz, conf):
        chiamate.append((len(images), images[0].shape[:2]))
        out = []
        for im in images:
            m = (im[..., 0] > 200).astype(np.uint8)
            n, _, st, _ = cv2.connectedComponentsWithStats(m)
            out.append(np.array([[x, y, x + w, y + h, 0.9, 0] for x, y, w, h, a in st[1:n] if a > 4], float).reshape(-1, 6))
        return out
    return detect


def vero(k):
    return np.array([200 + 6 * k, 300 + 2 * k, 240 + 6 * k, 330 + 2 * k], float)


def test_crop_segue_l_oggetto_e_riproietta_le_box():
    ch = []
    ct = CropTracker(detect_finto(ch), 4, Params(min_hits=3), scan_imgsz=640, crop=256, rescan=0)
    modi, err = [], []
    for k in range(40):
        shown, info = ct.step(frame_con(vero(k)), 1 / 25)
        modi.append(info["mode"])
        if info["mode"] == "TRACK":
            assert ch[-1][1] == (256, 256)                       # il modello vede il crop, non il fotogramma
            err.append(np.abs(info["dets"][0, :4] - vero(k)).max())
    assert modi[:3] == ["SCAN"] * 3 and set(modi[3:]) == {"TRACK"}
    assert max(err) <= 3                                          # riproiezione al pixel (scala del crop)
    assert len(shown) == 1 and shown[0].id == 1


def test_rescan_periodico_e_dopo_un_miss():
    ct = CropTracker(detect_finto([]), 4, Params(min_hits=2), rescan=10)
    modi = [ct.step(frame_con(vero(k)), 1 / 25)[1]["mode"] for k in range(25)]
    assert modi[10] == "SCAN" and modi[20] == "SCAN" and modi[11] == "TRACK"
    assert ct.step(frame_con(None), 1 / 25)[1]["mode"] == "TRACK"      # l'oggetto sparisce: crop vuoto
    assert ct.step(frame_con(None), 1 / 25)[1]["mode"] == "SCAN"       # e al fotogramma dopo si riguarda tutto


def test_crop_fuori_dal_fotogramma_si_completa_di_grigio():
    ct = CropTracker(detect_finto([]), 4, crop=256)
    img, (x0, y0), sc = ct.cut(frame_con(None), 10, 10, 200)
    assert img.shape == (256, 256, 3) and (x0, y0) == (-90, -90)
    assert (img[0, 0] == 114).all()
