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


def test_oggetto_oltre_max_side_va_in_scan_e_senza_limite_si_riduce():
    grande = np.array([300, 60, 900, 660], float)     # 600x600: lato del crop 1000 > 512
    ct = CropTracker(detect_finto([]), 4, Params(min_hits=3), scan_imgsz=640, crop=256, rescan=0)
    assert {ct.step(frame_con(grande), 1 / 25)[1]["mode"] for _ in range(12)} == {"SCAN"}
    ch = []
    libero = CropTracker(detect_finto(ch), 4, Params(min_hits=3), scan_imgsz=640, crop=256, rescan=0, max_side=None)
    out = [libero.step(frame_con(grande), 1 / 25)[1] for _ in range(12)]
    assert [o["mode"] for o in out][4:] == ["TRACK"] * 8 and ch[-1][1] == (256, 256)
    assert np.abs(out[-1]["dets"][0, :4] - grande).max() < 8
    medio = np.array([500, 250, 740, 470], float)     # 240x220: lato del crop 400 < 512
    ct2 = CropTracker(detect_finto([]), 4, Params(min_hits=3), scan_imgsz=640, crop=256, rescan=0)
    assert [ct2.step(frame_con(medio), 1 / 25)[1]["mode"] for _ in range(8)][4:] == ["TRACK"] * 4


def test_scan_riduce_con_area_e_riporta_le_box_nel_fotogramma():
    ch = []
    ct = CropTracker(detect_finto(ch), 4, Params(min_hits=3), scan_imgsz=640, crop=256, rescan=0)
    shown, info = ct.step(frame_con(vero(0)), 1 / 25)           # 1280x720 -> 640x360
    assert info["mode"] == "SCAN" and ch[-1][1] == (360, 640)
    assert np.abs(info["dets"][0, :4] - vero(0)).max() <= 2     # box in coordinate del fotogramma intero


def test_soglie_del_track_valgono_solo_nei_crop():
    # nei crop il modello da' conf 0,2: sotto il lo dello SCAN (0,3), sopra quello del TRACK (0,1)
    def detect(images, imgsz, conf):
        out = detect_finto([])(images, imgsz, conf)
        if imgsz == 256:
            for d in out:
                d[:, 4] = 0.2
        return [d[d[:, 4] >= conf] for d in out]
    for track, atteso in ((None, "SCAN"), ({"lo": 0.1}, "TRACK")):
        ct = CropTracker(detect, 4, Params(hi=0.5, lo=0.3, min_hits=2), crop=256, rescan=0, track=track)
        modi = [ct.step(frame_con(vero(k)), 1 / 25)[1]["mode"] for k in range(12)]
        assert modi[-1] == atteso and ct.tr.p is (ct.p_track if atteso == "TRACK" else ct.p_scan), (track, modi)
