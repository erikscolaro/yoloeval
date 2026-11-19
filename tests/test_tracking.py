import numpy as np

from src.tracking import KalmanTracker, Params


def _video(n=60, flips=(), weak=(), seed=0):
    """Un oggetto a velocita' costante con jitter; flips = fotogrammi con la classe sbagliata, weak = conf bassa."""
    rng = np.random.default_rng(seed)
    for k in range(n):
        cx, cy, w, h = 100 + 4 * k, 200 + 1 * k, 40, 30
        cx, cy = cx + rng.normal(0, 1), cy + rng.normal(0, 1)
        conf = 0.15 if k in weak else 0.8
        cls = 3 if k in flips else 0
        yield np.array([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf, cls]])


def test_id_stabile_e_classe_a_voto():
    tr = KalmanTracker(nc=4, p=Params(min_hits=3))
    out = [tr.step(d) for d in _video(flips={20, 21, 35})]
    shown = [o for o in out if o]
    assert len(shown) == 58                                # dal terzo fotogramma in poi
    assert {t.id for o in shown for t in o} == {1}         # un solo track per tutto il video
    assert all(t.cls == 0 for o in shown for t in o)       # i fotogrammi con la classe sbagliata non la cambiano


def test_detection_deboli_tengono_vivo_il_track_ma_non_lo_fanno_nascere():
    tr = KalmanTracker(nc=4, p=Params(hi=0.4, lo=0.1, max_age=2))
    out = [tr.step(d) for d in _video(weak=set(range(10, 30)))]
    assert all(len(o) == 1 and o[0].id == 1 for o in out[2:])     # 20 fotogrammi deboli di fila: track vivo
    tr2 = KalmanTracker(nc=4, p=Params(hi=0.4, lo=0.1))
    assert not any(tr2.step(d) for d in _video(n=10, weak=set(range(10))))   # solo deboli: nessun track


def test_box_filtrata_piu_precisa_della_misura():
    tr = KalmanTracker(nc=4)
    err_raw, err_kf = [], []
    for k, d in enumerate(_video(n=80, seed=1)):
        o = tr.step(d)
        if k > 20 and o:
            vero = np.array([100 + 4 * k, 200 + k])
            err_raw.append(np.abs((d[0, :2] + d[0, 2:4]) / 2 - vero).mean())
            err_kf.append(np.abs((o[0].box[:2] + o[0].box[2:]) / 2 - vero).mean())
    assert np.mean(err_kf) < np.mean(err_raw)


def test_track_cancellato_dopo_max_age():
    tr = KalmanTracker(nc=4, p=Params(max_age=3, min_hits=1))
    for d in _video(n=5):
        tr.step(d)
    for _ in range(3):
        assert tr.step(np.zeros((0, 6)))                   # predetto per max_age fotogrammi
    assert not tr.step(np.zeros((0, 6)))


def test_stesso_oggetto_con_due_classi_fa_un_solo_track():
    tr = KalmanTracker(nc=4, p=Params(min_hits=1))
    for d in _video(n=20):
        doppia = d.copy()
        doppia[0, 4], doppia[0, 5] = 0.6, 1                   # stessa box, classe airplane, conf piu' bassa
        o = tr.step(np.vstack([d, doppia]))
    assert len(tr.tracks) == 1 and o[0].cls == 0
