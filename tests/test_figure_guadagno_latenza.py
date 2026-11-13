from tools.figure_guadagno_latenza import EXP, LUT, pavimenti


def test_pavimento_nano_come_il_misurato():
    """La stima LUT+fit di N (w0.25 D126) sta entro 0,5 ms dal pavimento misurato sul Pi (22,3 ms), e il D170 sta sopra il D101."""
    p = pavimenti(EXP / "b4r_composizione/results/layers.csv", EXP / "pavimento_s/results/layers.csv", LUT / "fit.json",
                  LUT, ["w0.25_D101", "w0.25_D126", "w0.25_D170"])
    assert abs(p["w0.25_D126"] - 22.3) < 0.5
    assert p["w0.25_D101"] < p["w0.25_D126"] < p["w0.25_D170"]
