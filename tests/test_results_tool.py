"""tools.results: riepilogo, falliti, saltati e cella su un results/ finto."""
import json

from tools import results


def _rec(d, cid, status, **kw):
    (d / f"{cid}.json").write_text(json.dumps({"cell_id": cid, "status": status, "schema_version": 1,
                                               "axes": {"model": "m", "board": "rpi5", "backend": "openvino",
                                                        "quantization": "fp16"}, "config": {"x": 1}, **kw}))


def test_results(tmp_path, capsys):
    _rec(tmp_path, "a", "ok")
    _rec(tmp_path, "b", "failed", error="MissingExport: manca\nstack")
    _rec(tmp_path, "c", "skipped", reason="non supportato")
    assert results.main(["riepilogo", "--results-dir", str(tmp_path)]) == 0
    assert "3 misure" in capsys.readouterr().out
    results.main(["falliti", "--results-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "b  m fp16 openvino rpi5" in out and "MissingExport: manca" in out and "stack" not in out
    results.main(["saltati", "--results-dir", str(tmp_path)])
    assert "1  non supportato" in capsys.readouterr().out
    results.main(["cella", "b", "--results-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert '"cell_id": "b"' in out and '"config"' not in out
