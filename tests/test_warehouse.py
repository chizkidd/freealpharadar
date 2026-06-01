"""Offline tests for the bulk-fundamentals warehouse + discovery funnel.

Builds a tiny synthetic SEC Financial Statement Data Set ZIP in ``tmp_path`` and
exercises the loader, store, Stage-1 screen and the discovery promotion — with
**no network** and a stubbed scoring pipeline. Requires ``duckdb``/``pyarrow``
(skipped automatically if they are not installed).
"""

from __future__ import annotations

import zipfile
from types import SimpleNamespace

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from freealpharadar.discovery.discover import run_discovery  # noqa: E402
from freealpharadar.discovery.screen import (  # noqa: E402
    ScreenConfig,
    screen_candidates,
)
from freealpharadar.warehouse.loader import build_warehouse, load_quarter  # noqa: E402
from freealpharadar.warehouse.store import WarehouseStore  # noqa: E402

_CIK_MAP = {111: "AAA", 222: "BBB"}


def _make_zip(path):
    """Write a synthetic quarter ZIP with sub.txt + num.txt.

    AAA: small revenue (50M->~104M), 20%/yr growth -> should pass the screen.
    BBB: large revenue (5B), flat -> gated out (too big, no growth).
    """
    sub_rows = ["adsh\tcik\tname\tform\tfy\tfp\tperiod"]
    num_rows = ["adsh\ttag\tddate\tqtrs\tuom\tvalue"]

    def add(adsh, cik, name, fy, rev, gp, rnd, ni, assets):
        sub_rows.append(f"{adsh}\t{cik}\t{name}\t10-K\t{fy}\tFY\t{fy}1231")
        end = f"{fy}1231"
        num_rows.append(f"{adsh}\tRevenues\t{end}\t4\tUSD\t{rev}")
        num_rows.append(f"{adsh}\tGrossProfit\t{end}\t4\tUSD\t{gp}")
        num_rows.append(f"{adsh}\tResearchAndDevelopmentExpense\t{end}\t4\tUSD\t{rnd}")
        num_rows.append(f"{adsh}\tNetIncomeLoss\t{end}\t4\tUSD\t{ni}")
        num_rows.append(f"{adsh}\tAssets\t{end}\t0\tUSD\t{assets}")

    for i, fy in enumerate(range(2019, 2024)):
        rev = 50e6 * (1.2**i)
        add(
            f"a{fy}",
            111,
            "Alpha Co",
            fy,
            rev,
            rev * (0.4 + 0.02 * i),
            rev * 0.2,
            rev * 0.05,
            rev * 2,
        )
        add(f"b{fy}", 222, "Beta Co", fy, 5e9, 5e9 * 0.4, 5e9 * 0.1, 5e9 * 0.1, 5e9 * 2)

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("sub.txt", "\n".join(sub_rows) + "\n")
        zf.writestr("num.txt", "\n".join(num_rows) + "\n")
    return path


@pytest.fixture
def quarter_zip(tmp_path):
    return _make_zip(tmp_path / "2023q4.zip")


@pytest.fixture
def warehouse(tmp_path, quarter_zip):
    out = tmp_path / "facts.parquet"
    build_warehouse(zip_paths=[quarter_zip], cik_to_ticker=_CIK_MAP, out_path=out)
    return WarehouseStore(out)


class TestLoader:
    def test_load_quarter_wide(self, quarter_zip):
        df = load_quarter(quarter_zip, _CIK_MAP)
        assert not df.empty
        aaa = df[df["ticker"] == "AAA"].sort_values("fy")
        assert list(aaa["fy"]) == [2019, 2020, 2021, 2022, 2023]
        assert aaa.iloc[0]["revenue"] == pytest.approx(50e6)
        assert aaa.iloc[-1]["revenue"] == pytest.approx(50e6 * 1.2**4)
        # Balance-sheet (instant) and flow fields both captured.
        assert aaa.iloc[-1]["assets"] > 0
        assert {"revenue", "gross_profit", "rnd", "net_income"} <= set(df.columns)

    def test_build_and_store(self, warehouse):
        assert "AAA" in warehouse.tickers()
        s = warehouse.annual_series("AAA", "revenue")
        assert list(s["fy"]) == sorted(s["fy"])
        assert len(s) == 5


class TestScreen:
    def test_screen_keeps_small_growers_drops_megacaps(self, warehouse):
        cands = screen_candidates(warehouse, ScreenConfig())
        tickers = set(cands["ticker"])
        assert "AAA" in tickers  # small + 20%/yr growth qualifies
        assert "BBB" not in tickers  # $5B revenue exceeds max + no growth

    def test_screen_metrics_present(self, warehouse):
        cands = screen_candidates(warehouse, ScreenConfig())
        row = cands[cands["ticker"] == "AAA"].iloc[0]
        assert row["revenue_cagr"] == pytest.approx(0.20, abs=1e-6)
        assert row["years"] == 5


class TestDiscovery:
    def test_run_discovery_promotes_ranked_topN(self, warehouse, tmp_path):
        # Stub the expensive Stage-2 pipeline with deterministic scores.
        def fake_pipeline(tickers, **kwargs):
            results = [
                SimpleNamespace(
                    ticker="AAA", name="Alpha Co", score=80.0, market_cap=3e9
                ),
                SimpleNamespace(
                    ticker="ZZZ", name="Mega Co", score=99.0, market_cap=500e9
                ),  # excluded by cap ceiling
                SimpleNamespace(
                    ticker="CCC", name="Gamma Co", score=60.0, market_cap=None
                ),  # unknown cap -> kept
            ]
            return SimpleNamespace(results=results)

        uni = tmp_path / "universe.txt"
        snap = tmp_path / "prewarm.json"
        result = run_discovery(
            store=warehouse,
            screen_cfg=ScreenConfig(),
            top_n=5,
            max_market_cap=20e9,
            run_ml=False,
            write_outputs=True,
            universe_path=uni,
            report_dir=tmp_path,
            snapshot_path=snap,
            pipeline_fn=fake_pipeline,
        )
        # Mega-cap dropped; remaining ranked by score desc.
        assert [n.ticker for n in result.names] == ["AAA", "CCC"]
        assert result.names[0].rank == 1
        # universe.txt written with the ranked tickers (comments ignored).
        body = [
            ln.strip()
            for ln in uni.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        assert body == ["AAA", "CCC"]

    def test_run_discovery_dry_run_writes_nothing(self, warehouse, tmp_path):
        def fake_pipeline(tickers, **kwargs):
            return SimpleNamespace(
                results=[
                    SimpleNamespace(
                        ticker="AAA", name="Alpha Co", score=70.0, market_cap=1e9
                    )
                ]
            )

        uni = tmp_path / "u.txt"
        result = run_discovery(
            store=warehouse,
            write_outputs=False,
            universe_path=uni,
            pipeline_fn=fake_pipeline,
        )
        assert result.names and result.universe_path is None
        assert not uni.exists()
