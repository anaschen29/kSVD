"""Reporting-only regression tests: synthetic saved records, no optimizer runs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from ksvd.rankone_reporting import (build_report, case_events, deduplicate_controls,
                                    event_summary, summarize_rank_one)


def sample(id="a", seed=0, start="controlled"):
    rows = [
        dict(t=0, normalized_norm=1., trace_over_gap=1., factor_error=.1, product_error=.2),
        dict(t=1, normalized_norm=1., trace_over_gap=.5, factor_error=1e-8, product_error=2e-8),
        dict(t=2, normalized_norm=1., trace_over_gap=.2, factor_error=1e-9, product_error=1e-8),
    ]
    return {"id": id, "study": "jedra_shah_rank1", "parameters": {"seed": seed, "mu": .2, "start": start,
            "overlap": .01 if start == "controlled" else None}, "metadata": {"source_sha256": "test-source", "dtype": "float64"},
            "result": {"status": "max_steps", "initial_state": [[.01], [1.]],
                       "problem": {"n": 2, "k": 1, "positive_eigenvalues": [1., .8]},
                       "records": rows, "last_evaluated_step": 2, "final_state_evaluated": True,
                       "theory": {"a_js": .5, "b_js": 2., "tau_js": 1, "rho": .9, "initial_overlap": .01},
                       "rate_fit": {"status": "ok", "rho_fit": .9}}}


def archive(tmp_path, cases):
    source = tmp_path/"raw"
    source.mkdir()
    manifest = {"study": "jedra_shah_rank1", "cases": [], "planned_case_count": len(cases)}
    for case in cases:
        filename = case["id"]+".json"
        (source/filename).write_text(json.dumps(case))
        manifest["cases"].append({"id": case["id"], "file": filename})
    (source/"manifest.json").write_text(json.dumps(manifest))
    return source


def test_exact_control_replays_but_not_gaussians():
    cases = [sample("b", 1), sample("a"), sample("c", 0, "gaussian"), sample("d", 1, "gaussian")]
    kept, mapping = deduplicate_controls(cases)
    assert [c["id"] for c in kept] == ["a", "c", "d"]
    assert mapping[1]["representative_id"] == "a"
    assert mapping[1]["reason"] == "exact_control_replay"


@pytest.mark.parametrize("difference", ["initial", "records", "theory", "status", "problem", "source", "dtype", "parameter"])
def test_different_evidence_never_deduplicates(difference):
    a, b = sample("a"), sample("b", 1)
    if difference == "initial": b["result"]["initial_state"][0][0] = .02
    elif difference == "records": b["result"]["records"][1]["factor_error"] = 2e-8
    elif difference == "theory": b["result"]["theory"]["b_js"] = 3.
    elif difference == "status": b["result"]["status"] = "stagnated"
    elif difference == "problem": b["result"]["problem"]["n"] = 3
    elif difference == "source": b["metadata"]["source_sha256"] = "other"
    elif difference == "dtype": b["metadata"]["dtype"] = "float32"
    else: b["parameters"]["mu"] = .1
    assert len(deduplicate_controls([a,b])[0]) == 2


@pytest.mark.parametrize("missing", ["initial_state", "problem", "records", "source_sha256"])
def test_missing_evidence_preserved(missing):
    a, b = sample("a"), sample("b", 1)
    for case in (a,b):
        del case["metadata" if missing == "source_sha256" else "result"][missing]
    assert len(deduplicate_controls([a,b])[0]) == 2


def test_failures_not_deduplicated_or_reported_as_final_success():
    a, b = sample("a"), sample("b", 1)
    for c in (a,b):
        c["result"]["status"] = "numerical_failure"
        c["result"]["final_state_evaluated"] = False
    report = build_report([a,b])
    assert report["analysis_rows"] == 2
    event = report["cases"][0]["events"]["factor_accuracy"]
    assert event["first_hit_step"] == 1 and event["final_met"] is None


def test_strict_and_inclusive_boundaries_and_iteration_zero():
    events = case_events(sample())["events"]
    assert events["annulus"]["first_hit_step"] == 0
    assert events["strict_band"]["first_hit_step"] == 1  # equality at 1 excluded
    assert events["fixed_margin_band"]["first_hit_step"] == 1  # equality at .5 included
    assert events["factor_accuracy"]["first_hit_step"] == 1
    assert events["product_accuracy"]["first_hit_step"] == 2
    assert events["factor_accuracy"]["first_hit_has_complete_observed_prefix"]


def test_first_hit_is_not_permanence():
    rows = [dict(t=i, error=e) for i,e in enumerate([.1,.4,.05])]
    event = event_summary(rows, "error", lambda v:v<=.2, 2)
    assert event["first_hit_step"] == 0
    assert event["first_hit_with_no_later_observed_exit"] == 2
    assert not event["no_later_observed_exit_after_first_hit"]
    assert event["final_met"]


def test_sparse_and_missing_records_not_exact_hits_or_successes():
    case = sample()
    case["result"]["records"] = [case["result"]["records"][0], case["result"]["records"][2]]
    report = case_events(case)
    assert not report["saved_steps_complete_from_zero"]
    assert not report["events"]["factor_accuracy"]["first_hit_has_complete_observed_prefix"]
    del case["result"]["records"][-1]["factor_error"]
    e = case_events(case)["events"]["factor_accuracy"]
    assert e["first_hit_step"] is None and e["final_met"] is None
    assert e["missing_metric_records"] == 1


def test_missing_annulus_bounds_and_no_late_samples():
    c = sample()
    del c["result"]["theory"]["a_js"]
    r = case_events(c)
    assert r["events"]["annulus"]["first_hit_step"] is None
    assert r["annulus_at_all_observed_steps_from_tau"] is None
    c = sample(); c["result"]["theory"]["tau_js"] = 100
    assert case_events(c)["annulus_at_all_observed_steps_from_tau"] is None


@pytest.mark.parametrize("margin", [0.,1.,-1.,float("nan"), True])
def test_invalid_margin_even_with_empty_collection(margin):
    with pytest.raises(ValueError): build_report([], fixed_margin=margin)


@pytest.mark.parametrize("tol", [-1.,float("inf"), True])
def test_invalid_accuracy_tolerance(tol):
    with pytest.raises(ValueError): build_report([], factor_tolerance=tol)


def test_invalid_steps_and_negative_errors_rejected():
    c = sample(); c["result"]["records"][1]["t"] = 0
    with pytest.raises(ValueError): case_events(c)
    c = sample(); c["result"]["records"][0]["trace_over_gap"] = -1e-15
    with pytest.raises(ValueError, match="not silently clipped"): case_events(c)


def test_unique_groups_provenance_and_raw_bytes_untouched(tmp_path):
    source = archive(tmp_path, [sample("a"),sample("b",1),sample("g",0,"gaussian")])
    before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    out = tmp_path/"report"
    r = summarize_rank_one(source,out)
    assert (r["reviewed_files"],r["analysis_rows"],r["duplicate_files_excluded"]) == (3,2,1)
    assert r["groups"][0]["analysis_rows"] == 1 and r["groups"][0]["source_files"] == 2
    assert r["cases"][0]["source_ids"] == ["a","b"]
    assert r["cases"][0]["saved_rate_fit"] == sample()["result"]["rate_fit"]
    assert len(r["source_files"]) == 3 and len(r["duplicate_mapping"]) == 3
    assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()} == before
    assert {p.name for p in out.iterdir()} == {"summary.json","hitting_times.csv","events.csv","duplicate_mapping.csv","README.md"}
    assert len((out/"hitting_times.csv").read_text().splitlines()) == 3
    with pytest.raises(FileExistsError): summarize_rank_one(source,out)


@pytest.mark.parametrize("relative", ["inside", "same", "ancestor"])
def test_output_cannot_overlap_input(tmp_path, relative):
    source = archive(tmp_path, [sample()])
    out = source/"out" if relative == "inside" else source if relative == "same" else tmp_path
    with pytest.raises(ValueError): summarize_rank_one(source,out)


def test_manifest_escape_or_identity_mismatch_rejected(tmp_path):
    source = archive(tmp_path,[sample()]); p = source/"manifest.json"; m = json.loads(p.read_text())
    m["cases"][0]["file"] = "../a.json"; p.write_text(json.dumps(m))
    with pytest.raises(ValueError): summarize_rank_one(source,tmp_path/"report")
    m["cases"][0]["file"] = "a.json"; m["cases"][0]["id"] = "wrong"; p.write_text(json.dumps(m))
    with pytest.raises(ValueError): summarize_rank_one(source,tmp_path/"report")


def test_duplicate_ids_and_wrong_study_rejected():
    with pytest.raises(ValueError): build_report([sample(),sample()])
    c=sample();c["study"]="trace_contraction"
    with pytest.raises(ValueError): build_report([c])


def test_policy_override_and_empty_collection():
    r=build_report([sample()],fixed_margin=.1,factor_tolerance=1e-10)
    assert r["cases"][0]["events"]["factor_accuracy"]["first_hit_step"] is None
    assert r["cases"][0]["events"]["fixed_margin_band"]["first_hit_step"] is None
    assert build_report([])["analysis_rows"] == 0


def test_standalone_cli_without_loading_torch_or_optimizer(tmp_path):
    import ksvd.rankone_reporting as module
    source = archive(tmp_path,[sample()])
    code = '''import runpy, sys
class Block:
 def find_spec(self, fullname, path=None, target=None):
  if fullname.startswith(("torch", "ksvd.dynamics", "ksvd.runner")):
   raise RuntimeError("optimizer/numerical backend import forbidden")
sys.meta_path.insert(0, Block())
sys.argv=[sys.argv[1],"--input",sys.argv[2],"--out",sys.argv[3]]
runpy.run_path(sys.argv[0],run_name="__main__")
'''
    subprocess.run([sys.executable,"-c",code,module.__file__,str(source),str(tmp_path/"out")],check=True,capture_output=True)
    assert (tmp_path/"out"/"summary.json").exists()
