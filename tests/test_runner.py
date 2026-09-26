import json
from pathlib import Path
import pytest
import torch
from ksvd.problem import SpectralProblem
from ksvd.runner import RunConfig, run_trajectory
from ksvd.initialization import normal_start
from ksvd.metrics import estimate_rate
from ksvd.cli import run_study
from ksvd.studies.eventual_rates import high_precision_quadratic
from ksvd.studies.power_sketch_entry import wilson


def test_scalar_cycle_not_mistaken_for_convergence():
    p=SpectralProblem.make([1.])
    result=run_trajectory(p,torch.tensor([[2.]],dtype=torch.float64),RunConfig(eta=1.,max_steps=10,coordinates="y"))
    assert result["status"]=="cycle_detected"
    assert [r["t"] for r in result["records"]]==[0,1]


def test_stationary_saddle_is_not_labeled_global():
    p=SpectralProblem.make([2.,1.])
    y=torch.tensor([[0.],[1.]],dtype=torch.float64)
    result=run_trajectory(p,y,RunConfig(coordinates="y",stop_metric="product_error"))
    assert result["status"]=="stagnated"
    assert result["records"][0]["product_error"]>0


def test_rank_one_measured_rate():
    p=SpectralProblem.make([1.,.8])
    y=normal_start(p,1,"mixed")
    result=run_trajectory(p,y,RunConfig(eta=.5,max_steps=250,coordinates="y"),
                          extra=lambda s,x,t:{"factor_error":float((s-p.frame(1)).norm())})
    rate=estimate_rate(result["records"],"factor_error")
    assert rate["rho_fit"]==pytest.approx(.9,rel=1e-4)


def test_atomic_outputs_and_resume(tmp_path):
    cfg={"n":8,"r":6,"ks":[1],"ratios":[.5],"trials":4,"q_max":3,
         "theta":.5,"failure_probability":.1,"device":"cpu"}
    manifest=run_study("power_sketch_entry",cfg,tmp_path/"run")
    assert manifest["cases"][0]["status"]=="complete"
    again=run_study("power_sketch_entry",cfg,tmp_path/"run",resume=True)
    assert again["cases"][0]["reused"]
    with pytest.raises(ValueError,match="refusing resume"):
        run_study("power_sketch_entry",dict(cfg,trials=5),tmp_path/"run",resume=True)
    with pytest.raises(FileExistsError):
        run_study("power_sketch_entry",cfg,tmp_path/"run")
    from ksvd.plotting import plot_study
    plot_study(tmp_path/"run",tmp_path/"figs")
    assert len(list((tmp_path/"figs").glob("*.png")))==2


def test_wilson_endpoints():
    lo,hi=wilson(0,10)
    assert lo==0 and 0<hi<1
    lo,hi=wilson(10,10)
    assert 0<lo<1 and hi==1


def test_high_precision_reference():
    pytest.importorskip("mpmath")
    result=high_precision_quadratic(100)
    logs=[r["log_error"] for r in result["records"] if r["log_error"] is not None]
    assert len(logs)>=6
    assert logs[-1]<-100
    assert logs[-2]/logs[-3]==pytest.approx(2,rel=.05)


def test_damped_oscillations_are_not_cycles():
    p=SpectralProblem.make([1.])
    y=torch.tensor([[1.01]],dtype=torch.float64)
    result=run_trajectory(p,y,RunConfig(eta=.95,max_steps=500,coordinates="y"))
    assert result["status"] != "cycle_detected"
    assert result["records"][-1]["radial_error"] < 1e-11


def test_plot_decimal_names_do_not_collide(tmp_path):
    from ksvd.plotting import _save, plt
    for name in ("eta-mu0.1", "eta-mu0.2"):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [1, 0])
        _save(fig, tmp_path/name)
    assert (tmp_path/"eta-mu0.1.png").exists()
    assert (tmp_path/"eta-mu0.2.png").exists()
