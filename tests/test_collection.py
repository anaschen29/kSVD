import hashlib
import json
from pathlib import Path
import pytest
import torch

from ksvd.cli import run_study, planned_jobs
from ksvd.diagnostics import AccuracyTracker, component_rate_fits, accuracy_from_records
from ksvd.dynamics import reduced_step
from ksvd.initialization import coupled_normal_start, normal_start
from ksvd.problem import SpectralProblem
from ksvd.runner import RunConfig, run_trajectory
from ksvd.studies.eventual_rates import _run, _coupled_run, aligned_normal_metrics
from ksvd.summarize import summarize_study, summarize_case, group_reports


def test_first_hit_does_not_imply_final_accuracy():
    tracker = AccuracyTracker({'error': .1})
    tracker.observe({'t': 0, 'error': .2})
    tracker.observe({'t': 1, 'error': .05})
    tracker.observe({'t': 2, 'error': .4})
    report = tracker.summary(2)['error']
    assert report['first_hit_step'] == 1 and report['ever_met']
    assert report['final_value'] == .4 and not report['final_met']
    assert report['minimum_observed_value'] == .05


def test_unknown_terminal_value_and_missing_metric():
    tracker = AccuracyTracker({'error': .1, 'unavailable': 1e-8})
    tracker.observe({'t': 0, 'error': .01})
    report = tracker.summary(1)
    assert report['error']['ever_met'] and report['error']['final_met'] is None
    assert report['error']['final_value'] is None
    assert not report['unavailable']['available']


@pytest.mark.parametrize('targets', [{'x': -1}, {'x': float('nan')}, {'': .1}, {'x': True}])
def test_invalid_accuracy_targets(targets):
    with pytest.raises(ValueError):
        RunConfig(accuracy_targets=targets)


def test_sparse_recording_preserves_exact_first_hits_and_final_row():
    p = SpectralProblem.make([1., .8])
    y = normal_start(p, 1, 'mixed')
    kw = dict(eta=.5, max_steps=80, coordinates='y', accuracy_targets={'product_error': 1e-5})
    dense = run_trajectory(p, y, RunConfig(**kw))
    sparse = run_trajectory(p, y, RunConfig(record_every=17, **kw))
    assert dense['accuracy'] == sparse['accuracy']
    hit = dense['accuracy']['product_error']['first_hit_step']
    assert hit is not None and any(row['t'] == hit for row in sparse['records'])
    assert sparse['records'][-1]['t'] == sparse['last_evaluated_step'] == 80
    assert sparse['accuracy']['product_error']['final_met']
    assert sparse['status'] == 'max_steps'


def test_failure_does_not_inherit_previous_success(monkeypatch):
    import ksvd.runner as runner
    monkeypatch.setattr(runner, 'ambient_step', lambda p, x, eta: torch.full_like(x, float('nan')))
    p = SpectralProblem.make([1.])
    result = run_trajectory(p, p.target(1), RunConfig(max_steps=3))
    assert result['status'] == 'nonfinite'
    assert not result['final_state_evaluated']
    assert result['accuracy']['product_error']['ever_met']
    assert result['accuracy']['product_error']['final_met'] is None
    json.dumps(result, allow_nan=False)


def test_stationary_saddle_remains_unsuccessful():
    p = SpectralProblem.make([2., 1.])
    result = run_trajectory(p, torch.tensor([[0.], [1.]]), RunConfig(coordinates='y'))
    assert result['status'] == 'stagnated'
    assert not result['accuracy']['product_error']['ever_met']
    assert not result['accuracy']['product_error']['final_met']


def test_signed_mode_fit_keeps_negative_samples():
    records = [{'t': t, 'radial_signed': .005*(-.9)**t} for t in range(100)]
    fit = component_rate_fits(records, {'radial_multiplier': -.9})['radial_signed']
    assert fit['rho_fit'] == pytest.approx(.9, abs=1e-12)
    assert fit['sign_changes_in_fit_samples'] == fit['points']-1
    assert fit['reference_signed_multiplier'] == -.9
    assert fit['transform'] == 'absolute_value'


def test_short_or_zero_series_does_not_invent_fit():
    fits = component_rate_fits([{'t': 0, 'radial_signed': 0.}], {'radial_multiplier': 0.})
    assert fits['radial_signed']['status'] == 'insufficient_window'
    assert fits['radial_signed']['zero_samples'] == 1
    assert 'rho_fit' not in fits['radial_signed']


def test_angular_start_generates_slower_radial_error():
    result = _run({'n': 4, 'r': 3, 'max_steps': 280}, 1, .2, .95, 'angular', .01)
    fits = result['rate_fits']
    assert fits['angular_signed']['rho_fit'] == pytest.approx(.81, rel=1e-5)
    assert fits['radial_signed']['rho_fit'] == pytest.approx(.90, rel=1e-4)
    assert fits['factor_error']['rho_fit'] == pytest.approx(.90, rel=1e-4)
    assert result['first_resolved_radial_dominance_step'] > 0
    assert result['records'][1]['radial_signed'] == pytest.approx(-.95*.8*.01**2/(1+.8*.01**2), abs=1e-14)


def test_runner_does_not_modify_iterates_or_count_discarded_increment():
    p = SpectralProblem.make([1., .8])
    initial = normal_start(p, 1, 'mixed')
    result = run_trajectory(p, initial, RunConfig(eta=.95, max_steps=50, coordinates='y'))
    y = initial.clone()
    length = 0.
    for _ in range(result['completed_updates']):
        nxt = reduced_step(p, y, .95)
        length += float((nxt-y).norm())
        y = nxt
    torch.testing.assert_close(y, torch.tensor(result['final_state']), rtol=0, atol=0)
    assert result['records'][-1]['cumulative_length'] == length


@pytest.mark.parametrize('seed', [0, 1, 7])
def test_coupled_start_is_dense_seeded_and_has_prescribed_norm(seed):
    p = SpectralProblem.make([1.5, 1.25, 1., .8, .6, .4])
    y = coupled_normal_start(p, 3, .01, seed)
    torch.testing.assert_close(y, coupled_normal_start(p, 3, .01, seed), rtol=0, atol=0)
    assert float((y-p.frame(3)).norm()) == pytest.approx(.01, abs=1e-15)
    assert bool((y[3:].norm(dim=0) > 0).all())
    assert float(y[:3].triu(1).norm()) > 0
    assert torch.linalg.matrix_rank(y) == 3


def test_coupled_metrics_are_orbit_errors_not_point_limit_surrogates():
    p = SpectralProblem.make([1.5, 1., .8, .4])
    rot = torch.tensor([[.6, -.8], [.8, .6]])
    y = p.frame(2) @ rot
    out = aligned_normal_metrics(p, y, p.to_x(y))
    assert out['orbit_error_y'] < 1e-14
    assert (y-p.frame(2)).norm() > .5
    assert 'factor_error' not in out
    y = coupled_normal_start(p, 2)
    a = aligned_normal_metrics(p, y, p.to_x(y))
    b = aligned_normal_metrics(p, y@rot, p.to_x(y@rot))
    for key in ('orbit_error_y', 'radial_normal_norm', 'angular_normal_norm'):
        assert a[key] == pytest.approx(b[key], abs=1e-14)


def test_coupled_trial_has_noncommuting_gram_and_multiple_mode_fits():
    result = _coupled_run({'n': 8, 'r': 6, 'max_steps': 260}, 2, .2, .85, .01, 0)
    assert result['initial_gram_compression_commutator_relative'] > 1e-5
    assert result['accuracy']['orbit_error_y']['final_met']
    assert 'factor_error' not in result['records'][0]
    assert 'orbit_error_y' in result['rate_fits']
    assert 'angular_normal_norm' in result['rate_fits']


def test_high_precision_metadata_is_not_float64(tmp_path):
    pytest.importorskip('mpmath')
    cfg = {'n': 4, 'r': 3, 'ks': [1], 'mus': [.2], 'etas': [.5], 'modes': [],
           'amplitudes': [.01], 'gaussian_seeds': [], 'max_steps': 2, 'high_precision': True}
    manifest = run_study('eventual_rates', cfg, tmp_path/'run')
    hp = json.loads((tmp_path/'run'/'quadratic-high-precision.json').read_text())
    assert hp['metadata']['dtype'] == 'mpmath.mpf' and hp['metadata']['decimal_digits'] == 100
    assert hp['result']['termination_reason'] in {'precision_saturation', 'precision_floor'}
    assert hp['result']['empirical_orders'][-1]['order'] == pytest.approx(2, rel=.01)
    assert len(manifest['cases']) == 2


def legacy_case():
    return {'id': 'old', 'study': 'eventual_rates', 'parameters': {'eta': .95, 'k': 1, 'mu': .2},
            'metadata': {'dtype': 'float64'}, 'result': {'status': 'max_steps',
            'records': [{'t': t, 'factor_error': .005*.9**t, 'radial_signed': .005*(-.9)**t,
                         'angular_signed': .003*.81**t, 'product_error': .01*.9**t} for t in range(160)],
            'theory': {'rho': .9, 'radial_multiplier': -.9, 'angular_multiplier': .81}}}


def test_legacy_reanalysis_preserves_inputs_and_reports_mode_fits(tmp_path):
    raw, out = tmp_path/'raw', tmp_path/'report'
    raw.mkdir()
    c = legacy_case()
    (raw/'old.json').write_text(json.dumps(c))
    (raw/'manifest.json').write_text(json.dumps({'study': c['study'], 'cases': [{'id': 'old', 'file': 'old.json'}]}))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()}
    report = summarize_study(raw, out)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()}
    assert before == after
    entry = report['cases'][0]
    assert entry['accuracy']['product_error']['observation_scope'] == 'saved_records_only'
    assert entry['accuracy']['product_error']['final_met']
    assert entry['rate_fits']['radial_signed']['rho_fit'] == pytest.approx(.9, abs=1e-12)
    assert (out/'rate_fits.csv').exists() and (out/'cases.csv').exists()
    with pytest.raises(ValueError): summarize_study(raw, raw)


def test_legacy_sparse_first_hit_is_labeled_saved_only():
    records = [{'t': 0, 'e': 1.}, {'t': 10, 'e': .01}]
    report = accuracy_from_records(records, {'e': .1}, 10)['e']
    assert report['first_hit_step'] == 10
    assert report['observation_scope'] == 'saved_records_only'


def test_group_statistics_do_not_hide_nonattainment():
    hit = summarize_case(legacy_case())
    missed_case = legacy_case()
    missed_case['id'] = 'missed'
    missed_case['result']['records'] = missed_case['result']['records'][:2]
    missed = summarize_case(missed_case)
    group = group_reports([hit, missed])[0]
    assert group['runs'] == 2
    assert group['accuracy']['product_error']['ever_met_runs'] == 1
    assert group['accuracy']['product_error']['not_hit_before_end_runs'] == 1


def test_collection_case_counts_and_paired_seeds():
    config = json.loads((Path(__file__).parents[1]/'src/ksvd/presets/collection.json').read_text())
    rates = planned_jobs('eventual_rates', config['eventual_rates'])
    assert len(rates) == 253
    assert len(planned_jobs('trace_contraction', config['trace_contraction'])) == 144
    coupled = [j for j in rates if j.parameters.get('start') == 'coupled']
    assert len(coupled) == 60
    assert {j.parameters['seed'] for j in coupled} == set(range(5))
    assert {j.parameters['mu'] for j in coupled} == {.05, .2, .5}


def test_resume_preserves_raw_files_and_manifest_creation_time(tmp_path):
    cfg = {'n': 5, 'r': 3, 'ks': [1], 'ratios': [.5], 'trials': 2, 'q_max': 2,
           'theta': .5, 'failure_probability': .1}
    first = run_study('power_sketch_entry', cfg, tmp_path/'run')
    case_path = tmp_path/'run'/first['cases'][0]['file']
    original = case_path.read_bytes()
    second = run_study('power_sketch_entry', cfg, tmp_path/'run', resume=True)
    assert first['metadata']['created_utc'] == second['metadata']['created_utc']
    assert case_path.read_bytes() == original and second['cases'][0]['reused']
    with pytest.raises(ValueError, match='refusing resume'):
        run_study('power_sketch_entry', dict(cfg, q_max=3), tmp_path/'run', resume=True)


def test_dry_run_has_no_numerical_execution(monkeypatch, capsys):
    import ksvd.cli as cli
    import ksvd.studies.eventual_rates as rates
    monkeypatch.setattr(rates, 'run_trajectory', lambda *a, **k: pytest.fail('dry-run executed trajectory'))
    monkeypatch.setattr('sys.argv', ['ksvd-run', '--experiment', 'eventual_rates', '--preset', 'collection', '--dry-run'])
    cli.main()
    assert json.loads(capsys.readouterr().out)['case_count'] == 253


def test_full_rank_orbit_metrics_have_no_mixing_coordinate():
    p = SpectralProblem.make([2., 1.])
    y = p.frame(2)
    out = aligned_normal_metrics(p, y, p.to_x(y))
    assert out['slow_angular_signed'] is None
    assert out['angular_normal_norm'] == 0
