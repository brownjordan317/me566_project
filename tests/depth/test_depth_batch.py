"""The depth batch keeps each model only while queued work needs it."""
from recon.depth.batch import drop_points, needs_model, plan, weights_key


def job(stage, *argv):
    return {'stage': stage, 'argv': list(argv)}


def test_jobs_run_model_by_model_in_pipeline_order_keeping_each_stages_order():
    jobs = [job('da3', 'a'), job('moge', 'raw1'), job('vggt', 'v1'), job('moge', 'raw2'), job('da3', 'b')]
    order = [(stage, [j['argv'][0] for j in queue]) for stage, queue in plan(jobs)]
    assert order == [('moge', ['raw1', 'raw2']), ('vggt', ['v1']), ('da3', ['a', 'b'])]


def test_a_single_scan_drops_each_model_after_its_only_job():
    assert drop_points('vggt', [job('vggt', '--output-dir', 'x')]) == {0}


def test_a_batch_drops_only_after_the_last_scan_that_needs_the_model():
    queue = [job('vggt', '--output-dir', f's{i}') for i in range(18)]
    assert drop_points('vggt', queue) == {17}


def test_moge_grounding_jobs_need_no_model_so_the_model_goes_after_the_last_inference():
    queue = [job('moge', '--output-dir', 'a/raw'), job('moge', '--output-dir', 'a/scene', '--reuse-raw-dir', 'a/raw'),
             job('moge', '--output-dir', 'b/raw'), job('moge', '--output-dir', 'b/scene', '--reuse-raw-dir', 'b/raw')]
    assert not needs_model('moge', queue[1]['argv'])
    assert drop_points('moge', queue) == {2}


def test_different_weights_are_a_different_model():
    plain = job('da3', '--checkpoint', 'DA3-GIANT')
    adapted = job('da3', '--checkpoint', 'DA3-GIANT', '--head-weights', 'head.pt')
    assert weights_key('da3', plain['argv']) != weights_key('da3', adapted['argv'])
    assert drop_points('da3', [plain, adapted, plain]) == {1, 2}
