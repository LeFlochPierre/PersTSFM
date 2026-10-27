from scripts.phase2.audit_calibration_budget import common_feasibility, build_matrix, summarize
import pytest

CONTEXTS = [8,16,32,64,96,192,384,768,1536,3072]

def profile(name, history, horizon, n=10, family='family', total=None):
    return {'task_id':name,'dataset_config':name,'underlying_family_hint':family,
            'history_min':history,'history_max':history,'official_horizon':horizon,
            'entities_inspected':n,'entities_official_total':total if total is not None else n,
            'coverage':'FULL' if total is None or total==n else 'SAMPLED_NOT_FULL_CENSUS'}

def test_ercot_and_short_horizon():
    assert common_feasibility(151512,168,CONTEXTS,3)['max_common_feasible_context']==3072
    assert common_feasibility(39,13,CONTEXTS,3)['max_common_feasible_context'] is None
    assert common_feasibility(39,13,CONTEXTS,2)['max_common_feasible_context']==8
    assert common_feasibility(39,13,CONTEXTS,1)['max_common_feasible_context']==16

def test_hospital_k2_and_k1():
    assert common_feasibility(36,12,CONTEXTS,3)['max_common_feasible_context'] is None
    assert common_feasibility(36,12,CONTEXTS,2)['max_common_feasible_context']==8
    assert common_feasibility(36,12,CONTEXTS,1)['max_common_feasible_context']==16

def test_zero_is_no_probes():
    p=build_matrix([profile('a',11,5)],CONTEXTS,[0,1,2,3])
    assert p[0]['max_common_feasible_context']==8
    assert p[0]['no_probes_mode'] is True
    assert p[0]['can_support_k_calibration_origins'] is False
    assert p[1]['max_common_feasible_context'] is None

def test_sampled_cannot_prove_full_cohort():
    p=build_matrix([profile('a',100,12,n=5,total=20)],CONTEXTS,[0,1])
    assert not p[0]['full_cohort_verified']
    assert summarize(p,[0,1])[1]['full_cohort_tasks_with_context']==0

def test_provisional_family_deduplication():
    p=build_matrix([profile('ercot_h',1000,24,family='ercot'),
                    profile('ercot_d',1000,24,family='ercot'),
                    profile('usa',1000,24,family='us')],CONTEXTS,[3])
    out=summarize(p,[3])[0]
    assert out['full_cohort_tasks_with_context']==3
    assert out['provisional_family_hints_with_context']==2

def test_input_validation():
    with pytest.raises(ValueError): common_feasibility(100,0,CONTEXTS,1)
    with pytest.raises(ValueError): common_feasibility(100,10,[16,8],1)
    with pytest.raises(ValueError): build_matrix([profile('x',40,8),profile('x',40,8)],CONTEXTS,[0])
