"""Isolated explicit official-factory invocation; no global/process hook installed."""
from types import FunctionType
from ..data.temporal_raw_candidate import TemporalRawDatasetCandidate
from .temporal_policy_candidate import temporal_policy_class


def make_temporal_policy_candidate(official_factory, config, dataset, *, require_temporal_contract=False):
    """Invoke the original factory body with one local policy-class selection.

    Future unified train-smolvla integration must call this explicitly before
    optimizer creation. No module globals, installed classes or CLI are patched.
    The dataset wrapper and policy must use exactly the same source-tick spec.
    """
    if not isinstance(dataset, TemporalRawDatasetCandidate):
        raise ValueError('Temporal policy requires the source-tick RGB dataset wrapper')
    if config.type != 'smolvla' or config.use_peft:
        raise ValueError('Only the original non-PEFT SmolVLA candidate is supported')
    if require_temporal_contract and not config.pretrained_path:
        raise ValueError('Temporal resume requires a checkpoint directory')
    if not isinstance(official_factory, FunctionType) or official_factory.__closure__:
        raise ValueError('Expected the explicit official factory function')
    getter=official_factory.__globals__.get('get_policy_class')
    if not callable(getter):raise ValueError('Factory policy-class interface is absent')
    spec=dataset.spec
    Candidate=temporal_policy_class(getter(config.type))

    class SelectedTemporalPolicy(Candidate):
        def __init__(self, config, **kwargs):
            if kwargs.get('temporal_spec',spec) != spec:
                raise ValueError('Dataset and policy temporal sampling differ')
            kwargs['temporal_spec']=spec
            super().__init__(config, **kwargs)

        @classmethod
        def from_pretrained(cls, path=None, **kwargs):
            path=kwargs.pop('pretrained_name_or_path',path)
            return super().from_pretrained(path,temporal_spec=spec,
                require_temporal_contract=require_temporal_contract,**kwargs)

    def selected(name):
        if name!='smolvla':raise ValueError('Unexpected policy selection')
        return SelectedTemporalPolicy
    namespace=dict(official_factory.__globals__)
    namespace['get_policy_class']=selected
    invoke=FunctionType(official_factory.__code__,namespace,official_factory.__name__,
        official_factory.__defaults__)
    invoke.__kwdefaults__=official_factory.__kwdefaults__
    policy=invoke(cfg=config,ds_meta=dataset.meta)
    if not isinstance(policy, Candidate) or policy.model.spec != spec:
        raise ValueError('Returned policy sampling differs')
    return policy
