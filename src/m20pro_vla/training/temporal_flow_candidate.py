"""Isolated flow-core bridge, not registered with the official policy factory."""
import torch
from torch import nn
from ..data.temporal_rgb import TemporalRGBSpec
from .visual_prefix_identity import VisualPrefixIdentity, embed_prefix_with_identity


class TemporalFlowCoreCandidate(nn.Module):
    """Invoke base training and action sampling through the same local prefix view.

    Requires six prepared images ordered oldest-first/front-rear. The caller
    must supply them with identical train/runtime sampling and persist the
    spec. This class does not replace the policy's last-frame preprocessing,
    factory, runtime buffer, official checkpoint loader or optimizer factory.
    """
    def __init__(self, core, embedding_dim, spec=TemporalRGBSpec()):
        super().__init__()
        if len(spec.offsets) != 3 or spec.fps != 25:
            raise ValueError('Candidate supports exactly three 25-Hz RGB slots')
        self.core = core
        self.identity = VisualPrefixIdentity(embedding_dim)
        self.spec = spec

    def contract(self):
        return dict(schema='m20_temporal_flow_core_candidate_v1',
                    temporal_rgb=self.spec.to_dict(), identity=self.identity.contract())

    @classmethod
    def from_contract(cls, core, contract):
        if set(contract) != {'schema', 'temporal_rgb', 'identity'} or contract['schema'] != 'm20_temporal_flow_core_candidate_v1':
            raise ValueError('Unknown flow bridge contract')
        spec = TemporalRGBSpec.from_dict(contract['temporal_rgb'])
        identity = VisualPrefixIdentity.from_contract(contract['identity'])
        result = cls(core, identity.embedding_dim, spec)
        result.identity = identity
        return result

    def _view(self, images, masks):
        if len(images) != 6 or len(masks) != 6:
            raise ValueError('Expected all three temporal front/rear pairs')
        if not torch.all(masks[-1]) or not torch.all(masks[-2]):
            raise ValueError('Current camera observations must be valid')
        for slot in range(3):
            if not torch.equal(masks[2*slot], masks[2*slot+1]):
                raise ValueError('Camera history validity must match')
        valid = torch.stack(masks, dim=1)
        expected = torch.tensor([-offset for offset in self.spec.offsets for _ in range(2)],
                                dtype=torch.int64, device=valid.device)
        ages = torch.where(valid, expected[None, :], -1)
        candidate = self

        class CoreView:
            def __getattr__(self, name): return getattr(candidate.core, name)
            def embed_prefix(self, images, masks, lang_tokens, lang_masks, state=None):
                return embed_prefix_with_identity(candidate.core, candidate.identity,
                    images, masks, ages, [0, 1]*3, [0, 0, 1, 1, 2, 2], [0]*6,
                    lang_tokens, lang_masks, state)
        return CoreView()

    def forward(self, images, masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
        return self.core.forward.__func__(self._view(images, masks), images, masks,
            lang_tokens, lang_masks, state, actions, noise=noise, time=time)

    def sample_actions(self, images, masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        return self.core.sample_actions.__func__(self._view(images, masks), images, masks,
            lang_tokens, lang_masks, state, noise=noise, **kwargs)
