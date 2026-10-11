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
        self.binary_stop_spec = None
        self.binary_stop_head = None
        self.last_stop_logits = None

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

    def enable_binary_stop(self, spec):
        """Enable a saved, learned binary chunk head without altering the flow core.

        Call before strict loading a binary-head checkpoint or constructing an
        optimizer. The ordinary temporal policy retains its original dispatch.
        """
        from .binary_stop import validate_binary_stop_spec, make_binary_stop_head
        if self.binary_stop_spec is not None or self.binary_stop_head is not None:
            raise ValueError('Binary STOP head is already enabled')
        spec = validate_binary_stop_spec(spec)
        if spec['hidden_dim'] != self.core.state_proj.out_features:
            raise ValueError('Binary STOP head width differs from prefix state width')
        if spec['chunk_size'] != self.core.config.chunk_size or self.core.config.max_action_dim != 32:
            raise ValueError('Binary STOP head requires original 50 x 32 flow actions')
        head = make_binary_stop_head(spec)
        head.to(device=self.core.state_proj.weight.device, dtype=self.core.state_proj.weight.dtype)
        head.train(self.training)
        self.binary_stop_head = head
        self.binary_stop_spec = spec
        self.last_stop_logits = None

    def _canonical_binary_stop_logits(self, embeddings, padding, attention):
        """Canonical batch-one head branch, also used for checkpoint recomputation.

        Explicit immutable tensor inputs keep backward independent of the
        per-call capture bookkeeping. KV cache and full prefix outputs are not
        returned or stored; only the learned B x 50 logits leave this function.
        """
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        from .binary_stop import prefix_stop_logits
        if embeddings.shape[0] != 1 or padding.shape[0] != 1 or attention.shape[0] != 1:
            raise ValueError('Canonical binary STOP prefix requires exactly one observation')
        outputs, _ = self.core.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(padding, attention),
            position_ids=torch.cumsum(padding, dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[embeddings, None],
            use_cache=self.core.config.use_cache,
            fill_kv_cache=True,
        )
        if not isinstance(outputs[0], torch.Tensor):
            raise ValueError('VLM did not return its prefix hidden states')
        return prefix_stop_logits(self.binary_stop_head, outputs[0], padding)

    def _binary_stop_view(self, images, masks):
        """Use one canonical observation-only prefix output for the learned head.

        No installed method or shared core attribute is patched. Flow training
        retains its original joint prefix/suffix pass. Its learned head uses an
        extra text-prefix-only pass over the SAME prefix embeddings/gradient
        graph, without repeating any image encoding. The head uses batch-one
        text-prefix attention in both training and inference, since BF16 results
        otherwise depend on the prefix batch size. Actual batch-one inference
        reuses its existing prefix-cache pass. Batched inference computes only
        the head's prefix per sample while preserving its full flow/cache pass.
        Training checkpoints each head-prefix sample to bound retained and
        transient attention memory, recomputing that text prefix for backward.
        Flow batch, optimizer batch, loss reduction and labels remain unchanged.
        """
        from .binary_stop import prefix_stop_logits
        if self.binary_stop_spec is None or self.binary_stop_head is None:
            raise ValueError('Binary STOP view requires an enabled head')
        temporal_view = self._view(images, masks)
        candidate = self
        capture = {'mask': None, 'attention': None, 'embeddings': None, 'count': 0}
        self.last_stop_logits = None

        class PrefixOutputView:
            def __getattr__(self, name):
                return getattr(candidate.core.vlm_with_expert, name)

            def forward(self, *args, **kwargs):
                result = candidate.core.vlm_with_expert.forward(*args, **kwargs)
                inputs = kwargs.get('inputs_embeds')
                if inputs is None or len(inputs) != 2:
                    raise ValueError('Expected explicit prefix/suffix VLM inputs')
                if inputs[0] is not None:
                    if capture['mask'] is None or capture['count'] != 0:
                        raise ValueError('Expected exactly one prefix pass for binary STOP')
                    outputs, _ = result
                    if inputs[1] is not None or inputs[0].shape[0] != 1:
                        # Both joint suffix tokens and text-prefix batch size can
                        # change BF16 arithmetic. Canonicalize the learned head
                        # to the actual runtime's observation-only batch-one path.
                        if inputs[0] is not capture['embeddings']:
                            raise ValueError('Binary STOP prefix embeddings were replaced')
                        prefix_inputs = (capture['embeddings'], capture['mask'], capture['attention'])
                        parts = []
                        for sample in range(prefix_inputs[0].shape[0]):
                            sample_inputs = tuple(value[sample:sample + 1] for value in prefix_inputs)
                            if candidate.training and torch.is_grad_enabled():
                                from torch.utils.checkpoint import checkpoint
                                part = checkpoint(candidate._canonical_binary_stop_logits,
                                    *sample_inputs, use_reentrant=False, preserve_rng_state=True)
                            else:
                                part = candidate._canonical_binary_stop_logits(*sample_inputs)
                            parts.append(part)
                        # Preserve source order and one full-batch loss/update.
                        logits = torch.cat(parts, dim=0)
                    else:
                        if not isinstance(outputs[0], torch.Tensor):
                            raise ValueError('VLM did not return its prefix hidden states')
                        # Actual batch-one inference reuses its existing pass.
                        logits = prefix_stop_logits(
                            candidate.binary_stop_head, outputs[0], capture['mask'])
                    # Capture mutation happens once outside the checkpointed
                    # function. Backward recomputation never changes it.
                    candidate.last_stop_logits = logits
                    capture['count'] += 1
                return result

        prefix_output_view = PrefixOutputView()

        class BinaryCoreView:
            def __getattr__(self, name):
                if name == 'vlm_with_expert':
                    return prefix_output_view
                return getattr(temporal_view, name)

            def embed_prefix(self, images, masks, lang_tokens, lang_masks, state=None):
                result = temporal_view.embed_prefix(images, masks, lang_tokens, lang_masks, state=state)
                capture['embeddings'], capture['mask'], capture['attention'] = result
                return result

        return BinaryCoreView(), capture

    def _require_binary_stop_capture(self, capture, batch_size):
        if capture['count'] != 1 or self.last_stop_logits is None:
            raise ValueError('Binary STOP did not capture exactly one prefix output')
        if self.last_stop_logits.shape != (batch_size, self.core.config.chunk_size):
            raise ValueError('Binary STOP logits have an unexpected batch/chunk shape')

    def forward(self, images, masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
        if self.binary_stop_spec is None:
            return self.core.forward.__func__(self._view(images, masks), images, masks,
                lang_tokens, lang_masks, state, actions, noise=noise, time=time)
        view, capture = self._binary_stop_view(images, masks)
        losses = self.core.forward.__func__(view, images, masks,
            lang_tokens, lang_masks, state, actions, noise=noise, time=time)
        self._require_binary_stop_capture(capture, actions.shape[0])
        # Keep graph-connected logits for the explicit binary objective. The
        # inherited 50 x 32 flow losses are returned completely unchanged.
        return losses

    def sample_actions(self, images, masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        if self.binary_stop_spec is None:
            return self.core.sample_actions.__func__(self._view(images, masks), images, masks,
                lang_tokens, lang_masks, state, noise=noise, **kwargs)
        from .binary_stop import replace_stop_column
        view, capture = self._binary_stop_view(images, masks)
        actions = self.core.sample_actions.__func__(view, images, masks,
            lang_tokens, lang_masks, state, noise=noise, **kwargs)
        self._require_binary_stop_capture(capture, state.shape[0])
        # Only the saved normalized STOP column changes. Original flow noise,
        # dimensions, motion columns and sampling loop remain untouched.
        return replace_stop_column(actions, self.last_stop_logits, self.binary_stop_spec)
