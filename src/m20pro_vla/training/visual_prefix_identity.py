"""Isolated image-prefix identity candidate, not a policy/factory installation."""
import torch
from torch import nn


class VisualPrefixIdentity(nn.Module):
    """Trainable age/camera/slot identity added to individual image embeddings.

    Kinds: 0=fixed temporal RGB, 1=RGB colour evidence. Slots 0..2 represent
    the saved temporal order or colour bank order. Ages are actual past ticks
    at 25 Hz; -1 is masked missing evidence. No world state is introduced.
    Register on the future policy before optimizer creation and persist its
    contract and state_dict. This isolated helper does neither automatically.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError('Positive embedding dimension required')
        self.embedding_dim = embedding_dim
        self.age_projection = nn.Linear(4, embedding_dim, bias=False)
        self.camera_embedding = nn.Embedding(2, embedding_dim)
        self.slot_embedding = nn.Embedding(3, embedding_dim)
        self.kind_embedding = nn.Embedding(2, embedding_dim)

    def contract(self):
        return dict(schema='m20_visual_prefix_identity_candidate_v1',
                    embedding_dim=self.embedding_dim, fps=25,
                    age_features=['log1p_seconds', 'sin_seconds', 'cos_seconds', 'current'],
                    cameras=['front', 'rear'], kinds=['temporal', 'colour_evidence'],
                    initial_identity_scale=0.01)

    @classmethod
    def from_contract(cls, contract):
        if set(contract) != {'schema', 'embedding_dim', 'fps', 'age_features', 'cameras', 'kinds', 'initial_identity_scale'}:
            raise ValueError('Unknown identity contract fields')
        candidate = cls(contract['embedding_dim'])
        if candidate.contract() != contract:
            raise ValueError('Unknown identity contract')
        return candidate

    def tag(self, image_embedding, age_ticks, camera, slot, kind):
        if image_embedding.ndim != 3 or image_embedding.shape[-1] != self.embedding_dim:
            raise ValueError('Unexpected image embedding shape')
        if (age_ticks.shape != image_embedding.shape[:1] or age_ticks.dtype != torch.int64
                or (age_ticks < -1).any()):
            raise ValueError('Past ages must be int64 [B], with -1 reserved for missing')
        if type(camera) is not int or camera not in (0, 1) or type(slot) is not int or slot not in (0, 1, 2) or type(kind) is not int or kind not in (0, 1):
            raise ValueError('Unknown camera/slot/kind identity')
        seconds = age_ticks.clamp_min(0).to(self.age_projection.weight.dtype) / 25
        features = torch.stack([seconds.log1p(), seconds.sin(), seconds.cos(), (seconds == 0).to(seconds.dtype)], dim=-1)
        identity = (self.age_projection(features) + self.camera_embedding.weight[camera]
                    + self.slot_embedding.weight[slot] + self.kind_embedding.weight[kind]) * 0.01
        identity = identity.to(image_embedding.dtype) * (age_ticks >= 0)[:, None]
        return image_embedding + identity[:, None, :]


def embed_prefix_with_identity(base_model, identity, images, masks, age_ticks,
                               camera_ids, slot_ids, kind_ids, lang_tokens, lang_masks, state):
    """Run the base model's unchanged prefix method through a local image proxy.

    No base instance or installed class is mutated. Future loss/action sampling
    integration must explicitly call this same interface and register identity.
    Invalid history masks apply to its entire image block, including base image
    special tokens. Language, 32D state, and original attention grouping remain.
    """
    count = len(images)
    if count == 0 or not (len(masks) == len(camera_ids) == len(slot_ids) == len(kind_ids) == count):
        raise ValueError('Image identity cardinality mismatch')
    batch = images[0].shape[0]
    if age_ticks.dtype != torch.int64 or age_ticks.shape != (batch, count) or (age_ticks < -1).any():
        raise ValueError('Expected int64 [B,images] past ages')
    if state.shape != (batch, 32) or not torch.isfinite(state).all():
        raise ValueError('Expected finite existing 32D state')
    for i, (image, mask) in enumerate(zip(images, masks)):
        if (image.ndim != 4 or image.shape[1] != 3 or image.shape[0] != batch
                or mask.dtype != torch.bool or mask.shape != (batch,)
                or not torch.equal(mask, age_ticks[:, i] >= 0)):
            raise ValueError('Identity ages and image validity disagree')
        if not torch.isfinite(image).all():
            raise ValueError('Nonfinite RGB input')
    base_prefix = base_model.embed_prefix
    block_lengths = []

    class ImageProxy:
        def __init__(self): self.index = 0
        def __getattr__(self, name): return getattr(base_model.vlm_with_expert, name)
        def embed_image(self, image):
            i = self.index
            self.index += 1
            embedding = base_model.vlm_with_expert.embed_image(image)
            length = embedding.shape[1]
            if base_model.add_image_special_tokens:
                length += base_model.global_image_start_token.numel() + base_model.image_end_token.numel()
            block_lengths.append(length)
            return identity.tag(embedding, age_ticks[:, i], camera_ids[i], slot_ids[i], kind_ids[i])

    class ModelProxy:
        vlm_with_expert = ImageProxy()
        def __getattr__(self, name): return getattr(base_model, name)

    embeddings, padding, attention = base_prefix.__func__(
        ModelProxy(), images, masks, lang_tokens, lang_masks, state=state)
    if len(block_lengths) != count:
        raise ValueError('Base prefix did not consume all image slots')
    # The base method marks image special tokens valid even for a padded image.
    # Mask those tokens too; doing this on returned tensors avoids base mutation.
    padding = padding.clone()
    offset = 0
    for length, mask in zip(block_lengths, masks):
        padding[:, offset:offset+length] &= mask[:, None]
        offset += length
    embeddings = embeddings * padding[:, :, None].to(embeddings.dtype)
    return embeddings, padding, attention
