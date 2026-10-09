import io
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from m20pro_vla.training.visual_prefix_identity import VisualPrefixIdentity, embed_prefix_with_identity


class TinyVision:
    """Explicit tensor substitute, not a pretrained vision model."""
    def embed_image(self, image):
        return image.mean(dim=(-2, -1)).repeat(1, 2)[:, None, :].expand(-1, 2, -1)


class TinyPrefix:
    add_image_special_tokens = True
    global_image_start_token = torch.tensor([1])
    image_end_token = torch.tensor([2])
    def __init__(self): self.vlm_with_expert = TinyVision()
    def embed_prefix(self, images, masks, lang_tokens, lang_masks, state=None):
        pieces, validity = [], []
        for image, mask in zip(images, masks):
            embedded = self.vlm_with_expert.embed_image(image)
            special = torch.ones((len(image), 1, 6))
            pieces.extend([special, embedded, special])
            validity.extend([torch.ones((len(image), 1), dtype=torch.bool),
                             mask[:, None].expand(-1, 2), torch.ones((len(image), 1), dtype=torch.bool)])
        pieces.append(torch.full((len(state), 1, 6), 3.0))
        validity.append(torch.ones((len(state), 1), dtype=torch.bool))
        padding = torch.cat(validity, dim=1)
        return torch.cat(pieces, dim=1), padding, torch.zeros_like(padding)


class VisualPrefixIdentityTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(42)

    def test_age_camera_slot_kind_each_change_embedding(self):
        identity = VisualPrefixIdentity(6)
        embedding = torch.zeros((1, 2, 6))
        original = identity.tag(embedding, torch.tensor([0]), 0, 0, 0)
        variants = [identity.tag(embedding, torch.tensor([400]), 0, 0, 0),
                    identity.tag(embedding, torch.tensor([0]), 1, 0, 0),
                    identity.tag(embedding, torch.tensor([0]), 0, 1, 0),
                    identity.tag(embedding, torch.tensor([0]), 0, 0, 1)]
        self.assertTrue(all(not torch.equal(original, value) for value in variants))

    def test_masked_evidence_including_special_tokens_has_no_input_gradient(self):
        model = TinyPrefix(); identity = VisualPrefixIdentity(6)
        images = [torch.zeros((1, 3, 2, 2), requires_grad=True) for _ in range(2)]
        masks = [torch.tensor([False]), torch.tensor([True])]
        output, padding, _ = embed_prefix_with_identity(model, identity, images, masks,
            torch.tensor([[-1, 0]]), [0, 1], [0, 0], [0, 0], None, None, torch.zeros((1, 32)))
        self.assertFalse(padding[:, :4].any())
        self.assertTrue(torch.all(output[:, :4] == 0))
        output.sum().backward()
        self.assertTrue(torch.all(images[0].grad == 0))
        self.assertTrue(torch.all(images[1].grad != 0))

    def test_valid_early_image_influences_prefix_without_mutating_base_model(self):
        model = TinyPrefix(); identity = VisualPrefixIdentity(6)
        old_method = model.embed_prefix.__func__
        image = torch.zeros((1, 3, 2, 2))
        args = ([torch.tensor([True])], torch.tensor([[400]]), [0], [0], [0], None, None, torch.zeros((1, 32)))
        before = embed_prefix_with_identity(model, identity, [image], *args)[0]
        after = embed_prefix_with_identity(model, identity, [image + 1], *args)[0]
        self.assertFalse(torch.equal(before[:, 1:3], after[:, 1:3]))
        self.assertIs(model.embed_prefix.__func__, old_method)
        self.assertIsInstance(model.vlm_with_expert, TinyVision)

    def test_identity_has_finite_gradients_and_optimizer_registration_is_explicit(self):
        identity = VisualPrefixIdentity(6)
        tagged = identity.tag(torch.zeros((2, 2, 6)), torch.tensor([400, 100]), 0, 0, 0)
        tagged.square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in identity.parameters()))
        self.assertGreater(identity.age_projection.weight.grad.abs().sum().item(), 0)

    def test_contract_plus_tensor_state_roundtrip_reconstructs_same_tags(self):
        identity = VisualPrefixIdentity(6)
        data = io.BytesIO(); torch.save(identity.state_dict(), data); data.seek(0)
        restored = VisualPrefixIdentity.from_contract(identity.contract())
        restored.load_state_dict(torch.load(data, weights_only=True))
        args = (torch.zeros((1, 2, 6)), torch.tensor([700]), 1, 2, 1)
        self.assertTrue(torch.equal(identity.tag(*args), restored.tag(*args)))

    def test_future_age_privileged_contract_and_validity_mismatch_rejected(self):
        identity = VisualPrefixIdentity(6)
        with self.assertRaises(ValueError): identity.tag(torch.zeros((1, 2, 6)), torch.tensor([-2]), 0, 0, 0)
        with self.assertRaises(ValueError): VisualPrefixIdentity.from_contract(identity.contract() | {'target_xy': [1, 2]})
        with self.assertRaises(ValueError): embed_prefix_with_identity(TinyPrefix(), identity,
            [torch.zeros((1, 3, 2, 2))], [torch.tensor([True])], torch.tensor([[-1]]),
            [0], [0], [0], None, None, torch.zeros((1, 32)))


if __name__ == '__main__': unittest.main()
