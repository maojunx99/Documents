"""CPU loss/gradient references for the actual GPU and NPU selector modules.

Requires a working CPU-capable torch; native config tests also use
pydantic-settings and PyYAML. No accelerator or fused kernel is initialized.
"""

import ast
from contextlib import contextmanager
import importlib
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def native_modules(platform):
    """Load real math modules without importing model/accelerator registries."""
    source = ROOT / platform / "source/src/speculators"
    with patch.dict(sys.modules):
        for name in list(sys.modules):
            if name == "speculators" or name.startswith("speculators."):
                del sys.modules[name]
        for name in ["", ".models", ".models.dflash2", ".models.dspark", ".train"]:
            module = types.ModuleType("speculators" + name)
            module.__path__ = [
                str(source.joinpath(*name.strip(".").split(".")) if name else source)
            ]
            sys.modules[module.__name__] = module
        yield source


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "Numerical tests require torch on CPU"
)
class SelectorLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, F
        import torch
        from torch.nn import functional as F

        cls.modules = {}
        for platform in ["gpu", "npu"]:
            with native_modules(platform):
                cls.modules[platform] = (
                    importlib.import_module("speculators.models.dflash2.metrics"),
                    importlib.import_module(
                        "speculators.models.dflash2.selector_losses"
                    ),
                    importlib.import_module("speculators.losses.utils"),
                    importlib.import_module("speculators.losses.eager"),
                )

    def inputs(self, dtype=None):
        generator = torch.Generator().manual_seed(73)
        teacher = torch.randn(
            1, 24, 13, generator=generator, dtype=dtype or torch.float32
        )
        scores = torch.randn(
            1, 24, 4, generator=generator, dtype=dtype or torch.float32
        )
        ids = torch.rand(1, 24, 13, generator=generator).topk(4, dim=-1).indices
        # Includes masked tokens and an entirely padded anchor block.
        mask = torch.tensor([[1, 1, 1, 0, 1, 1, 1, 0] * 2 + [0] * 8])
        labels = torch.arange(24).reshape(1, -1) % 4
        return teacher, scores, ids, mask, labels

    def test_tv_loss_and_backbone_gradients_match_explicit_full_vocabulary(self):
        for platform, (metrics, helpers, utils, _) in self.modules.items():
            for dtype in [torch.float32, torch.bfloat16]:
                for sample_from_anchor in [False, True]:
                    with self.subTest(
                        platform=platform, dtype=dtype, anchor=sample_from_anchor
                    ):
                        teacher, scores, ids, mask, labels = self.inputs(dtype)
                        teacher.requires_grad_()
                        # Candidate scores remain connected to trainable draft features.
                        features = scores.clone().requires_grad_()
                        scores = features * 0.7
                        log_probs = teacher.gather(
                            -1, ids
                        ).float() - helpers.teacher_log_normalizer(teacher).unsqueeze(
                            -1
                        )
                        actual = metrics.compute_selector_loss(
                            scores,
                            labels,
                            mask,
                            8,
                            gamma=4.0,
                            per_position_loss_weight="fixed-exp-decay",
                            dpace_alpha=0.5,
                            sample_from_anchor=sample_from_anchor,
                            selector_loss_type="full_teacher_tv",
                            teacher_candidate_log_probs=log_probs,
                        )
                        p = teacher.detach().float().softmax(-1)
                        q = torch.zeros_like(p).scatter(
                            -1, ids, scores.float().softmax(-1)
                        )
                        elements = 0.5 * (p - q).abs().sum(-1)
                        pos = torch.arange(24).reshape(1, -1) % 8
                        decay = torch.exp(
                            -(
                                (
                                    pos
                                    if sample_from_anchor
                                    else (pos - 1).clamp(min=0)
                                ).float()
                            )
                            / 4
                        )
                        if not sample_from_anchor:
                            decay *= pos != 0
                        expected = (elements * mask * decay).sum() / (mask.sum() + 1e-5)
                        torch.testing.assert_close(
                            actual, expected, atol=2e-7, rtol=2e-6
                        )
                        grad, teacher_grad = torch.autograd.grad(
                            actual,
                            (features, teacher),
                            retain_graph=True,
                            allow_unused=True,
                        )
                        reference_grad = torch.autograd.grad(expected, features)[0]
                        torch.testing.assert_close(
                            grad,
                            reference_grad,
                            atol=2e-5 if dtype == torch.bfloat16 else 1e-8,
                            rtol=0.01 if dtype == torch.bfloat16 else 2e-6,
                        )
                        self.assertGreater(grad.abs().sum().item(), 0)
                        self.assertEqual(grad[:, 16:].count_nonzero().item(), 0)
                        self.assertIsNone(teacher_grad)

    def test_hard_ce_default_is_exactly_the_previous_loss_and_gradient(self):
        for platform, (metrics, _, utils, _) in self.modules.items():
            for weighting in ["fixed-exp-decay", "dpace"]:
                with self.subTest(platform=platform, weighting=weighting):
                    _, scores, _, mask, labels = self.inputs()
                    scores.requires_grad_()
                    actual = metrics.compute_selector_loss(
                        scores,
                        labels,
                        mask,
                        8,
                        gamma=4.0,
                        per_position_loss_weight=weighting,
                        dpace_alpha=0.5,
                    )
                    ce = F.cross_entropy(
                        scores.flatten(0, -2), labels.flatten(), reduction="none"
                    ).view_as(labels)
                    elements = ce * mask
                    pos = (torch.arange(24) % 8).unsqueeze(0).float()
                    decay = (
                        utils.dpace_loss_decay(pos, mask, 8, 0.5, elements)
                        if weighting == "dpace"
                        else utils.dflash_loss_decay(pos, 4.0)
                    )
                    previous = (
                        (elements * decay).sum(dim=1) / (mask.sum(dim=1) + 1e-5)
                    ).mean()
                    self.assertTrue(torch.equal(actual, previous))
                    self.assertTrue(
                        torch.equal(
                            torch.autograd.grad(actual, scores)[0],
                            torch.autograd.grad(previous, scores)[0],
                        )
                    )

    def test_tv_dpace_retains_ce_confidence_weights(self):
        for metrics, helpers, utils, _ in self.modules.values():
            teacher, scores, ids, mask, labels = self.inputs()
            log_probs = teacher.gather(-1, ids) - helpers.teacher_log_normalizer(
                teacher
            ).unsqueeze(-1)
            actual = metrics.compute_selector_loss(
                scores,
                labels,
                mask,
                8,
                gamma=4.0,
                per_position_loss_weight="dpace",
                dpace_alpha=0.5,
                selector_loss_type="full_teacher_tv",
                teacher_candidate_log_probs=log_probs,
            )
            ce = F.cross_entropy(
                scores.flatten(0, -2), labels.flatten(), reduction="none"
            ).view_as(labels)
            decay = utils.dpace_loss_decay(
                torch.arange(24).reshape(1, -1) % 8, mask, 8, 0.5, ce * mask
            )
            p = teacher.softmax(-1)
            q = torch.zeros_like(p).scatter(-1, ids, scores.softmax(-1))
            expected = ((p - q).abs().sum(-1) * 0.5 * mask * decay).sum() / (
                mask.sum() + 1e-5
            )
            torch.testing.assert_close(actual, expected)

    def test_full_teacher_mass_is_not_renormalized_within_candidates(self):
        for _, helpers, _, _ in self.modules.values():
            teacher = torch.zeros(1, 1, 10)
            ids = torch.tensor([[[0, 1]]])
            p_log = teacher.gather(-1, ids) - helpers.teacher_log_normalizer(
                teacher
            ).unsqueeze(-1)
            loss = helpers.full_teacher_tv_elements(torch.zeros(1, 1, 2), p_log)
            torch.testing.assert_close(loss, torch.tensor([[0.8]]))

    def test_chunked_normalizer_matches_dense_and_bounds_fp32_temporary(self):
        for _, helpers, _, _ in self.modules.values():
            logits = (torch.arange(513 * 13).reshape(1, 513, 13) % 47 - 20).to(
                torch.bfloat16
            )
            with patch.object(torch, "logsumexp", wraps=torch.logsumexp) as operation:
                actual = helpers.teacher_log_normalizer(
                    logits.requires_grad_(), chunk_rows=256
                )
                self.assertEqual(
                    [call.args[0].shape[0] for call in operation.call_args_list],
                    [256, 256, 1],
                )
                self.assertTrue(
                    all(
                        call.args[0].dtype == torch.float32
                        for call in operation.call_args_list
                    )
                )
            self.assertFalse(actual.requires_grad)
            self.assertTrue(torch.equal(actual, torch.logsumexp(logits.float(), -1)))
            with self.assertRaises(ValueError):
                helpers.teacher_log_normalizer(logits, 0)

    def test_combined_metrics_use_selected_loss_and_keep_unary_and_candidate_mass(self):
        for metrics, helpers, utils, eager in self.modules.values():
            teacher, scores, _, mask, _ = self.inputs()
            unary = -teacher.clone().requires_grad_()
            ids, labels, contains = metrics.selector_training_candidates(
                unary.topk(4, -1).indices, teacher.argmax(-1)
            )
            loss_config = utils.resolve_loss_config('{"ce": 0.1, "tv": 0.9}', "eager")
            common = dict(
                unary_logits=unary,
                targets=teacher,
                training_candidate_ids=ids,
                candidate_logits=scores.requires_grad_(),
                target_positions=labels,
                contains_target=contains,
                loss_mask=mask,
                block_size=8,
                top_k=4,
                loss_config=loss_config,
                tv_loss_fn=eager.tv_loss,
                selector_loss_alpha=0.1,
            )
            _, ce_metrics = metrics.compute_metrics(**common)
            loss, tv_metrics = metrics.compute_metrics(
                **common, selector_loss_type="full_teacher_tv"
            )
            self.assertTrue(
                torch.equal(ce_metrics["unary_loss_sum"], tv_metrics["unary_loss_sum"])
            )
            log_probs = teacher.gather(-1, ids) - helpers.teacher_log_normalizer(
                teacher
            ).unsqueeze(-1)
            selector = metrics.compute_selector_loss(
                scores,
                labels,
                mask,
                8,
                gamma=4.0,
                per_position_loss_weight="fixed-exp-decay",
                dpace_alpha=0.5,
                selector_loss_type="full_teacher_tv",
                teacher_candidate_log_probs=log_probs,
            )
            torch.testing.assert_close(
                loss.detach(), tv_metrics["unary_loss_sum"] + 0.1 * selector.detach()
            )
            expected_mass = teacher.float().softmax(-1).gather(-1, ids).sum(-1)
            torch.testing.assert_close(
                tv_metrics["unary_candidate_target_mass_at_4_sum"],
                (expected_mass * mask).sum(),
            )
            loss.backward()
            self.assertGreater(scores.grad.abs().sum().item(), 0)
            # Missing teacher argmax is still injected, with no duplicate IDs.
            self.assertTrue(
                ids.gather(-1, labels.unsqueeze(-1))
                .squeeze(-1)
                .eq(teacher.argmax(-1))
                .all()
            )
            self.assertTrue(ids.sort(-1).values.diff(dim=-1).ne(0).all())

    def test_invalid_objective_or_missing_probabilities_fails(self):
        for metrics, _, _, _ in self.modules.values():
            _, scores, _, mask, labels = self.inputs()
            for extra in [
                dict(selector_loss_type="unknown"),
                dict(selector_loss_type="full_teacher_tv"),
                dict(
                    selector_loss_type="full_teacher_tv",
                    teacher_candidate_log_probs=scores[..., :1],
                ),
            ]:
                with self.assertRaises(ValueError):
                    metrics.compute_selector_loss(
                        scores,
                        labels,
                        mask,
                        8,
                        gamma=4.0,
                        per_position_loss_weight="fixed-exp-decay",
                        dpace_alpha=0.5,
                        **extra,
                    )

    @unittest.skipUnless(
        importlib.util.find_spec("pydantic_settings")
        and importlib.util.find_spec("yaml"),
        "Native schema/CLI checks require pydantic-settings and PyYAML",
    )
    def test_native_schema_cli_flatten_and_train_val_kwargs(self):
        for platform in self.modules:
            with self.subTest(platform=platform), native_modules(platform) as source:
                # These two external I/O registries are not used by the loss/config test.
                client = types.ModuleType("speculators.data_generation.vllm_client")
                client.DEFAULT_MAX_RETRIES = 3
                client.DEFAULT_REQUEST_TIMEOUT = 600
                sys.modules[client.__name__] = client
                backend = types.ModuleType("hs_connectors")
                backend.HiddenStatesBackend = types.SimpleNamespace(registry={})
                sys.modules["hs_connectors"] = backend
                schema = importlib.import_module("speculators.train.config.schema")
                resolution = importlib.import_module(
                    "speculators.train.config.resolution"
                )
                core = ast.parse((source / "models/dflash2/core.py").read_text())
                model = next(
                    n
                    for n in core.body
                    if isinstance(n, ast.ClassDef) and n.name == "DFlash2DraftModel"
                )
                method = next(
                    n
                    for n in model.body
                    if isinstance(n, ast.FunctionDef) and n.name == "get_trainer_kwargs"
                )
                method.decorator_list = []
                env = {
                    "resolve_loss_config": lambda value, implementation: {
                        "tv": (None, 1.0)
                    }
                }
                exec(
                    compile(
                        ast.Module(body=[method], type_ignores=[]), str(source), "exec"
                    ),
                    env,
                )
                self.assertEqual(
                    schema.TrainConfig().flatten()["selector_loss_type"], "hard_ce"
                )
                for objective in ["hard_ce", "full_teacher_tv"]:
                    args = resolution.build_parser().parse_args(
                        ["--selector-loss-type", objective]
                    )
                    cfg = schema.TrainConfig(
                        dflash2={"selector_loss_type": args.selector_loss_type}
                    )
                    train, val = env["get_trainer_kwargs"](**cfg.flatten())
                    self.assertEqual(train["selector_loss_type"], objective)
                    self.assertEqual(val["selector_loss_type"], objective)
                    self.assertIsNot(train, val)
                with self.assertRaises(ValueError):
                    schema.DFlash2Args(selector_loss_type="conditional_tv")


if __name__ == "__main__":
    unittest.main()
