"""CPU numerical tests; run with python -m unittest discover -s tests."""

import io
import unittest
import torch
from training.objectives import (
    teacher_support,
    student_baseline,
    combine_advantages,
    clipped_objective,
)


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.teacher = torch.randn(11, 23)
        self.student = torch.randn(11, 23)
        self.sample = torch.arange(11)

    def test_sparse_matches_dense(self):
        ids, p, r, hit = teacher_support(self.teacher, self.sample, 7)
        dense = torch.zeros_like(self.teacher).scatter(1, ids, p)
        b = student_baseline(self.student, ids, p)
        torch.testing.assert_close(b, (self.student.softmax(-1) * dense).sum(-1))
        torch.testing.assert_close(r, dense.gather(1, self.sample[:, None]).squeeze(1))
        self.assertTrue(torch.equal(hit, r > 0))

    def test_outside_topk_reward_zero(self):
        ids, p, r, hit = teacher_support(
            torch.tensor([[3.0, 2.0, 1.0]]), torch.tensor([2]), 1
        )
        self.assertEqual(r.item(), 0.0)
        self.assertFalse(hit.item())
        self.assertGreater(student_baseline(torch.zeros(1, 3), ids, p).item(), 0.0)

    def test_no_topk_renormalization(self):
        ids, p, r, _ = teacher_support(torch.zeros(3, 10), torch.arange(3), 2)
        torch.testing.assert_close(p, torch.full((3, 2), 0.1))
        torch.testing.assert_close(p.sum(-1), torch.full((3,), 0.2))

    def test_full_vocabulary(self):
        ids, p, _, _ = teacher_support(self.teacher, self.sample, 23)
        b = student_baseline(self.student, ids, p)
        torch.testing.assert_close(
            b, (self.teacher.softmax(-1) * self.student.softmax(-1)).sum(-1)
        )

    def test_zero_expected_local_advantage(self):
        ids, p, _, _ = teacher_support(self.teacher, self.sample, 5)
        dense = torch.zeros_like(self.teacher).scatter(1, ids, p)
        b = student_baseline(self.student, ids, p)
        expected = (self.student.softmax(-1) * (dense - b[:, None])).sum(-1)
        torch.testing.assert_close(expected, torch.zeros(11), atol=2e-8, rtol=0)

    def test_rewards_and_baseline_detached(self):
        ids, p, r, _ = teacher_support(self.teacher.requires_grad_(), self.sample, 7)
        b = student_baseline(self.student.requires_grad_(), ids, p)
        combined, local = combine_advantages(
            torch.ones(11, requires_grad=True), r, b, 1.0
        )
        self.assertFalse(any(x.requires_grad for x in (p, r, b, combined, local)))

    def test_lambda_zero_matches_grpo_loss_and_gradient(self):
        from training.grpo_core import grpo_loss

        new = torch.tensor([-0.2, -1.1, -0.8], requires_grad=True)
        old = torch.tensor([-0.3, -0.8, -0.5])
        ref = old - 0.1
        owners = torch.tensor([0, 1, 1])
        counts = torch.tensor([1.0, 2.0])
        weights = 1 / counts[owners] / 2
        ga = torch.tensor([1.0, -1.0, -1.0])
        a, _ = combine_advantages(ga, torch.rand(3), torch.rand(3), 0.0)
        actual, *_ = clipped_objective(new, old, ref, a, weights)
        expected, _ = grpo_loss(new, old, ref, ga, owners, counts, 2)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, new, retain_graph=True)[0],
            torch.autograd.grad(expected, new)[0],
        )

    def test_combination_before_clipping(self):
        # Opposite-sign contributions must be combined before the nonlinear min.
        new = torch.tensor([0.4], requires_grad=True)
        old = torch.zeros(1)
        weights = torch.ones(1)
        combined, _ = combine_advantages(
            torch.tensor([1.0]), torch.zeros(1), torch.ones(1), 2.0
        )
        loss, *_ = clipped_objective(new, old, old, combined, weights, beta=0.0)
        separate = sum(
            clipped_objective(new, old, old, a, weights, beta=0.0)[0]
            for a in (torch.ones(1), torch.tensor([-2.0]))
        )
        torch.testing.assert_close(loss, new.exp().sum())
        self.assertFalse(torch.isclose(loss, separate).item())

    def test_length_weighting_and_split_segments(self):
        new = torch.tensor([-0.4, -0.5, -0.6, -0.7], requires_grad=True)
        old = new.detach()
        ref = old
        adv = torch.tensor([1.0, -0.5, -0.5, -0.5])
        weights = torch.tensor([0.5, 1 / 6, 1 / 6, 1 / 6])
        all_loss = clipped_objective(new, old, ref, adv, weights)[0]
        split_loss = sum(
            clipped_objective(new[i:j], old[i:j], ref[i:j], adv[i:j], weights[i:j])[0]
            for i, j in [(0, 2), (2, 4)]
        )
        torch.testing.assert_close(all_loss, split_loss)
        torch.testing.assert_close(
            torch.autograd.grad(all_loss, new, retain_graph=True)[0],
            torch.autograd.grad(split_loss, new)[0],
        )

    def test_expected_policy_gradient(self):
        # Exhaustive action enumeration gives gradient of E_p[r], with detached b.
        theta = torch.tensor([0.2, -0.6, 0.8], requires_grad=True)
        p = theta.softmax(-1)
        reward = torch.tensor([0.6, 0.2, 0.0])
        b = (p.detach() * reward).sum()
        sampled_surrogate = (p.detach() * (reward - b) * theta.log_softmax(-1)).sum()
        expected = (p * reward).sum()
        torch.testing.assert_close(
            torch.autograd.grad(sampled_surrogate, theta, retain_graph=True)[0],
            torch.autograd.grad(expected, theta)[0],
        )

    def test_exact_prefix_packing(self):
        from training.grpo_core import pack_episode, feature_segments

        # Re-serialized second prompt differs from first raw output: retain it separately.
        e = dict(
            item_id="x",
            attempt=1,
            advantage=-1.0,
            trace=[
                dict(
                    prompt_ids=[10, 11],
                    output_ids=[12, 13],
                    behavior_logprobs=[-0.2, -0.3],
                ),
                dict(
                    prompt_ids=[10, 11, 99, 14],
                    output_ids=[15],
                    behavior_logprobs=[-0.4],
                ),
            ],
        )
        pieces = [
            f for f in feature_segments([pack_episode(e)]) if f["target_positions"]
        ]
        pairs = {
            (tuple(f["input_ids"][:p]), f["input_ids"][p])
            for f in pieces
            for p in f["target_positions"]
        }
        expected = {
            (tuple(t["prompt_ids"] + t["output_ids"][:j]), y)
            for t in e["trace"]
            for j, y in enumerate(t["output_ids"])
        }
        self.assertEqual(pairs, expected)
        self.assertTrue(all(f["trajectory_tokens"] == 3 for f in pieces))

    def test_zero_group_keeps_local_signal(self):
        combined, local = combine_advantages(
            torch.zeros(3),
            torch.tensor([0.8, 0.0, 0.2]),
            torch.tensor([0.2, 0.2, 0.2]),
            0.5,
        )
        torch.testing.assert_close(combined, 0.5 * local)
        self.assertGreater(combined.abs().sum().item(), 0.0)


def run_suite():
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ObjectiveTests)
    )
    if not result.wasSuccessful():
        raise AssertionError(stream.getvalue())
    return dict(passed=True, tests=result.testsRun, details=stream.getvalue())


if __name__ == "__main__":
    print(run_suite()["details"])
