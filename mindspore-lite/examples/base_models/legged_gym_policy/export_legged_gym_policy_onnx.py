#!/usr/bin/env python3
"""Export a Legged Gym / rsl_rl locomotion policy (MLP actor) to a single ONNX.

Legged Gym trains an MLP actor-critic with `rsl_rl`; only the deterministic
actor (observation -> action mean, bounded by tanh) is needed for deployment.

The exported module maps:

  - input  : observation   [batch, obs_dim]    float32
  - output : action        [batch, action_dim] float32   (tanh-bounded)

The script is self-contained (no rsl_rl / legged_gym source import required):
  * `--checkpoint <actor_critic.pt>`: load a real rsl_rl ActorCritic state_dict
    and copy the actor trunk weights into a standalone ``PolicyMLP``.
  * `--random-init` (default): build a demo policy with seeded random weights so
    the full export/convert/infer/align pipeline can be exercised end-to-end.
    Swap in a trained checkpoint for real locomotion behavior.

Default shape (ANYmal-like quadruped): obs_dim=235, action_dim=18,
hidden_dims=(512, 256, 128), activation=elu, output tanh. Override via CLI.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


_ACT = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


class PolicyMLP(nn.Module):
    """Standalone MLP actor mapping an observation vector to a tanh-bounded action."""

    def __init__(self, obs_dim, action_dim, hidden_dims, activation="elu", output_tanh=True):
        super().__init__()
        act = _ACT[activation]
        layers = []
        in_dim = int(obs_dim)
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, int(h)), act()]
            in_dim = int(h)
        layers += [nn.Linear(in_dim, int(action_dim))]
        if output_tanh:
            layers += [nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, observation):
        """Return the deterministic action mean for a batch of observations."""
        return self.net(observation)


def _load_from_checkpoint(ckpt_path, obs_dim, action_dim, hidden_dims, activation, output_tanh):
    """Build a ``PolicyMLP`` and copy an rsl_rl ActorCritic actor trunk into it."""
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("model_state_dict", state)
    # rsl_rl actor naming variants: "actor.0.linear.weight", "actor_net.0.0.weight", ...
    actor_sd = {}
    for key, val in sd.items():
        if not key.startswith("actor") or "std" in key or "log" in key:
            continue
        actor_sd[key.replace("actor_net", "").replace("actor.", "", 1).lstrip(".")] = val
    model = PolicyMLP(obs_dim, action_dim, hidden_dims, activation, output_tanh)
    missing, unexpected = model.load_state_dict(actor_sd, strict=False)
    print(f"checkpoint loaded from {ckpt_path}: matched={len(actor_sd)} "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if len(actor_sd) == 0:
        raise RuntimeError("No actor weights matched in checkpoint; check rsl_rl naming.")
    return model


def _parse_hidden(s):
    """Parse a comma-separated hidden-dims string into a tuple of ints."""
    return tuple(int(x) for x in str(s).split(",") if x.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Export a Legged Gym / rsl_rl locomotion policy (MLP actor) to ONNX."
    )
    parser.add_argument("--output-dir", type=str, default="./legged_gym_policy_onnx")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--checkpoint", type=str, default="",
                        help="rsl_rl ActorCritic .pt; omit to use --random-init demo.")
    parser.add_argument("--random-init", action="store_true",
                        help="Build a demo policy with seeded random weights (default).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-dim", type=int, default=235)
    parser.add_argument("--action-dim", type=int, default=18)
    parser.add_argument("--hidden-dims", type=str, default="512,256,128")
    parser.add_argument("--activation", type=str, default="elu", choices=list(_ACT.keys()))
    parser.add_argument("--no-output-tanh", action="store_true",
                        help="Disable tanh on the action output (unbounded actions).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden_dims = _parse_hidden(args.hidden_dims)
    output_tanh = not args.no_output_tanh

    if args.checkpoint:
        model = _load_from_checkpoint(args.checkpoint, args.obs_dim, args.action_dim,
                                      hidden_dims, args.activation, output_tanh)
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = PolicyMLP(args.obs_dim, args.action_dim, hidden_dims,
                          args.activation, output_tanh)
        print(f"random-init demo policy (seed={args.seed}). "
              f"Use --checkpoint for a real trained policy.")

    model = model.to(args.device).eval()

    dummy_obs = torch.randn(1, args.obs_dim, dtype=torch.float32, device=args.device)
    onnx_path = output_dir / "legged_gym_policy.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_obs,),
            str(onnx_path),
            input_names=["observation"],
            output_names=["action"],
            opset_version=int(args.opset),
            do_constant_folding=False,
            dynamo=False,
        )

    print("Export complete.")
    print(f"ONNX saved to: {onnx_path}")
    print(f"  obs_dim={args.obs_dim} action_dim={args.action_dim} "
          f"hidden={hidden_dims} act={args.activation} output_tanh={output_tanh}")


if __name__ == "__main__":
    main()
