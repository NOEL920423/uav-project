"""Fine-tune the latent BC actor with clipped PPO in the Isaac city env."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import signal
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bc-checkpoint", default="training_runs/latent_bc_city_v0/best.pt")
parser.add_argument("--output", default="training_runs/ppo_city_v0")
parser.add_argument("--timesteps", type=int, default=8192)
parser.add_argument("--rollout-steps", type=int, default=512)
parser.add_argument("--update-epochs", type=int, default=5)
parser.add_argument("--minibatch-size", type=int, default=128)
parser.add_argument("--learning-rate", type=float, default=3e-5)
parser.add_argument("--evaluation-episodes", type=int, default=20)
parser.add_argument("--seed", type=int, default=614420090)
parser.add_argument(
    "--resume",
    help="resume from a latest.pt checkpoint in the same output directory",
)
parser.add_argument(
    "--no-progress",
    action="store_true",
    help="disable interactive progress bars (useful when redirecting logs)",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from uav_ml.inference.latent_city_policy import load_bc_actor  # noqa: E402
from uav_ml.isaac.fixed_height_city_env import IsaacFixedHeightCityEnv  # noqa: E402
from uav_ml.models import LatentActorCritic  # noqa: E402


GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEFFICIENT = 0.2
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.001
MAX_GRADIENT_NORM = 0.5


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _save_latest(
    path: Path,
    model,
    optimizer,
    generator,
    elapsed: int,
    training_seed: int,
    history: list[dict],
    episode_count: int,
    episode_successes: int,
    initial_digest: str,
) -> None:
    """Atomically save one rollout boundary for interruption recovery."""
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "format_version": "bc_initialized_ppo_resume_v0.1",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "generator_state": generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            "elapsed_timesteps": elapsed,
            # A rollout may end in the middle of an episode. Resume at a fresh
            # deterministic seed instead of pretending the simulator state was saved.
            "resume_training_seed": training_seed + 1,
            "history": history,
            "episode_count": episode_count,
            "episode_successes": episode_successes,
            "initial_bc_actor_digest": initial_digest,
            "target_timesteps": args.timesteps,
        },
        temporary,
    )
    temporary.replace(path)


@torch.inference_mode()
def _evaluate(
    model, encoder, env, seed_base: int, episodes: int, show_progress: bool
) -> dict:
    records = []
    offsets = tqdm(
        range(episodes),
        desc="Final evaluation",
        unit="episode",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for offset in offsets:
        observation, _ = env.reset(seed=seed_base + offset)
        episode_return = 0.0
        while True:
            normalized = encoder.encode(observation)
            action = model.deterministic_action(normalized)[0].cpu().numpy()
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            if terminated or truncated:
                break
        records.append(
            {
                "seed": seed_base + offset,
                "success": bool(info["success"]),
                "collision": bool(info["collision"] or info["out_of_bounds"]),
                "timeout": bool(info["timeout"]),
                "steps": int(info["step_index"]),
                "return": episode_return,
                "final_distance_m": float(info["goal_distance_m"]),
            }
        )
        offsets.set_postfix(
            success=f"{sum(item['success'] for item in records) / len(records):.1%}",
            collision=f"{sum(item['collision'] for item in records) / len(records):.1%}",
        )
    return {
        "episodes": episodes,
        "seed_base": seed_base,
        "success_rate": sum(x["success"] for x in records) / episodes,
        "collision_rate": sum(x["collision"] for x in records) / episodes,
        "timeout_rate": sum(x["timeout"] for x in records) / episodes,
        "mean_return": sum(x["return"] for x in records) / episodes,
        "mean_final_distance_m": sum(x["final_distance_m"] for x in records) / episodes,
        "records": records,
    }


def _ppo_update(model, optimizer, batch, generator) -> dict:
    observations, actions, old_log_probs, advantages, returns = batch
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    indices = torch.arange(len(observations), device=observations.device)
    totals = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "updates": 0}
    for _ in range(args.update_epochs):
        order = indices[torch.randperm(len(indices), generator=generator, device=indices.device)]
        for start in range(0, len(order), args.minibatch_size):
            selected = order[start : start + args.minibatch_size]
            distribution = model.distribution(observations[selected])
            log_probs = distribution.log_prob(actions[selected]).sum(dim=1)
            ratio = (log_probs - old_log_probs[selected]).exp()
            unclipped = ratio * advantages[selected]
            clipped = ratio.clamp(1.0 - CLIP_COEFFICIENT, 1.0 + CLIP_COEFFICIENT) * advantages[selected]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(model.value(observations[selected]), returns[selected])
            entropy = distribution.entropy().sum(dim=1).mean()
            loss = policy_loss + VALUE_COEFFICIENT * value_loss - ENTROPY_COEFFICIENT * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRADIENT_NORM)
            optimizer.step()
            totals["policy"] += float(policy_loss.detach())
            totals["value"] += float(value_loss.detach())
            totals["entropy"] += float(entropy.detach())
            totals["updates"] += 1
    count = totals.pop("updates")
    return {name: value / count for name, value in totals.items()}


def main() -> None:
    if args.timesteps < 1 or args.rollout_steps < 2:
        raise ValueError("timesteps and rollout-steps must be positive")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output}", flush=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    bc_actor, encoder, bc_checkpoint = load_bc_actor(args.bc_checkpoint, device)
    model = LatentActorCritic(bc_actor.config).to(device)
    model.actor.load_state_dict(bc_actor.state_dict())
    initial_actor_state = copy.deepcopy(model.actor.state_dict())
    initial_digest = _state_digest(initial_actor_state)
    if initial_digest != _state_digest(bc_checkpoint["model_state"]):
        raise RuntimeError("PPO actor was not initialized exactly from BC")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    torch.save(
        {"model_state": model.state_dict(), "bc_actor_digest": initial_digest},
        output / "initialized_from_bc.pt",
    )
    env = IsaacFixedHeightCityEnv(device=args.device)

    training_seed = 40000
    elapsed = 0
    history = []
    episode_successes = episode_count = 0
    if args.resume:
        resume_path = Path(args.resume)
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if resume["initial_bc_actor_digest"] != initial_digest:
            raise ValueError("resume checkpoint uses a different BC initialization")
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        generator.set_state(resume["generator_state"].cpu())
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in resume["cuda_rng_state_all"]]
        )
        elapsed = int(resume["elapsed_timesteps"])
        training_seed = int(resume["resume_training_seed"])
        history = list(resume["history"])
        episode_count = int(resume["episode_count"])
        episode_successes = int(resume["episode_successes"])
        if elapsed >= args.timesteps:
            raise ValueError(
                f"resume already has {elapsed} timesteps; target must be larger"
            )
        print(f"Resuming {resume_path} at timestep {elapsed}", flush=True)
    observation, _ = env.reset(seed=training_seed)
    normalized = encoder.encode(observation)[0]
    # Isaac Kit installs its own signal handling during startup. Restore a
    # Python-level interruption here so Ctrl+C reaches the recovery path below.
    interruption_requested = False

    def _interrupt_training(_signal_number, _frame) -> None:
        # Raising directly here is unsafe because Isaac may invoke Python from
        # a Kit event callback that swallows exceptions. Set a flag and raise
        # from the main rollout loop instead.
        nonlocal interruption_requested
        interruption_requested = True

    signal.signal(signal.SIGINT, _interrupt_training)
    signal.signal(signal.SIGTERM, _interrupt_training)
    progress = tqdm(
        total=args.timesteps,
        initial=elapsed,
        desc="PPO training",
        unit="step",
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    interrupted = False
    try:
        while elapsed < args.timesteps:
            rollout_size = min(args.rollout_steps, args.timesteps - elapsed)
            observations, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
            for _ in range(rollout_size):
                if interruption_requested:
                    raise KeyboardInterrupt
                with torch.no_grad():
                    distribution = model.distribution(normalized.unsqueeze(0))
                    sampled_action = distribution.sample()[0]
                    log_prob = distribution.log_prob(sampled_action.unsqueeze(0)).sum(dim=1)[0]
                    value = model.value(normalized.unsqueeze(0))[0]
                next_observation, reward, terminated, truncated, info = env.step(
                    sampled_action.clamp(-1.0, 1.0).cpu().numpy()
                )
                done = terminated or truncated
                observations.append(normalized)
                actions.append(sampled_action)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(done)
                values.append(value)
                if done:
                    episode_count += 1
                    episode_successes += int(info["success"])
                    training_seed += 1
                    next_observation, _ = env.reset(seed=training_seed)
                normalized = encoder.encode(next_observation)[0]
                progress.update(1)
            with torch.no_grad():
                next_value = model.value(normalized.unsqueeze(0))[0]
            reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            done_tensor = torch.tensor(dones, dtype=torch.float32, device=device)
            value_tensor = torch.stack(values)
            advantages = torch.zeros_like(reward_tensor)
            gae = torch.zeros((), device=device)
            for index in reversed(range(rollout_size)):
                following_value = next_value if index == rollout_size - 1 else value_tensor[index + 1]
                nonterminal = 1.0 - done_tensor[index]
                delta = reward_tensor[index] + GAMMA * following_value * nonterminal - value_tensor[index]
                gae = delta + GAMMA * GAE_LAMBDA * nonterminal * gae
                advantages[index] = gae
            returns = advantages + value_tensor
            losses = _ppo_update(
                model,
                optimizer,
                (
                    torch.stack(observations),
                    torch.stack(actions),
                    torch.stack(log_probs),
                    advantages,
                    returns,
                ),
                generator,
            )
            elapsed += rollout_size
            row = {
                "timesteps": elapsed,
                **{f"loss_{name}": value for name, value in losses.items()},
                "training_episode_success_rate": episode_successes / max(episode_count, 1),
                "action_std": model.log_std.exp().detach().cpu().tolist(),
            }
            history.append(row)
            progress.set_postfix(
                episodes=episode_count,
                success=f"{row['training_episode_success_rate']:.1%}",
                policy_loss=f"{losses['policy']:.3f}",
                value_loss=f"{losses['value']:.1f}",
            )
            if args.no_progress:
                print(json.dumps(row), flush=True)
            _save_latest(
                output / "latest.pt",
                model,
                optimizer,
                generator,
                elapsed,
                training_seed,
                history,
                episode_count,
                episode_successes,
                initial_digest,
            )
    except KeyboardInterrupt:
        interrupted = True
        _save_latest(
            output / "latest.pt",
            model,
            optimizer,
            generator,
            elapsed,
            training_seed,
            history,
            episode_count,
            episode_successes,
            initial_digest,
        )
        print("\nTraining interrupted by user; recovery checkpoint saved.", flush=True)
        print(f"Checkpoint: {output / 'latest.pt'}", flush=True)
        print(f"Output directory: {output}", flush=True)
        print(
            f"Resume: ./train_ppo.sh resume {output} {args.timesteps}",
            flush=True,
        )
    except BaseException:
        _save_latest(
            output / "latest.pt",
            model,
            optimizer,
            generator,
            elapsed,
            training_seed,
            history,
            episode_count,
            episode_successes,
            initial_digest,
        )
        print("\nTraining failed; recovery checkpoint saved.", flush=True)
        print(f"Checkpoint: {output / 'latest.pt'}", flush=True)
        print(f"Output directory: {output}", flush=True)
        print(
            f"Resume: ./train_ppo.sh resume {output} {args.timesteps}",
            flush=True,
        )
        raise
    finally:
        progress.close()

    if interrupted:
        env.close()
        return

    final_digest = _state_digest(model.actor.state_dict())
    evaluation = _evaluate(
        model,
        encoder,
        env,
        50000,
        args.evaluation_episodes,
        show_progress=not args.no_progress,
    )
    checkpoint = {
        "format_version": "bc_initialized_ppo_v0.1",
        "model_state": model.state_dict(),
        "model_config": model.config.to_dict(),
        "observation_mean": bc_checkpoint["observation_mean"],
        "observation_std": bc_checkpoint["observation_std"],
        "autoencoder_checkpoint": bc_checkpoint["autoencoder_checkpoint"],
        "bc_checkpoint": str(Path(args.bc_checkpoint).resolve()),
        "initial_bc_actor_digest": initial_digest,
        "final_actor_digest": final_digest,
        "reward": "environment_reward_v0.1",
        "loss": {
            "policy": "negative_clipped_surrogate",
            "value_coefficient": VALUE_COEFFICIENT,
            "entropy_coefficient": ENTROPY_COEFFICIENT,
            "clip_coefficient": CLIP_COEFFICIENT,
        },
    }
    torch.save(checkpoint, output / "final.pt")
    summary = {
        "bc_initialization_exact": True,
        "initial_bc_actor_digest": initial_digest,
        "final_actor_digest": final_digest,
        "actor_updated": final_digest != initial_digest,
        "timesteps": elapsed,
        "training_episodes": episode_count,
        "training_success_rate": episode_successes / max(episode_count, 1),
        "evaluation": evaluation,
        "history": history,
        "checkpoint": str((output / "final.pt").resolve()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Training complete. Output directory: {output}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
