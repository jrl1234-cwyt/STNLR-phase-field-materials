from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def pad_neumann(x: torch.Tensor) -> torch.Tensor:
    return F.pad(x[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]


def grad_x(x: torch.Tensor, dx: float) -> torch.Tensor:
    padded = pad_neumann(x)
    return (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / (2.0 * dx)


def grad_y(x: torch.Tensor, dx: float) -> torch.Tensor:
    padded = pad_neumann(x)
    return (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / (2.0 * dx)


def laplace(x: torch.Tensor, dx: float) -> torch.Tensor:
    padded = pad_neumann(x)
    return (
        padded[:, 1:-1, 2:]
        + padded[:, 1:-1, :-2]
        + padded[:, 2:, 1:-1]
        + padded[:, :-2, 1:-1]
        - 4.0 * padded[:, 1:-1, 1:-1]
    ) / (dx * dx)


def finite_volume_anisotropic_force(
    phi: torch.Tensor,
    *,
    dx: float,
    epsilon4: float,
    theta0: float,
    w0: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return div(W^2 grad(phi) + cross flux) and the anisotropy.

    A cell-face finite-volume divergence is used here instead of applying two
    centered first-derivative operators.  The latter has an odd-even null mode
    and produces a checkerboard artifact during long dendrite rollouts.
    """
    phi_p = pad_neumann(phi)
    center = phi_p[:, 1:-1, 1:-1]
    left = phi_p[:, 1:-1, :-2]
    right = phi_p[:, 1:-1, 2:]
    up = phi_p[:, :-2, 1:-1]
    down = phi_p[:, 2:, 1:-1]
    phi_x = (right - left) / (2.0 * dx)
    phi_y = (down - up) / (2.0 * dx)
    theta = torch.atan2(phi_y, phi_x)
    anisotropy = 1.0 + epsilon4 * torch.cos(4.0 * (theta - theta0))
    anisotropy_theta = -4.0 * epsilon4 * torch.sin(4.0 * (theta - theta0))
    width = w0 * anisotropy
    coeff = width.square()
    cross = width * (w0 * anisotropy_theta)

    coeff_p = pad_neumann(coeff)
    cross_p = pad_neumann(cross)
    gx_p = pad_neumann(phi_x)
    gy_p = pad_neumann(phi_y)
    coeff_l, coeff_r = coeff_p[:, 1:-1, :-2], coeff_p[:, 1:-1, 2:]
    coeff_u, coeff_d = coeff_p[:, :-2, 1:-1], coeff_p[:, 2:, 1:-1]
    cross_l, cross_r = cross_p[:, 1:-1, :-2], cross_p[:, 1:-1, 2:]
    cross_u, cross_d = cross_p[:, :-2, 1:-1], cross_p[:, 2:, 1:-1]
    gx_u, gx_d = gx_p[:, :-2, 1:-1], gx_p[:, 2:, 1:-1]
    gy_l, gy_r = gy_p[:, 1:-1, :-2], gy_p[:, 1:-1, 2:]

    flux_x_right = 0.5 * (coeff + coeff_r) * (right - center) / dx
    flux_x_right -= 0.5 * (cross * phi_y + cross_r * gy_r)
    flux_x_left = 0.5 * (coeff_l + coeff) * (center - left) / dx
    flux_x_left -= 0.5 * (cross_l * gy_l + cross * phi_y)
    flux_y_down = 0.5 * (coeff + coeff_d) * (down - center) / dx
    flux_y_down += 0.5 * (cross * phi_x + cross_d * gx_d)
    flux_y_up = 0.5 * (coeff_u + coeff) * (center - up) / dx
    flux_y_up += 0.5 * (cross_u * gx_u + cross * phi_x)

    # Enforce the prescribed no-flux boundary condition on boundary faces.
    flux_x_left[:, :, 0] = 0.0
    flux_x_right[:, :, -1] = 0.0
    flux_y_up[:, 0, :] = 0.0
    flux_y_down[:, -1, :] = 0.0
    force = (flux_x_right - flux_x_left + flux_y_down - flux_y_up) / dx
    return force, anisotropy


@torch.no_grad()
def rhs(
    phi: torch.Tensor,
    temperature: torch.Tensor,
    *,
    dx: float,
    epsilon4: float,
    theta0: float,
    diffusion: float,
    w0: float,
    tau0: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    gradient_force, anisotropy = finite_volume_anisotropic_force(
        phi,
        dx=dx,
        epsilon4=epsilon4,
        theta0=theta0,
        w0=w0,
    )

    coupling = diffusion * tau0 / (0.6267 * w0 * w0)
    one_minus_phi2 = 1.0 - phi.square()
    chemical_force = (phi - coupling * temperature * one_minus_phi2) * one_minus_phi2
    phi_t = (chemical_force + gradient_force) / (tau0 * anisotropy.square())
    temperature_t = diffusion * laplace(temperature, dx) + 0.5 * phi_t
    return phi_t, temperature_t


def initial_conditions(
    trajectories: int,
    grid: int,
    dx: float,
    undercooling: float,
    seed_radius: float,
    seed_jitter: float,
    seed_shape_noise: float,
    temperature_noise: float,
    exact_index: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coordinates = torch.arange(grid, device=device, dtype=torch.float32) * dx
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    radii = seed_radius + seed_jitter * torch.randn(
        trajectories, generator=generator, device=device
    )
    distance = torch.sqrt(xx.square() + yy.square())[None]
    angle = torch.atan2(yy, xx)[None]
    amplitude4 = seed_shape_noise * torch.randn(
        trajectories, generator=generator, device=device
    )
    amplitude8 = 0.5 * seed_shape_noise * torch.randn(
        trajectories, generator=generator, device=device
    )
    if 0 <= exact_index < trajectories:
        radii[exact_index] = seed_radius
        amplitude4[exact_index] = 0.0
        amplitude8[exact_index] = 0.0
    local_radius = radii[:, None, None] * (
        1.0
        + amplitude4[:, None, None] * torch.cos(4.0 * angle)
        + amplitude8[:, None, None] * torch.cos(8.0 * angle)
    )
    phi = -torch.tanh((distance - local_radius) / math.sqrt(2.0))
    temperature = torch.full_like(phi, undercooling)
    if temperature_noise > 0:
        noise = torch.randn(
            phi.shape, generator=generator, device=device, dtype=phi.dtype
        )
        noise = F.avg_pool2d(noise[:, None], 5, stride=1, padding=2)[:, 0]
        if 0 <= exact_index < trajectories:
            noise[exact_index] = 0.0
        temperature = temperature + temperature_noise * noise
    return phi, temperature, radii


@torch.no_grad()
def generate(args: argparse.Namespace) -> dict[str, np.ndarray]:
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    phi, temperature, radii = initial_conditions(
        args.trajectories,
        args.grid,
        args.dx,
        args.undercooling,
        args.seed_radius,
        args.seed_jitter,
        args.seed_shape_noise,
        args.temperature_noise,
        args.exact_index,
        generator,
        device,
    )
    initial_tensor = torch.stack((phi, temperature), dim=1)
    if args.save_grid and args.save_grid != args.grid:
        initial_tensor = F.interpolate(
            initial_tensor,
            size=(args.save_grid, args.save_grid),
            mode="bilinear",
            align_corners=False,
        )
    initial = initial_tensor.cpu().numpy().astype(np.float32)
    output_times = torch.linspace(0.0, args.t_end, args.outputs, device=device)
    output_steps = torch.round(output_times / args.dt).long()
    total_steps = int(output_steps[-1])
    def snapshot() -> torch.Tensor:
        state = torch.stack((phi, temperature), dim=1)
        if args.save_grid and args.save_grid != args.grid:
            state = F.interpolate(
                state,
                size=(args.save_grid, args.save_grid),
                mode="bilinear",
                align_corners=False,
            )
        return state.cpu()

    snapshots = [snapshot()]
    next_output = 1
    started = time.time()
    for step in range(1, total_steps + 1):
        phi_t, temperature_t = rhs(
            phi,
            temperature,
            dx=args.dx,
            epsilon4=args.epsilon4,
            theta0=args.theta0,
            diffusion=args.diffusion,
            w0=args.w0,
            tau0=args.tau0,
        )
        phi = phi + args.dt * phi_t
        temperature = temperature + args.dt * temperature_t
        if not torch.isfinite(phi).all() or not torch.isfinite(temperature).all():
            raise FloatingPointError(f"non-finite state at integration step {step}")
        if next_output < len(output_steps) and step == int(output_steps[next_output]):
            snapshots.append(snapshot())
            next_output += 1
        if step % max(1, total_steps // 10) == 0:
            print(
                f"step={step}/{total_steps} phi=[{phi.min().item():.3f},{phi.max().item():.3f}] "
                f"U=[{temperature.min().item():.3f},{temperature.max().item():.3f}] "
                f"elapsed={time.time()-started:.1f}s",
                flush=True,
            )
    fields = torch.stack(snapshots, dim=1).numpy().astype(np.float32)
    return {
        "initial": initial,
        "fields": fields,
        "times": output_times.cpu().numpy().astype(np.float32),
        "seed_radius": radii.cpu().numpy().astype(np.float32),
        "epsilon4": np.full(args.trajectories, args.epsilon4, np.float32),
        "theta0": np.full(args.trajectories, args.theta0, np.float32),
        "diffusion": np.full(args.trajectories, args.diffusion, np.float32),
        "undercooling": np.full(args.trajectories, args.undercooling, np.float32),
        "dx": np.array(args.dx, dtype=np.float32),
        "dt": np.array(args.dt, dtype=np.float32),
        "w0": np.array(args.w0, dtype=np.float32),
        "tau0": np.array(args.tau0, dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=64)
    parser.add_argument("--grid", type=int, default=96)
    parser.add_argument(
        "--save-grid",
        type=int,
        default=0,
        help="Optionally downsample stored trajectories while integrating on the full grid.",
    )
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--t-end", type=float, default=120.0)
    parser.add_argument("--outputs", type=int, default=9)
    parser.add_argument("--epsilon4", type=float, default=0.05)
    parser.add_argument("--theta0", type=float, default=0.0)
    parser.add_argument("--diffusion", type=float, default=10.0)
    parser.add_argument("--undercooling", type=float, default=-0.3)
    parser.add_argument("--w0", type=float, default=1.0)
    parser.add_argument("--tau0", type=float, default=1.0)
    parser.add_argument("--seed-radius", type=float, default=8.0)
    parser.add_argument("--seed-jitter", type=float, default=0.35)
    parser.add_argument("--seed-shape-noise", type=float, default=0.015)
    parser.add_argument("--temperature-noise", type=float, default=0.0)
    parser.add_argument(
        "--exact-index",
        type=int,
        default=-1,
        help="Optional trajectory index with the exact radius-8 unperturbed PFHub initial seed.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = generate(args)
    np.savez_compressed(args.out, **payload)
    metadata = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    physical_extent = args.grid * args.dx
    if args.t_end >= 1500.0 and physical_extent >= 960.0:
        metadata["pfhub_scope"] = (
            "Full-domain, t=1500 PFHub Benchmark 3-type long-horizon dataset; "
            "the strict reference parameter row is included alongside a declared "
            "parameter-shifted transfer row."
        )
    else:
        metadata["pfhub_scope"] = (
            "Reduced-domain/short-horizon pilot of PFHub Benchmark 3 equations; "
            "not the full-domain, t=1500 reference calculation."
        )
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"saved {args.out} fields={payload['fields'].shape}", flush=True)


if __name__ == "__main__":
    main()
