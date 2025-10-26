#!/usr/bin/env python3
"""
Model visualization script using torchview
Creates an SVG visualization of the FlowMatchingModel architecture
"""

from collections import OrderedDict

import torch
from torchview import draw_graph

from model import FlowMatchingModel


def main():
    # Configuration
    CKPT_PATH = "models/model_1min.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")

    # Load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state = OrderedDict(
        (k.replace("_orig_mod.", ""), v) for k, v in ckpt["model_state"].items()
    )
    cfg = ckpt.get(
        "model_cfg",
        dict(
            d_model=512,
            nhead=8,
            enc_layers=6,
            dec_layers=8,
            ff=4 * 512,
            dropout=0.1,
            in_dim=7,
            context_dim=8,
        ),
    )
    model = FlowMatchingModel(**cfg).to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    print("Loaded checkpoint:", CKPT_PATH)

    # Create example inputs for torchview (batch_size=1 for visualization)
    B = 1
    x_hist = torch.randn(B, 60, 7)  # history sequence: 60 timesteps, 7 features
    x_t = torch.randn(B, 24, 7)  # future sequence: 24 timesteps, 7 features
    t = torch.rand(B, 1)  # time scalar
    ctx = torch.randn(B, 8)  # context vector: 8 features

    print("Example input shapes:")
    print(f"x_hist: {x_hist.shape}")
    print(f"x_t: {x_t.shape}")
    print(f"t: {t.shape}")
    print(f"ctx: {ctx.shape}")

    # torchview needs a callable and example inputs
    # Use positional arguments (model expects: x_hist, x_t, t_scalar, context)
    input_data = (
        x_hist,
        x_t,
        t,
        ctx,
    )  # (history_1min, future_trajectory, time_step, context)

    print("Creating torchview visualization...")
    graph = draw_graph(
        model,
        input_data=input_data,
        expand_nested=True,  # show main submodules as boxes
        graph_name="CFM",
        depth=1,  # show only top-level submodules
        roll=True,  # collapse repeated layers within each submodule
        show_shapes=True,  # show tensor shapes
    )
    output_file = graph.visual_graph.render("cfm_torchview", format="svg", cleanup=True)
    graph = draw_graph(
        model,
        input_data=input_data,
        expand_nested=True,  # show main submodules as boxes
        graph_name="CFM",
        depth=2,  # show only top-level submodules
        roll=True,  # collapse repeated layers within each submodule
        show_shapes=True,  # show tensor shapes
    )
    output_file = graph.visual_graph.render(
        "cfm_torchview_depth2", format="svg", cleanup=True
    )
    print(f"Visualization saved to: {output_file}")


if __name__ == "__main__":
    main()
