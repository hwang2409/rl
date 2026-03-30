"""Export a trained PPO model to ONNX for browser inference."""

import sys

import numpy as np
import torch
import torch.nn as nn

from config import cartpole_config
from network import DiscreteActorCritic


class ActorOnly(nn.Module):
    """Wraps just the actor network for ONNX export."""

    def __init__(self, actor: nn.Sequential):
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


def export(checkpoint_path: str, output_path: str = "cartpole.onnx"):
    config = cartpole_config()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = DiscreteActorCritic(4, 2, config.hidden_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Export only the actor (policy network) — we don't need the critic for inference
    actor_only = ActorOnly(model.actor)
    actor_only.eval()

    dummy_input = torch.randn(1, 4)

    torch.onnx.export(
        actor_only, dummy_input, output_path,
        input_names=["obs"],
        output_names=["logits"],
        dynamic_axes={"obs": {0: "batch"}},
        opset_version=17,
    )

    # Ensure all weights are inlined (no external .data file)
    import onnx
    model_proto = onnx.load(output_path)
    onnx.save(model_proto, output_path, save_as_external_data=False)

    # Remove stale .data file if it was created
    import os
    data_file = output_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)

    # Verify
    with torch.no_grad():
        torch_logits = actor_only(dummy_input)

    import onnxruntime as ort
    session = ort.InferenceSession(output_path)
    ort_logits = session.run(None, {"obs": dummy_input.numpy()})[0]

    assert np.allclose(torch_logits.numpy(), ort_logits, atol=1e-5), "Output mismatch!"

    size_kb = round(len(open(output_path, "rb").read()) / 1024, 1)
    print(f"Exported to {output_path} ({size_kb} KB)")
    print("Verification passed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_onnx.py <checkpoint_path> [output_path]")
        print("Example: python export_onnx.py checkpoints/step_0099840.pt cartpole.onnx")
        sys.exit(1)

    cp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "cartpole.onnx"
    export(cp, out)
