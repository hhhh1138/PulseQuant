"""Strict in-memory adapter from official Wan2.1 T2V checkpoints to Diffusers."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanPipeline, WanTransformer3DModel
from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
from safetensors.torch import load_file
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel


def bundled_assets_root() -> Path:
    """Use the component configurations shipped with this model-family runner."""
    root = Path(__file__).resolve().parents[3] / "configs" / "wan_assets" / "wan21"
    for component in ("text_encoder", "vae"):
        path = root / component / "config.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing bundled Wan configuration: {path}")
    return root


def _validate(expected, converted, component: str) -> None:
    missing = sorted(set(expected) - set(converted))
    unexpected = sorted(set(converted) - set(expected))
    mismatched = sorted(
        key for key in set(expected) & set(converted) if tuple(expected[key].shape) != tuple(converted[key].shape)
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"{component} conversion mismatch: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}, shape={mismatched[:8]}"
        )


def _transformer_key(key: str) -> str:
    top_level = {
        "head.modulation": "scale_shift_table",
        "head.head.weight": "proj_out.weight",
        "head.head.bias": "proj_out.bias",
        "patch_embedding.weight": "patch_embedding.weight",
        "patch_embedding.bias": "patch_embedding.bias",
        "time_embedding.0.weight": "condition_embedder.time_embedder.linear_1.weight",
        "time_embedding.0.bias": "condition_embedder.time_embedder.linear_1.bias",
        "time_embedding.2.weight": "condition_embedder.time_embedder.linear_2.weight",
        "time_embedding.2.bias": "condition_embedder.time_embedder.linear_2.bias",
        "time_projection.1.weight": "condition_embedder.time_proj.weight",
        "time_projection.1.bias": "condition_embedder.time_proj.bias",
        "text_embedding.0.weight": "condition_embedder.text_embedder.linear_1.weight",
        "text_embedding.0.bias": "condition_embedder.text_embedder.linear_1.bias",
        "text_embedding.2.weight": "condition_embedder.text_embedder.linear_2.weight",
        "text_embedding.2.bias": "condition_embedder.text_embedder.linear_2.bias",
    }
    if key in top_level:
        return top_level[key]
    key = key.replace(".modulation", ".scale_shift_table")
    key = key.replace(".self_attn.", ".attn1.").replace(".cross_attn.", ".attn2.")
    key = key.replace(".attn1.q.", ".attn1.to_q.").replace(".attn2.q.", ".attn2.to_q.")
    key = key.replace(".attn1.k.", ".attn1.to_k.").replace(".attn2.k.", ".attn2.to_k.")
    key = key.replace(".attn1.v.", ".attn1.to_v.").replace(".attn2.v.", ".attn2.to_v.")
    key = key.replace(".attn1.o.", ".attn1.to_out.0.").replace(".attn2.o.", ".attn2.to_out.0.")
    key = key.replace(".norm3.", ".norm2.")
    key = key.replace(".ffn.0.", ".ffn.net.0.proj.").replace(".ffn.2.", ".ffn.net.2.")
    return key


def load_transformer(model_root: Path) -> WanTransformer3DModel:
    source_config = json.loads((model_root / "config.json").read_text())
    with init_empty_weights():
        model = WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=int(source_config.get("num_heads", 12)),
            attention_head_dim=128,
            in_channels=16,
            out_channels=16,
            text_dim=4096,
            freq_dim=256,
            ffn_dim=int(source_config.get("ffn_dim", 8960)),
            num_layers=int(source_config.get("num_layers", 30)),
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            eps=1e-6,
        )
    single = model_root / "diffusion_pytorch_model.safetensors"
    shards = [single] if single.is_file() and single.stat().st_size > 0 else sorted(
        model_root.glob("diffusion_pytorch_model-*-of-*.safetensors")
    )
    if not shards:
        raise FileNotFoundError(f"No transformer safetensors under {model_root}")
    source = {}
    for shard in shards:
        source.update(load_file(str(shard), device="cpu"))
    converted = {_transformer_key(key): value for key, value in source.items()}
    _validate(model.state_dict(), converted, "transformer")
    model.load_state_dict(converted, strict=True, assign=True)
    return model.to(dtype=torch.bfloat16).eval().requires_grad_(False)


def _text_key(key: str) -> str:
    if key == "norm.weight":
        return "encoder.final_layer_norm.weight"
    prefix, layer, suffix = key.split(".", 2)
    if prefix != "blocks":
        raise KeyError(key)
    base = f"encoder.block.{layer}."
    replacements = {
        "norm1.weight": "layer.0.layer_norm.weight",
        "attn.q.weight": "layer.0.SelfAttention.q.weight",
        "attn.k.weight": "layer.0.SelfAttention.k.weight",
        "attn.v.weight": "layer.0.SelfAttention.v.weight",
        "attn.o.weight": "layer.0.SelfAttention.o.weight",
        "pos_embedding.embedding.weight": "layer.0.SelfAttention.relative_attention_bias.weight",
        "norm2.weight": "layer.1.layer_norm.weight",
        "ffn.gate.0.weight": "layer.1.DenseReluDense.wi_0.weight",
        "ffn.fc1.weight": "layer.1.DenseReluDense.wi_1.weight",
        "ffn.fc2.weight": "layer.1.DenseReluDense.wo.weight",
    }
    return base + replacements[suffix]


def load_text_encoder(model_root: Path, assets_root: Path) -> UMT5EncoderModel:
    config_data = json.loads((assets_root / "text_encoder" / "config.json").read_text())
    with init_empty_weights():
        model = UMT5EncoderModel(UMT5Config.from_dict(config_data))
    source = torch.load(
        model_root / "models_t5_umt5-xxl-enc-bf16.pth",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    converted = {_text_key(key): value for key, value in source.items() if key != "token_embedding.weight"}
    token_embedding = source["token_embedding.weight"]
    converted["shared.weight"] = token_embedding
    converted["encoder.embed_tokens.weight"] = token_embedding
    _validate(model.state_dict(), converted, "text_encoder")
    model.load_state_dict(converted, strict=True, assign=True)
    model.tie_weights()
    return model.eval().requires_grad_(False)


def load_vae(model_root: Path, assets_root: Path) -> AutoencoderKLWan:
    config_data = json.loads((assets_root / "vae" / "config.json").read_text())
    config_data = {key: value for key, value in config_data.items() if not key.startswith("_")}
    with init_empty_weights():
        model = AutoencoderKLWan(**config_data)
    source = torch.load(
        model_root / "Wan2.1_VAE.pth",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    converted = convert_wan_vae_to_diffusers(dict(source))
    _validate(model.state_dict(), converted, "vae")
    model.load_state_dict(converted, strict=True, assign=True)
    return model.to(dtype=torch.float32).eval().requires_grad_(False)


def load_local_pipeline(model_root: str) -> WanPipeline:
    model_root_path = Path(model_root)
    assets_root_path = bundled_assets_root()
    tokenizer = AutoTokenizer.from_pretrained(model_root_path / "google" / "umt5-xxl")
    text_encoder = load_text_encoder(model_root_path, assets_root_path)
    transformer = load_transformer(model_root_path)
    vae = load_vae(model_root_path, assets_root_path)
    scheduler = FlowMatchEulerDiscreteScheduler()
    return WanPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
    )
