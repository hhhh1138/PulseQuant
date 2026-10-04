# Bundled Wan component configurations

Wan 2.1 (1.3B and 14B) and Wan 2.2-A14B loaders automatically select the
corresponding `wan21/` or `wan22/` directory. These files configure conversion
of the original T5 and VAE checkpoint weights; no model weights are bundled.
Both source repositories use the Apache-2.0 license.

The following files were retrieved on 2026-10-04 from pinned Hugging Face revisions:

| File | Source | Revision | SHA-256 |
|---|---|---|---|
| `wan21/text_encoder/config.json` | [Wan-AI/Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/blob/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/text_encoder/config.json) | `38ec498cb3208fb688890f8cc7e94ede2cbd7f68` | `4087b6192155a4643f6d29fd326c4610103130674011ef4d0a53f8bce5de967d` |
| `wan21/vae/config.json` | [Wan-AI/Wan2.1-T2V-14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers/blob/38ec498cb3208fb688890f8cc7e94ede2cbd7f68/vae/config.json) | `38ec498cb3208fb688890f8cc7e94ede2cbd7f68` | `f0c1cc1d7decb5badc384f54691746a27a9aeff49f7ebca974e583389342d527` |
| `wan22/text_encoder/config.json` | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers/blob/5be7df9619b54f4e2667b2755bc6a756675b5cd7/text_encoder/config.json) | `5be7df9619b54f4e2667b2755bc6a756675b5cd7` | `a2bcb24699f6c009a2427432bdd483ef8b2b42a712abc9503759cdc77d171f07` |
| `wan22/vae/config.json` | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers/blob/5be7df9619b54f4e2667b2755bc6a756675b5cd7/vae/config.json) | `5be7df9619b54f4e2667b2755bc6a756675b5cd7` | `47e8bcf55e93e9c182e1962a8c7a0650faeb34ea0f66826d6f8aaa9f73e08ec9` |
