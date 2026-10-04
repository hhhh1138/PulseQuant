# Self-Forcing note

The audited Self-Forcing implementation calibrates its PulseQuant wrappers and
then generates videos in the same Python process. The calibration section is in
`runtime/self_forcing/self_forcing_quant_runner_walsh.py` immediately before
the prompt-generation loop. Unlike the Wan and H3 runners, this version has no
stable cache export/import interface; therefore the launcher is intentionally
named `calibrate_and_infer.sh`.

## Additional dependencies

Use a separate Python 3.10 environment with PyTorch and the runtime dependencies.
Install OmegaConf in that environment for the Self Forcing YAML configurations:

```bash
python -m pip install omegaconf
```

Install FlashAttention after PyTorch:

```bash
python -m pip install flash-attn --no-build-isolation
```
