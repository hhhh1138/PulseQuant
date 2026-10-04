from typing import Tuple
import torch

from model.base import BaseModel
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CausalDiffusion(BaseModel):
    def __init__(self, args, device):
        """
        Initialize the Diffusion loss module.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Step 2: Initialize all hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.guidance_scale = args.guidance_scale
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.teacher_forcing = getattr(args, "teacher_forcing", False)
        # Noise augmentation in teacher forcing, we add small noise to clean context latents
        self.noise_augmentation_max_timestep = getattr(args, "noise_augmentation_max_timestep", 0)
        self.planner_enabled = bool(getattr(args, "planner_enabled", False))
        weights = getattr(args, "planner_auxiliary_weights", {})
        self.planner_auxiliary_weights = {
            "event_coverage": float(getattr(weights, "event_coverage", 1.0)),
            "pairwise_order": float(getattr(weights, "pairwise_order", 0.5)),
            "initial_state": float(getattr(weights, "initial_state", 0.5)),
            "terminal_state_sustained": float(getattr(weights, "terminal_state_sustained", 1.0)),
            "relative_pacing": float(getattr(weights, "relative_pacing", 0.5)),
        }

    def _initialize_models(self, args, device=None):
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        if device is not None:
            self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        noise = torch.randn_like(clean_latent)
        batch_size, num_frame = image_or_video_shape[:2]

        # Step 2: Randomly sample a timestep and add noise to denoiser inputs
        index = self._get_timestep(
            0,
            self.scheduler.num_train_timesteps,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=False
        )
        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)
        noisy_latents = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))
        training_target = self.scheduler.training_target(clean_latent, noise, timestep)

        # Step 3: Noise augmentation, also add small noise to clean context latents
        if self.noise_augmentation_max_timestep > 0:
            index_clean_aug = self._get_timestep(
                0,
                self.noise_augmentation_max_timestep,
                image_or_video_shape[0],
                image_or_video_shape[1],
                self.num_frame_per_block,
                uniform_timestep=False
            )
            timestep_clean_aug = self.scheduler.timesteps[index_clean_aug].to(dtype=self.dtype, device=self.device)
            clean_latent_aug = self.scheduler.add_noise(
                clean_latent.flatten(0, 1),
                noise.flatten(0, 1),
                timestep_clean_aug.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
        else:
            clean_latent_aug = clean_latent
            timestep_clean_aug = None

        # Compute loss
        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=clean_latent_aug if self.teacher_forcing else None,
            aug_t=timestep_clean_aug if self.teacher_forcing else None
        )
        # loss = torch.nn.functional.mse_loss(flow_pred.float(), training_target.float())
        loss = torch.nn.functional.mse_loss(
            flow_pred.float(), training_target.float(), reduction='none'
        ).mean(dim=(2, 3, 4))
        loss = loss * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
        loss = loss.mean()

        auxiliary = {}
        plan_mask = conditional_dict.get("plan_event_mask")
        plan_bins = conditional_dict.get("plan_time_bins")
        if self.planner_enabled and plan_mask is not None and plan_mask.any():
            # Differentiable latent-space supervision. Event centers come from
            # the formal relative schedule; targets remain the clean latent so
            # no external vision model is in the training graph.
            frame_count = x0_pred.shape[1]
            per_frame = (x0_pred.float() - clean_latent.float()).pow(2).mean((2, 3, 4))
            centers = (plan_bins.float() / 32 * (frame_count - 1)).round().long()
            centers = centers.clamp(0, frame_count - 1)
            center_loss = per_frame.gather(1, centers)
            mask_float = plan_mask.float()
            auxiliary["event_coverage"] = ((center_loss * mask_float).sum()
                                            / mask_float.sum().clamp_min(1))
            auxiliary["initial_state"] = per_frame[:, :max(1, frame_count // 10)].mean()
            tail = max(2, frame_count // 8)
            auxiliary["terminal_state_sustained"] = per_frame[:, -tail:].mean()

            predicted_states = []
            target_states = []
            for batch_index in range(x0_pred.shape[0]):
                active = centers[batch_index][plan_mask[batch_index]]
                predicted_states.append(x0_pred[batch_index, active].float().flatten(1))
                target_states.append(clean_latent[batch_index, active].float().flatten(1))
            order_losses, pacing_losses = [], []
            for predicted, target in zip(predicted_states, target_states):
                if len(predicted) < 2:
                    continue
                pred_delta = predicted[1:] - predicted[:-1]
                target_delta = target[1:] - target[:-1]
                order_losses.append(1 - torch.nn.functional.cosine_similarity(
                    pred_delta, target_delta, dim=1).mean())
                pred_speed = pred_delta.norm(dim=1)
                target_speed = target_delta.norm(dim=1)
                pred_speed = pred_speed / pred_speed.sum().clamp_min(1e-6)
                target_speed = target_speed / target_speed.sum().clamp_min(1e-6)
                pacing_losses.append(torch.nn.functional.l1_loss(pred_speed, target_speed))
            zero = loss.new_zeros(())
            auxiliary["pairwise_order"] = torch.stack(order_losses).mean() if order_losses else zero
            auxiliary["relative_pacing"] = torch.stack(pacing_losses).mean() if pacing_losses else zero
            loss = loss + sum(self.planner_auxiliary_weights[name] * value
                              for name, value in auxiliary.items())

        log_dict = {
            "x0": clean_latent.detach(),
            "x0_pred": x0_pred.detach(),
            **{f"loss_{name}": value.detach() for name, value in auxiliary.items()},
        }
        return loss, log_dict
