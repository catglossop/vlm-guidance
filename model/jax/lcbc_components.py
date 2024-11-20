import enum
from typing import Dict, Optional, Tuple

# Jax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

# Modules
import model.jax.efficientnet
import model.jax.film 
from model.jax.transformer import Transformer
import model.jax.diffusion
from model.jax.unet import ConditionalUnet1D, unet_squaredcos_cap_v2

# Other
from model.jax.utils import TokenGroup, PRNGKey

default_init = nn.initializers.xavier_uniform

class LCBCEncoder(nn.MKodule):

    """ Vision Encoder module for language conditioning"""
    context_size: int
    observation_shape: Tuple[int, int, int, int]
    language_shape: int
    num_heads: int = 8
    num_layers: int = 4
    dim_factor: int = 4
    num_tokens: int = 8

    @nn.compact
    def __call__(self, obs, goal, language, mask, train : bool = False):
        """ Applies the vision encoder to the input

        Args:
            obs: array of shape (B, S, C, H, W) with the observation.
            goal: array of shape (B, C, H, W) with the goal.
            language: array of shape (B, language_shape) with the language. 
            mask: array of shape (B) with the mask."""
        
        #### Compute the filmed encoding of the observation sequence ####
        bs, seqlen, *_ = obs.shape

        # The efficientnet-b3 model uses 300x300 images.
        efficientnet_config = efficientnet.MODEL_CONFIGS['efficientnet-b3']
        obs = jnp.reshape(obs, [bs * seqlen, 300, 300, 3])
        obs -= jnp.array(efficientnet.MEAN_RGB)
        obs /= jnp.array(efficientnet.STDDEV_RGB)

        # Apply film in EfficientNet.
        x = efficientnet.EfficientNetWithFilm(efficientnet_config)(
            obs, context_input=language, train=train
        )

        # 1x1 conv. This corresponds to the 1x1 conv here:
        # google-research/robotics_transformer/film_efficientnet/pretrained_efficientnet_encoder.py
        var_init = nn.initializers.variance_scaling(
            scale=1.0,
            mode='fan_in',
            distribution='truncated_normal',
        )
        x = nn.Conv(
            features=self.num_features,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='SAME',
            use_bias=False,
            kernel_init=var_init,
        )(x)

        x = film.FilmConditioning(num_channels=self.num_features)(
            x, language
        )

        # End with images tokens over the input context (filmed with language, just images)
        full_tokens = jnp.reshape(x, [bs, seqlen, self.num_tokens, -1]) 
        attn_mask = jnp.broadcast_to(
            jnp.reshape(mask, [bs, 1, 1, 1]), [bs, seqlen, self.num_tokens, 1]
        )
        # Pass encoded tokens through transformer
        output_tokens = Transformer(
            num_layers=self.num_layers,
            layer_size=self.layer_size,
            num_heads=self.num_heads,
            feed_forward_hidden_size=self.feed_forward_hidden_size,
            feed_forward_output_size=self.feed_forward_output_size,
            dropout_rate=self.dropout_rate,
            vocab_size=self.vocab_size,
            ffn_option=self.ffn_option,
        )(full_tokens, attn_mask=attn_mask, train=train)

        return output_tokens

class DenseActionHead(nn.Module):
    """ Dense action head for language conditioning"""
    embedding_dim: int
    control_horizon: int
    activation: nn.activation = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    action_dim: int
    dropout_rate: Optional[float] = None

    def __call__(self, x, train: bool = False):
        hidden_dims = [self.embedding_dim//4, self.embedding_dim//16, self.action_dim*self.control_horizon]
        for i, size in enumerate(hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)

            if i <= len(hidden_dims):
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activation(x)
            if self.activate_final:
                x = nn.sigmoid(x)
        x = jnp.reshape(x, [x.shape[0], self.control_horizon, self.action_dim])
        return x

def masked_mean(x, mask):
    mask = jnp.broadcast_to(mask, x.shape)
    return jnp.mean(x * mask) / jnp.clip(jnp.mean(mask), a_min=1e-5, a_max=None)

def continuous_loss(
    pred_value: ArrayLike,
    ground_truth_value: ArrayLike,
    mask: ArrayLike,
    loss_type: str = "mse",
) -> Array:
    """
    Args:
        pred_value: shape (batch_dims...)
        ground_truth_value: continuous values w/ shape (batch_dims...)
        mask: broadcastable to ground_truth
    """
    if loss_type == "mse":
        loss = jnp.square(pred_value - ground_truth_value)
    elif loss_type == "l1":
        loss = jnp.abs(pred_value - ground_truth_value)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    loss = masked_mean(loss, mask)

    mse = jnp.square(pred_value - ground_truth_value)
    mse = masked_mean(mse, mask)
    return loss, {
        "loss": loss,
        "mse": mse,
    }

class DiffusionActionHead(nn.Module):
    """Predicts actions using a diffusion process and a U-Net architecture (unlike MLP above)

    Only a single pass through the transformer is done to obtain an action embedding at each timestep. The
    actions are then predicted using a diffusion process conditioned on this embedding. The diffusion model
    architecture is an 1D unet based on the implementation from Chi et al: https://arxiv.org/abs/2303.04137

    You may create an embedding by either mean-pooling across tokens (use_map=False) or using multi-head
    attention pooling (use_map=True). It is recommended to use MAP when decoding from the observation token
    stream.
    """

    readout_key: str
    action_dim: int
    action_horizon: int

    use_map: bool = (False,)
    flatten_tokens: bool = (False,)
    timesteps: int = 100
    max_action: float = 1.0
    clip_sample: Optional[float] = None
    variance_type: str = "fixed_large"

    def setup(self):
        self.action_proj = nn.Dense(self.action_dim)
        betas = unet_squaredcos_cap_v2(self.timesteps).astype(jnp.float32)
        self.alphas = 1.0 - betas  # So betas = 1 - alphas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)

        self.model = ConditionalUnet1D(
            down_features=(256, 512, 1024),
            mid_layers=2,
            time_features=128,
            kernel_size=5,
        )

    def __call__(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        time: Optional[ArrayLike] = None,
        noisy_actions: Optional[ArrayLike] = None,
        train: bool = True,
    ) -> jax.Array:
        """Performs a single forward pass through the diffusion model."""
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )

        if self.use_map:  # Multi-head attention pooling
            assert not self.flatten_tokens, "Cannot use MAP token and flattening!"
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        elif self.flatten_tokens:  # concatenate tokens in final dim
            embeddings = token_group.tokens.reshape((*token_group.tokens.shape[:2], -1))
        else:  # mean pooling
            embeddings = token_group.tokens.mean(axis=-2)
        # Now, embeddings is (batch_size, window_size, embedding_size)

        # time and noisy_actions are None during initialization, so we replace them with a dummy array
        if (time is None or noisy_actions is None) and not self.is_initializing():
            raise ValueError(
                "Must provide time and noisy_actions when calling diffusion action head"
            )
        elif self.is_initializing():
            time = jnp.zeros((*embeddings.shape[:2], 1), dtype=jnp.float32)
            noisy_actions = jnp.zeros(
                (*embeddings.shape[:2], self.action_horizon, self.action_dim),
                dtype=jnp.float32,
            )  # (b, w, p, a)
        pred_eps = self.model(embeddings, action=noisy_actions, time=time, train=train)
        pred_eps = self.action_proj(pred_eps)
        return pred_eps

    def loss(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        actions: ArrayLike,
        action_pad_mask: ArrayLike,
        timestep_pad_mask: ArrayLike,
        train: bool = True,
    ) -> Tuple[Array, Dict[str, Array]]:
        """Computes the loss for the diffusion objective.

        Args:
            transformer_ouputs: must contain self.readout_key with shape (batch_size, window_size, num_tokens,
                embedding_size)
            actions: shape (batch_size, >= window_size + action_horizon - 1, action_dim)
            action_pad_mask: boolean array (same shape as actions) which is True if the action dimension is not a padding dimension
            timestep_pad_mask: boolean array (batch, window_size) which is True if the timestep is not a padding timestep

        Returns:
            loss: float
            metrics: dict
        """
        batch_size, window_size = timestep_pad_mask.shape[:2]

        actions = jnp.clip(actions, -self.max_action, self.max_action)

        # piggy-back on the dropout rng chain for diffusion rng
        rng = self.make_rng("dropout")
        time_key, noise_key = jax.random.split(rng)
        time = jax.random.randint(
            time_key,
            (batch_size, window_size, 1),
            0,
            self.timesteps,
        )
        noise = jax.random.normal(noise_key, actions.shape)

        # Add noise to the action according to the schedule
        sqrt_alpha_prod = jnp.sqrt(self.alphas_cumprod[time[:, None]])  # (B, 1, 1)
        sqrt_one_minus_alpha_prod = jnp.sqrt(
            1 - self.alphas_cumprod[time[:, None]]
        )  # (B, 1, 1)
        noisy_actions = sqrt_alpha_prod * actions + sqrt_one_minus_alpha_prod * noise

        pred_eps = self(
            transformer_outputs, train=train, time=time, noisy_actions=noisy_actions
        )

        # combine the timestep-level pad mask with the action-dimension-level pad mask
        mask = (
            jnp.broadcast_to(action_pad_mask[:, None, None, :], actions.shape)
            * timestep_pad_mask
        )

        loss, metrics = continuous_loss(pred_eps, noise, mask, loss_type="mse")
        # Sum over action dimension instead of averaging
        loss = loss * self.action_dim
        metrics["loss"] = metrics["loss"] * self.action_dim
        metrics["mse"] = metrics["mse"] * self.action_dim
        return loss, metrics

    def predict_action(
        self,
        transformer_outputs: Dict[str, TokenGroup],
        rng: PRNGKey,
        train: bool = True,
        embodiment_action_dim: Optional[int] = None,
        *args,
        **kwargs,
    ) -> jax.Array:
        """
        Code inspired by diffusers:
        https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm_flax.py
        """
        batch_size, window_size = transformer_outputs[self.readout_key].tokens.shape[:2]
        module, variables = self.unbind()

        action_mask = jnp.ones(
            (
                batch_size,
                window_size,
                self.action_horizon,
                self.action_dim,
            ),
            dtype=bool,
        )

        if embodiment_action_dim is not None:
            action_mask = action_mask.at[..., embodiment_action_dim:].set(False)
        else:
            print(
                "embodiment_action_dim is highly recommended for diffusion action head"
                " if any action dimensions were masked during training"
            )

        def loop_body(i, args):
            sample, rng = args
            time = self.timesteps - 1 - i
            # Note that here time is (B, 1, 1) where as in loss in is (B, 1)
            time = jnp.broadcast_to(time, (sample.shape[0], 1, 1))
            alpha = self.alphas[time]
            alpha_prod_t = self.alphas_cumprod[time]
            alpha_prod_t_prev = jnp.where(
                time > 0,
                self.alphas_cumprod[time - 1],
                jnp.array(1.0, dtype=jnp.float32),
            )

            # Run the model. Reduce time to (B, 1) for the model.
            eps = module.apply(
                variables,
                transformer_outputs,
                time=time,
                noisy_actions=sample,
                train=train,
            )

            # Predict x_0, clip if desired.
            orig = (sample - jnp.sqrt(1 - alpha_prod_t) * eps) / jnp.sqrt(alpha_prod_t)
            if self.clip_sample is not None:
                orig = jnp.clip(orig, -self.clip_sample, self.clip_sample)

            # Compute x_{t-1} using x_0
            orig_coeff = jnp.sqrt(alpha_prod_t_prev) * (1 - alpha) / (1 - alpha_prod_t)
            current_coeff = (
                jnp.sqrt(alpha) * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
            )

            prev = orig_coeff * orig + current_coeff * sample

            # Add noise according to the schedule
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha)
            if self.variance_type == "fixed_large":
                variance = 1 - alpha
            elif self.variance_type == "fixed_small":
                variance = jnp.clip(variance, a_min=1e-20)
            else:
                raise ValueError("Invalid schedule provided")

            rng, key = jax.random.split(rng)
            variance = jnp.where(
                time > 0, variance, jnp.zeros(eps.shape, dtype=jnp.float32)
            )
            z = jax.random.normal(key, shape=sample.shape, dtype=jnp.float32)
            prev = prev + jnp.sqrt(variance) * z

            # set non-eval actions to the noise that would have been seen during training
            prev = jnp.where(action_mask, prev, jnp.sqrt(1 - alpha_prod_t) * z)

            return (prev, rng)

        rng, key = jax.random.split(rng)
        noisy_action = jax.random.normal(
            key,
            (
                batch_size,
                window_size,
                self.action_horizon,
                self.action_dim,
            ),
        )

        noisy_action, _ = jax.lax.fori_loop(
            0, self.timesteps, loop_body, (noisy_action, rng)
        )

        return noisy_action

    






        