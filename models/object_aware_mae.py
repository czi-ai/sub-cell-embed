import collections
import math
from copy import deepcopy
from typing import Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm
from transformers import ViTMAEConfig
from transformers.models.vit_mae.modeling_vit_mae import (
    _CONFIG_FOR_DOC,
    VIT_MAE_INPUTS_DOCSTRING,
    BaseModelOutput,
    ViTMAEAttention,
    ViTMAEDecoderOutput,
    ViTMAEIntermediate,
    ViTMAEModelOutput,
    ViTMAEOutput,
    ViTMAEPatchEmbeddings,
    ViTMAEPreTrainedModel,
    ViTMAESdpaAttention,
    get_2d_sincos_pos_embed,
)
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

# class LayerNorm(nn.LayerNorm):
#     def forward(self, x):
#         t = x.dtype
#         x = super().forward(x.type(torch.float32))
#         return x.type(t)


class ViTMAEMaskAwareConfig(ViTMAEConfig):
    def __init__(self, object_mask_ratio: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.object_mask_ratio = object_mask_ratio


class ViTMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = ViTMAEPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        # fixed sin-cos embedding
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.hidden_size),
            requires_grad=False,
        )
        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1],
            int(self.patch_embeddings.num_patches**0.5),
            add_cls_token=True,
        )
        self.position_embeddings.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

    def random_sampling(self, sequence, noise, batch_size, seq_length, len_keep):
        if noise is None:
            noise = torch.rand(
                batch_size, seq_length, device=sequence.device
            )  # noise in [0, 1]

            # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        return ids_restore, ids_keep

    def object_aware_sampling(
        self, sequence, object_mask, batch_size, seq_length, len_keep
    ):
        # Calculate the number of masked tokens to keep based on the object mask ratio
        mask_len = object_mask.sum(dim=1).long()
        mask_len_keep = torch.clamp(
            mask_len * (1 - self.config.object_mask_ratio), max=len_keep
        ).long()

        non_mask_len = (1 - object_mask).sum(dim=1).long()
        non_mask_len_keep = torch.clamp(len_keep - mask_len_keep, max=non_mask_len)
        actual_mask_len_keep = len_keep - non_mask_len_keep

        assert (
            torch.eq(actual_mask_len_keep + non_mask_len_keep, len_keep).all() == True
        ), "Mask length mismatch"

        ids_shuffle = []
        ids_keep = []
        for i in range(batch_size):
            object_mask_ids = object_mask[i].nonzero().squeeze()
            object_mask_ids_shuffle = object_mask_ids[
                torch.randperm(object_mask_ids.size(0))
            ]
            object_mask_ids_keep = object_mask_ids_shuffle[: actual_mask_len_keep[i]]
            rest_object_ids = object_mask_ids_shuffle[actual_mask_len_keep[i] :]

            non_object_ids = torch.arange(seq_length, device=sequence.device)[
                ~object_mask[i].bool()
            ]
            non_object_ids_shuffle = non_object_ids[
                torch.randperm(non_object_ids.size(0))
            ]
            non_object_ids_keep = non_object_ids_shuffle[: non_mask_len_keep[i]]
            rest_non_object_ids = non_object_ids_shuffle[non_mask_len_keep[i] :]

            sample_ids_keep = torch.cat([object_mask_ids_keep, non_object_ids_keep])
            assert sample_ids_keep.size(0) == len_keep, "Sample ids keep size mismatch"
            ids_keep.append(sample_ids_keep)

            sample_ids_rest = torch.cat([rest_object_ids, rest_non_object_ids])
            sample_all_ids = torch.cat([sample_ids_keep, sample_ids_rest])
            ids_shuffle.append(sample_all_ids)

        ids_shuffle = torch.stack(ids_shuffle)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = torch.stack(ids_keep)
        return ids_restore, ids_keep

    def random_masking(self, sequence, mask_ratio=None, noise=None, object_mask=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        mask_ratio = mask_ratio if mask_ratio is not None else self.config.mask_ratio
        len_keep = int(seq_length * (1 - mask_ratio))

        if object_mask is not None and self.config.object_mask_ratio > 0:
            ids_restore, ids_keep = self.object_aware_sampling(
                sequence, object_mask, batch_size, seq_length, len_keep
            )
        else:
            ids_restore, ids_keep = self.random_sampling(
                sequence, noise, batch_size, seq_length, len_keep
            )

        sequence_unmasked = torch.gather(
            sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim)
        )

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def forward(self, pixel_values, mask_ratio=None, noise=None, object_mask=None):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        # add position embeddings w/o cls token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]

        # patchify mask for object masking
        if object_mask is not None and self.config.object_mask_ratio > 0:
            w_f, h_f = (
                width // self.patch_embeddings.patch_size[0],
                height // self.patch_embeddings.patch_size[0],
            )
            object_mask = F.interpolate(
                object_mask, size=(w_f, h_f), mode="nearest"
            ).reshape(batch_size, w_f * h_f)

        # masking: length -> length * config.mask_ratio
        embeddings, mask, ids_restore = self.random_masking(
            embeddings, mask_ratio, noise, object_mask
        )

        # append cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore


VITMAE_ATTENTION_CLASSES = {
    "eager": ViTMAEAttention,
    "sdpa": ViTMAESdpaAttention,
}


class ViTMAELayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = VITMAE_ATTENTION_CLASSES[config._attn_implementation](config)
        self.intermediate = ViTMAEIntermediate(config)
        self.output = ViTMAEOutput(config)
        self.layernorm_before = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(
                hidden_states
            ),  # in ViTMAE, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViTMAE, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class ViTMAEEncoder(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [ViTMAELayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, layer_head_mask, output_attentions
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ViTMAEModel(ViTMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTMAEEmbeddings(config)
        self.encoder = ViTMAEEncoder(config)

        self.layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(VIT_MAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=ViTMAEModelOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        mask_ratio: Optional[float] = None,
        object_mask: Optional[float] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ViTMAEModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ViTMAEModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        >>> model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, mask, ids_restore = self.embeddings(
            pixel_values,
            mask_ratio=mask_ratio,
            noise=noise,
            object_mask=object_mask,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            return (sequence_output, mask, ids_restore) + encoder_outputs[1:]

        return ViTMAEModelOutput(
            last_hidden_state=sequence_output,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ViTMAEDecoder(nn.Module):
    def __init__(self, config, num_patches):
        super().__init__()
        self.decoder_embed = nn.Linear(
            config.hidden_size, config.decoder_hidden_size, bias=True
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size),
            requires_grad=False,
        )  # fixed sin-cos embedding

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [
                ViTMAELayer(decoder_config)
                for _ in range(config.decoder_num_hidden_layers)
            ]
        )

        self.decoder_norm = LayerNorm(
            config.decoder_hidden_size, eps=config.layer_norm_eps
        )
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size,
            config.patch_size**2 * config.num_channels,
            bias=True,
        )  # encoder to decoder
        self.gradient_checkpointing = False
        self.config = config
        self.initialize_weights(num_patches)

    def initialize_weights(self, num_patches):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(num_patches**0.5), add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

    def forward(
        self,
        hidden_states,
        ids_restore,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # embed tokens
        x = self.decoder_embed(hidden_states)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # unshuffle
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]).to(x_.device),
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        hidden_states = x + self.decoder_pos_embed

        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    None,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, head_mask=None, output_attentions=output_attentions
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)

        # remove cls token
        logits = logits[:, 1:, :]

        if not return_dict:
            return tuple(
                v
                for v in [logits, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return ViTMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
