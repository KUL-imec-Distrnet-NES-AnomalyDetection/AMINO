from typing import Optional

import torch
import torch.nn as nn
from transformers import DeiTModel

from src.models.components.layer.embedding import PositionalEncoding

DEIT_MODEL_NAMES = [
    "facebook/deit-tiny-distilled-patch16-224",
    "facebook/deit-small-distilled-patch16-224",
    "facebook/deit-base-distilled-patch16-224",
    "facebook/deit-base-distilled-patch16-384",
    "facebook/deit-tiny-patch16-224" "facebook/deit-small-patch16-224",
    "facebook/deit-base-patch16-384",
]


class AudioDeiTEmbeddings(nn.Module):
    def __init__(self, patch_embedding, position_embedding, dropout):
        super().__init__()
        self.patch_embedding = patch_embedding
        self.position_embedding = position_embedding
        self.dropout = dropout

    def forward(self, x):
        # x: Batch, 1, Time, Features

        # Batch, 1, Time, Features
        # -> Batch, Channel, Time, Features
        # -> Batch, Channel, Patch
        # -> Batch, Patch, Channel
        embeddings = self.patch_embedding(x)
        num_patch_in_time = embeddings.size(-2)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        position_embeddings = self.position_embedding(embeddings.size(1))
        embeddings = self.dropout(embeddings + position_embeddings)
        return embeddings, num_patch_in_time

    @classmethod
    def from_DeiTEmbeddings(cls, ori_embeddings):
        patch_embedding = nn.Conv2d(
            1,
            ori_embeddings.patch_embeddings.projection.out_channels,
            kernel_size=ori_embeddings.patch_embeddings.patch_size,
            stride=ori_embeddings.patch_embeddings.patch_size,
        )
        position_embedding = PositionalEncoding(
            ori_embeddings.patch_embeddings.projection.out_channels,
            ori_embeddings.patch_embeddings.num_patches * 10,
        )
        return cls(
            patch_embedding=patch_embedding,
            position_embedding=position_embedding,
            dropout=ori_embeddings.dropout,
        )


class AudioDeiT(nn.Module):
    def __init__(
        self,
        model_name="facebook/deit-tiny-distilled-patch16-224",
    ):
        super().__init__()
        assert (
            model_name in DEIT_MODEL_NAMES
        ), f"model_name should be in {DEIT_MODEL_NAMES}, now it is {model_name}"
        model = DeiTModel.from_pretrained(model_name)
        self.embeddings = AudioDeiTEmbeddings.from_DeiTEmbeddings(model.embeddings)
        self.encoder = model.encoder
        self.layernorm = model.layernorm

    def forward(
        self,
        spectral,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        embedding_output, num_patch_in_time = self.embeddings(spectral)
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        return (
            sequence_output.unflatten(-2, (num_patch_in_time, -1))
            .flatten(2)
            .unsqueeze(1)
        )


class WavAudioDeiT(AudioDeiT):
    def __init__(
        self,
        spectrogram,
        model_name="facebook/deit-tiny-distilled-patch16-224",
    ):
        super().__init__(model_name)
        self.spectrogram = spectrogram

    def forward(self, audio, *args, **kwargs):
        spectral = self.spectrogram(audio)
        sequence_output = super().forward(spectral, *args, **kwargs)
        return sequence_output


if __name__ == "__main__":
    from torchaudio.transforms import MelSpectrogram

    from src.models.components.nets import spectrogram_hook

    fs = 16000
    duration = 5
    batch_size = 5
    spectral_tranform = MelSpectrogram()
    spectral_tranform.register_forward_hook(spectrogram_hook)
    model = AudioDeiT()

    audio = torch.randn(1, duration * fs).unsqueeze(1)
    spectrogram = spectral_tranform(audio)
    sequence_output = model(spectrogram)
    print(sequence_output.shape)

    audio = torch.randn(batch_size, 1, duration * fs)
    spectrogram = spectral_tranform(audio)
    sequence_output = model(spectrogram)
    print(sequence_output.shape)
