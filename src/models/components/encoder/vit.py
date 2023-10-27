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


def deit_patch_embedding_model_convert(ori_patch_embedding_model):
    # convert the deit embedding model to
    # the audio deit embedding model
    # the input of the audio deit embedding model is just the spectrogram with 1 channel bactchx1xtimexfeature
    # the input of the original deit embedding model is the image with 3 channel batchx3x224x224.
    # the output of the audio deit embedding model is the same as the original deit embedding model, batchx768.
    patch_embedding_model = AusdioDeiTPatchEmbeddings(
        1,
        ori_patch_embedding_model.projection.out_channels,
        ori_patch_embedding_model.patch_size,
    )
    return patch_embedding_model


class AudioDeiTEmbeddings(nn.Module):
    def __init__(self, patch_embedding_model, position_embedding_model, dropout):
        super().__init__()
        self.patch_embeddings = patch_embedding_model
        self.position_embeddings = position_embedding_model
        self.dropout = dropout

    def forward(self, x):
        embeddings = self.patch_embeddings(x)
        embeddings += self.position_embeddings(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    @classmethod
    def from_DeiTEmbeddings(cls, ori_embeddings):
        return cls(
            patch_embedding_model=deit_patch_embedding_model_convert(
                ori_embeddings.patch_embeddings
            ),
            position_embedding_model=PositionalEncoding(
                ori_embeddings.position_embeddings.size(-1), 
                ori_embeddings.patch_embeddings.num_patches * 10,
            ),
            dropout=ori_embeddings.dropout,
        )


class AusdioDeiTPatchEmbeddings(nn.Module):
    def __init__(self, num_channels, hidden_size, patch_size):
        super().__init__()
        self.projection = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x


class AudioDeiTModel(nn.Module):
    def __init__(
        self,
        model_name="facebook/deit-tiny-distilled-patch16-224",
        num_classes=1000,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=1024,
    ):
        super().__init__()
        assert (
            model_name in DEIT_MODEL_NAMES
        ), f"model_name should be in {DEIT_MODEL_NAMES}, now it is {model_name}"
        model = DeiTModel.from_pretrained(model_name)
        self.embeddings = AudioDeiTEmbeddings.from_DeiTEmbeddings(model.embeddings)
        self.encoder = model.encoder
        self.layernorm = model.layernorm
        self.pooler = model.pooler
        # self.model.embeddings.patch_embeddings = deit_patch_embedding_model_convert(
        #     self.model.embeddings.patch_embeddings
        # )
        self.model = model

    def forward(
            self,
            spectral,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
        # spectral: batchx1x128x1024
        # output: batchx768
        embedding_output = self.embeddings(spectral)
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        return sequence_output, pooled_output


if __name__ == "__main__":
    import torchaudio

    audio = torch.randn(1, 16000)
    fs = 16000

    speectral_tranform = torchaudio.transforms.MelSpectrogram()
    spectrogram = speectral_tranform(audio)
    model = AudioDeiTModel()
    output = model(spectrogram.unsqueeze(1))
    for each in output:
        print(each.shape)
