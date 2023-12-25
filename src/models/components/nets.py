import torch
import torch.nn as nn

from src.models.components.decoder.reconstruction import compute_reconstruction


def spectrogram_hook(model, input, output):
    return output.transpose(-1, -2)

# https://github.com/qmpzzpmq/AMINO/blob/main/examples/audioset/conf/HF_enc_classifier.yaml
class TimerSeriesClassifier(nn.Module):
    def __init__(self, encoder, classifier, buncher):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.buncher = buncher

    def forward(self, x):
        # x: Batch, Time, Features
        x = self.encoder(x)
        # x: Batch, Time, Features

        x = self.classifier(x)
        # x: Batch, Time, Classify
        x = self.buncher(x)
        # x: Batch, Classify
        return x


class SingleEncMultiDec(nn.Module):
    def __init__(self, spectrogram, encoder, decoders) -> None:
        super().__init__()
        spectrogram.register_forward_hook(spectrogram_hook)
        self.spectrogram = spectrogram
        self.encoder = encoder
        self.decoders = nn.ModuleDict(decoders)

    def forward(self, x):
        # x: Batch, Time, Features
        x = self.spectrogram(x)
        h = self.encoder(x)
        output = {
            "features": x,
            "encoder_features": h, # for cpc
        }
        for name, decoder in self.decoders.items():
            output[name] = decoder(h)
        return output

def classify_decoder_post(decoder_dict):
    output_dict = dict()
    merged = torch.cat(
        [
            decoder_dict["encoder_features"],
            decoder_dict["classify"]["classify"],
        ],
        dim=-1,
    )
    for key, value in decoder_dict.items():
        if key == "classify":
            output_dict[key] = value["bunch"]
        else:
            output_dict[key] = value
    return merged, output_dict


class SingleEncMultiDecWithGMM(SingleEncMultiDec):
    def __init__(self, spectrogram, encoder, decoders, gmm, decoder_post) -> None:
        super().__init__(spectrogram, encoder, decoders)
        self.gmm = gmm
        self.decoder_post = decoder_post

    def forward(self, x):
        decoder_dict = super().forward(x)
        gmm_input, decoder_dict = self.decoder_post(decoder_dict)
        decoder_dict["gmm_input"] = gmm_input
        decoder_dict["gmm_output"] = self.gmm(gmm_input)
        return decoder_dict


if __name__ == "__main__":
    from functools import partial

    from torchaudio.transforms import MelSpectrogram

    from src.models.components.encoder.vit import AudioDeiTModel

    fs = 16000
    duration = 7
    batch_size = 5

    hidden_size = 192
    classify_size = 12

    encoder = AudioDeiTModel()
    classifier = nn.Linear(hidden_size, classify_size)
    buncher = partial(torch.mean, dim=-2)

    spectral_tranform = MelSpectrogram()
    model = TimerSeriesClassifier(encoder, classifier, buncher)
    audio1 = torch.randn(1, duration * fs)
    audio2 = torch.randn(5, 1, duration * fs)

    audio = torch.randn(1, duration * fs).unsqueeze(1)
    spectrogram = spectral_tranform(audio)
    output = model(spectrogram)
    print(output.shape)

    audio = torch.randn(batch_size, 1, duration * fs)
    spectrogram = spectral_tranform(audio)
    output = model(spectrogram)
    print(output.shape)
