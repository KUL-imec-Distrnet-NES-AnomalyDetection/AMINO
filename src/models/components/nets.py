import torch
import torch.nn as nn


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
    def __init__(self, encoder, decoders) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleDict(decoders)

    def forward(self, x):
        # x: Batch, Time, Features
        x = self.encoder(x)
        # x: Batch, Time, Features
        output = dict()
        for name, decoder in self.decoders.items():
            output[name] = decoder(x)
        return output


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
