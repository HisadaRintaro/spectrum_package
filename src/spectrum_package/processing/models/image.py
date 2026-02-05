from dataclasses import dataclass
from typing import Self
from ...util.reader import read_fits
from .header import HeaderRaw
import matplotlib.pyplot as plt
import numpy as np

@dataclass(frozen=True)
class SpectrumBase:
    data: np.ndarray
    error: np.ndarray
    quality: np.ndarray
    wavelength: np.ndarray    

    def __repr__(self) -> str:
        return f"SpectrumBase(data={self.data.shape}, error={self.error.shape}, quality={self.quality.shape}, wavelength={self.wavelength.shape})"

    def plot_spectrum(self,point: int,ax: plt.Axes=None) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.wavelength, self.data[point,:])
        return ax

@dataclass(frozen=True)
class ImageModel:
    header: HeaderRaw
    spectrum: SpectrumBase

    @classmethod
    def load(cls, filename: str) -> Self:
        header, data, error, quality = read_fits(filename)
        header = HeaderRaw.parse_header(header)
        return cls(
            header=header,
            spectrum=SpectrumBase(data=data, error=error, quality=quality, wavelength=header.wavelength),
        )
