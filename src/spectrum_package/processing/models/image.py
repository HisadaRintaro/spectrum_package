"""スペクトル画像モデル.

FITS ファイルのヘッダー情報とスペクトルデータを統合し、
スペクトルの描画などの操作を提供する。
"""

from dataclasses import dataclass
from typing import Self
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from .header import HeaderProfile
from .spectrum import SpectrumBase
from ...util.constants import ANGSTROM_TO_METER
from ...util.fits_reader import STISFitsReader


@dataclass(frozen=True)
class ImageModel:
    """スペクトル画像の統合モデル.

    ヘッダー情報（WCS、観測メタデータ）とスペクトルデータ（科学データ、
    誤差、品質フラグ）を一つのモデルとして保持する。

    Attributes
    ----------
    header : HeaderProfile
        FITS ヘッダー情報（Primary + Spectrogram）
    spectrum : SpectrumBase
        2次元スペクトルデータ
    """

    header: HeaderProfile
    spectrum: SpectrumBase

    def __repr__(self) -> str:
        return f"ImageModel( \n header={self.header}, \n spectrum={self.spectrum} \n )"

    @classmethod
    def load(cls, filename: Path) -> Self:
        """FITS ファイルからスペクトル画像モデルをロードする.

        内部で STISFitsReader を使い、ファイルを1回だけ開いて
        ヘッダーとデータを同時に読み込む。

        Parameters
        ----------
        filename : Path
            FITS ファイルのパス

        Returns
        -------
        ImageModel
            ロードされたスペクトル画像モデル
        """
        reader = STISFitsReader.open(filename)
        return cls(
            header=HeaderProfile.from_reader(reader),
            spectrum=SpectrumBase.from_reader(reader),
        )

    def plot_spectrum(
        self,
        point: int,
        center_wave: float = 5007,
        width: float = 10,
        ax: Axes | None = None,
    ) -> Axes:
        """指定した空間位置のスペクトルを描画する.

        Parameters
        ----------
        point : int
            空間方向のピクセルインデックス（data[:, point] を描画）
        center_wave : float, optional
            表示中心波長 [Å]（デフォルト: 5007）
        width : float, optional
            表示波長幅の半値 [Å]（デフォルト: 10）。
            表示範囲は center_wave ± width となる。
        ax : Axes or None, optional
            描画先の matplotlib Axes。None の場合は新規作成。

        Returns
        -------
        Axes
            スペクトルが描画された Axes オブジェクト
        """
        wave = self.header.spectrogram.wavelength_array / ANGSTROM_TO_METER
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(wave, self.spectrum.data[:, point])
        ax.set_xlim(center_wave - width, center_wave + width)
        ax.set_xlabel(r"Wavelength [$\AA$]")
        ax.set_ylabel("Counts")
        return ax