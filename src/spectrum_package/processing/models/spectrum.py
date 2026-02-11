"""スペクトルデータモデル.

STIS FITS ファイルから読み取った2次元スペクトルデータ
（科学データ・統計的誤差・品質フラグ）を保持する。
"""

from dataclasses import dataclass
from typing import Self
from pathlib import Path

import numpy as np

from ...util.reader import read_image


@dataclass(frozen=True)
class SpectrumBase:
    """2次元スペクトルデータの基底モデル.

    STIS の FITS ファイルから読み取った科学データ（HDU 1）、
    統計的誤差（HDU 2）、品質フラグ（HDU 3）を保持する。
    データ配列の形状は (波長ピクセル数, 空間ピクセル数) である。

    Attributes
    ----------
    data : np.ndarray
        科学データ配列（2D: [波長, 空間位置]）
    error : np.ndarray
        統計的誤差配列（2D: data と同形状）
    quality : np.ndarray
        品質フラグ配列（2D: data と同形状）
    """

    data: np.ndarray
    error: np.ndarray
    quality: np.ndarray

    def __repr__(self) -> str:
        return f"SpectrumBase(data={self.data.shape}, error={self.error.shape}, quality={self.quality.shape})"

    @classmethod
    def load(cls, filename: Path) -> Self:
        """FITS ファイルからスペクトルデータをロードする.

        Parameters
        ----------
        filename : Path
            FITS ファイルのパス

        Returns
        -------
        SpectrumBase
            ロードされたスペクトルデータ
        """
        data, error, quality = read_image(filename)
        return cls(data=data, error=error, quality=quality)
