"""FITS ヘッダー解析モデル.

STIS FITS ファイルのヘッダー情報を構造化されたデータクラスとして提供する。
Primary HDU（HDU 0）と Spectrogram HDU（HDU 1）のメタデータを
それぞれ専用クラスで保持し、`HeaderProfile` で集約する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, cast, Any, TYPE_CHECKING
from pathlib import Path

from astropy.io import fits  # type: ignore
from astropy.wcs import WCS  # type: ignore
import numpy as np

if TYPE_CHECKING:
    from ..util.fits_reader import STISFitsReader


@dataclass(frozen=True)
class HeaderSpectrogram:
    """スペクトログラム HDU（HDU 1）のヘッダー情報.

    WCS 情報を用いて波長配列および空間座標配列への変換機能を提供する。

    Attributes
    ----------
    rootname : str
        観測のルート名（ROOTNAME キーワード）
    optical_element : str
        使用された光学素子（OPT_ELEM キーワード、例: G430L）
    wcs : WCS
        World Coordinate System オブジェクト
    shape : tuple[int, int]
        データ配列の形状 (NAXIS1, NAXIS2)
    """

    rootname: str
    optical_element: str
    wcs: WCS
    shape: tuple[int, int]

    @classmethod
    def parse_header(cls, header: fits.Header) -> Self:
        """FITS ヘッダーからスペクトログラム情報を解析する.

        Parameters
        ----------
        header : fits.Header
            HDU 1 の FITS ヘッダー

        Returns
        -------
        HeaderSpectrogram
            解析されたスペクトログラムヘッダー情報
        """
        h = cast(Any, header)
        return cls(
            rootname=h.get("ROOTNAME", ""),
            optical_element=cast(str, h.get("OPT_ELEM", "")),
            wcs=WCS(header),
            shape=(h.get("NAXIS1", 0), h.get("NAXIS2", 0)),
        )

    @property
    def wavelength_array(self) -> np.ndarray:
        """WCS から波長配列を計算する.

        WCS の spectral サブシステムを使用して、ピクセルインデックスから
        波長値 [m] への変換を行う。

        Returns
        -------
        np.ndarray
            波長配列 [m]

        Raises
        ------
        ValueError
            WCS の CTYPE に WAVE 軸が見つからない場合
        """
        if self.wcs.wcs.ctype[0] == "WAVE":  # type: ignore
            nx = self.shape[1]
        elif self.wcs.wcs.ctype[1] == "WAVE":  # type: ignore
            nx = self.shape[0]
        else:
            raise ValueError(f"CTYPE is not WAVE {self.wcs.wcs.ctype}")  # type: ignore
        pixel_indices = np.arange(nx)
        spec_wcs = self.wcs.sub(["spectral"])  # type: ignore
        return spec_wcs.pixel_to_world_values(pixel_indices)  # type: ignore

    @property
    def spatial_array(self) -> np.ndarray:
        """空間方向のピクセルインデックス配列を返す.

        空間軸のピクセル数に基づいた 0 始まりのインデックス配列を返す。
        実際の物理座標（arcsec 等）への変換はスリット間隔の情報と
        組み合わせて行う。

        Returns
        -------
        np.ndarray
            空間方向のピクセルインデックス配列 (0, 1, 2, ...)
        """
        if self.wcs.wcs.ctype[0] == "WAVE":  # type: ignore
            n_spatial = self.shape[0]
        else:
            n_spatial = self.shape[1]
        return np.arange(n_spatial)


@dataclass(frozen=True)
class HeaderPrimary:
    """Primary HDU（HDU 0）のヘッダー情報.

    観測の基本メタデータを保持する。

    Attributes
    ----------
    filename : str
        ファイル名（FILENAME キーワード）
    optical_element : str
        使用された光学素子（OPT_ELEM キーワード）
    bandwidth : float
        バンド幅 [m]（BANDWID キーワード）
    """

    filename: str
    optical_element: str
    bandwidth: float

    @classmethod
    def parse_header(cls, header: fits.Header) -> Self:
        """FITS ヘッダーから Primary 情報を解析する.

        Parameters
        ----------
        header : fits.Header
            HDU 0 の FITS ヘッダー

        Returns
        -------
        HeaderPrimary
            解析された Primary ヘッダー情報
        """
        h = cast(Any, header)
        return cls(
            filename=h.get("FILENAME", ""),
            optical_element=cast(str, h.get("OPT_ELEM", "")),
            bandwidth=cast(float, h.get("BANDWID", np.nan)),
        )


@dataclass(frozen=True)
class HeaderProfile:
    """FITS ヘッダー情報の集約モデル.

    Primary HDU と Spectrogram HDU のヘッダー情報をまとめて保持する。

    Attributes
    ----------
    primary : HeaderPrimary
        Primary HDU のヘッダー情報
    spectrogram : HeaderSpectrogram
        Spectrogram HDU のヘッダー情報
    """

    primary: HeaderPrimary
    spectrogram: HeaderSpectrogram

    def __repr__(self) -> str:
        return f"HeaderProfile(primary={self.primary.__class__.__name__}, spectrogram={self.spectrogram.__class__.__name__})"

    @classmethod
    def from_reader(cls, reader: STISFitsReader) -> Self:
        """STISFitsReader からヘッダー情報を生成する.

        Parameters
        ----------
        reader : STISFitsReader
            読み込み済みの Reader インスタンス

        Returns
        -------
        HeaderProfile
            ロードされたヘッダー情報
        """
        return cls(
            primary=HeaderPrimary.parse_header(reader.header(0)),
            spectrogram=HeaderSpectrogram.parse_header(reader.header(1)),
        )