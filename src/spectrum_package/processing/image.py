"""スペクトル画像モデル.

FITS ファイルの HDU データ（ヘッダー＋画像）を型安全に管理し、
波長配列・空間配列の取得およびスペクトル描画機能を提供する。

クラス構成:
- ImageUnit   : data / header のペア（HDU の型安全なラッパー）
- ImageModel  : 1 ファイルに対応する sci / err / dq の集合体
- ImageCollection : 複数 ImageModel のコレクション
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, Iterator, TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from astropy.io import fits  # type: ignore
from astropy.wcs import WCS  # type: ignore

from ..util.constants import ANGSTROM_TO_METER
from ..util.fits_reader import STISFitsReader, ReaderCollection


# ---------------------------------------------------------------------------
# ImageUnit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageUnit:
    """data と header のペア（HDU の型安全なラッパー）.

    Attributes
    ----------
    data : np.ndarray
        画像データ配列
    header : fits.Header
        対応する FITS ヘッダー
    """

    data: np.ndarray
    header: fits.Header

    @property
    def naxis1(self) -> int:
        """列数（NAXIS1）."""
        return int(self.header.get("NAXIS1", 0))  # type: ignore[arg-type]

    @property
    def naxis2(self) -> int:
        """行数（NAXIS2）."""
        return int(self.header.get("NAXIS2", 0))  # type: ignore[arg-type]

    def to_hdu(self) -> fits.ImageHDU:
        """fits.ImageHDU に変換する.

        Returns
        -------
        fits.ImageHDU
            data と header を格納した ImageHDU
        """
        return fits.ImageHDU(data=self.data, header=self.header)


# ---------------------------------------------------------------------------
# ImageModel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageModel:
    """STIS スペクトル画像の単一フレームモデル.

    1 つの FITS ファイルに対応する科学データ・誤差・品質フラグを
    ImageUnit として保持し、波長配列・空間配列の取得および
    スペクトル描画機能を提供する。

    Attributes
    ----------
    primary_header : fits.Header
        Primary HDU（HDU 0）のヘッダー
    sci : ImageUnit
        科学画像（HDU 1）の data / header ペア
    err : ImageUnit or None
        統計的誤差（HDU 2）の data / header ペア。存在しない場合は None
    dq : ImageUnit or None
        品質フラグ（HDU 3）の data / header ペア。存在しない場合は None
    """

    primary_header: fits.Header
    sci: ImageUnit
    err: ImageUnit | None = None
    dq: ImageUnit | None = None

    def __repr__(self) -> str:
        return (
            f"ImageModel(\n"
            f"  sci={self.sci.data.shape},\n"
            f"  err={self.err is not None},\n"
            f"  dq={self.dq is not None}\n"
            f")"
        )

    @classmethod
    def from_reader(cls, reader: STISFitsReader) -> Self:
        """STISFitsReader からスペクトル画像モデルを生成する.

        Parameters
        ----------
        reader : STISFitsReader
            読み込み済みの Reader インスタンス

        Returns
        -------
        ImageModel
            生成されたスペクトル画像モデル
        """
        try:
            err = ImageUnit(
                data=reader.image_data(2),
                header=reader.header(2),
            )
        except KeyError:
            err = None

        try:
            dq = ImageUnit(
                data=reader.image_data(3),
                header=reader.header(3),
            )
        except KeyError:
            dq = None

        return cls(
            primary_header=reader.header(0),
            sci=ImageUnit(
                data=reader.image_data(1),
                header=reader.header(1),
            ),
            err=err,
            dq=dq,
        )

    # ------------------------------------------------------------------
    # プロパティ
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """科学データ配列の形状 (行数, 列数) を返す."""
        return self.sci.data.shape  # type: ignore[return-value]

    @property
    def wavelength_array(self) -> np.ndarray:
        """WCS から波長配列を計算する.

        sci.header の WCS 情報（CTYPE = 'WAVE'）を用いて、
        ピクセルインデックスから波長値 [m] へ変換する。

        Returns
        -------
        np.ndarray
            波長配列 [m]

        Raises
        ------
        ValueError
            WCS の CTYPE に WAVE 軸が見つからない場合
        """
        wcs = WCS(self.sci.header)
        naxis1 = self.sci.naxis1
        naxis2 = self.sci.naxis2

        ctype = wcs.wcs.ctype  # type: ignore[attr-defined]
        if ctype[0] == "WAVE":
            nx = int(naxis2)
        elif ctype[1] == "WAVE":
            nx = int(naxis1)
        else:
            raise ValueError(f"CTYPE に WAVE 軸が見つかりません: {ctype}")

        pixel_indices = np.arange(nx)
        spec_wcs = wcs.sub(["spectral"])  # type: ignore[attr-defined]
        return spec_wcs.pixel_to_world_values(pixel_indices)  # type: ignore[attr-defined]

    @property
    def spatial_array(self) -> np.ndarray:
        """空間方向のピクセルインデックス配列を返す.

        空間軸のピクセル数に基づいた 0 始まりのインデックス配列を返す。

        Returns
        -------
        np.ndarray
            空間方向のピクセルインデックス配列 (0, 1, 2, ...)
        """
        wcs = WCS(self.sci.header)
        naxis1 = self.sci.naxis1
        naxis2 = self.sci.naxis2

        ctype = wcs.wcs.ctype  # type: ignore[attr-defined]
        if ctype[0] == "WAVE":
            n_spatial = int(naxis1)
        else:
            n_spatial = int(naxis2)
        return np.arange(n_spatial)

    # ------------------------------------------------------------------
    # 描画
    # ------------------------------------------------------------------

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
        wave = self.wavelength_array / ANGSTROM_TO_METER
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(wave, self.sci.data[:, point])
        ax.set_xlim(center_wave - width, center_wave + width)
        ax.set_xlabel(r"Wavelength [$\AA$]")
        ax.set_ylabel("Counts")
        return ax


# ---------------------------------------------------------------------------
# ImageCollection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageCollection:
    """複数の ImageModel をまとめて管理するコレクション.

    ReaderCollection から一括で生成し、解析パイプラインへ渡す。

    Attributes
    ----------
    images : list[ImageModel]
        ロード済み ImageModel のリスト
    """

    images: list[ImageModel]

    @classmethod
    def from_readers(cls, reader_collection: ReaderCollection) -> Self:
        """ReaderCollection から全ファイルの ImageModel を一括生成する.

        Parameters
        ----------
        reader_collection : ReaderCollection
            読み込み済み Reader コレクション

        Returns
        -------
        ImageCollection
            生成済みコレクション
        """
        return cls(
            images=[ImageModel.from_reader(r) for r in reader_collection]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> ImageModel:
        return self.images[index]

    def __iter__(self) -> Iterator[ImageModel]:
        return iter(self.images)