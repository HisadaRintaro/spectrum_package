"""FITS ファイル I/O ユーティリティ.

STIS の FITS ファイルからヘッダーおよび画像データを読み取る関数群。
STIS の FITS ファイルは主に以下の4つの HDU で構成される:

- HDU 0: PrimaryHDU（観測メタデータ）
- HDU 1: ImageHDU（科学データ）
- HDU 2: ImageHDU（統計的誤差）
- HDU 3: ImageHDU（品質フラグ）
"""

from astropy.io import fits  # type: ignore
import numpy as np
from typing import Any, cast
from pathlib import Path


def print_info(filename: Path) -> None:
    """FITS ファイルの HDU 情報を標準出力に表示する.

    Parameters
    ----------
    filename : Path
        FITS ファイルのパス
    """
    with fits.open(filename) as hdul:  # type: ignore
        print(hdul.info())  # type: ignore


def read_header(filename: Path, hdu_number: int = 0) -> fits.Header:
    """FITS ファイルから指定した HDU のヘッダーを読み取る.

    Parameters
    ----------
    filename : Path
        FITS ファイルのパス
    hdu_number : int, optional
        読み取る HDU の番号（デフォルト: 0）

    Returns
    -------
    fits.Header
        指定した HDU のヘッダーオブジェクト
    """
    with fits.open(filename) as hdul:  # type: ignore
        header: fits.Header = cast(fits.Header, hdul[hdu_number].header)  # type: ignore
        return header


def read_data(hdu: Any) -> np.ndarray:
    """単一 HDU からデータ配列を読み取る.

    Parameters
    ----------
    hdu : Any
        astropy HDU オブジェクト

    Returns
    -------
    np.ndarray
        HDU のデータ配列

    Raises
    ------
    TypeError
        HDU のデータが ndarray でない場合
    """
    if not isinstance(hdu.data, np.ndarray):
        raise TypeError("hdu.data is not ndarray")
    return hdu.data  # type: ignore


def read_image(filename: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STIS FITS ファイルから科学データ・誤差・品質フラグを読み取る.

    HDU 1（科学データ）、HDU 2（統計的誤差）、HDU 3（品質フラグ）を
    それぞれ ndarray として返す。

    Parameters
    ----------
    filename : Path
        FITS ファイルのパス

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (科学データ, 統計的誤差, 品質フラグ) のタプル
    """
    with fits.open(filename) as hdul:  # type: ignore
        image = read_data(hdul[1])
        error = read_data(hdul[2])
        quality = read_data(hdul[3])
        return image, error, quality
