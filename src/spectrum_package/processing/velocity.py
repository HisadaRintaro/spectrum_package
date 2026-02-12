"""後退速度モデル.

スペクトルデータから輝線のガウスフィッティングを行い、
赤方偏移と後退速度を計算する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from scipy.optimize import curve_fit  # type: ignore

from ..util.constants import ANGSTROM_TO_METER, SPEED_OF_LIGHT

if TYPE_CHECKING:
    from .image import ImageModel


def _gaussian(x: np.ndarray, amp: float, center: float, sigma: float, offset: float) -> np.ndarray:
    """ガウス関数モデル.

    Parameters
    ----------
    x : np.ndarray
        入力値（波長配列）
    amp : float
        振幅
    center : float
        中心位置
    sigma : float
        標準偏差
    offset : float
        ベースラインオフセット

    Returns
    -------
    np.ndarray
        ガウス関数の値
    """
    return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2) + offset  # type: ignore


def _fit_gaussian(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    rest_wavelength: float,
    window_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """指定した静止波長付近の輝線にガウスフィッティングを行う.

    Parameters
    ----------
    wavelengths : np.ndarray
        波長配列 [m]
    flux : np.ndarray
        フラックス配列（1D, 波長方向）
    rest_wavelength : float
        輝線の静止波長 [m]
    window_width : float
        フィッティングウィンドウの半幅 [m]

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (フィッティングパラメータ popt,
         ウィンドウ内の波長配列, ウィンドウ内のフラックス配列)

    Raises
    ------
    RuntimeError
        フィッティングが収束しなかった場合
    """
    mask = (wavelengths >= rest_wavelength - window_width) & (
        wavelengths <= rest_wavelength + window_width
    )
    wave_window = wavelengths[mask]
    flux_window = flux[mask]

    if len(wave_window) < 4:
        raise RuntimeError(
            f"フィッティングウィンドウ内のデータ点が不足しています "
            f"({len(wave_window)} 点, 最低 4 点必要)"
        )

    offset_guess = float(np.median(flux_window))
    amp_guess = float(np.max(flux_window) - offset_guess)
    center_guess = float(wave_window[np.argmax(flux_window)])
    sigma_guess = window_width / 4.0

    p0 = [amp_guess, center_guess, sigma_guess, offset_guess]

    try:
        popt, _ = curve_fit(
            _gaussian,
            wave_window,
            flux_window,
            p0=p0,
            maxfev=5000,
        )
    except RuntimeError as e:
        raise RuntimeError(f"ガウスフィッティングが収束しませんでした: {e}") from e

    return popt, wave_window, flux_window


def _fit_emission_line(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    rest_wavelength: float,
    window_width: float,
) -> float:
    """指定した静止波長付近の輝線にガウスフィッティングを行い、観測波長を返す.

    Parameters
    ----------
    wavelengths : np.ndarray
        波長配列 [m]
    flux : np.ndarray
        フラックス配列（1D, 波長方向）
    rest_wavelength : float
        輝線の静止波長 [m]
    window_width : float
        フィッティングウィンドウの半幅 [m]

    Returns
    -------
    float
        フィッティングで得られた輝線の観測波長 [m]

    Raises
    ------
    RuntimeError
        フィッティングが収束しなかった場合
    """
    popt, _, _ = _fit_gaussian(wavelengths, flux, rest_wavelength, window_width)
    return float(popt[1])


@dataclass(frozen=True)
class VelocityModel:
    """輝線の後退速度モデル.

    各空間位置における輝線の観測波長・赤方偏移・後退速度を保持する。

    Attributes
    ----------
    rest_wavelength : float
        使用した静止波長 [m]
    observed_wavelengths : np.ndarray
        各空間位置での観測波長 [m] (1D)
    redshifts : np.ndarray
        各空間位置での赤方偏移 z (1D)
    velocities : np.ndarray
        各空間位置での後退速度 [m/s] (1D)
    spatial_positions : np.ndarray
        スリット方向の空間座標 [arcsec] (1D)
    slit_offset : float
        スリットの垂直方向オフセット [arcsec]。
        VelocityMap で2Dマップを構成する際に使用。
    """

    rest_wavelength: float
    observed_wavelengths: np.ndarray
    redshifts: np.ndarray
    velocities: np.ndarray
    spatial_positions: np.ndarray
    slit_offset: float

    def __repr__(self) -> str:
        return (
            f"VelocityModel(\n"
            f"  rest_wavelength={self.rest_wavelength:.4e} m,\n"
            f"  n_positions={len(self.velocities)},\n"
            f"  slit_offset={self.slit_offset:.2f} arcsec,\n"
            f"  v_mean={np.nanmean(self.velocities):.2f} m/s,\n"
            f"  v_range=[{np.nanmin(self.velocities):.2f}, {np.nanmax(self.velocities):.2f}] m/s\n"
            f")"
        )

    @classmethod
    def from_image(
        cls,
        image: ImageModel,
        rest_wavelength: float,
        window_width: float,
        slit_offset: float = 0.0,
        pixel_scale: float = 0.05,
    ) -> Self:
        """ImageModel から後退速度モデルを生成する.

        Parameters
        ----------
        image : ImageModel
            スペクトルデータを持つ ImageModel
        rest_wavelength : float
            輝線の静止波長 [m]
        window_width : float
            フィッティングウィンドウの半幅 [m]
        slit_offset : float, optional
            スリットの垂直方向オフセット [arcsec]（デフォルト: 0.0）
        pixel_scale : float, optional
            空間方向の1ピクセルあたりのスケール [arcsec/pixel]
            （STIS デフォルト: 0.05 arcsec/pixel）

        Returns
        -------
        VelocityModel
            後退速度モデル
        """
        wavelengths = image.header.spectrogram.wavelength_array
        data = image.spectrum.data
        n_spatial = data.shape[1]

        observed = np.full(n_spatial, np.nan)

        for i in range(n_spatial):
            flux = data[:, i]
            try:
                observed[i] = _fit_emission_line(
                    wavelengths, flux, rest_wavelength, window_width
                )
            except RuntimeError:
                continue

        redshifts = (observed - rest_wavelength) / rest_wavelength
        velocities = SPEED_OF_LIGHT * redshifts

        spatial_pixels = image.header.spectrogram.spatial_array
        spatial_positions = spatial_pixels * pixel_scale

        return cls(
            rest_wavelength=rest_wavelength,
            observed_wavelengths=observed,
            redshifts=redshifts,
            velocities=velocities,
            spatial_positions=spatial_positions,
            slit_offset=slit_offset,
        )

    @staticmethod
    def plot_fit(
        image: ImageModel,
        spatial_pixel: int,
        rest_wavelength: float,
        window_width: float,
        ax: Axes | None = None,
    ) -> Axes:
        """指定した空間ピクセルの curve_fit 結果を可視化する診断プロット.

        データ点とフィッティングされたガウス曲線を重ねて描画し、
        フィッティングの品質を目視確認できる。

        Parameters
        ----------
        image : ImageModel
            スペクトルデータを持つ ImageModel
        spatial_pixel : int
            スリット長方向（空間方向）のピクセルインデックス
        rest_wavelength : float
            輝線の静止波長 [m]
        window_width : float
            フィッティングウィンドウの半幅 [m]
        ax : Axes or None, optional
            描画先の matplotlib Axes。None の場合は新規作成。

        Returns
        -------
        Axes
            診断プロットが描画された Axes オブジェクト
        """
        wavelengths = image.header.spectrogram.wavelength_array
        flux = image.spectrum.data[:, spatial_pixel]

        popt, wave_window, flux_window = _fit_gaussian(
            wavelengths, flux, rest_wavelength, window_width
        )
        observed_wave = float(popt[1])

        # Å 単位に変換して描画
        wave_window_angstrom = wave_window / ANGSTROM_TO_METER
        observed_angstrom = observed_wave / ANGSTROM_TO_METER
        rest_angstrom = rest_wavelength / ANGSTROM_TO_METER

        # フィッティング曲線用の細かい波長グリッド（元のピクセル数の10倍）
        wave_fine = np.linspace(wave_window.min(), wave_window.max(), len(wave_window) * 10)
        flux_fit = _gaussian(wave_fine, *popt)
        wave_fine_angstrom = wave_fine / ANGSTROM_TO_METER

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(
            wave_window_angstrom, flux_window,
            color="steelblue", zorder=3, label="Data",
        )
        ax.plot(
            wave_fine_angstrom, flux_fit,
            color="tomato", linewidth=1.5, label="Gaussian fit",
        )
        ax.axvline(
            observed_angstrom, color="tomato", linestyle="--",
            linewidth=0.8, alpha=0.7, label=f"Obs. {observed_angstrom:.2f} Å",
        )
        ax.axvline(
            rest_angstrom, color="gray", linestyle=":",
            linewidth=0.8, alpha=0.7, label=f"Rest {rest_angstrom:.2f} Å",
        )

        ax.set_xlabel(r"Wavelength [$\AA$]")
        ax.set_ylabel("Counts")
        ax.set_title(f"Gaussian Fit — Spatial Pixel {spatial_pixel}")
        ax.legend(fontsize="small")

        return ax
