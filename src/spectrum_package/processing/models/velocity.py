from __future__ import annotations

from dataclasses import dataclass
from typing import Self, TYPE_CHECKING

import numpy as np
from scipy.optimize import curve_fit  # type: ignore

from ...util.constants import SPEED_OF_LIGHT

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
    # ウィンドウ内のデータを切り出し
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

    # 初期推定値
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

    # popt[1] = center (観測波長)
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
    """

    rest_wavelength: float
    observed_wavelengths: np.ndarray
    redshifts: np.ndarray
    velocities: np.ndarray

    def __repr__(self) -> str:
        return (
            f"VelocityModel(\n"
            f"  rest_wavelength={self.rest_wavelength:.4e} m,\n"
            f"  n_positions={len(self.velocities)},\n"
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
            静止波長の前後この幅の範囲でガウスフィッティングを行う

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
                # フィッティング失敗時は NaN を維持
                continue

        redshifts = (observed - rest_wavelength) / rest_wavelength
        velocities = SPEED_OF_LIGHT * redshifts

        return cls(
            rest_wavelength=rest_wavelength,
            observed_wavelengths=observed,
            redshifts=redshifts,
            velocities=velocities,
        )
