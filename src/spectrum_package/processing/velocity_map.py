"""2D 速度マップモデル.

複数のスリットの VelocityModel を集約し、
補間によって2次元速度マップを生成する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, TYPE_CHECKING

if TYPE_CHECKING:
    from .instrument import InstrumentModel

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from .velocity import VelocityModel
from .image import ImageCollection
from ..util.interpolation import Interpolator, get_interpolator


@dataclass(frozen=True)
class VelocityMap:
    """2次元速度マップモデル.

    複数スリットの VelocityModel を補間して2次元の速度場を保持する。

    Attributes
    ----------
    velocity_models : list[VelocityModel]
        各スリットの VelocityModel リスト
    velocity_2d : np.ndarray
        補間済み 2D 速度マップ [m/s]
    x_coords : np.ndarray
        X 軸座標グリッド（スリット方向）[arcsec]
    y_coords : np.ndarray
        Y 軸座標グリッド（スリット垂直方向）[arcsec]
    interpolation_method : str
        使用した補間方法名
    """

    velocity_models: list[VelocityModel]
    velocity_2d: np.ndarray
    x_coords: np.ndarray
    y_coords: np.ndarray
    interpolation_method: str

    @classmethod
    def _from_velocity_models(
        cls,
        models: list[VelocityModel],
        method: str = "linear",
        grid_resolution: int | None = None,
    ) -> Self:
        """複数の VelocityModel から 2D 速度マップを生成する.

        Parameters
        ----------
        models : list[VelocityModel]
            各スリットの VelocityModel リスト
        method : str, optional
            補間メソッド名（"linear", "nearest", "cubic"）。
            デフォルト: "linear"
        grid_resolution : int, optional
            グリッドの解像度（各軸のピクセル数）。

        Returns
        -------
        VelocityMap
            補間された 2D 速度マップ
        """
        interpolator = get_interpolator(method)

        # 全モデルから散布データを収集
        all_x: list[np.ndarray] = []  # スリット方向
        all_y: list[np.ndarray] = []  # スリット垂直方向（slit_offset）
        all_v: list[np.ndarray] = []

        for model in models:
            n = len(model.velocities)
            valid = ~np.isnan(model.velocities)
            all_x.append(model.spatial_positions[valid])
            all_y.append(np.full(int(np.sum(valid)), model.slit_offset))
            all_v.append(model.velocities[valid])

        points_x = np.concatenate(all_x)
        points_y = np.concatenate(all_y)
        values = np.concatenate(all_v)
        points = np.column_stack([points_x, points_y])

        x_min, x_max = points_x.min(), points_x.max()
        y_min, y_max = points_y.min(), points_y.max()

        if grid_resolution is None:
            grid_resolution = 4 * len(models)

        grid_x_1d = np.linspace(x_min, x_max, len(models[0].velocities))
        grid_y_1d = np.linspace(y_min, y_max, grid_resolution)
        grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)

        # 補間実行
        velocity_2d = interpolator.interpolate(points, values, grid_x, grid_y)

        return cls(
            velocity_models=models,
            velocity_2d=velocity_2d,
            x_coords=grid_x,
            y_coords=grid_y,
            interpolation_method=method,
        )

    @classmethod
    def from_image_collection(
        cls,
        image_collection: ImageCollection,
        rest_wavelength: float,
        window_width: float,
        slit_step: float = 0.2,
        method: str = "linear",
        grid_resolution: int | None = None,
    ) -> Self:
        """ImageCollection から VelocityMap を直接生成する.

        Parameters
        ----------
        image_collection : ImageCollection
            画像コレクション
        rest_wavelength : float
            輝線の静止波長 [m]
        window_width : float
            フィッティングウィンドウの半幅 [m]
        slit_step : float, optional
            スリット間のオフセットステップ [arcsec]（デフォルト: 0.2）
        method : str, optional
            補間メソッド名（"linear", "nearest", "cubic"）。
            デフォルト: "linear"
        grid_resolution : int, optional
            グリッドの解像度（各軸のピクセル数）。

        Returns
        -------
        VelocityMap
            補間された 2D 速度マップ
        """
        models = [
            VelocityModel.from_image(
                image,
                rest_wavelength=rest_wavelength,
                window_width=window_width,
                slit_offset=i * slit_step,
            )
            for i, image in enumerate(image_collection)
        ]

        return cls._from_velocity_models(
            models, method=method, grid_resolution=grid_resolution
        )

    def plot(
        self,
        ax: Axes | None = None,
        cmap: str = "RdBu_r",
        center_slit: int | None = None,
        center_pixel: int | None = None,
        **kwargs: object,
    ) -> Axes:
        """2D 速度マップをカラーマップとして描画する.

        ``center_slit`` と ``center_pixel`` を両方指定すると、
        その座標の後退速度を基準 (v=0) とした相対速度マップを描画する。

        Parameters
        ----------
        ax : Axes or None, optional
            描画先の matplotlib Axes。None の場合は新規作成。
        cmap : str, optional
            カラーマップ名（デフォルト: "RdBu_r"）
        center_slit : int or None, optional
            基準点のスリットインデックス（``velocity_models`` のリスト
            インデックス）。``center_pixel`` と併せて指定する。
        center_pixel : int or None, optional
            基準点の空間ピクセルインデックス（各 VelocityModel の
            ``velocities`` 配列インデックス）。``center_slit`` と
            併せて指定する。
        **kwargs : object
            `pcolormesh` に渡す追加キーワード引数

        Returns
        -------
        Axes
            速度マップが描画された Axes オブジェクト

        Raises
        ------
        IndexError
            指定されたインデックスが範囲外の場合
        ValueError
            ``center_slit`` と ``center_pixel`` の一方のみが指定された場合
        """
        if (center_slit is None) != (center_pixel is None):
            raise ValueError(
                "center_slit と center_pixel は両方同時に指定してください。"
            )

        if ax is None:
            _, ax = plt.subplots()

        # ---- 基準速度の取得と相対速度計算 ----
        if center_slit is not None and center_pixel is not None:
            ref_model = self.velocity_models[center_slit]
            v_ref = ref_model.velocities[center_pixel]  # [m/s]
            v_km_s = (self.velocity_2d - v_ref) / 1e3

            # 基準点の座標
            ref_x = ref_model.spatial_positions[center_pixel]  # arcsec
            ref_y = ref_model.slit_offset  # arcsec
            v_ref_km_s = v_ref / 1e3

            title = (
                f"Relative Velocity Map\n"
                f"ref: slit={center_slit}, pixel={center_pixel} "
                f"({ref_x:.2f}\", {ref_y:.2f}\")  "
                f"$v_{{sys}}$={v_ref_km_s:.1f} km/s"
            )
            colorbar_label = r"$\Delta v$ [km/s]"
        else:
            v_km_s = self.velocity_2d / 1e3
            title = "2D Velocity Map"
            colorbar_label = "Velocity [km/s]"
            ref_x = ref_y = None

        im = ax.pcolormesh(
            self.x_coords,
            self.y_coords,
            v_km_s,
            cmap=cmap,
            shading="auto",
            **kwargs,  # type: ignore
        )
        plt.colorbar(im, ax=ax, label=colorbar_label)

        # 基準点マーカー
        if ref_x is not None and ref_y is not None:
            ax.plot(
                ref_x, ref_y,
                marker="x", color="black", markersize=10,
                markeredgewidth=2, zorder=5,
            )

        ax.set_xlabel("Slit position [arcsec]")
        ax.set_ylabel("Slit offset [arcsec]")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        return ax
