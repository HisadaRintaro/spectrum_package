"""2D 速度マップモデル.

複数のスリットの VelocityModel を集約し、
補間によって2次元速度マップを生成する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from .velocity import VelocityModel
from ...util.interpolation import Interpolator, get_interpolator


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
    def from_velocity_models(
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

    def plot(
        self,
        ax: Axes | None = None,
        cmap: str = "RdBu_r",
        **kwargs: object,
    ) -> Axes:
        """2D 速度マップをカラーマップとして描画する.

        Parameters
        ----------
        ax : Axes or None, optional
            描画先の matplotlib Axes。None の場合は新規作成。
        cmap : str, optional
            カラーマップ名（デフォルト: "RdBu_r"）
        **kwargs : object
            `pcolormesh` に渡す追加キーワード引数

        Returns
        -------
        Axes
            速度マップが描画された Axes オブジェクト
        """
        if ax is None:
            _, ax = plt.subplots()

        # 速度を km/s に変換して表示
        v_km_s = self.velocity_2d / 1e3

        im = ax.pcolormesh(
            self.x_coords,
            self.y_coords,
            v_km_s,
            cmap=cmap,
            shading="auto",
            **kwargs,  # type: ignore
        )
        plt.colorbar(im, ax=ax, label="Velocity [km/s]")
        ax.set_xlabel("Slit position [arcsec]")
        ax.set_ylabel("Slit offset [arcsec]")
        ax.set_title("2D Velocity Map")
        ax.set_aspect("equal")
        return ax
