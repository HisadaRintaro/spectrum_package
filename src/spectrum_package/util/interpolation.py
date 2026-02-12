"""補間アルゴリズムモジュール.

2D 速度マップ生成時に使用する補間アルゴリズムを提供する。
Protocol による抽象化により、補間方法の差し替えが容易に行える。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.interpolate import griddata  # type: ignore


class Interpolator(Protocol):
    """補間アルゴリズムのプロトコル.

    `VelocityMap` で使用される補間メソッドが満たすべきインターフェース。
    """

    def interpolate(
        self,
        points: np.ndarray,
        values: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
    ) -> np.ndarray:
        """散布データをグリッドに補間する.

        Parameters
        ----------
        points : np.ndarray
            既知点の座標 (N, 2)
        values : np.ndarray
            既知点の値 (N,)
        grid_x : np.ndarray
            補間先グリッドの X 座標（2D meshgrid）
        grid_y : np.ndarray
            補間先グリッドの Y 座標（2D meshgrid）

        Returns
        -------
        np.ndarray
            補間された 2D 配列
        """
        ...


class LinearInterpolator:
    """線形補間.

    `scipy.interpolate.griddata` の `method='linear'` を使用する。
    """

    def interpolate(
        self,
        points: np.ndarray,
        values: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
    ) -> np.ndarray:
        """線形補間で散布データをグリッドに補間する.

        Parameters
        ----------
        points : np.ndarray
            既知点の座標 (N, 2)
        values : np.ndarray
            既知点の値 (N,)
        grid_x : np.ndarray
            補間先グリッドの X 座標（2D meshgrid）
        grid_y : np.ndarray
            補間先グリッドの Y 座標（2D meshgrid）

        Returns
        -------
        np.ndarray
            線形補間された 2D 配列
        """
        return griddata(points, values, (grid_x, grid_y), method="linear")  # type: ignore


class NearestInterpolator:
    """最近傍補間.

    `scipy.interpolate.griddata` の `method='nearest'` を使用する。
    """

    def interpolate(
        self,
        points: np.ndarray,
        values: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
    ) -> np.ndarray:
        """最近傍補間で散布データをグリッドに補間する.

        Parameters
        ----------
        points : np.ndarray
            既知点の座標 (N, 2)
        values : np.ndarray
            既知点の値 (N,)
        grid_x : np.ndarray
            補間先グリッドの X 座標（2D meshgrid）
        grid_y : np.ndarray
            補間先グリッドの Y 座標（2D meshgrid）

        Returns
        -------
        np.ndarray
            最近傍補間された 2D 配列
        """
        return griddata(points, values, (grid_x, grid_y), method="nearest")  # type: ignore


class CubicInterpolator:
    """3次補間.

    `scipy.interpolate.griddata` の `method='cubic'` を使用する。
    """

    def interpolate(
        self,
        points: np.ndarray,
        values: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
    ) -> np.ndarray:
        """3次補間で散布データをグリッドに補間する.

        Parameters
        ----------
        points : np.ndarray
            既知点の座標 (N, 2)
        values : np.ndarray
            既知点の値 (N,)
        grid_x : np.ndarray
            補間先グリッドの X 座標（2D meshgrid）
        grid_y : np.ndarray
            補間先グリッドの Y 座標（2D meshgrid）

        Returns
        -------
        np.ndarray
            3次補間された 2D 配列
        """
        return griddata(points, values, (grid_x, grid_y), method="cubic")  # type: ignore


#: 補間メソッド名から Interpolator インスタンスへのマッピング
INTERPOLATORS: dict[str, Interpolator] = {
    "linear": LinearInterpolator(),
    "nearest": NearestInterpolator(),
    "cubic": CubicInterpolator(),
}


def get_interpolator(method: str = "linear") -> Interpolator:
    """補間メソッド名から対応する Interpolator を取得する.

    Parameters
    ----------
    method : str, optional
        補間メソッド名（"linear", "nearest", "cubic"）。デフォルト: "linear"

    Returns
    -------
    Interpolator
        対応する補間アルゴリズム

    Raises
    ------
    ValueError
        未知の補間メソッド名が指定された場合
    """
    if method not in INTERPOLATORS:
        raise ValueError(
            f"未知の補間メソッド: '{method}'. "
            f"利用可能: {list(INTERPOLATORS.keys())}"
        )
    return INTERPOLATORS[method]
