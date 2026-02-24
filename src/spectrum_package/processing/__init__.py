"""処理モデルパッケージ.

STIS スペクトルデータの解析に使用するモデルクラスを提供する。
"""

from .instrument import InstrumentModel
from .image import ImageUnit, ImageModel, ImageCollection

__all__ = [
    "InstrumentModel",
    "ImageUnit",
    "ImageModel",
    "ImageCollection",
]
