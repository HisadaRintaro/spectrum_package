"""処理モデルパッケージ.

STIS スペクトルデータの解析に使用するモデルクラスを提供する。
"""

from .instrument import InstrumentModel
from .image import ImageModel, ImageCollection
from .header import HeaderProfile
from .spectrum import SpectrumBase
from .velocity import VelocityModel
from .velocity_map import VelocityMap

__all__ = [
    "InstrumentModel",
    "ImageModel",
    "ImageCollection",
    "HeaderProfile",
    "SpectrumBase",
    "VelocityModel",
    "VelocityMap",
]
