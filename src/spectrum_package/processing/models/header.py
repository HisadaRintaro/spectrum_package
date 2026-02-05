from dataclasses import dataclass
from astropy.io import fits # type: ignore
from typing import Self, cast, Any
import numpy as np

@dataclass
class HeaderRaw:
    filename: str
    optical_element: str
    bandwidth: float
    spec_res: float
    center_wavelength: float
    wave_min: float
    wave_max: float
    plate_scale: float #arcsec/pixel
    size_axis1: int
    size_axis2: int
    

    @classmethod
    def parse_header(cls,header: fits.Header) -> Self:
        h = cast(Any, header)
        return cls(
            filename=h.get("FILENAME",""),
            optical_element= cast(str,h.get("OPT_ELEM","")),
            bandwidth= cast(float,h.get("BANDWID",np.nan)),
            spec_res= cast(float,h.get("SPECRES",np.nan)),
            center_wavelength= cast(float,h.get("CENTRWV",np.nan)),
            wave_min= cast(float,h.get("MINWAVE",np.nan)),
            wave_max= cast(float,h.get("MAXWAVE",np.nan)),
            plate_scale= cast(float,h.get("PLATESC",np.nan)),
            size_axis1= cast(int,h.get("SIZAXIS1",np.nan)),
            size_axis2= cast(int,h.get("SIZAXIS2",np.nan)),
            )

    @property
    def wavelength(self) -> np.ndarray:
        return np.linspace(self.wave_min, self.wave_max, int(self.bandwidth))