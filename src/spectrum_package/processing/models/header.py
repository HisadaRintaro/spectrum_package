from dataclasses import dataclass
from astropy.io import fits
from typing import Self, cast
import numpy as np

@dataclass
class HeaderRaw:
    optical_element: str
    bandwidth: float
    spec_res: float
    center_wavelength: float
    wave_min: float
    wave_max: float
    plate_scale: float #arcsec/pixel
    

    @classmethod
    def parse_header(cls,header: fits.Header, ) -> Self:
        return cls(
                optical_element= cast(str,header.get("OPT_ELEM",np.nan)),
                bandwidth= cast(float,header.get("BANDWID",np.nan)),
                spec_res= cast(float,header.get("SPECRES",np.nan)),
                center_wavelength= cast(float,header.get("CENTRWV",np.nan)),
                wave_min= cast(float,header.get("MINWAVE",np.nan)),
                wave_max= cast(float,header.get("MAXWAVE",np.nan)),
                plate_scale= cast(float,header.get("PLATESC",np.nan)),
                )

    @property
    def wavelength(self) -> np.ndarray:
        return np.linspace(self.wave_min, self.wave_max, int(self.bandwidth))