from astropy.io import fits # type: ignore
import numpy as np
from typing import Any
from pathlib import Path

def print_info(filename: Path) -> None:
    with fits.open(filename) as hdul: # type: ignore
        print(hdul.info()) # type: ignore

def read_header(hdu: Any) -> fits.Header:
    if not isinstance(hdu.header, fits.Header):
        raise TypeError("hdu.header is not Header")
    return hdu.header

def read_data(hdu: Any) -> np.ndarray:
    if not isinstance(hdu.data, np.ndarray):
        raise TypeError("hdu.data is not ndarray")
    return hdu.data # type: ignore

def read_fits(filename: Path) -> tuple[fits.Header, np.ndarray, np.ndarray, np.ndarray]:
    #STISのfitsファイルは主に4つのHDUで構成されている
    #HDU0: PrimaryHDU
    #HDU1: ImageHDU(science values)
    #HDU2: ImageHDU(statistical errors)
    #HDU3: ImageHDU(quality flags)
    with fits.open(filename) as hdul: # type: ignore
        header = read_header(hdul[1])
        image = read_data(hdul[1])
        error = read_data(hdul[2])
        quality = read_data(hdul[3])
        return header, image, error, quality 
