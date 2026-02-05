from astropy.io import fits
import numpy as np
from typing import cast

def print_info(filename: str) -> None:
    with fits.open(filename) as hdul:
        print(hdul.info())

def read_fits(filename: str) -> tuple[fits.Header, np.ndarray, np.ndarray, np.ndarray]:
    #STISのfitsファイルは主に4つのHDUで構成されている
    #HDU0: PrimaryHDU
    #HDU1: ImageHDU(science values)
    #HDU2: ImageHDU(statistical errors)
    #HDU3: ImageHDU(quality flags)
    with fits.open(filename) as hdul:
        primary_hdu = cast(fits.PrimaryHDU, hdul[0])
        image_hdu = cast(fits.ImageHDU, hdul[1])
        error_hdu = cast(fits.ImageHDU, hdul[2])
        quality_hdu = cast(fits.ImageHDU, hdul[3])
        return primary_hdu.header, image_hdu.data, error_hdu.data, quality_hdu.data
