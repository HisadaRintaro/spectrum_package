from spectrum_package.processing import InstrumentModel, ImageModel, VelocityModel, VelocityMap
from astropy.io import fits
from astropy.wcs import WCS

inst = InstrumentModel("HST/", "_flt", ".fits", depth=1,exclude_files=("o56503010_flt.fits",))
image = ImageModel.load(inst.path_list[0])

# [OIII] 5007Å = 5.007e-7 m
vel = VelocityModel.from_image(image, rest_wavelength=5.007e-7, window_width=1e-8)
print(vel.velocities)  # 各空間位置の後退速度 [m/s]

vmap = VelocityMap.from_instrument_model(
    inst,
    rest_wavelength=5.007e-7,
    window_width=1e-8,
    slit_step=0.2,
    method="linear",
    grid_resolution=4 * 6,
)
vmap.plot()

# スリット方向ピクセル50番の curve_fit 結果を可視化
VelocityModel.plot_fit(image, spatial_pixel=50, rest_wavelength=5.007e-7, window_width=1e-8)