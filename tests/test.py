from spectrum_package.processing import InstrumentModel, ImageModel, VelocityModel, VelocityMap
from astropy.io import fits
from astropy.wcs import WCS

inst = InstrumentModel("HST/", "_flt", ".fits", depth=1,exclude_files=("o56503010_flt.fits",))

image = ImageModel.load(inst.path_list[0])
ax = image.plot_spectrum(0,center_wave=5100,width=100)

hdul = fits.open(inst.path_list[0])
header = hdul[1].header
w = WCS(header)

image = ImageModel.load(inst.path_list[0])

# [OIII] 5007Å = 5.007e-7 m
vel = VelocityModel.from_image(image, rest_wavelength=5.007e-7, window_width=1e-8)
print(vel.velocities)  # 各空間位置の後退速度 [m/s]

images = [ImageModel.load(path) for path in inst.path_list]
models = [
    VelocityModel.from_image(img, rest_wavelength=5.007e-7, window_width=1e-8,
                              slit_offset=i * 0.2)
    for i, img in enumerate(images)
]
vmap = VelocityMap.from_velocity_models(models, method="linear",grid_resolution=4*6)
vmap.plot()

# 1. VelocityModel plot_fit
from spectrum_package.processing import InstrumentModel, ImageModel, VelocityModel

inst = InstrumentModel("HST/", "_flt", ".fits", depth=1)
image = ImageModel.load(inst.path_list[0])

# スリット方向ピクセル50番の curve_fit 結果を可視化
VelocityModel.plot_fit(image, spatial_pixel=50, rest_wavelength=5.007e-7, window_width=1e-8)

# 2. InstrumentModel exclude_files
inst_ex = InstrumentModel("HST/", "_flt", ".fits", depth=1, exclude_files=("o56502010_flt",))
print(inst_ex.path_list)  # 除外されたファイルがリストに含まれないことを確認
