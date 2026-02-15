from spectrum_package.processing import InstrumentModel, ImageCollection, VelocityModel, VelocityMap
from spectrum_package.util import ReaderCollection

inst = InstrumentModel("HST/", "_flt", ".fits", depth=1,exclude_files=("o56503010_flt.fits",))
reader_collection = ReaderCollection.from_paths(inst.path_list)
image_collection = ImageCollection.from_readers(reader_collection)


# [OIII] 5007Å = 5.007e-7 m
vel = VelocityModel.from_image(image_collection[0], rest_wavelength=5.007e-7, window_width=1e-8)
#print(vel.velocities)  # 各空間位置の後退速度 [m/s]

vmap = VelocityMap.from_image_collection(
    image_collection,
    rest_wavelength=5.007e-7,
    window_width=1e-8,
    slit_step=0.2,
    method="linear",
    grid_resolution=4 * 6,
)
vmap.plot()

# スリット方向ピクセル50番の curve_fit 結果を可視化
VelocityModel.plot_fit(image_collection[0], spatial_pixel=50, rest_wavelength=5.007e-7, window_width=1e-8)