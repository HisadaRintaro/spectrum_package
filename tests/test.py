from spectrum_package.processing import VelocityModel, ImageModel, InstrumentModel

inst = InstrumentModel("HST/", "_flt", ".fits", depth=1)
image = ImageModel.load(inst.path_list()[0])

# [OIII] 5007Å = 5.007e-7 m
vel = VelocityModel.from_image(image, rest_wavelength=5.007e-7, window_width=1e-8)
print(vel.velocities)  # 各空間位置の後退速度 [m/s]