from spectrum_package.processing import InstrumentModel, ImageModel
from spectrum_package.util.reader import read_fits

inst = InstrumentModel("HST/", "_flt", ".fits", depth=1)

header, data, error, quality = read_fits(inst.path_list()[0])
image = ImageModel.load(inst.path_list()[0])
#image.spectrum.plot_spectrum(0)
