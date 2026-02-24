"""リファクタリング後のクラス群に対するユニットテスト.

テスト対象:
- ImageUnit   : data / header のペア管理
- ImageModel  : 単一 FITS フレームモデル（sci / err / dq）
- ImageCollection : 複数 ImageModel のコレクション
- STISFitsReader  : FITS ファイル読み取り
- ReaderCollection: 複数 Reader のコレクション

実際の FITS ファイルへの依存を避けるため、
pytest + unittest.mock でヘッダー・データをスタブ化している。
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from spectrum_package.processing.image import ImageUnit, ImageModel, ImageCollection
from spectrum_package.util.fits_reader import STISFitsReader, ReaderCollection
from spectrum_package.util.constants import ANGSTROM_TO_METER


# ---------------------------------------------------------------------------
# ヘルパー: モック用ヘッダーの生成
# ---------------------------------------------------------------------------

def _make_wcs_header(naxis1: int = 100, naxis2: int = 50) -> fits.Header:
    """WCS 情報を持つシンプルな sci ヘッダーを生成する.

    STIS の典型的なレイアウトを模倣する:
      CTYPE1 = 'WAVE', CTYPE2 = 'SPATIAL'
    """
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["WAVE", "PIXEL"]
    wcs.wcs.crpix = [1.0, 1.0]
    wcs.wcs.crval = [5000e-10, 0.0]   # 5000 Å in meters
    wcs.wcs.cdelt = [1e-10, 1.0]      # 1 Å/pixel, 1 pixel/pixel
    header = wcs.to_header()
    header["NAXIS"] = 2
    header["NAXIS1"] = naxis1
    header["NAXIS2"] = naxis2
    return header


def _make_primary_header() -> fits.Header:
    """Primary HDU 用のシンプルなヘッダーを返す."""
    h = fits.Header()
    h["TELESCOP"] = "HST"
    h["INSTRUME"] = "STIS"
    return h


def _make_image_unit(naxis1: int = 100, naxis2: int = 50) -> ImageUnit:
    """テスト用の ImageUnit を生成する."""
    data = np.ones((naxis2, naxis1), dtype=np.float32)
    header = _make_wcs_header(naxis1, naxis2)
    return ImageUnit(data=data, header=header)


def _make_reader_stub(
    naxis1: int = 100,
    naxis2: int = 50,
    filename: Path | None = None,
) -> STISFitsReader:
    """実際のファイルを使わずに STISFitsReader のスタブを返す."""
    if filename is None:
        filename = Path("dummy.fits")
    sci_data = np.ones((naxis2, naxis1), dtype=np.float32)
    err_data = np.full((naxis2, naxis1), 0.1, dtype=np.float32)
    dq_data = np.zeros((naxis2, naxis1), dtype=np.int16)

    headers: dict[int, fits.Header] = {
        0: _make_primary_header(),
        1: _make_wcs_header(naxis1, naxis2),
        2: _make_wcs_header(naxis1, naxis2),
        3: _make_wcs_header(naxis1, naxis2),
    }
    data_dict: dict[int, np.ndarray] = {
        1: sci_data,
        2: err_data,
        3: dq_data,
    }
    return STISFitsReader(filename=filename, headers=headers, data=data_dict)


# ===========================================================================
# ImageUnit のテスト
# ===========================================================================

class TestImageUnit:
    """ImageUnit の基本機能を検証する."""

    def test_data_and_header_stored_correctly(self) -> None:
        """data / header が正しく保持されるか."""
        unit = _make_image_unit(naxis1=80, naxis2=40)
        assert unit.data.shape == (40, 80)
        assert isinstance(unit.header, fits.Header)

    def test_naxis1_returns_column_count(self) -> None:
        """naxis1 が NAXIS1 の値を返すか."""
        unit = _make_image_unit(naxis1=120, naxis2=60)
        assert unit.naxis1 == 120

    def test_naxis2_returns_row_count(self) -> None:
        """naxis2 が NAXIS2 の値を返すか."""
        unit = _make_image_unit(naxis1=120, naxis2=60)
        assert unit.naxis2 == 60

    def test_naxis_defaults_zero_when_keyword_missing(self) -> None:
        """NAXIS1/NAXIS2 キーワードがない場合は 0 を返すか."""
        data = np.zeros((10, 10))
        header = fits.Header()   # NAXIS キーワードなし
        unit = ImageUnit(data=data, header=header)
        assert unit.naxis1 == 0
        assert unit.naxis2 == 0

    def test_to_hdu_returns_image_hdu(self) -> None:
        """to_hdu が fits.ImageHDU を返すか."""
        unit = _make_image_unit()
        hdu = unit.to_hdu()
        assert isinstance(hdu, fits.ImageHDU)
        np.testing.assert_array_equal(hdu.data, unit.data)

    def test_frozen_dataclass_raises_on_set(self) -> None:
        """frozen=True により属性への代入が拒否されるか."""
        unit = _make_image_unit()
        with pytest.raises((AttributeError, TypeError)):
            unit.data = np.zeros((5, 5))  # type: ignore[misc]


# ===========================================================================
# ImageModel のテスト
# ===========================================================================

class TestImageModel:
    """ImageModel の生成・プロパティを検証する."""

    @pytest.fixture
    def model_full(self) -> ImageModel:
        """sci / err / dq すべて持つ ImageModel."""
        sci = _make_image_unit(naxis1=100, naxis2=50)
        err = _make_image_unit(naxis1=100, naxis2=50)
        dq = _make_image_unit(naxis1=100, naxis2=50)
        return ImageModel(
            primary_header=_make_primary_header(),
            sci=sci,
            err=err,
            dq=dq,
        )

    @pytest.fixture
    def model_sci_only(self) -> ImageModel:
        """sci のみ持つ ImageModel（err / dq = None）."""
        sci = _make_image_unit(naxis1=100, naxis2=50)
        return ImageModel(
            primary_header=_make_primary_header(),
            sci=sci,
        )

    # ---- 生成 ----

    def test_from_reader_creates_model(self) -> None:
        """from_reader が正常に ImageModel を生成するか."""
        reader = _make_reader_stub(naxis1=100, naxis2=50)
        model = ImageModel.from_reader(reader)
        assert isinstance(model, ImageModel)
        assert model.sci.data.shape == (50, 100)

    def test_from_reader_with_missing_err_dq(self) -> None:
        """HDU 2 / 3 がない場合に err / dq が None になるか."""
        reader = _make_reader_stub()
        # HDU 2, 3 のデータを除去
        data_without_err_dq = {1: reader.data[1]}
        headers_without_err_dq = {0: reader.headers[0], 1: reader.headers[1]}
        reader_stub = STISFitsReader(
            filename=Path("dummy.fits"),
            headers=headers_without_err_dq,
            data=data_without_err_dq,
        )
        model = ImageModel.from_reader(reader_stub)
        assert model.err is None
        assert model.dq is None

    # ---- プロパティ ----

    def test_shape_returns_sci_data_shape(self, model_full: ImageModel) -> None:
        """shape が sci データの形状を返すか."""
        assert model_full.shape == (50, 100)

    def test_wavelength_array_length_matches_naxis(
        self, model_full: ImageModel
    ) -> None:
        """wavelength_array の長さが CTYPE=WAVE 軸のピクセル数と一致するか.

        ヘッダー CTYPE1='WAVE', NAXIS2=50 → naxis2 方向が wave 軸のため
        wavelength_array の長さは 50 であることを期待する。
        """
        wave = model_full.wavelength_array
        # CTYPE1 == "WAVE" のとき nx = naxis2
        assert len(wave) == model_full.sci.naxis2

    def test_wavelength_array_unit_in_meters(
        self, model_full: ImageModel
    ) -> None:
        """wavelength_array の値がメートル単位か（可視光帯域に収まるか）."""
        wave = model_full.wavelength_array
        # 可視光は 380–700 nm = 3.8e-7 – 7e-7 m
        assert float(wave[0]) == pytest.approx(5000e-10, rel=1e-2)

    def test_spatial_array_length_matches_spatial_axis(
        self, model_full: ImageModel
    ) -> None:
        """spatial_array の長さが空間軸のピクセル数と一致するか.

        CTYPE1='WAVE' のとき空間軸は NAXIS1=100。
        """
        spatial = model_full.spatial_array
        assert len(spatial) == model_full.sci.naxis1

    def test_repr_contains_shape(self, model_full: ImageModel) -> None:
        """__repr__ に形状情報が含まれるか."""
        r = repr(model_full)
        assert "(50, 100)" in r

    # ---- sci / err / dq アクセス ----

    def test_err_none_when_not_provided(self, model_sci_only: ImageModel) -> None:
        """err を指定しなかった場合に None か."""
        assert model_sci_only.err is None

    def test_dq_none_when_not_provided(self, model_sci_only: ImageModel) -> None:
        """dq を指定しなかった場合に None か."""
        assert model_sci_only.dq is None

    def test_err_data_when_provided(self, model_full: ImageModel) -> None:
        """err が None でないとき data にアクセスできるか."""
        assert model_full.err is not None
        assert model_full.err.data.shape == (50, 100)


# ===========================================================================
# ImageCollection のテスト
# ===========================================================================

class TestImageCollection:
    """ImageCollection の生成・インターフェースを検証する."""

    @pytest.fixture
    def collection(self) -> ImageCollection:
        """3 枚の ImageModel を持つ ImageCollection."""
        readers = [_make_reader_stub(naxis1=100, naxis2=50) for _ in range(3)]
        rc = ReaderCollection(readers=readers)
        return ImageCollection.from_readers(rc)

    def test_len_matches_number_of_images(
        self, collection: ImageCollection
    ) -> None:
        """len() がファイル数を返すか."""
        assert len(collection) == 3

    def test_getitem_returns_image_model(
        self, collection: ImageCollection
    ) -> None:
        """インデックスアクセスで ImageModel が返るか."""
        model = collection[0]
        assert isinstance(model, ImageModel)

    def test_iter_yields_all_models(
        self, collection: ImageCollection
    ) -> None:
        """for ループで全モデルを走査できるか."""
        models = list(collection)
        assert len(models) == 3
        assert all(isinstance(m, ImageModel) for m in models)

    def test_from_readers_sci_shapes_consistent(
        self, collection: ImageCollection
    ) -> None:
        """全モデルの sci.data の形状が一致するか."""
        shapes = [m.shape for m in collection]
        assert len(set(shapes)) == 1  # すべて同じ形状

    def test_frozen_prevents_attribute_mutation(
        self, collection: ImageCollection
    ) -> None:
        """frozen=True により images リストへの代入が拒否されるか."""
        with pytest.raises((AttributeError, TypeError)):
            collection.images = []  # type: ignore[misc]


# ===========================================================================
# STISFitsReader のテスト
# ===========================================================================

class TestSTISFitsReader:
    """STISFitsReader の基本機能を検証する."""

    @pytest.fixture
    def reader(self) -> STISFitsReader:
        return _make_reader_stub(naxis1=100, naxis2=50)

    def test_header_returns_correct_hdu(self, reader: STISFitsReader) -> None:
        """header() が正しい HDU のヘッダーを返すか."""
        h0 = reader.header(0)
        assert h0["TELESCOP"] == "HST"

    def test_header_raises_key_error_for_missing_hdu(
        self, reader: STISFitsReader
    ) -> None:
        """存在しない HDU 番号で KeyError が発生するか."""
        with pytest.raises(KeyError):
            reader.header(99)

    def test_image_data_returns_array(self, reader: STISFitsReader) -> None:
        """image_data() が ndarray を返すか."""
        d = reader.image_data(1)
        assert isinstance(d, np.ndarray)
        assert d.shape == (50, 100)

    def test_image_data_raises_key_error_for_missing_hdu(
        self, reader: STISFitsReader
    ) -> None:
        """データのない HDU 番号で KeyError が発生するか."""
        with pytest.raises(KeyError):
            reader.image_data(0)  # primary は data なし

    def test_spectrum_data_returns_three_arrays(
        self, reader: STISFitsReader
    ) -> None:
        """spectrum_data() が (sci, err, dq) の 3 タプルを返すか."""
        sci, err, dq = reader.spectrum_data()
        assert sci.shape == err.shape == dq.shape

    def test_info_contains_filename(self, reader: STISFitsReader) -> None:
        """info() の出力にファイル名が含まれるか."""
        info_str = reader.info()
        assert "dummy.fits" in info_str

    def test_open_reads_real_fits_file(self, tmp_path: Path) -> None:
        """open() が実際の FITS ファイルを正しく読み込めるか."""
        fits_path = tmp_path / "test.fits"
        sci_data = np.ones((50, 100), dtype=np.float32)
        err_data = np.full((50, 100), 0.1, dtype=np.float32)
        dq_data = np.zeros((50, 100), dtype=np.int16)

        hdul = fits.HDUList([
            fits.PrimaryHDU(header=_make_primary_header()),
            fits.ImageHDU(data=sci_data, header=_make_wcs_header(100, 50)),
            fits.ImageHDU(data=err_data, header=_make_wcs_header(100, 50)),
            fits.ImageHDU(data=dq_data, header=_make_wcs_header(100, 50)),
        ])
        hdul.writeto(fits_path)

        reader = STISFitsReader.open(fits_path)
        assert reader.filename == fits_path
        np.testing.assert_array_equal(reader.image_data(1), sci_data)
        np.testing.assert_array_equal(reader.image_data(2), err_data)


# ===========================================================================
# ReaderCollection のテスト
# ===========================================================================

class TestReaderCollection:
    """ReaderCollection の基本機能を検証する."""

    @pytest.fixture
    def collection(self) -> ReaderCollection:
        readers = [_make_reader_stub(filename=Path(f"file{i}.fits")) for i in range(4)]
        return ReaderCollection(readers=readers)

    def test_len_matches_reader_count(self, collection: ReaderCollection) -> None:
        """len() が Reader の数を返すか."""
        assert len(collection) == 4

    def test_getitem_returns_reader(self, collection: ReaderCollection) -> None:
        """インデックスアクセスで STISFitsReader が返るか."""
        r = collection[0]
        assert isinstance(r, STISFitsReader)

    def test_iter_yields_all_readers(self, collection: ReaderCollection) -> None:
        """for ループで全 Reader を走査できるか."""
        readers = list(collection)
        assert len(readers) == 4

    def test_info_contains_all_filenames(self, collection: ReaderCollection) -> None:
        """info() に全ファイル名が含まれるか."""
        info_str = collection.info()
        for i in range(4):
            assert f"file{i}.fits" in info_str

    def test_from_paths_opens_real_fits_files(self, tmp_path: Path) -> None:
        """from_paths() が複数の FITS ファイルを正しく開けるか."""
        paths: list[Path] = []
        for i in range(2):
            p = tmp_path / f"obs{i}.fits"
            fits.HDUList([
                fits.PrimaryHDU(header=_make_primary_header()),
                fits.ImageHDU(
                    data=np.ones((50, 100), dtype=np.float32),
                    header=_make_wcs_header(100, 50),
                ),
            ]).writeto(p)
            paths.append(p)

        rc = ReaderCollection.from_paths(paths)
        assert len(rc) == 2
        assert isinstance(rc[0], STISFitsReader)


# ===========================================================================
# constants のテスト
# ===========================================================================

class TestConstants:
    """物理定数・変換係数の値を検証する."""

    def test_angstrom_to_meter(self) -> None:
        """ANGSTROM_TO_METER が 1e-10 か."""
        assert ANGSTROM_TO_METER == pytest.approx(1e-10)

    def test_speed_of_light(self) -> None:
        """光速が物理値に近い値か."""
        from spectrum_package.util.constants import SPEED_OF_LIGHT
        assert SPEED_OF_LIGHT == pytest.approx(2.998e8, rel=1e-3)
