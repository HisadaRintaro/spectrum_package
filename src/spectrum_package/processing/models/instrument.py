"""ファイル探索モデル.

観測データの FITS ファイルをディレクトリ構造から検索するためのモデル。
ディレクトリ、ファイル接尾辞、拡張子、およびディレクトリ深度を指定して
glob パターンによるファイル探索を行う。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Self


@dataclass(frozen=True)
class InstrumentModel:
    """観測装置のファイル構成モデル.

    指定されたディレクトリ内から、接尾辞と拡張子のパターンに一致する
    FITS ファイルを検索する。

    Attributes
    ----------
    file_directory : str
        データファイルのルートディレクトリパス
    suffix : str
        ファイル名の接尾辞（例: "_flt"）
    extension : str
        ファイルの拡張子（例: ".fits"）
    depth : int
        ディレクトリ探索の深度（デフォルト: 1）
    """

    file_directory: str
    suffix: str
    extension: str
    depth: int = 1

    @classmethod
    def load(cls, file_directory: str, suffix: str = "", extension: str = "", depth: int = 1) -> Self:
        """InstrumentModel を生成する.

        Parameters
        ----------
        file_directory : str
            データファイルのルートディレクトリパス
        suffix : str, optional
            ファイル名の接尾辞（デフォルト: ""）
        extension : str, optional
            ファイルの拡張子（デフォルト: ""）
        depth : int, optional
            ディレクトリ探索の深度（デフォルト: 1）

        Returns
        -------
        InstrumentModel
            生成されたモデル
        """
        return cls(
            file_directory=file_directory,
            suffix=suffix,
            extension=extension,
            depth=depth,
        )

    @staticmethod
    def get_path_list(file_directory: str, suffix: str, extension: str, depth: int = 1) -> list[Path]:
        """指定条件に一致するファイルパスの一覧を取得する.

        Parameters
        ----------
        file_directory : str
            データファイルのルートディレクトリパス
        suffix : str
            ファイル名の接尾辞
        extension : str
            ファイルの拡張子
        depth : int, optional
            ディレクトリ探索の深度（デフォルト: 1）

        Returns
        -------
        list[Path]
            パターンに一致するファイルパスのソート済みリスト
        """
        path = Path(file_directory)
        pattern = "*/" * depth + f"*{suffix}{extension}"
        path_list = list(path.glob(pattern))
        path_list.sort()
        return path_list

    def path_list(self) -> list[Path]:
        """現在の設定に基づいてファイルパスの一覧を取得する.

        Returns
        -------
        list[Path]
            パターンに一致するファイルパスのソート済みリスト
        """
        return self.get_path_list(
            file_directory=self.file_directory,
            suffix=self.suffix,
            extension=self.extension,
            depth=self.depth,
        )
