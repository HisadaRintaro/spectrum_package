from dataclasses import dataclass
from pathlib import Path
from typing import Self

@dataclass(frozen=True)
class InstrumentModel:
    file_directry:str
    suffix: str
    extension: str 
    depth: int = 1

    @classmethod
    def load(cls, file_directry: str, suffix: str = "", extension: str = "", depth: int = 1) -> Self:
        return cls(
            file_directry=file_directry,
            suffix=suffix,
            extension=extension,
            depth=depth,
        )

    @staticmethod
    def get_path_list(file_directry: str, suffix: str, extension: str, depth: int = 1) -> list[Path]:
        path = Path(file_directry)
        pattern = "*/" * depth + f"*{suffix}{extension}"
        path_list = list(path.glob(pattern))
        path_list.sort()
        return path_list

    def path_list(self) -> list[Path]:
        return self.get_path_list(
            file_directry=self.file_directry,
            suffix=self.suffix,
            extension=self.extension,
            depth=self.depth,
        )
