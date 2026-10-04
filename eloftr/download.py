"""Download the released EfficientLoFTR weights (eloftr_outdoor.ckpt) from the upstream Google Drive folder."""
import shutil
import tempfile
from pathlib import Path

import gdown

from eloftr.upstream import DEFAULT_CKPT

FOLDER_URL = 'https://drive.google.com/drive/folders/1GOw6iVqsB-f1vmG6rNmdCcgwfB4VZ7_Q'


def main():
    if DEFAULT_CKPT.exists():
        print(f'Weights already present: {DEFAULT_CKPT}')
        return
    DEFAULT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=DEFAULT_CKPT.parent) as tmp:
        gdown.download_folder(FOLDER_URL, output=tmp, quiet=False)
        found = sorted(Path(tmp).rglob(DEFAULT_CKPT.name))
        if not found:
            raise RuntimeError(f'{DEFAULT_CKPT.name} not found in {FOLDER_URL}')
        shutil.move(str(found[0]), DEFAULT_CKPT)
    DEFAULT_CKPT.chmod(0o644)
    print(f'Saved {DEFAULT_CKPT}')


if __name__ == '__main__':
    main()
