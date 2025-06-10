import pytest
import pandas as pd
import tempfile
import os
from PIL import Image

from src.databases.database import MoldDataset

@pytest.fixture
def test_with_images():
    # Create a temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Define image filenames
        filenames = ['fooA.jpg', 'fooB.jpg', 'barC.jpg', 'barD.jpg']
        # Create dummy images
        for fname in filenames:
            img = Image.new('RGB', (10, 10), color='white')
            img.save(os.path.join(tmpdir, fname))

        # DataFrame with expected image filenames
        df = pd.DataFrame({
        'filename': filenames,
        'ID': ['A', 'A', 'B', 'B'],
        'rotation_index': [0, 0, 1, 1],
        'class': ['foo', 'foo', 'bar', 'bar'],
        'top/bottom': ["top", "bottom", "top", "bottom"]
        })

        # Save CSV
        csv_path = os.path.join(tmpdir, 'fold.csv')
        df.to_csv(csv_path, index=False)
        yield tmpdir, csv_path

def test_molddataset_with_images(test_with_images):
    root, csv_file = test_with_images
    dataset = MoldDataset(
        root=root,
        fold_csv=csv_file,
        group_col="ID",
        pos_col="rotation_index"
    )
    assert len(dataset) == 2
    item = dataset[0]
    assert item is not None