import re
from bs4 import BeautifulSoup as Soup

import numpy as np
import pandas as pd
from tifffile import TiffFile, TiffWriter
from skimage import util
from skimage.transform import pyramid_gaussian

from renderapi.image_pyramid import ImagePyramid, MipMapLevel


# TU Delft storage server
HOST = 'https://sonic.tnw.tudelft.nl'


def create_tile_dict(fp, d_tile=None, host=HOST, **kwargs):
    """Parses Odemis OME-TIFF metadata to create a dict resembling a tile specification

    Parameters
    ----------
    fp : `pathlib.Path`
        Filepath
    d_tile : dict, (optional)
        Pre-populated tile dictionary
    host : str (optional)
        Host url for the image pyramid

    Returns
    -------
    d_tile : dict
        Almighty dictionary containing lots of juicy info about the image tile
    """
    # Extract metadata
    # ----------------
    tif = TiffFile(fp.as_posix())
    metadata = tif.pages[0].description
    soup = Soup(metadata, 'lxml')

    # Initialize tile dict
    # --------------------
    if d_tile is None:
        d_tile = {}
    # Infer stack, z value, and sectionId
    if 'stack' not in d_tile:
        d_tile['stack'] = fp.parents[2].name
    if 'z' not in d_tile:
        d_tile['z'] = -1
    if 'sectionId' not in d_tile:
        d_tile['sectionId'] = fp.parents[1].name

    # Layout parameters
    # -----------------
    # Infer image row and column from filepath
    col, row = [int(i) for i in re.findall(r'\d+', fp.parent.name)[:2]]
    d_tile['imageRow'] = row
    d_tile['imageCol'] = col
    # Parse metadata for stage coordinates
    d_tile['stageX'] = 1e6 * float(soup.plane['positionx'])  # m --> um
    d_tile['stageY'] = 1e6 * float(soup.plane['positiony'])  # m --> um
    # Parse metadata for pixel size
    psx = 1e3 * float(soup.pixels['physicalsizex'])  # um --> nm
    psy = 1e3 * float(soup.pixels['physicalsizey'])  # um --> nm
    d_tile['pixelsize'] = (psx + psy) / 2            # nm/px
    # Parse metadata for acquisition timestamp
    d_tile['acqTime'] = pd.to_datetime(soup.acquisitiondate.text)

    # Image pyramid
    # -------------
    # Create nested MipMapLevels
    mmls = []
    for mmfp in sorted(fp.parent.glob('[0-9].tif')):
        level = mmfp.stem
        imageUrl = f"{host}{mmfp.as_posix()}"
        mml = MipMapLevel(level, imageUrl=imageUrl, maskUrl=None)
        mmls.append(mml)
    # Create ImagePyramid from MipMapLevels
    ip = ImagePyramid({mm.level: mm.mipmap for mm in mmls})
    d_tile['imagePyramid'] = ip

    # Remaining tile specifications
    # -----------------------------
    # Set unique tileId
    d_tile['tileId'] = f"{d_tile['stack']}-{d_tile['sectionId']}-{col:05d}x{row:05d}"
    # Parse metadata for width and height
    d_tile['width'] = int(soup.pixels['sizex'])
    d_tile['height'] = int(soup.pixels['sizey'])
    # Set min, max intensity levels at 16bit uint limits
    d_tile['minint'] = 0
    d_tile['maxint'] = 2**16 - 1
    # Set empty list of transforms
    d_tile['tforms'] = []

    return d_tile


def create_mipmaps(image, dir_out, metadata=None, dtype=np.uint16, invert=False,
                   downscale=2, max_layer=8, preserve_range=True,
                   **pyramid_kwargs):
    """Generates and writes an image pyramid to disk (mipmaps)

    Parameters
    ----------
    image : ndarray
        Base image of the pyramid
    dir_out : `pathlib.Path`
        Output directory for mipmaps
        Individual mipmaps are saved as `dir_out/level.tif`
    metadata : str or encoded bytes (optional)
        Description of the image as 7-bit ASCII
        In practice this is the ome-xml metadata written by Odemis
        Passed (confusingly) to `description` argument of TiffWriter.save
    dtype : dtype (optional)
        The type of the output array
    invert : bool
        Whether to invert the contrast
    downscale : scalar (optional)
        Downscale factor
    max_layer : scalar (optional)
        Number of layers for the pyramid
    preserve_range : bool (optional)
        Whether to keep the original range of values
    pyramid_kwargs : dict (optional)
        Additional keyword arguments passed to `skimage.transform.pyramid_gaussian`

    Notes
    -----
    * Currently only writes .tif files
    """
    # Conditionally invert contrast
    if invert:
        image = util.invert(image)

    # Create image pyramid
    pyramid = pyramid_gaussian(image,
                               downscale=downscale,
                               max_layer=max_layer,
                               preserve_range=preserve_range,
                               **pyramid_kwargs)
    # Format pyramid as dict {0: image, 1: image//2, 2: image//4}
    d_pyramid = {level: mipmap.astype(dtype) for level, mipmap in enumerate(pyramid)}

    # Write mipmaps to disk
    for level, mipmap in d_pyramid.items():
        # Only write metadata to level 0 mipmap
        if level != 0:
            metadata = None
        # Set filepath for mipmap
        fp = dir_out / f"{int(level)}.tif"
        with TiffWriter(fp.as_posix()) as _tif:
            _tif.save(mipmap, description=metadata)
