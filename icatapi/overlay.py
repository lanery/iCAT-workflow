import ast
from curses import meta

import numpy as np
from bs4 import BeautifulSoup as Soup
from matplotlib.transforms import Affine2D as AffineMPL

from skimage import feature, filters, measure
from skimage.transform import AffineTransform as AffineSkimage
from tifffile import TiffFile

from .utils import rescale


def get_transform_metadata(filepath):
    """Parse Odemis tif file for transform data

    Parameters
    ----------
    filepath : `Path`
        Path to image file

    Returns
    -------
    tform_md : tuple
        All the relevant Odemis transform data
        * pixelsize     | pixel size in x, y [m]
        * rotation      | rotation angle [rad]
        * shear         | shear [?]
        * translation   | stage-based translation in x, y [m]
    """
    # Gather metadata as `Soup`
    tif = TiffFile(filepath.as_posix())
    xml_data = tif.pages[0].description
    metadata = Soup(xml_data, 'lxml')

    # Parse the transform metadata for each image in tif
    tform_md = {}
    for md in metadata.find_all('image'):
        if md['name'] != 'Composited image preview':
            tform_md[md['name']] = parse_transform_metadata(md)
    return tform_md


def parse_transform_metadata(metadata):
    """Parse Odemis metadata for transform data

    Parameters
    ----------
    metadata : `Soup`
        Odemis metadata in a warm bowl of soup

    Returns
    -------
    pixelsize : tuple
        Image pixel size in x, y [m]
    rotation : float
        Image rotation angle [rad]
    shear : float
        Image shear [?]
    translation : tuple
        Stage-based translation in x, y [m]
    """
    # Calculate pixel size in x & y
    md = metadata.pixels
    psx = 1e-6 * float(md['physicalsizex'])  # um --> m
    psy = 1e-6 * float(md['physicalsizey'])  # um --> m
    pixelsize = (psx, psy)

    # Parse out rotation matrix
    md = metadata.transform
    if md is not None:
        A00 = float(md['a00'])  # /         \
        A01 = float(md['a01'])  # | a00  a01 |
        A10 = float(md['a10'])  # | a10  a11 |
        A11 = float(md['a11'])  # \         /
        # QR decomposition into Rotation and Scale matrices
        A = np.array([[A00, A10],
                      [A01, A11]])
        R, S = np.linalg.qr(A)
        mask = np.diag(S) < 0.
        R[:, mask] *= -1.
        S[mask, :] *= -1.
        # Calculate rotation angle and shear
        rotation = np.arctan2(R[1, 0], R[0, 0])
        rotation %= (2*np.pi)  # Odemis convention
        shear = S[0, 1] / S[0, 0]
    else:
        rotation = 0
        shear = 0

    # Translation
    md = metadata.plane
    x0 = float(md['positionx'])
    y0 = float(md['positiony'])
    translation = (x0, y0)

    return pixelsize, rotation, shear, translation


def compute_relative_transform(psx_EM, psy_EM,
                               ro_EM, sh_EM,
                               trx_EM, try_EM,
                               psx_FM, psy_FM,
                               ro_FM, sh_FM,
                               trx_FM, try_FM):
    """Compute relative affine transformation

    Parameters
    ----------
    psx_EM, psy_EM : float
        EM pixel size in x, y [m]
    ro_EM : float
        EM rotation (should be ~0)
    sh_EM : float
        EM shear
    trx_EM, try_EM : float
        EM translation in x, y [m]
    psx_FM, psy_FM : float
        FM pixel size in x, y [m]
    ro_FM : float
        FM rotation
    sh_FM : float
        FM shear (should be ~0)
    trx_FM, try_FM : float
        FM translation in x, y [m]

    Returns
    -------
    A : 3x3 array
        Relative affine transformation
    """
    A = AffineMPL().rotate(-ro_FM)\
                   .skew(0, -sh_EM)\
                   .scale(psx_FM / psx_EM,
                          psy_FM / psy_EM)\
                   .translate((trx_FM - trx_EM) / psx_EM,
                              (try_FM - try_EM) / -psy_EM)
    return A.get_matrix()


def compute_relative_transform_from_filepaths(fp_EM, fp_FM):
    """Compute affine transformation between correlative EM and FM image tiles

    Parameters
    ----------
    fp_EM : `Path`
        Filepath to EM image tile

    fp_FM : `Path`
        Filepath to FM image tile

    Returns
    -------
    A : 3x3 array
        Relative affine transformation
    """
    # Parse transform data
    tform_md_EM = list(get_transform_metadata(fp_EM).values())[0]
    tform_md_FM = list(get_transform_metadata(fp_FM).values())[0]
    (psx_EM, psy_EM), ro_EM, sh_EM, (trx_EM, try_EM) = tform_md_EM
    (psx_FM, psy_FM), ro_FM, sh_FM, (trx_FM, try_FM) = tform_md_FM

    # Pass transform data to `compute_relative_transform`
    A = compute_relative_transform(psx_EM, psy_EM,
                                   ro_EM, sh_EM,
                                   trx_EM, try_EM,
                                   psx_FM, psy_FM,
                                   ro_FM, sh_FM,
                                   trx_FM, try_FM)
    return A


def split_CLEM_image(filepath,
                     page_name_FM='Filtered colour 1',
                     page_name_EM='Secondary electrons'):
    """Split CLEM image into FM and EM images

    Parameters
    ----------
    filepath : `pathlib.Path`
        Filepath to multipage TIFF CLEM image
    page_name_FM : str
        TIFF page name for FM data
    page_name_EM : str
        TIFF page name for EM data

    Returns
    -------
    image_FM : (M, N) array
        FM image as numpy array
    metadata_FM : `Soup`
        FM metadata as xml
    image_EM : (M, N) array
        EM image as numpy array
    metadata_EM : `Soup`
        EM metadata as xml
    """
    tif = TiffFile(filepath)
    md = Soup(tif.pages[0].description, features='xml')

    # Split CLEM image
    image_FM = next(page.asarray()
                    for page in tif.pages if page.tags['PageName'].value == page_name_FM)
    image_EM = next(page.asarray()
                    for page in tif.pages if page.tags['PageName'].value == page_name_EM)

    # Get respective metadata
    metadata_FM = next(d for d in md.find_all(
        'Image') if d['Name'] == page_name_FM)
    metadata_EM = next(d for d in md.find_all(
        'Image') if d['Name'] == page_name_EM)

    return image_FM, metadata_FM, image_EM, metadata_EM


def get_transform_from_metadata(filepath):
    """Get CLEM overlay transform from metadata

    Parameters
    ----------
    filepath : `pathlib.Path`
        Filepath to CLEM image

    Returns
    -------
    T : (3, 3) array
        Affine transform as numpy array
    """
    # Split CLEM image
    image_FM, metadata_FM, image_EM, metadata_EM = split_CLEM_image(filepath)

    # Parse metadata for relevant transformation data
    (psx_FM, psy_FM), ro_FM, sh_FM, (trx_FM,
                                     try_FM) = parse_transform_metadata(metadata_FM)
    (psx_EM, psy_EM), ro_EM, sh_EM, (trx_EM,
                                     try_EM) = parse_transform_metadata(metadata_EM)

    # Translations to center images
    x0_FM, y0_FM = (d/2 for d in image_FM.shape)
    x0_EM, y0_EM = (d/2 for d in image_EM.shape)

    # Compute relative transform
    sx = psx_FM / psx_EM
    sy = psy_FM / psy_EM
    theta = -ro_FM
    shear = -sh_EM
    tx = (trx_FM - trx_EM) / psx_EM
    ty = -(try_FM - try_EM) / psy_EM

    # Chain transformations
    T = AffineMPL().translate(-x0_FM, -y0_FM)\
                   .rotate(theta)\
                   .skew(0, shear)\
                   .scale(sx, sy)\
                   .translate(tx, ty)\
                   .translate(x0_EM, y0_EM)._mtx
    return T


def parse_overlay_report(filepath):
    """Parses overlay report for EM spot coordinates and converts
    them into EM pixel space.

    The automated CL registration procedure involves the ccd recording
    a grid of CL spots generated by the e-beam in the absence of any
    excitation light. Spot coordinates depend on the dimensions of the
    EM image and the grid size. For the typical case of a (4, 4) grid
    and (4096, 4096) resolution, the coordinates are

        [(-1536,  1536), (-512,  1536), (512,  1536), (1536,  1536),
         (-1536,   512), (-512,   512), (512,   512), (1536,   512),
         (-1536,  -512), (-512,  -512), (512,  -512), (1536,  -512),
         (-1536, -1536), (-512, -1536), (512, -1536), (1536, -1536)]

    These are then translated to "EM pixel space" i.e. the pixel coordinates
    of where the EM spot actually falls in the EM image. Again for the typical
    case of a (4, 4) grid and (4096, 4096) resoultion, the pixel coordinates
    are

        [(512,  512), (1536,  512), (2560,  512), (3584,  512),
         (512, 1536), (1536, 1536), (2560, 1536), (3584, 1536),
         (512, 2560), (1536, 2560), (2560, 2560), (3584, 2560),
         (512, 3584), (1536, 3584), (2560, 3584), (3584, 3584)]

    Note that the ordering of coordinates in this docstring (which matches
    the scan direction) does not match the order in the `report.txt` file
    (which is sorted ascending y, x).

    Parameters
    ----------
    filepath : `pathlib.Path` or str
        Filepath to `report.txt` file output by Odemis

    Returns
    -------
    coords_EM : (N, N) array
        EM spot coordinates in EM pixel space
    """
    # Parse overlay report for EM spot coordinates and other goodies
    with open(filepath) as txt:
        for line in txt.readlines():
            if 'Grid size' in line:
                grid_shape_str = line
            if 'SEM pixel size' in line:
                pixelsize_str = line
            if 'SEM FoV' in line:
                fieldwidth_str = line
            if 'Spots coordinates in SEM ref' in line:
                coords_str = line

    # Get dimensions of EM image
    pixelsize = ast.literal_eval(pixelsize_str.split(':\t')[1])
    fieldwidth = ast.literal_eval(fieldwidth_str.split(':\t')[1])
    Nx, Ny = (int(fieldwidth[0] / pixelsize[0]),
              int(fieldwidth[1] / pixelsize[1]))

    # Convert coordinate data to numpy array
    grid_shape = ast.literal_eval(grid_shape_str.split(':\t')[1])
    coords = np.array(ast.literal_eval(coords_str.split(':\t')[1]))
    # Convert coords to EM pixel space
    coords_EM = np.zeros_like(coords)
    coords_EM[:, 0] = (coords[:, 0] - coords[:, 0].min()) / \
        (coords[:, 0] - coords[:, 0].min()).max() * Nx
    coords_EM[:, 1] = (coords[:, 1] - coords[:, 1].min()) / \
        (coords[:, 1] - coords[:, 1].min()).max() * Ny
    # Scale EM coordinates
    # | window in which CL spots are placed are "shrunk" wrt the full SEM FoV
    # | by a factor `(n-1)/n` where `n` is the number of CL spots per row
    # | https://github.com/delmic/odemis/blob/master/src/odemis/acq/align/find_overlay.py#L694
    scale = tuple((n - 1) / n for n in grid_shape)
    coords_EM[:, 0] = scale[0] * \
        (coords_EM[:, 0] - coords_EM[:, 0].max()/2) + coords_EM[:, 0].max()/2
    coords_EM[:, 1] = scale[1] * \
        (coords_EM[:, 1] - coords_EM[:, 1].max()/2) + coords_EM[:, 1].max()/2

    return coords_EM


def get_distances_to_points(points, n=2):
    """Find distance to nearest `n` points for each point in a set of points.

    References
    ----------
    [1] https://codereview.stackexchange.com/a/28210
    """
    points = points.astype(float)
    distances = [np.sort(np.sum((point - points)**2, axis=1))[1:1+n]
                 for point in points]
    return np.array(distances)


def detect_CL_peaks(filepath, grid_shape=(4, 4), min_distance=25):
    """Detect peaks from CL spots in CCD image.

    Parameters
    ----------
    filepath : `pathlib.Path`
        Filepath to `OpticalGrid.tiff` file output by Odemis
    grid_shape : tuple
        Dimensions of CL spot grid
    min_distance : scalar
        Input to `skimage.feature.peak_local_max`.
        The minimal allowed distance separating peaks. To find the
        maximum number of peaks, use `min_distance=1`.

    Returns
    -------
    peaks : (N, 2) array
        Coordinates of detected peaks in CL grid
    """
    # Load CL spot image
    tif = TiffFile(filepath)
    image_CL = tif.pages[0].asarray()

    # Smooth and rescale intensity
    image_CL = filters.gaussian(image_CL, sigma=2)
    image_CL = rescale(image_CL, (0, 100))

    # Find peaks (find a few extra in case of outliers)
    N_peaks = np.multiply(*grid_shape)
    peaks = feature.peak_local_max(image_CL,
                                   min_distance=min_distance,
                                   num_peaks=int(1.3*N_peaks),
                                   exclude_border=True)

    # Swap x, y
    peaks = peaks[:, ::-1]  # (not totally sure why this has to be done)
    # Filter outliers based on point-to-point distances
    # | Idea is that sometimes there are bright spots/artefacts in the CL image
    # | not from the e-beam grid. The distance from each spot in the grid to its
    # | nearest >= 2 neighbors (2 for corner spots, 3 for edge spots, 4 for
    # | central spots) should be uniform, so filter on this basis.
    distances = get_distances_to_points(peaks, n=2)
    peaks = peaks[np.abs(distances - np.median(distances)
                         ).sum(axis=1).argsort()][:N_peaks]
    # Sort by ascending y, x to match EM spot coordinates
    peaks = peaks[np.lexsort((peaks[:, 1], peaks[:, 0]))]

    return peaks


def get_transform_from_CLgrid(filepath, model_class=AffineSkimage):
    """Get CLEM overlay transform from CL grid and Odemis overlay report

    Parameters
    ----------
    filepath : `pathlib.Path`
        Filepath to directory containing the Odemis overlay report
        and `OpticalGrid.tiff`

    Returns
    -------
    tform : (3, 3) array
        Affine transformation matrix as (3, 3) numpy array
    """
    # Parse overlay report for EM spot coordinates
    coords_EM = parse_overlay_report(filepath / 'report.txt')
    # Detect peaks from CL spots in CCD image
    peaks_FM = detect_CL_peaks(filepath / 'OpticalGrid.tiff')

    # RANSAC FTW
    src = peaks_FM.copy()
    tgt = coords_EM.copy()
    model, inliers = measure.ransac((src, tgt),
                                    model_class=model_class,
                                    min_samples=7,          # not sure how optimal
                                    residual_threshold=25)  # these parameters are
    return model.params
