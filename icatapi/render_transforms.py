import numpy as np
import pandas as pd
from matplotlib.transforms import Affine2D as AffineMPL

from renderapi.stack import get_stack_bounds
from renderapi.transform import AffineModel as AffineRender

from .render_pandas import create_stack_DataFrame, upload_stack_DataFrame


__all__ = ['AffineMPL_2_AffineRender',
           'scale_stack',
           'rotate_stack',
           'translate_stack',
           'transform_stack']


def AffineMPL_2_AffineRender(T):
    """Convert `AffineMPL` object to `AffineRender` object"""
    A = AffineRender()
    A.M = T.get_matrix()
    return A


def scale_stack(stack_in, stack_out=None, sx=1.0, sy=1.0, translate=False,
                render=None, **kwargs):
    """Scale stack by an arbitrary scale factor

    Parameters
    ----------
    stack_in : str
        Input stack name
    stack_out : str (optional)
        Output stack name
        Overwrites input stack if not provided
    sx : float
        x scale factor
    sy : float
        y scale factor
    translate : bool
        Whether to reposition the stack such that (minX, minY) = (0, 0)
    """
    # Create DataFrame for input stack
    df_stack = create_stack_DataFrame(stack=stack_in,
                                      render=render)

    # Determine stack center
    bounds = get_stack_bounds(stack=stack_in,
                              render=render)
    x0 = (bounds['minX'] + bounds['maxX']) / 2
    y0 = (bounds['minY'] + bounds['maxY']) / 2

    # Loop through tiles
    for i, tile in df_stack.iterrows():

        # Translate to center
        T1 = AffineRender(B0=-x0,
                          B1=-y0)
        # Scale
        S = AffineRender(M00=sx,
                         M11=sy)

        # Translate back such that (minX, minY) = (0, 0)
        tx = np.abs(bounds['minX'] - x0) * sx
        ty = np.abs(bounds['minY'] - y0) * sy
        T2 = AffineRender(B0=tx,
                          B1=ty)

        # Add transform to tile
        if translate:
            df_stack.at[i, 'tforms'] += [T1, S, T2]
        else:
            df_stack.at[i, 'tforms'] += [S]

    # Set output stack name
    stack_out = stack_out if stack_out is not None else stack_in

    # Create scaled stack
    upload_stack_DataFrame(df=df_stack,
                           name=stack_out,
                           render=render, **kwargs)


def rotate_stack(stack_in, stack_out=None, r=0,
                 render=None):
    """Rotate stack by an arbitrary rotation angle

    Parameters
    ----------
    stack_in : str
        Input stack name
    stack_out : str (optional)
        Output stack name
        Overwrites input stack if not provided
    r : float
        rotation angle [rad]
    """
    # Create DataFrame for input stack
    df_stack = create_stack_DataFrame(stack=stack_in,
                                      render=render)

    # Create rotation transform
    T = AffineMPL().rotate(r)
    A = AffineRender()
    A.M = T.get_matrix()

    # Add rotation transform to DataFrame
    for i, tile in df_stack.iterrows():
        df_stack.at[i, 'tforms'] += [A]

    # Set output stack name
    stack_out = stack_out if stack_out is not None else stack_in

    # Create scaled stack
    upload_stack_DataFrame(df=df_stack,
                           name=stack_out,
                           render=render)


def translate_stack(stack_in, stack_out=None, tx=0.0, ty=0.0,
                    render=None):
    """Translate stack by an arbitrary translation

    Parameters
    ----------
    stack_in : str
        Input stack name
    stack_out : str (optional)
        Output stack name
        Overwrites input stack if not provided
    tx : float
        x translation [px]
    ty : float
        y translation [px]
    """
    # Create DataFrame for input stack
    df_stack = create_stack_DataFrame(stack=stack_in,
                                      render=render)

    # Create scaling transform
    T = AffineMPL().translate(tx, ty)
    A = AffineRender()
    A.M = T.get_matrix()

    # Add translation transform to DataFrame
    for i, tile in df_stack.iterrows():
        df_stack.at[i, 'tforms'] += [A]

    # Set output stack name
    stack_out = stack_out if stack_out is not None else stack_in

    # Create scaled stack
    upload_stack_DataFrame(df=df_stack,
                           name=stack_out,
                           render=render)


def transform_stack(stack_in, stack_out=None, T=None,
                    render=None):
    """Transform stack by an arbitrary affine transformation

    Parameters
    ----------
    stack_in : str
        Input stack name
    stack_out : str (optional)
        Output stack name
        Overwrites input stack if not provided
    T : 3x3 array, `AffineMPL`, or `AffineRender` object
        Affine transform
    """
    # Create DataFrame for input stack
    df_stack = create_stack_DataFrame(stack=stack_in,
                                      render=render)

    # Create `AffineRender` transform object from given transform
    A = AffineRender()
    if isinstance(T, AffineMPL):
        A.M = T.get_matrix()
    elif isinstance(T, np.ndarray) and (T.shape == (3, 3)):
        A.M = T
    elif isinstance(T, AffineRender):
        pass
    else:
        raise ValueError("Improper form for transform.")

    # Add scale transform to DataFrame
    for i, tile in df_stack.iterrows():
        df_stack.at[i, 'tforms'] += [A]

    # Set output stack name
    stack_out = stack_out if stack_out is not None else stack_in

    # Create scaled stack
    upload_stack_DataFrame(df=df_stack,
                           name=stack_out,
                           render=render)
