import numpy as np
import pandas as pd

from renderapi.render import get_stacks_by_owner_project
from renderapi.tilespec import TileSpec, get_tile_specs_from_stack
from renderapi.stack import create_stack, set_stack_state
from renderapi.client import import_tilespecs


__all__ = ['create_stack_DataFrame',
           'create_stacks_DataFrame',
           'create_project_DataFrame',
           'upload_stack_DataFrame']


def create_stack_DataFrame(stack, render, **kwargs):
    """Create DataFrame of `TileSpec`s from a given stack

    Parameters
    ----------
    stack : str
        Stack from which to generate DataFrame
    render : `renderapi.render.RenderClient`
        `render-ws` instance

    Returns
    -------
    df_stack : `pd.DataFrame`
        DataFrame of all `TileSpec`s from given stack
    """
    # Loop through tile specifications
    tile_specs = []
    for ts in get_tile_specs_from_stack(stack=stack,
                                        render=render, **kwargs):
        # Convert to dict
        d_tile = ts.to_dict()
        # Adjust certain specifications
        d_tile['minint'] = ts.minint
        d_tile['maxint'] = ts.maxint
        d_tile['imagePyramid'] = ts.ip
        d_tile['tforms'] = ts.tforms
        # Remove problematic keys
        d_tile.pop('minIntensity', None)
        d_tile.pop('maxIntensity', None)
        d_tile.pop('mipmapLevels', None)
        d_tile.pop('transforms', None)
        # Add to collection
        tile_specs.append(d_tile)

    # Create DataFrame from tile specifications
    df_stack = pd.DataFrame(tile_specs)
    # Add stack name to DataFrame
    df_stack['stack'] = stack

    # Expand `layout` column
    if 'layout' in df_stack.columns:
        df_stack = pd.concat([df_stack.drop('layout', axis=1),
                              df_stack['layout'].apply(pd.Series)], axis=1)

    return df_stack


def create_stacks_DataFrame(stacks, render, **kwargs):
    """Create DataFrame of `TileSpec`s from multiple stacks within a project

    Parameters
    ----------
    stacks : list
        List of stacks from which to generate DataFrame
    render : `renderapi.render.RenderClient`
        `render-ws` instance

    Returns
    -------
    df_stacks : `pd.DataFrame`
        DataFrame of all `TileSpec`s from given stacks
    """
    # Initialize DataFrame
    df_stacks = pd.DataFrame()
    # Create and append DataFrames from each given stack
    for stack in stacks:
        df_stack = create_stack_DataFrame(stack, render=render, **kwargs)
        df_stacks = pd.concat([df_stacks, df_stack])
    return df_stacks.reset_index(drop=True)


def create_project_DataFrame(render):
    """Create DataFrame of `TileSpec`s from all stacks within a project

    Parameters
    ----------
    render : `renderapi.render.RenderClient`
        `render-ws` instance

    Returns
    -------
    df_project : `pd.DataFrame`
        DataFrame of all `TileSpec`s from within a project
    """
    # Get all stacks within project
    stacks = get_stacks_by_owner_project(render=render)
    df_project = create_stacks_DataFrame(stacks, render=render)
    return df_project


def upload_stack_DataFrame(df, render, name=None,
                           stackResolutionX=None,
                           stackResolutionY=None,
                           stackResolutionZ=None, **kwargs):
    """Creates a `render-ws` stack from given DataFrame

    Parameters
    ----------
    df : `pd.DataFrame`
        DataFrame of tile data
    render : `renderapi.render.RenderClient`
        `render-ws` instance
    name : str
        Name of stack
        Looks for 'stack' column in `df` if not provided
    """
    # Set stack name
    if name is None:
        stack = df.iloc[0]['stack']
    else:
        stack = name

    # Loop through tiles
    out = f"Creating tile specifications for \033[1m{stack}\033[0m..."
    print(out)
    tile_specs = []
    for i, tile in df.iterrows():
        # Create `TileSpec`s
        ts = TileSpec(**tile.to_dict())
        # Ensure integer min, max intensity
        ts.minint = int(tile['minint'])
        ts.maxint = int(tile['maxint'])
        # Collect `TileSpec`s
        tile_specs.append(ts)

    # Create stack
    create_stack(stack=stack,
                 stackResolutionX=stackResolutionX,
                 stackResolutionY=stackResolutionY,
                 stackResolutionZ=stackResolutionZ,
                 render=render)

    # Import TileSpecs to render
    out = f"Importing tile specifications to \033[1m{stack}\033[0m..."
    print(out)
    import_tilespecs(stack=stack,
                     tilespecs=tile_specs,
                     render=render, **kwargs)

    # Close stack
    set_stack_state(stack=stack,
                    state='COMPLETE',
                    render=render, **kwargs)
    out = f"Stack \033[1m{stack}\033[0m created successfully."
    print(out)
