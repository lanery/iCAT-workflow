import requests
from itertools import product

from tqdm.notebook import tqdm
import numpy as np
import altair as alt
from seaborn import color_palette
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.geometry import box
from shapely import affinity

from renderapi.render import format_preamble
from renderapi.stack import (get_stack_bounds,
                             get_bounds_from_z,
                             get_z_values_for_stack)
from renderapi.tilespec import get_tile_spec
from renderapi.image import get_bb_image
from renderapi.errors import RenderError

from .render_pandas import create_stacks_DataFrame


__all__ = ['clear_image_cache',
           'render_bbox_image',
           'render_tileset_image',
           'render_layer_images',
           'render_stack_images',
           'render_neighborhood_image',
           'plot_tile_map',
           'plot_stacks',
           'plot_neighborhoods',
           'plot_stacks_interactive',
           'plot_matches_within_section',
           'plot_matches_across_sections']


def clear_image_cache():
    url = 'https://sonic.tnw.tudelft.nl/render-ws/v1/imageProcessorCache/allEntries'
    response = requests.delete(url)
    return response


def render_bbox_image(stack, z, bbox, width=1024, render=None,
                      **renderapi_kwargs):
    """Renders an image given the specified bounding box

    Parameters
    ----------
    stack : str
        Input stack from which to render bbox image
    z : float
        Z layer at which to render bbox image
    bbox : list, tuple, array-like
        Coordinates of bounding box (minx, miny, maxx, maxy)
    width : float
        Width of rendered tileset image in pixels
    render : `renderapi.render.RenderClient`
        `render-ws` instance

    Returns
    -------
    image : ndarray
        Rendered bounding box image

    Notes
    -----
    Differs from `renderapi.image.get_bb_image` parameters:
        (x0, y0, width, height, scale)
    """
    # Unpack bbox
    x = bbox[0]
    y = bbox[1]
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    s = np.round(width / (bbox[2] - bbox[0]), decimals=6)

    # Render image bounding box image as tif
    image = get_bb_image(stack=stack, z=z, x=x, y=y,
                         width=w, height=h, scale=s,
                         render=render,
                         **renderapi_kwargs)

    # Sometimes it overloads the system
    if isinstance(image, RenderError):
        # Recreate requested url
        request_url = format_preamble(
            host=render.DEFAULT_HOST,
            port=render.DEFAULT_PORT,
            owner=render.DEFAULT_OWNER,
            project=render.DEFAULT_PROJECT,
            stack=stack) + \
            f"/z/{z:.0f}/box/{x:.0f},{y:.0f},{w:.0f},{h:.0f},{s}/png-image"
        # Tell 'em the bad news
        print(f"Failed to load {request_url}.")

    return image


def render_tileset_image(stack, z, width=1024, render=None,
                         **renderapi_kwargs):
    """Renders an image of a tileset

    Parameters
    ----------
    stack : str
        Stack with which to render the tileset image
    z : float
        Z value of stack at which to render tileset image
    width : float
        Width of rendered tileset image in pixels
    render : `renderapi.render.RenderClient`
        `render-ws` instance

    Returns
    -------
    image : ndarray
        Rendered image of the specified tileset
    """
    # Get bbox for z layer from stack bounds
    bounds = get_bounds_from_z(stack=stack,
                               z=z,
                               render=render)
    bbox = [bounds[k] for k in ['minX', 'minY', 'maxX', 'maxY']]
    # Render bbox image
    image = render_bbox_image(stack=stack,
                              z=z,
                              bbox=bbox,
                              width=width,
                              render=render,
                              **renderapi_kwargs)
    return image


def render_stack_images(stack, width=1024, render=None,
                        **renderapi_kwargs):
    """Renders tileset images for a given stack

    Parameters
    ----------
    stack : str
        Stack with which to render images for all z values
    width : float
        Width of rendered tileset image in pixels
    render : `renderapi.render.RenderClient`
        `render-ws` instance

    Returns
    -------
    images : dict
        Dictionary of tileset images comprising the stack with z value as key
    """
    # Get z values of stack
    z_values = get_z_values_for_stack(stack=stack,
                                      render=render)
    # Get bbox of stack from stack bounds
    bounds = get_stack_bounds(stack=stack,
                              render=render)
    bbox = [bounds[k] for k in ['minX', 'minY', 'maxX', 'maxY']]
    # Loop through z values and collect images
    images = {}
    for z in tqdm(z_values, leave=False):
        image = render_bbox_image(stack=stack,
                                  z=z,
                                  bbox=bbox,
                                  width=width,
                                  render=render,
                                  **renderapi_kwargs)
        images[z] = image
    return images


def render_layer_images(stacks, z, width=1024, render=None,
                        **renderapi_kwargs):
    """Renders tileset images for a given layer

    Parameters
    ----------
    stacks : list
        List of stacks to with which to render layer images
    z : float
        Z value of stacks at which to render layer images
    width : float
        Width of rendered layer images in pixels
    render : `renderapi.render.RenderClient`
        `render-ws` instance

    Returns
    -------
    images : dict
        Dictionary of tileset images comprising the layer with stack name as key
    """
    # Get bbox of layer from bounds of each stack
    boundss = []
    for stack in stacks:
        bounds = get_bounds_from_z(stack=stack, z=z, render=render)
        boundss.append(bounds)
    bbox = [min([bounds['minX'] for bounds in boundss]),
            min([bounds['minY'] for bounds in boundss]),
            max([bounds['maxX'] for bounds in boundss]),
            max([bounds['maxY'] for bounds in boundss])]

    # Loop through stacks and collect images
    images = {}
    for stack in tqdm(stacks, leave=False):
        image = render_bbox_image(stack=stack,
                                  z=z,
                                  bbox=bbox,
                                  width=width,
                                  render=render,
                                  **renderapi_kwargs)
        images[stack] = image
    return images


def render_neighborhood_image(stack, tileId, neighborhood=1, width=1024,
                              render=None, **renderapi_kwargs):
    """Renders an image of the local neighborhood surrounding a tile

    Parameters
    ----------
    stack : str
        Stack from which to render neighborhood image
    tileId : str
        tileId (duh)
    neighborhood : float
        Number of tiles surrounding center tile from which to render the image
    width : float
        Width of rendered layer images in pixels
    render : `renderapi.render.RenderClient`
        `render-ws` instance
    """
    # Alias for neighborhood
    N = neighborhood

    # Get bounding box of specified tile
    tile_spec = get_tile_spec(stack=stack,
                              tile=tileId,
                              render=render)

    # Get width of bbox
    bbox = tile_spec.bbox
    w = bbox[2] - bbox[0]

    # Assume surrounding tiles are squares with ~same width as the center tile
    bbox_neighborhood = (bbox[0] - N*w, bbox[1] - N*w,
                         bbox[2] + N*w, bbox[3] + N*w)

    # Render image of neighborhood
    image = render_bbox_image(stack=stack,
                              z=tile_spec.z,
                              bbox=bbox_neighborhood,
                              width=width,
                              render=render,
                              **renderapi_kwargs)
    return image



def plot_tile_map(stacks, render=None, **kwargs):
    """Plots tiles (as matplotlib patches) in `render-ws`

    Parameters
    ----------
    stacks : list
        List of stacks to plot
    render : `renderapi.render.RenderClient`
        `render-ws` instance
    """

    # Create stacks DataFrame
    df_stacks = create_stacks_DataFrame(stacks=stacks,
                                        render=render, **kwargs)

    # Specify stacks and sections for plotting
    stacks_2_plot = df_stacks['stack'].unique().tolist()
    sections_2_plot = df_stacks['sectionId'].unique().tolist()

    # Set up figure
    ncols = len(sections_2_plot)
    fig, axes = plt.subplots(ncols=ncols, sharex=True, sharey=True,
                             squeeze=False, figsize=(8*ncols, 8))
    axmap = {k: v for k, v in zip(sections_2_plot, axes.flat)}
    cmap = {k: v for k, v in zip(stacks_2_plot,
                                 color_palette(n_colors=len(stacks_2_plot)))}
    # Collect all tiles in each layer to determine bounds
    boxes = []

    # Iterate through layers
    for sectionId, layer in tqdm(df_stacks.groupby('sectionId')):
        # Set axis
        ax = axmap[sectionId]
        # Collect legend handles
        handles = []

        # Loop through tilesets within each layer
        for stack, tileset in layer.groupby('stack'):

            # Loop through each tile
            for i, tile in tileset.reset_index().iterrows():

                # Create `shapely.box` resembling raw image tile
                b = box(0, 0, tile['width'], tile['height'])
                # Apply transforms to `shapely.box`
                for tform in tile['tforms']:
                    A = (tform.M[:2, :2].ravel().tolist() +
                         tform.M[:2,  2].ravel().tolist())
                    b = affinity.affine_transform(b, A)
                boxes.append(b)

                # Get coordinates of `shapely.box` to plot matplotlib polygon patch
                xy = np.array(b.exterior.xy).T
                p = Polygon(xy, color=cmap[stack], alpha=0.2, label=stack)
                # Add patch to axis
                if i != 0: ax.add_patch(p)             # Only add first patch
                else: handles.append(ax.add_patch(p))  # to legend handles
                # Label first tile in tileset
                x, y = np.array(b.centroid.xy).ravel()
                s = f"{stack}\n{sectionId}\n"\
                    f"{tile['imageCol']:02.0f}x{tile['imageRow']:02.0f}"
                if i == 0: ax.text(x, y, s, ha='center', va='center')

        # Axis aesthetics
        ax.set_title(sectionId)
        ax.legend(handles=handles, loc='lower right')
        ax.set_xlabel('X [px]')
        ax.set_ylabel('Y [px]')
        ax.grid(ls=':', alpha=0.7)
        ax.set_aspect('equal')

    # Set axis limits based on bounding boxes
    bounds = np.swapaxes([b.exterior.xy for b in boxes], 1, 2).reshape(-1, 2)
    for ax in axmap.values():
        ax.set_xlim(bounds[:, 0].min(), bounds[:, 0].max())
        ax.set_ylim(bounds[:, 1].min(), bounds[:, 1].max())
        ax.invert_yaxis()


def plot_stacks(stacks, z_values=None, width=1024, render=None,
                **renderapi_kwargs):
    """Renders and plots tileset images for the given stacks"""
    # Create DataFrame from stacks
    df_stacks = create_stacks_DataFrame(stacks=stacks,
                                        render=render)

    # Plot all z values if none are provided
    if z_values is None:
        z_values = df_stacks['z'].unique().tolist()

    # Set up figure
    nrows = len(stacks)
    ncols = len(z_values)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(8*ncols, 8*nrows))
    axmap = {k: v for k, v in zip(product(stacks, z_values), axes.flat)}

    # Iterate through tilesets
    df_2_plot = df_stacks.loc[df_stacks['z'].isin(z_values)]
    for (stack, z), tileset in tqdm(df_2_plot.groupby(['stack', 'z'])):

        # Render tileset image
        image = render_tileset_image(stack=stack,
                                     z=z,
                                     width=width,
                                     render=render,
                                     **renderapi_kwargs)

        # Get extent of tileset image in render-space
        bounds = get_stack_bounds(stack=stack,
                                  render=render)
        extent = [bounds[k] for k in ['minX', 'maxX', 'minY', 'maxY']]

        # Plot image
        ax = axmap[(stack, z)]
        ax.imshow(image, origin='lower', extent=extent)
        # Axis aesthetics
        ax.invert_yaxis()
        sectionId = tileset['sectionId'].iloc[0]
        ax.set_title(f"{stack}\nz = {z:.0f} | {sectionId}")
        ax.set_xlabel('X [px]')
        ax.set_ylabel('Y [px]')


def plot_neighborhoods(stacks, z_values=None, neighborhood=1, width=1024,
                       render=None, **renderapi_kwargs):
    """Renders and plots a neighborhood image around the given tile"""
    # Create DataFrame from stacks
    df_stacks = create_stacks_DataFrame(stacks=stacks,
                                        render=render)

    # Plot all z values if none are provided
    if z_values is None:
        z_values = df_stacks['z'].unique().tolist()

    # Set up figure
    nrows = len(stacks)
    ncols = len(z_values)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(8*ncols, 8*nrows))
    axmap = {k: v for k, v in zip(product(stacks, z_values), axes.flat)}

    # Iterate through tilesets
    df_2_plot = df_stacks.loc[df_stacks['z'].isin(z_values)]
    for (stack, z), tileset in tqdm(df_2_plot.groupby(['stack', 'z'])):

        # Select a tile from the tileset randomly
        tileId = tileset.sample(1).iloc[0]['tileId']

        # Render neighborhood image
        image, bbox = render_neighborhood_image(stack=stack,
                                                tileId=tileId,
                                                neighborhood=neighborhood,
                                                width=width,
                                                return_bbox=True,
                                                render=render,
                                                **renderapi_kwargs)

        # Get extent of neighborhood image in render-space (L, R, B, T)
        extent = (bbox[0], bbox[2], bbox[1], bbox[3])

        # Plot image
        ax = axmap[(stack, z)]
        ax.imshow(image, origin='lower', extent=extent)
        # Axis aesthetics
        ax.invert_yaxis()
        sectionId = tileset['sectionId'].iloc[0]
        ax.set_title(f"{stack}\nz = {z:.0f} | {sectionId}\n{tileId}")
        ax.set_xlabel('X [px]')
        ax.set_ylabel('Y [px]')


def plot_stacks_interactive(z, stack_images, render=None):
    """Plot stacks interactively (imshow) with a slider to scroll through z value

    Parameters
    ----------
    z : scalar
        Z value to plot
    stack_images : dict
        Collection of images in {stack1: {'z_n': image_n},
                                         {'z_n+1': image_n+1},
                                 stack2: {'z_n': image_n},
                                         {'z_n+1': image_n+1}} form
    """
    # Get stack names as keys
    stacks = list(stack_images.keys())
    # Setup figure
    ncols=len(stacks)
    fig, axes = plt.subplots(ncols=ncols, sharex=True, sharey=True,
                             squeeze=False, figsize=(7*ncols, 7))
    # Map each stack to an axis
    axmap = {k: v for k, v in zip(stacks, axes.flat)}
    # Get extent from global bounds
    bounds = np.array([list(get_stack_bounds(
              stack=stack, render=render).values()) for stack in stacks])
    extent = [bounds[:, 0].min(axis=0), bounds[:, 3].max(axis=0),  # minx, maxx
              bounds[:, 1].min(axis=0), bounds[:, 4].max(axis=0)]  # miny, maxy
    # Loop through stacks to plot images
    for stack, images in stack_images.items():
        cmap = 'magma' if 'EM' not in stack else 'Greys'
        axmap[stack].imshow(images[z], origin='lower', extent=extent, cmap=cmap)
        axmap[stack].set_title(stack)
        axmap[stack].set_aspect('equal')
        axmap[stack].invert_yaxis()


def plot_matches_within_section(df_matches, direction, width=200, height=200):
    """Plot point matches within each section

    Parameters
    ----------
    df_matches : pd.DataFrame
        DataFrame of point matches from a given stack (or stacks)
    direction : str
        Direction along which to plot point matches
        Can either be EAST-WEST or NORTH-SOUTH
    width : scalar (optional)
        Width of each subplot
    height : scalar (optional)
        Height of each subplot

    Returns
    -------
    chart : `alt.FacetChart`
        (altair) plot of point matches within each section
    """
    # Filter point matches DataFrame to East-West (LEFT, RIGHT) tile pairs
    if direction.lower() in ['ew', 'east-west', 'left-right']:
        source = df_matches.loc[df_matches['pr'] == df_matches['qr']]\
                           .copy()
        source['pqc'] = source.loc[:, ['pc', 'qc']].min(axis=1)
        # Create base of chart
        base = alt.Chart(source).encode(
            x='pqc:O',
            y='pr:O')

    # Filter point matches DataFrame to North-South (TOP, BOTTOM) tile pairs
    else:
        source = df_matches.loc[df_matches['pc'] == df_matches['qc']]\
                           .copy()
        source['pqr'] = source.loc[:, ['pr', 'qr']].min(axis=1)
        # Create base of chart
        base = alt.Chart(source).encode(
            x='pc:O',
            y='pqr:O')

    # Make heatmap
    heatmap = base.mark_rect().encode(
        color=alt.Color('N:Q'),
    ).properties(
        width=width,
        height=height
    )
    text = base.mark_text(baseline='middle').encode(
        text='N:Q',
    )
    # Facet heatmaps across sections and montage stacks
    heatmap = alt.layer(heatmap, text, data=source).facet(
        column=r'pGroupId:N',
        row='stack:N'
    )
    return heatmap


def plot_matches_across_sections(df_matches, width=200, height=200):
    """Plot point matches within each section

    Parameters
    ----------
    df_matches : pd.DataFrame
        DataFrame of point matches from a given stack (or stacks)
    width : scalar (optional)
        Width of each subplot
    height : scalar (optional)
        Height of each subplot

    Returns
    -------
    chart : `alt.FacetChart`
        (altair) plot of point matches within each section
    """
    # Filter DataFrame of point matches to cross section tile pairs
    source = df_matches.loc[df_matches['pGroupId'] != df_matches['qGroupId']].copy()
    # Add column specifying section pair
    source['sections'] = [f"{pId} -- {qId}" for (pId, qId) in\
                          zip(source['pGroupId'], source['qGroupId'])]

    # Initialize chart by attempting to make a heatmap along rows, cols
    if ('pc' in df_matches) and ('pr' in df_matches):
        base = alt.Chart(source).encode(
            x='pc:N',
            y='pr:N'
        )
    # Rough alignment matches may not have 'pc', 'pr' data
    else:
        base = alt.Chart(source)

    # Create heatmap from base
    heatmap = base.mark_rect().encode(
        color=alt.Color('N:Q'),
    ).properties(
        width=width,
        height=height
    )
    text = base.mark_text(baseline='middle').encode(
        text='N:Q',
    )
    # Facet heatmaps across sections and montage stacks
    heatmap = alt.layer(heatmap, text, data=source).facet(
        row='stack:N',
        column='sections:N',
    )
    return heatmap
