import re
from functools import partial

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from renderapi.client import (tilePairClient, pointMatchClient,
                              SiftPointMatchOptions, WithPool)
from renderapi.stack import get_z_values_for_stack, get_section_z_value
from renderapi.pointmatch import get_matches_within_group, get_match_groupIds


def get_tile_pairs_4_montage(stack, render,
                             **tilePairClient_kwargs):
    """Collect tile pairs from stack one section at a time for montaging

    Parameters
    ----------
    stack : str
        Stack from which to generate DataFrame
    render : `renderapi.render.RenderClient`
        `render-ws` instance
    tilePairClient_kwargs : dict (optional)
        stack : str
            stack from which tilepairs should be considered
        minz : str
            minimum z bound from which tile 'p' is selected
        maxz : str
            maximum z bound from which tile 'p' is selected
        outjson : str or None
            json to which tile pair file should be written
            (defaults to using temporary file and deleting after completion)
        delete_json : bool
            whether to delete outjson on function exit (True if outjson is None)
        baseowner : str
            owner of stack from which stack was derived
        baseproject : str
            project of stack from which stack was derived
        basestack : str
            stack from which stack was derived
        xyNeighborFactor : float
            factor to multiply by max(width, height) of tile 'p' in order
            to generate search radius in z (0.9 if None)
        zNeighborDistance : int
            number of z sections defining the half-height of search cylinder
            for tile 'p' (2 if None)
        excludeCornerNeighbors : bool
            whether to exclude potential 'q' tiles based on center points
            falling outside search (True if None)
        excludeCompletelyObscuredTiles : bool
            whether to exclude potential 'q' tiles that are obscured by other tiles
            based on Render's sorting (True if None)
        excludeSameLayerNeighbors : bool
            whether to exclude potential 'q' tiles in the same z layer as 'p'
        excludeSameSectionNeighbors : bool
            whether to exclude potential 'q' tiles with the same sectionId as 'p'
        excludePairsInMatchCollection : str
            a matchCollection whose 'p' and 'q' pairs will be ignored
            if generated using this client
        minx : float
            minimum x bound from which tile 'p' is selected
        maxx : float
            maximum x bound from wich tile 'p' is selected
        miny : float
            minimum y bound from which tile 'p' is selected
        maxy : float
            maximum y bound from wich tile 'p' is selected

    Returns
    -------
    df_pairs : `pd.DataFrame`
        DataFrame of tile pairs from a given stack
    """
    # Initialize tile pairs DataFrame
    df_pairs = pd.DataFrame(columns=['stack', 'z'])

    # Iterate through stack's z values
    z_values = get_z_values_for_stack(stack=stack,
                                      render=render)
    for z in z_values:
        # Generate tile pairs
        tile_pairs_json = tilePairClient(stack=stack,
                                         minz=z,
                                         maxz=z,
                                         render=render,
                                         **tilePairClient_kwargs)
        # Create DataFrame from json
        df = pd.json_normalize(tile_pairs_json['neighborPairs'])
        df['z'] = z
        df_pairs = pd.concat([df_pairs, df])

    # Add stack info and reset index
    df_pairs['stack'] = stack
    return df_pairs.reset_index(drop=True)


def run_point_match_client(data, stack, collection, render,
                           **pointMatchClient_kwargs):
    """Point match client wrapper for use in multiprocessing"""
    tile_pair_batch, sift_options = data
    pointMatchClient(stack=stack,
                     collection=collection,
                     tile_pairs=tile_pair_batch,
                     sift_options=sift_options,
                     render=render,
                     **pointMatchClient_kwargs)


def generate_point_matches(df_pairs, match_collections, sift_options, render,
                           N_cores=4, batch_size=12, **pointMatchClient_kwargs):
    """Generate point matches for a set of tile pairs

    Parameters
    ----------
    df_pairs : `pd.DataFrame`
        DataFrame of tile pairs from a given stack (or stacks)
    match_collections : dict
        Mapping of stack names to names of
        e.g. {'lil_EM': 'zebrafish_lil_EM_points',
              'hoechst': 'zebrafish_hoechst_points'}
    sift_options : `renderapi.client.params.SiftPointMatchOptions`
        Parameter set for SIFT and RANSAC algorithms
    N_cores : scalar (optional)
        Number of cores to run point match generation in parallel
    batch_size : scalar (optional)
        Number of tile pairs to include in each batch
    pointMatchClient_kwargs : dict
        stack : str
            stack containing the tiles
        stack2 : str
            second optional stack containing tiles (if stack2 is not none, then
            tile_pair['p'] comes from stack and tile_pair['q'] comes from stack2)
        collection : str
            point match collection to save results into
        tile_pairs : iterable
            list of iterables of length 2 containing tileIds to calculate point matches between
        sift_options: SiftOptions
            options for running point matching
        pointMatchRender : renderapi.render.renderaccess
            renderaccess object specifying the render server to store point matches in
            defaults to values specified by render and its keyword argument overrides
        debugDirectory : str
            directory to store debug results (optional)
        filter: bool
            whether to apply default filtering to tile (default=False)
        renderWithoutMask: bool
            whether to exclude the mask when rendering tile (default=False)
        normalizeForMatching: bool
            whether to apply traditional 'normalizeForMatching' transform manipulation
            to image this removes the last transform from the transformList, then if
            there are more than 3 transforms continues to remove transforms until
            there are exactly 3.  Then assumes the image will be near 0,0 with a
            width/height that is about equal to the raw image width/height. This is
            true for Janelia's conventions for transformation alignment, but use at
            your own risk. (default=True)
        excludeTransformsAfterLast: str or None
            alternative to normalizeForMatching, which uses transformLabels.  Will
            remove all transformations after the last transformation with this transform
            label. i.e. if all lens corrections have a 'lens' label.  Then this will
            remove all non-lens transformations from the list. This is more general
            than normalizeForMatching=true, but requires you have transform labels applied.
            (default = None)
        excludeFirstTransformAndAllAfter: str
            alternative to normalizeForMatching which finds the first transform in the list with a given label
            and then removes that transform and all transforms that follow it. i.e. if you had a compound list
            of transformations, and you had labelled the first non-local transform 'montage' then setting
            excludeFirstTransformAndAllAfter='montage' would remove that montage transform and any other
            transforms that you had applied after it. (default=None).
        excludeAllTransforms: bool
            alternative to normalizeForMatching which simply removes all transforms from the list.
            (default=False)
        stackChannels: str or None
            If specified, option to select which channel is used for the stack.
            (default=None)
        stack2Channels: str or None
            If specified, option to select which channel is used for stack2, if specified.
            (default=None)
    """
    # Loop through sections of each montage stack
    for (stack, z), tile_pairs in tqdm(df_pairs.groupby(['stack', 'z'])):

        # Group tile pairs into batches
        grouping = np.arange(len(tile_pairs)) // batch_size
        for _, batch in tqdm(tile_pairs.groupby(grouping), leave=False):

            # Set up `pointMatchClient` partial
            point_match_client_partial = partial(run_point_match_client,
                                                 stack=stack,
                                                 collection=match_collections[stack],
                                                 render=render,
                                                 **pointMatchClient_kwargs)

            # Create batch of tile pairs
            tp_batch = [[tuple(tp)] for tp in batch[['p.id', 'q.id']].values.tolist()]

            # Create corresponding batch of SIFT options (updating `firstCanvasPosition` arg)
            sift_options_batch = []
            for i in batch.index:
                # Create new instance of `SiftPointMatchOptions`
                sift_options = SiftPointMatchOptions(**sift_options.__dict__)
                # Update canvas position -- only parameter that is tile pair dependent
                sift_options.firstCanvasPosition = batch.loc[i, 'p.relativePosition']
                sift_options_batch.append(sift_options)

            # Run `pointMatchClient` on `N_cores`
            with WithPool(N_cores) as pool:
                pool.map(point_match_client_partial, zip(tp_batch, sift_options_batch))


def get_matches_within_section(stack, sectionId, match_collection, render, **kwargs):
    """Wrapper for renderapi.pointmatches.get_matches_within_group

    Parameters
    ----------
    sectionId : str
        Section name (aka `groupId` in `renderapi` terminology)
    match_collection : str
        Name of match collection
    render : `renderapi.render.RenderClient`
        `render-ws` instance

    Returns
    -------
    df_matches : `pd.DataFrame`
        DataFrame of point matches from a given section
    """
    # Get z value from sectionId
    z = get_section_z_value(stack=stack,
                            sectionId=sectionId,
                            render=render, **kwargs)
    # Get matches within section
    matches_json = get_matches_within_group(matchCollection=match_collection,
                                            groupId=sectionId,
                                            render=render, **kwargs)
    # Convert json data to DataFrame and add z value
    df = pd.json_normalize(matches_json)
    df['z'] = z
    return df.rename(columns={'matchCount': 'N'})


def get_matches_within_stack(stack, match_collection, render, **kwargs):
    """Collect point matches across each section in a stack

    Parameters
    ----------
    stack : str
        Stack name
    match_collection : str
        Name of match collection
    render : `renderapi.render.RenderClient`
        `render-ws` instance

    Returns
    -------
    df_matches : `pd.DataFrame`
        DataFrame of point matches from a given stack
    """
    # Initialize DataFrame for point matches
    df_matches = pd.DataFrame(columns=['stack', 'z',
                                       'pGroupId', 'pId', 'pc', 'pr',
                                       'qGroupId', 'qId', 'qc', 'qr'])

    # Get sectionIds per stack
    sectionIds = get_match_groupIds(matchCollection=match_collection,
                                    render=render, **kwargs)
    # Loop through sectionIds
    for sectionId in tqdm(sectionIds, leave=False):

        # Get matches within section
        df = get_matches_within_section(stack=stack,
                                        sectionId=sectionId,
                                        match_collection=match_collection,
                                        render=render, **kwargs)
        # Aggregate matches
        df_matches = pd.concat([df_matches, df])

    # Format DataFrame
    df_matches['stack'] = stack
    # Add row/col indices
    df_matches[['pc', 'pr']] = np.stack(df_matches['pId'].apply(lambda x:
                                        [int(i) for i in re.findall(r'\d+', x)[-2:]]))
    df_matches[['qc', 'qr']] = np.stack(df_matches['qId'].apply(lambda x:
                                        [int(i) for i in re.findall(r'\d+', x)[-2:]]))
    return df_matches
