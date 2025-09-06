from libpysal.weights import Queen
from esda import G_Local

def run_getis_ord_analysis(gdf, column):
    """
    Perform Getis-Ord G* local hotspot analysis on a given column of a GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing county geometries and relevant attributes.
    column : str
        Column name on which to perform the Getis-Ord G* analysis.

    Returns
    -------
    geopandas.GeoDataFrame
        Original GeoDataFrame with added columns:
        - G_Statistic_<column>
        - Z_Scores_<column>
        - P_Values_<column>
    """

    # Filter to CONUS counties (exclude AK, HI, PR, territories)
    exclude_states = {"02", "15", "72", "60", "66", "69", "78"}
    gdf = gdf[~gdf["STATEFP"].isin(exclude_states)].copy()

    # Define Queen contiguity-based spatial weights
    w = Queen.from_dataframe(gdf, use_index=True)
    w.transform = "r"  # Row-standardized weights

    # Fill missing values in the target column
    gdf[column] = gdf[column].fillna(0)

    # Run Getis-Ord G* analysis
    g_local = G_Local(gdf[column].values, w)

    # Add results back into GeoDataFrame
    gdf[f'G_Statistic_{column}'] = g_local.Gs
    gdf[f'Z_Scores_{column}'] = g_local.z_sim
    gdf[f'P_Values_{column}'] = g_local.p_sim

    return gdf
