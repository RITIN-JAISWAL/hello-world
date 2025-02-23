import pandas as pd

# -----------------------------------
# 1) Function to create journey paths
# -----------------------------------
def create_journey_paths(df):
    """
    Group by 'channel_visit_id' and create ordered paths
    based on 'page_number'.
    """
    journey_paths = (
        df.sort_values(by=['channel_visit_id', 'page_number'])
          .groupby('channel_visit_id')['page']
          .apply(list)
          .reset_index()
    )
    
    # Rename 'page' to 'path' to represent ordered list of pages
    journey_paths.rename(columns={'page': 'path'}, inplace=True)
    
    return journey_paths

# --------------------------------------------
# 2) Function to map page names to numbers
# --------------------------------------------
def map_pages_to_numbers(df, path_column='path'):
    """
    Map page names in paths to numerical values.
    """
    # Flatten all pages to get unique pages
    unique_pages = pd.Series([page for path in df[path_column] for page in path]).unique()
    
    # Create a mapping dictionary {page_name: number}
    page_to_number = {page: idx + 1 for idx, page in enumerate(unique_pages)}
    
    # Map pages in each journey path
    df[path_column] = df[path_column].apply(lambda path: [page_to_number[page] for page in path])
    
    return df, page_to_number

# --------------------------------------------
# 3) Sample DataFrame
# --------------------------------------------
data = {
    'channel_visit_id': [1, 1, 1, 2, 2, 3, 3],
    'page_number': [1, 2, 3, 1, 2, 1, 2],
    'page': ['Home', 'Products', 'Checkout', 'Home', 'Cart', 'Home', 'About']
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
print("\n" + "-"*50 + "\n")

# --------------------------------------------
# 4) Apply create_journey_paths
# --------------------------------------------
journey_paths_df = create_journey_paths(df)
print("Journey Paths DataFrame:")
print(journey_paths_df)
print("\n" + "-"*50 + "\n")

# --------------------------------------------
# 5) Apply map_pages_to_numbers
# --------------------------------------------
mapped_journeys_df, page_mapping = map_pages_to_numbers(journey_paths_df)
print("Mapped Journeys DataFrame:")
print(mapped_journeys_df)
print("\nPage Mapping:")
print(page_mapping)
