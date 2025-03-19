import itertools
import pandas as pd
import umap
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Define hyperparameter grids for Graph2Vec, UMAP, and HDBSCAN
graph2vec_params = {
    "dimensions": [32, 64, 128],  
    "wl_iterations": [2, 3, 5]  
}

umap_params = {
    "n_components": [5, 10, 15],  
    "min_dist": [0.1, 0.5, 0.8],  
    "n_neighbors": [10, 15, 30]  
}

hdbscan_params = {
    "min_cluster_size": [5, 10, 15], 
    "min_samples": [5, 10, 15], 
    "cluster_selection_epsilon": [0.1, 0.5, 1.0]
}

# Track the best score and parameters
best_score = float('-inf')
best_params = None
results = []

# Iterate over Graph2Vec hyperparameters
for graph2vec_setting in itertools.product(*graph2vec_params.values()):
    dimensions, wl_iterations = graph2vec_setting
    print(f"Testing Graph2Vec with dimensions={dimensions}, wl_iterations={wl_iterations}")

    # Train Graph2Vec
    model = Graph2Vec(dimensions=dimensions, wl_iterations=wl_iterations)
    model.fit(graph_list)
    embeddings = model.get_embedding()

    # Iterate over UMAP and HDBSCAN hyperparameters
    for umap_setting in itertools.product(*umap_params.values()):
        for hdbscan_setting in itertools.product(*hdbscan_params.values()):
            umap_n_components, umap_min_dist, umap_n_neighbors = umap_setting
            hdb_min_cluster_size, hdb_min_samples, hdb_cluster_selection_epsilon = hdbscan_setting

            print(f"Testing UMAP: n_components={umap_n_components}, min_dist={umap_min_dist}, n_neighbors={umap_n_neighbors}")
            print(f"Testing HDBSCAN: min_cluster_size={hdb_min_cluster_size}, min_samples={hdb_min_samples}, cluster_selection_epsilon={hdb_cluster_selection_epsilon}")

            # Apply UMAP
            umap_reducer = umap.UMAP(n_components=umap_n_components, 
                                    min_dist=umap_min_dist, 
                                    n_neighbors=umap_n_neighbors, 
                                    random_state=42)
            reduced_embeddings = umap_reducer.fit_transform(embeddings)

            # Apply HDBSCAN
            hdb = hdbscan.HDBSCAN(min_cluster_size=hdb_min_cluster_size, 
                                  min_samples=hdb_min_samples, 
                                  cluster_selection_epsilon=hdb_cluster_selection_epsilon)
            labels = hdb.fit_predict(reduced_embeddings)

            # Evaluate clustering with silhouette score
            if len(set(labels)) > 1:  # Avoid silhouette score error with single clusters
                silhouette = silhouette_score(reduced_embeddings, labels)
            else:
                silhouette = -1  # Assign a poor score if only one cluster

            print(f"Silhouette Score: {silhouette}")

            # Store results
            results.append({
                "graph2vec_dimensions": dimensions,
                "graph2vec_wl_iterations": wl_iterations,
                "umap_n_components": umap_n_components,
                "umap_min_dist": umap_min_dist,
                "umap_n_neighbors": umap_n_neighbors,
                "hdb_min_cluster_size": hdb_min_cluster_size,
                "hdb_min_samples": hdb_min_samples,
                "hdb_cluster_selection_epsilon": hdb_cluster_selection_epsilon,
                "silhouette_score": silhouette
            })

            # Update best parameters
            if silhouette > best_score:
                best_score = silhouette
                best_params = {
                    "graph2vec": {
                        "dimensions": dimensions,
                        "wl_iterations": wl_iterations
                    },
                    "umap": {
                        "n_components": umap_n_components,
                        "min_dist": umap_min_dist,
                        "n_neighbors": umap_n_neighbors
                    },
                    "hdbscan": {
                        "min_cluster_size": hdb_min_cluster_size,
                        "min_samples": hdb_min_samples,
                        "cluster_selection_epsilon": hdb_cluster_selection_epsilon
                    }
                }
                best_embeddings = reduced_embeddings
                best_labels = labels

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results and best model artifacts
io.save(
    data=results_df,
    path=model_artifact.path.replace("_artifact", "_graph2vec_umap_hdbscan_results.parquet"),
)

best_model_data = pd.DataFrame(best_embeddings, columns=[f"dim_{i}" for i in range(best_embeddings.shape[1])])
best_model_data["cluster_label"] = best_labels

io.save(
    data=best_model_data,
    path=model_artifact.path.replace("_artifact", "_best_graph2vec_umap_hdbscan_model.parquet"),
)

# Register model artifact with lineage tracking
lineage_client.register_model_artifact(
    artifact_name="best_graph2vec_umap_hdbscan_model",
    artifact_uri=model_artifact.uri.replace("_artifact", "_best_graph2vec_umap_hdbscan_model.parquet"),
)

lineage_client.register_model_artifact(
    artifact_name="graph2vec_umap_hdbscan_tuning_results",
    artifact_uri=model_artifact.uri.replace("_artifact", "_graph2vec_umap_hdbscan_results.parquet"),
)

print("Best parameters found:", best_params)
