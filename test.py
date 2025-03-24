import itertools
import pandas as pd
import numpy as np
import umap
import hdbscan
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Load training data
io = StorageHelperV2()
data = io.load(data_artifact.path.replace("_artifact", "_train.parquet"))

# Restrict to 50,000 rows
if len(data) > 50000:
    data = data.sample(n=50000, random_state=42)

# Prepare graph list
graph_list, session_ids = [], []
for index, nodes in enumerate(data.path):
    G = nx.Graph()
    G.add_nodes_from([(i - 1) for i in nodes])
    edges = [(nodes[i] - 1, nodes[i+1] - 1) for i in range(len(nodes) - 1)]
    G.add_edges_from(edges)
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    graph_list.append(G)
    session_ids.append(data.channel_visit_id.iloc[index])

# Define hyperparameters
graph2vec_params = {"dimensions": [4096], "wl_iterations": [100]}
umap_params = {"n_components": [10], "min_dist": [0.1], "n_neighbors": [10]}
hdbscan_params = {"min_cluster_size": [3], "min_samples": [5], "cluster_selection_epsilon": [0.5]}

# Track best model
best_score = float('-inf')
best_params = None
results = []

# Hyperparameter search
for (dimensions, wl_iterations) in itertools.product(*graph2vec_params.values()):
    model = Graph2Vec(dimensions=dimensions, wl_iterations=wl_iterations)
    model.fit(graph_list)
    embeddings = model.get_embedding()
    embedding_df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(dimensions)])
    embedding_df["session_id"] = session_ids

    for (n_components, min_dist, n_neighbors) in itertools.product(*umap_params.values()):
        umap_reducer = umap.UMAP(n_components=n_components, min_dist=min_dist,
                                 n_neighbors=n_neighbors, random_state=42)
        reduced_embeddings = umap_reducer.fit_transform(embedding_df.drop("session_id", axis=1))

        for (min_cluster_size, min_samples, cluster_selection_epsilon) in itertools.product(*hdbscan_params.values()):
            hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                  cluster_selection_epsilon=cluster_selection_epsilon)
            labels = hdb.fit_predict(reduced_embeddings)

            # Compute silhouette score for HDBSCAN clusters
            valid_mask = labels != -1
            valid_embeddings = reduced_embeddings[valid_mask]
            valid_labels = labels[valid_mask]

            if len(set(valid_labels)) > 1:
                hdbscan_silhouette = silhouette_score(valid_embeddings, valid_labels)
            else:
                hdbscan_silhouette = -1

            # Save HDBSCAN Clusters Plot
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette="viridis", s=50, alpha=0.7)
            plt.title("HDBSCAN Clustering Visualization")
            plt.savefig(model_artifact.path.replace("_artifact", "_hdbscan_clusters.png"))
            plt.close()

            # Hybrid Clustering: KMeans on HDBSCAN centroids
            hdbscan_clusters = {cid: valid_embeddings[valid_labels == cid].mean(axis=0) for cid in np.unique(valid_labels)}
            centroid_array = np.vstack(list(hdbscan_clusters.values()))
            kmeans = KMeans(n_clusters=5, random_state=42)
            super_labels = kmeans.fit_predict(centroid_array)

            hdb_to_kmeans = {cid: super_labels[i] for i, cid in enumerate(hdbscan_clusters.keys())}
            final_labels = np.full_like(labels, -1)
            for i, hdb_label in enumerate(labels):
                if hdb_label in hdb_to_kmeans:
                    final_labels[i] = hdb_to_kmeans[hdb_label]

            # Compute silhouette score for Hybrid Clusters
            if len(set(final_labels[final_labels != -1])) > 1:
                hybrid_silhouette = silhouette_score(reduced_embeddings[final_labels != -1], final_labels[final_labels != -1])
            else:
                hybrid_silhouette = -1

            # Save Hybrid Clusters Plot
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=final_labels, palette="coolwarm", s=50, alpha=0.7)
            plt.title("Hybrid Clustering (HDBSCAN + KMeans)")
            plt.savefig(model_artifact.path.replace("_artifact", "_hybrid_clusters.png"))
            plt.close()

            results.append({
                "hdbscan_silhouette": hdbscan_silhouette,
                "hybrid_silhouette": hybrid_silhouette
            })

            if hybrid_silhouette > best_score:
                best_score = hybrid_silhouette
                best_embeddings = reduced_embeddings
                best_labels = final_labels

# Save results
results_df = pd.DataFrame(results)
io.save(data=results_df, path=model_artifact.path.replace("_artifact", "_hyperparam_results.parquet"))

plot_df = pd.DataFrame(best_embeddings[:, :2], columns=["dim_0", "dim_1"])
plot_df["cluster_label"] = best_labels
io.save(data=plot_df, path=model_artifact.path.replace("_artifact", "_hybrid_clusters.parquet"))

hdbscan_df = pd.DataFrame(reduced_embeddings[:, :2], columns=["dim_0", "dim_1"])
hdbscan_df["cluster_label"] = labels
io.save(data=hdbscan_df, path=model_artifact.path.replace("_artifact", "_hdbscan_clusters.parquet"))

# Model Lineage Tracking
if not init_kfp_variables(kfp_vars=kfp_vars):
    lineage_client = LineageClient(**lineage_dict)

    # Register model
    lineage_client.register_model(algorithm_name="Graph2Vec-UMAP-HDBSCAN+KMeans", package_name="Graph Embedding & Clustering",
                                  **user_dict, **git_dict)

    # Register artifacts
    lineage_client.register_model_artifact(artifact_name="hdbscan_clusters",
                                           artifact_uri=model_artifact.uri.replace("_artifact", "_hdbscan_clusters.parquet"))
    lineage_client.register_model_artifact(artifact_name="hybrid_clusters",
                                           artifact_uri=model_artifact.uri.replace("_artifact", "_hybrid_clusters.parquet"))
    lineage_client.register_model_artifact(artifact_name="hyperparam_results",
                                           artifact_uri=model_artifact.uri.replace("_artifact", "_hyperparam_results.parquet"))

print("✅ Training complete. Best silhouette score:", best_score)







import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

io = StorageHelperV2()

# Load clustered data
plot_df = io.load(model_artifact.path.replace("_artifact", "_hybrid_clusters.parquet"))
hdbscan_df = io.load(model_artifact.path.replace("_artifact", "_hdbscan_clusters.parquet"))

# Compute silhouette scores
valid_hdbscan = hdbscan_df["cluster_label"] != -1
hdbscan_silhouette = silhouette_score(hdbscan_df[["dim_0", "dim_1"]][valid_hdbscan], hdbscan_df["cluster_label"][valid_hdbscan]) if len(set(hdbscan_df["cluster_label"][valid_hdbscan])) > 1 else None

valid_hybrid = plot_df["cluster_label"] != -1
hybrid_silhouette = silhouette_score(plot_df[["dim_0", "dim_1"]][valid_hybrid], plot_df["cluster_label"][valid_hybrid]) if len(set(plot_df["cluster_label"][valid_hybrid])) > 1 else None

print(f"HDBSCAN Silhouette Score: {hdbscan_silhouette:.4f}")
print(f"Hybrid Clustering Silhouette Score: {hybrid_silhouette:.4f}")

# Save evaluation results
df_results = pd.DataFrame({"hdbscan_silhouette": [hdbscan_silhouette], "hybrid_silhouette": [hybrid_silhouette]})
io.save(data=df_results, path=model_artifact.path.replace("_artifact", "_hybrid_eval_summary.csv"))



















