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





import pandas as pd
import numpy as np

# -----------------------------------------------------
# 1) Example DataFrame Setup (Simulated Example)
# -----------------------------------------------------
data = {
    'journey_name':      ['JourneyA','JourneyA','JourneyA','JourneyB','JourneyB','JourneyC','JourneyC'],
    'channel_visit_id':  [100,      100,       100,       200,       200,       300,       300],
    'page_number':       [1,        2,         3,         1,         2,         1,         2],
    'account_num':       [123,      123,       123,       456,       456,       789,       789],
    'page_referrer':     [None,     'Home',    'Splash',  None,      'Login',   None,      'Home'],
    'page':              ['Home',   'Splash',  'Products','Login',   'Dashboard','Home',   'Checkout'],
    'next_page':         ['Splash','Products', 'Exit','Dashboard','Exit','Checkout','Exit'],
    'time_spent_seconds':[10,       0.5,       5,         20,        15,        2,         0.3],
    'is_exit':           [False,    False,     False,     False,     False,     False,     True]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
print("-"*50)

# -----------------------------------------------------
# 2) Sort the DataFrame
#    We'll use 'journey_name', 'channel_visit_id', and 'page_number'
#    to ensure the visit order is correct.
# -----------------------------------------------------
df_sorted = df.sort_values(['journey_name','channel_visit_id','page_number'])
print("Sorted DataFrame:")
print(df_sorted)
print("-"*50)

# -----------------------------------------------------
# 3) Define the build_valid_journey function
#    We'll illustrate how to incorporate page_referrer (previous page),
#    page (current), next_page (future), time_spent_seconds, and is_exit.
# -----------------------------------------------------
def build_valid_journey(group):
    """
    Build a valid journey for each user/group.
    We'll create transitions of the form:
      (page_referrer -> page) if not exit,
      (page -> "Exit") if is_exit,
    skipping rows with time_spent_seconds < 1 (example).
    You could also incorporate next_page if desired.
    """
    journey = []

    for row in group.itertuples():
        # Convert columns to strings (in case of None)
        ref_page = str(row.page_referrer) if row.page_referrer else "None"
        current_page = str(row.page)
        next_pg = str(row.next_page)

        # Skip pages with very short time_spent_seconds (example logic)
        if row.time_spent_seconds < 1:
            continue

        # Check if this row is an exit
        is_exit = getattr(row, 'is_exit', False)

        if is_exit:
            # Mark the current page as leading to an exit
            journey.append((current_page, "Exit"))
        else:
            # Normal transition using page_referrer -> page
            # Alternatively, you could do (current_page -> next_page).
            journey.append((ref_page, current_page))

    return journey

# -----------------------------------------------------
# 4) Group by account_num and apply build_valid_journey
# -----------------------------------------------------
journeys = (
    df_sorted
    .groupby('account_num')
    .apply(build_valid_journey)
    .reset_index(name='journey')
)

print("Resulting 'journeys' DataFrame:")
print(journeys)






# Required Libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------------------------------
# 1️⃣ Sample DataFrame: Customer Journeys
# --------------------------------------------

data = {
    'journey_name': ['JourneyA', 'JourneyA', 'JourneyA', 'JourneyB', 'JourneyB', 'JourneyC', 'JourneyC'],
    'channel_visit_id': [100, 100, 100, 200, 200, 300, 300],
    'page_number': [1, 2, 3, 1, 2, 1, 2],
    'account_num': [123, 123, 123, 456, 456, 789, 789],
    'page_referrer': [None, 'Home', 'Splash', None, 'Login', None, 'Home'],
    'page': ['Home', 'Splash', 'Products', 'Login', 'Dashboard', 'Home', 'Checkout'],
    'next_page': ['Splash', 'Products', 'Exit', 'Dashboard', 'Exit', 'Checkout', 'Exit'],
    'time_spent_seconds': [10.0, 0.5, 5.0, 20.0, 15.0, 2.0, 0.3],
    'is_exit': [False, False, False, False, False, False, True]
}

df = pd.DataFrame(data)
print("Sample DataFrame:\n", df)

# --------------------------------------------
# 2️⃣ Build the Graph from Journeys
# --------------------------------------------

# Create Directed Graph
G = nx.DiGraph()

# Add edges with weights (time spent)
for idx, row in df.iterrows():
    referrer = row['page_referrer'] if row['page_referrer'] else 'Start'
    G.add_edge(referrer, row['page'], weight=row['time_spent_seconds'])

# Visualize the Graph
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G, seed=42)
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, width=weights)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)})
plt.title("Customer Journey Graph")
plt.show()

# --------------------------------------------
# 3️⃣ Create Transition Matrices
# --------------------------------------------

# Pages and mapping
pages = list(G.nodes())
page_indices = {page: idx for idx, page in enumerate(pages)}

# Initialize transition matrix
transition_matrix = np.zeros((len(df['account_num'].unique()), len(pages)**2))

# Build transition matrices for each account
for idx, account in enumerate(df['account_num'].unique()):
    user_df = df[df['account_num'] == account]
    user_matrix = np.zeros((len(pages), len(pages)))

    for _, row in user_df.iterrows():
        referrer = row['page_referrer'] if row['page_referrer'] else 'Start'
        i = page_indices[referrer]
        j = page_indices[row['page']]
        user_matrix[i, j] += 1

    # Flatten the matrix to a vector
    transition_matrix[idx, :] = user_matrix.flatten()

# --------------------------------------------
# 4️⃣ Scale and Apply Clustering (K-Means)
# --------------------------------------------

# Scale the data
scaler = StandardScaler()
scaled_matrix = scaler.fit_transform(transition_matrix)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_matrix)
labels = kmeans.labels_

# Display Clustering Results
print("\nClustering Results:")
for idx, account in enumerate(df['account_num'].unique()):
    print(f"Account: {account}, Cluster: {labels[idx]}")

# --------------------------------------------
# 5️⃣ Visualize Clustered Customer Journeys
# --------------------------------------------

# Color map for clusters
colors = ['red', 'green', 'blue']
account_clusters = {acc: colors[labels[idx]] for idx, acc in enumerate(df['account_num'].unique())}

# Plot user journeys with cluster colors
plt.figure(figsize=(10, 7))
for idx, account in enumerate(df['account_num'].unique()):
    user_df = df[df['account_num'] == account]
    user_G = nx.DiGraph()

    for _, row in user_df.iterrows():
        ref = row['page_referrer'] if row['page_referrer'] else 'Start'
        user_G.add_edge(ref, row['page'])

    nx.draw(user_G, pos, with_labels=True, node_size=2000, node_color=account_clusters[account], font_size=10)

plt.title("Clustered Customer Journeys")
plt.show()

# --------------------------------------------
# ✅ Done! You've clustered customer journeys.
# --------------------------------------------








import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import networkx as nx

# Step 1: Prepare the DataFrame
data = {
    'journey_name': ['JourneyA', 'JourneyA', 'JourneyA', 'JourneyB', 'JourneyB', 'JourneyC', 'JourneyC'],
    'channel_visit_id': [100, 100, 100, 200, 200, 300, 300],
    'page_number': [1, 2, 3, 1, 2, 1, 2],
    'account_num': [123, 123, 123, 456, 456, 789, 789],
    'page_referrer': [None, 'Home', 'Splash', None, 'Login', None, 'Home'],
    'page': ['Home', 'Splash', 'Products', 'Login', 'Dashboard', 'Home', 'Checkout'],
    'next_page': ['Splash', 'Products', 'Exit', 'Dashboard', 'Exit', 'Checkout', 'Exit'],
    'time_spent_seconds': [10.0, 0.5, 5.0, 20.0, 15.0, 2.0, 0.3],
    'is_exit': [False, False, False, False, False, False, True]
}

df = pd.DataFrame(data)

# Step 2: Encode page names to integers
le = LabelEncoder()
df['page_encoded'] = le.fit_transform(df['page'])

# Step 3: Build sequences per account
sequences = df.groupby('account_num')['page_encoded'].apply(list).tolist()
account_ids = df['account_num'].unique()

# Pad sequences to have the same length
max_len = max(len(seq) for seq in sequences)
sequences_padded = [seq + [0]*(max_len - len(seq)) for seq in sequences]

# Convert to 3D numpy array for tslearn
from tslearn.preprocessing import TimeSeriesResampler
X = np.array(sequences_padded).reshape(len(sequences_padded), max_len, 1)

# Step 4: Apply DTW-based KMeans Clustering
n_clusters = 2  # Adjust as needed
model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
labels = model.fit_predict(X)

# Step 5: Visualize Clustered Sequences
plt.figure(figsize=(10, 5))
for yi in range(n_clusters):
    plt.subplot(n_clusters, 1, yi + 1)
    for xx in X[labels == yi]:
        plt.plot(xx.ravel(), "k-", alpha=0.4)
    plt.title(f"Cluster {yi + 1}")

plt.tight_layout()
plt.show()

# Print cluster assignments
for idx, label in enumerate(labels):
    print(f"Account: {account_ids[idx]}, Cluster: {label}")

# ------------------------------------------------------------
# Optional Step 6: Build Markov Chains for each cluster
# ------------------------------------------------------------
def build_markov_chain(sequence):
    transitions = {}
    for i in range(len(sequence) - 1):
        current_page = sequence[i]
        next_page = sequence[i + 1]
        if current_page not in transitions:
            transitions[current_page] = {}
        transitions[current_page][next_page] = transitions[current_page].get(next_page, 0) + 1

    # Normalize probabilities
    for current_page, next_pages in transitions.items():
        total = sum(next_pages.values())
        for next_page in next_pages:
            transitions[current_page][next_page] /= total

    return transitions

# Build Markov Chains per cluster
for cluster_id in range(n_clusters):
    print(f"\nMarkov Chain for Cluster {cluster_id + 1}:")
    cluster_sequences = [sequences[i] for i in range(len(labels)) if labels[i] == cluster_id]
    combined_sequence = [page for seq in cluster_sequences for page in seq]
    markov_chain = build_markov_chain(combined_sequence)

    # Visualize as graph
    G = nx.DiGraph()
    for current_page, next_pages in markov_chain.items():
        for next_page, prob in next_pages.items():
            G.add_edge(le.inverse_transform([current_page])[0],
                       le.inverse_transform([next_page])[0],
                       weight=round(prob, 2))

    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color='skyblue')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(f"Markov Chain for Cluster {cluster_id + 1}")
    plt.show()






























import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gc
import matplotlib.pyplot as plt

# ============================
# Dataset Preparation
# ============================

class MatrixDataset(Dataset):
    def __init__(self, df, n_nodes):
        super().__init__()
        self.df = df
        self.n_nodes = n_nodes

    def get_adj(self, path):
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        adj[path[0], 0] = 1  # From source
        adj[self.n_nodes - 1, path[-1]] = 1  # To sink
        adj[self.n_nodes - 1, self.n_nodes - 1] = 1  # Sink absorbs

        for i in range(len(path) - 1):
            adj[path[i + 1], path[i]] = 1

        col_sums = adj.sum(axis=0)
        col_sums[col_sums == 0] = 1e-6  # Avoid division by zero
        adj /= col_sums

        return np.nan_to_num(adj)

    def __getitem__(self, index):
        path = self.df.at[index, 'path']
        adj = self.get_adj(path)
        return torch.tensor(adj, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

# ============================
# Enhanced Model Definition
# ============================

class MatrixEmbedder(nn.Module):
    def __init__(self, n_nodes, embed_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_nodes ** 2, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(500, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(250, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(100, embed_dim)
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.layers(x)

# ============================
# Distance Functions
# ============================

def dme(mat1, mat2, vector):
    mat = np.matmul(mat1.T, mat2)
    v = np.matmul(mat, vector)
    out = np.matmul(vector.T, v)
    return out[0, 0]

def markov_distance(mata, matb):
    v1 = np.ones((mata.shape[0], 1))
    out = dme(mata, matb, v1) - 0.5 * dme(mata, mata, v1) - 0.5 * dme(matb, matb, v1)
    return out

# ============================
# Training Loop
# ============================

def train_model(mapped_journeys_df, embed_dim=20, batch_size=16, epochs=50, lr=1e-3):
    n_nodes = mapped_journeys_df['path'].apply(max).max() + 2
    dataset = MatrixDataset(df=mapped_journeys_df, n_nodes=n_nodes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    embedder = MatrixEmbedder(n_nodes=n_nodes, embed_dim=embed_dim)
    optimizer = AdamW(embedder.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    losses = []

    for ep in range(epochs):
        epoch_loss = 0
        print(f"Epoch {ep + 1}/{epochs}")
        for batch in dataloader:
            optimizer.zero_grad()

            # Calculate Markov distances
            distances = []
            with torch.no_grad():
                for i in range(batch.size(0)):
                    for j in range(i + 1, batch.size(0)):
                        d = markov_distance(batch[i].cpu().numpy(), batch[j].cpu().numpy())
                        distances.append(d)
            distances = torch.tensor(distances, dtype=torch.float32)

            # Embed batch and calculate loss
            batch_flat = batch.view(batch.size(0), -1)
            embeddings = embedder(batch_flat)
            left_vecs = embeddings[:-1]
            right_vecs = embeddings[1:]
            diff = left_vecs - right_vecs

            loss = torch.pow(((diff ** 2).sum(axis=1) - distances ** 2), 2).sum()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(embedder.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            # Memory cleanup
            del batch, left_vecs, right_vecs, diff, distances
            gc.collect()
            torch.cuda.empty_cache()

        epoch_loss /= len(dataloader)
        scheduler.step(epoch_loss)
        losses.append(epoch_loss)
        print(f"Loss: {epoch_loss:.4f}")

    print("Training completed successfully!")
    return embedder, losses

# ============================
# Training Execution
# ============================

# Example: Assuming mapped_journeys_df is already created
# embedder, losses = train_model(mapped_journeys_df)

# ============================
# Visualization
# ============================

def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()

# plot_losses(losses)



















import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gc
import matplotlib.pyplot as plt

# ============================
# Dataset Preparation
# ============================

class MatrixDataset(Dataset):
    def __init__(self, df, n_nodes):
        super().__init__()
        self.df = df
        self.n_nodes = n_nodes

    def get_adj(self, path):
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        adj[path[0], 0] = 1  # From source
        adj[self.n_nodes - 1, path[-1]] = 1  # To sink
        adj[self.n_nodes - 1, self.n_nodes - 1] = 1  # Sink absorbs

        for i in range(len(path) - 1):
            adj[path[i + 1], path[i]] = 1

        col_sums = adj.sum(axis=0)
        col_sums[col_sums == 0] = 1e-6  # Avoid division by zero
        adj /= col_sums

        return np.nan_to_num(adj)

    def __getitem__(self, index):
        path = self.df.at[index, 'path']
        adj = self.get_adj(path)
        return torch.tensor(adj, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

# ============================
# Model Definition
# ============================

class MatrixEmbedder(nn.Module):
    def __init__(self, n_nodes, embed_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_nodes ** 2, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(500, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(250, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(100, embed_dim)
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.layers(x)

# ============================
# Distance Functions
# ============================

def dme(mat1, mat2, vector):
    mat = np.matmul(mat1.T, mat2)
    v = np.matmul(mat, vector)
    out = np.matmul(vector.T, v)
    return out[0, 0]

def markov_distance(mata, matb):
    v1 = np.ones((mata.shape[0], 1))
    out = dme(mata, matb, v1) - 0.5 * dme(mata, mata, v1) - 0.5 * dme(matb, matb, v1)
    return out

# ============================
# Optimized Training Loop
# ============================

def train_model(mapped_journeys_df, embed_dim=20, batch_size=8, epochs=50, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_nodes = mapped_journeys_df['path'].apply(max).max() + 2
    dataset = MatrixDataset(df=mapped_journeys_df, n_nodes=n_nodes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    embedder = MatrixEmbedder(n_nodes=n_nodes, embed_dim=embed_dim).to(device)
    optimizer = AdamW(embedder.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    losses = []

    scaler = torch.cuda.amp.GradScaler()  # Mixed precision

    for ep in range(epochs):
        epoch_loss = 0
        print(f"Epoch {ep + 1}/{epochs}")

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():  # Mixed precision
                batch_flat = batch.view(batch.size(0), -1)
                embeddings = embedder(batch_flat)

                # Calculate distances within the batch
                distances = torch.cdist(embeddings, embeddings, p=2)

                # Loss: minimize intra-batch distances
                loss = torch.mean(distances)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(embedder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # Memory cleanup
            del batch, embeddings, distances, loss
            gc.collect()
            torch.cuda.empty_cache()

        epoch_loss /= len(dataloader)
        scheduler.step(epoch_loss)
        losses.append(epoch_loss)
        print(f"Loss: {epoch_loss:.4f}")

    print("Training completed successfully!")
    return embedder, losses

# ============================
# Visualization
# ============================

def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()

# ============================
# Example Usage
# ============================

# Sample DataFrame
import pandas as pd

data = {
    'channel_visit_id': [1001, 1002, 1003],
    'path': [
        [1, 2, 3, 4, 5],
        [2, 3, 5, 6, 7],
        [1, 3, 4, 6, 8]
    ]
}

mapped_journeys_df = pd.DataFrame(data)

# Run training
embedder, losses = train_model(mapped_journeys_df, embed_dim=20, batch_size=4, epochs=10, lr=1e-3)

# Plot the loss
plot_losses(losses)





















from dtaidistance import dtw
from sklearn_extra.cluster import KMedoids
import numpy as np

# Step 1: Prepare sequences
sequences = mapped_journeys_df['path'].tolist()

# Step 2: Compute pairwise DTW distances
distance_matrix = np.zeros((len(sequences), len(sequences)))

for i in range(len(sequences)):
    for j in range(i+1, len(sequences)):
        distance = dtw.distance(sequences[i], sequences[j])
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

# Step 3: Apply K-Medoids clustering
kmedoids = KMedoids(n_clusters=5, metric='precomputed', random_state=42)
labels = kmedoids.fit_predict(distance_matrix)

# Add labels to dataframe
mapped_journeys_df['cluster'] = labels
print(mapped_journeys_df.head())







from karateclub import Graph2Vec
from sklearn.cluster import KMeans
import networkx as nx

# Step 1: Convert paths to graphs
graphs = []
for path in mapped_journeys_df['path']:
    G = nx.DiGraph()
    for i in range(len(path) - 1):
        G.add_edge(path[i], path[i+1])
    graphs.append(G)

# Step 2: Graph2Vec Embedding
model = Graph2Vec(dimensions=128)
model.fit(graphs)
embeddings = model.get_embedding()

# Step 3: KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Add labels to dataframe
mapped_journeys_df['cluster'] = labels
print(mapped_journeys_df.head())





import matplotlib.pyplot as plt

# MDS Projection
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_features = mds.fit_transform(distance_matrix)

# Plot clusters
plt.figure(figsize=(10, 6))
for cluster_id in np.unique(labels):
    indices = labels == cluster_id
    plt.scatter(mds_features[indices, 0], mds_features[indices, 1], label=f'Cluster {cluster_id}')

plt.title("MDS Projection of DTW + K-Medoids Clusters")
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
plt.legend()
plt.show()



def cluster_stats(distance_matrix, labels):
    clusters = np.unique(labels)
    intra_dists = []
    inter_dists = []

    # Intra-cluster distances
    for cluster in clusters:
        indices = np.where(labels == cluster)[0]
        intra = np.mean([distance_matrix[i, j] for i in indices for j in indices if i != j])
        intra_dists.append(intra)

    # Inter-cluster distances
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            indices_i = np.where(labels == clusters[i])[0]
            indices_j = np.where(labels == clusters[j])[0]
            inter = np.mean([distance_matrix[p, q] for p in indices_i for q in indices_j])
            inter_dists.append(inter)

    print(f"Average Intra-cluster Distance: {np.mean(intra_dists):.3f}")
    print(f"Average Inter-cluster Distance: {np.mean(inter_dists):.3f}")

cluster_stats(distance_matrix, labels)









# ===============================
# Step 1: Import Required Libraries
# ===============================
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# Step 2: Assume You Have:
# ava_mapped: 2D data points (UMAP projection)
# labels: Cluster labels assigned (e.g., from KMeans, DBSCAN, etc.)
# ===============================

# Example:
# ava_mapped = np.array([[2, 3], [5, 6], [3, 2], [8, 7], [7, 5]])  # Replace with your data
# labels = np.array([0, 1, 0, 1, 1])  # Replace with your cluster labels

# ===============================
# Step 3: Calculate Silhouette Score
# ===============================
def calculate_silhouette(ava_mapped, labels):
    # Compute average silhouette score
    avg_score = silhouette_score(ava_mapped, labels)
    print(f'Average Silhouette Score: {avg_score:.3f}')
    return avg_score

# ===============================
# Step 4: Plot Silhouette Analysis
# ===============================
def plot_silhouette(ava_mapped, labels):
    n_clusters = len(np.unique(labels))
    silhouette_vals = silhouette_samples(ava_mapped, labels)
    avg_score = silhouette_score(ava_mapped, labels)
    
    y_lower = 10
    plt.figure(figsize=(10, 6))

    for i in range(n_clusters):
        # Aggregate silhouette scores for samples in cluster i
        ith_cluster_silhouette_vals = silhouette_vals[labels == i]
        ith_cluster_silhouette_vals.sort()

        size_cluster_i = ith_cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_vals,
                          facecolor=color, edgecolor=color, alpha=0.7)

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # Add space between clusters

    # Plot formatting
    plt.axvline(x=avg_score, color="red", linestyle="--", label=f'Avg Silhouette Score: {avg_score:.3f}')
    plt.xlabel("Silhouette Coefficient Values")
    plt.ylabel("Cluster Label")
    plt.title("Silhouette Plot for the Clusters")
    plt.legend(loc='best')
    plt.show()

# ===============================
# Step 5: Run the Silhouette Evaluation
# ===============================
# Calculate silhouette score
avg_score = calculate_silhouette(ava_mapped, labels)

# Plot silhouette analysis
plot_silhouette(ava_mapped, labels)






from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import torch
import numpy as np
from tqdm import tqdm

# ---------------------------
# Step 1: Extract Embeddings
# ---------------------------
def extract_embeddings(embedder, dataloader):
    embedder.eval()  # Set model to evaluation mode
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Embeddings"):
            embeddings = embedder(batch.float())
            all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)

# ---------------------------
# Step 2: Perform Clustering
# ---------------------------
def perform_kmeans(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels

# ---------------------------
# Step 3: Compute Silhouette Score
# ---------------------------
def evaluate_clustering(embeddings, labels):
    score = silhouette_score(embeddings, labels)
    print(f"Silhouette Score: {score:.4f}")
    return score

# ---------------------------
# Execute the Evaluation
# ---------------------------
# Assuming 'embedder' is trained and 'dataloader' is ready
embeddings = extract_embeddings(embedder, dataloader)

# Try different cluster numbers
best_score = -1
best_k = None

for n_clusters in range(2, 11):
    print(f"\nEvaluating for n_clusters = {n_clusters}")
    labels = perform_kmeans(embeddings, n_clusters=n_clusters)
    score = evaluate_clustering(embeddings, labels)
    
    if score > best_score:
        best_score = score
        best_k = n_clusters

print(f"\nBest Silhouette Score: {best_score:.4f} with n_clusters = {best_k}")























import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import gc

# ----------------------
# 1. Dataset Preparation
# ----------------------
class MatrixDataset(Dataset):
    def __init__(self, paths, n_nodes):
        self.paths = paths
        self.n_nodes = n_nodes

    def get_adj(self, path):
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        for i in range(len(path) - 1):
            adj[path[i], path[i+1]] += 1
        col_sums = adj.sum(axis=0)
        col_sums[col_sums == 0] = 1e-6  # Avoid division by zero
        adj = adj / col_sums
        adj = np.nan_to_num(adj)
        return adj

    def __getitem__(self, idx):
        path = self.paths[idx]
        adj = self.get_adj(path)
        return torch.tensor(adj, dtype=torch.float32)

    def __len__(self):
        return len(self.paths)

# ----------------------
# 2. Neural Network Embedder
# ----------------------
class MatrixEmbedder(nn.Module):
    def __init__(self, n_nodes, embed_dim):
        super(MatrixEmbedder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_nodes * n_nodes, 500),
            nn.Tanh(),
            nn.Linear(500, 250),
            nn.Tanh(),
            nn.Linear(250, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, embed_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)

# ----------------------
# 3. Distance Functions
# ----------------------
def dme(mat1, mat2, vector):
    mat = np.matmul(mat1.T, mat2)
    v = np.matmul(mat, vector)
    out = np.matmul(vector.T, v)
    return out[0, 0]

def markov_distance(mata, matb):
    v1 = np.ones((mata.shape[0], 1))
    return dme(mata, matb, v1) - 0.5 * dme(mata, mata, v1) - 0.5 * dme(matb, matb, v1)

# ----------------------
# 4. Data Generation
# ----------------------
# Example data: list of paths
np.random.seed(42)
n_samples = 100
n_nodes = 10
paths = [np.random.choice(n_nodes, size=np.random.randint(5, 10), replace=False).tolist() for _ in range(n_samples)]

dataset = MatrixDataset(paths, n_nodes)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ----------------------
# 5. Training Setup
# ----------------------
embed_dim = 20
embedder = MatrixEmbedder(n_nodes, embed_dim)
optim = Adam(embedder.parameters(), lr=0.001)
epochs = 50
losses = []

# ----------------------
# 6. Training Loop
# ----------------------
for ep in range(epochs):
    total_loss = 0
    for batch in dataloader:
        optim.zero_grad()
        
        # Pairwise distances
        batch_size = batch.size(0)
        left_indices = np.random.randint(0, batch_size, size=batch_size)
        right_indices = np.random.randint(0, batch_size, size=batch_size)
        
        left_batch = batch[left_indices]
        right_batch = batch[right_indices]
        
        # Forward pass
        left_vecs = embedder(left_batch)
        right_vecs = embedder(right_batch)
        
        # Compute pairwise distances
        diff = left_vecs - right_vecs
        distances = torch.norm(diff, dim=1)
        
        # Loss (Contrastive)
        loss = torch.mean(distances)
        loss.backward()
        optim.step()
        
        total_loss += loss.item()
    
    losses.append(total_loss / len(dataloader))
    print(f"Epoch {ep+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# ----------------------
# 7. Loss Plot
# ----------------------
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.show()

# ----------------------
# 8. Clustering and Silhouette Score
# ----------------------
# Generate embeddings for all data
all_vecs = []
with torch.no_grad():
    for batch in dataloader:
        embeddings = embedder(batch)
        all_vecs.append(embeddings)

all_vecs = torch.cat(all_vecs).numpy()

# KMeans Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(all_vecs)

# Silhouette Score
score = silhouette_score(all_vecs, labels)
print(f"Silhouette Score: {score:.4f}")

# ----------------------
# 9. Visualization (t-SNE)
# ----------------------
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
all_vecs_2d = tsne.fit_transform(all_vecs)

plt.figure(figsize=(8,6))
plt.scatter(all_vecs_2d[:,0], all_vecs_2d[:,1], c=labels, cmap='viridis', s=50)
plt.title("t-SNE Visualization of Path Embeddings")
plt.colorbar()
plt.show()







from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
import umap

# ---------------------------
# Step 1: Extract Embeddings
# ---------------------------
def extract_embeddings(embedder, dataloader):
    embedder.eval()  # Set model to evaluation mode
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Embeddings"):
            embeddings = embedder(batch.float())
            all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)

# ---------------------------
# Step 2: Perform KMeans Clustering
# ---------------------------
def perform_kmeans(embeddings, n_clusters=9):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans

# ---------------------------
# Step 3: Clustering Evaluation Metrics
# ---------------------------
def evaluate_clustering(embeddings, labels):
    silhouette = silhouette_score(embeddings, labels)
    calinski = calinski_harabasz_score(embeddings, labels)
    davies = davies_bouldin_score(embeddings, labels)

    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Calinski-Harabasz Index: {calinski:.4f}")
    print(f"Davies-Bouldin Index: {davies:.4f}")

    return {
        "silhouette_score": silhouette,
        "calinski_harabasz_score": calinski,
        "davies_bouldin_score": davies
    }

# ---------------------------
# Step 4: Visualization
# ---------------------------
def visualize_clusters(embeddings, labels, method='umap'):
    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced_embeddings = reducer.fit_transform(embeddings)
        title = "PCA Visualization of Clusters"
    elif method == 'umap':
        reducer = umap.UMAP(random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        title = "UMAP Visualization of Clusters"
    else:
        raise ValueError("Invalid reduction method. Choose 'pca' or 'umap'.")

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster Label')
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()

# ---------------------------
# Execute the Clustering and Evaluation
# ---------------------------

# Step 1: Extract embeddings
embeddings = extract_embeddings(embedder, dataloader)

# Step 2: Perform clustering with best n_clusters = 9
labels, kmeans_model = perform_kmeans(embeddings, n_clusters=9)

# Step 3: Evaluate clustering performance
evaluation_metrics = evaluate_clustering(embeddings, labels)

# Step 4: Visualize the clusters using UMAP
visualize_clusters(embeddings, labels, method='umap')

# Optionally, visualize using PCA as well
visualize_clusters(embeddings, labels, method='pca')


import pickle

# Save the trained KMeans model
with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans_model, f)

# Save cluster labels
np.save("cluster_labels.npy", labels)




import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from dtaidistance import dtw
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# ==========================
# Step 1: Define the Dataset Class
# ==========================
class MatrixDataset(Dataset):
    def __init__(self, df, n_nodes):
        """
        Initializes the dataset with a DataFrame containing paths and the number of nodes.
        :param df: DataFrame containing a 'path' column with sequences.
        :param n_nodes: Total number of unique nodes in the dataset.
        """
        self.df = df
        self.n_nodes = n_nodes

    def get_adj(self, path):
        """
        Converts a given path into an adjacency matrix representation.
        """
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)

        # Source node
        adj[path[0], 0] = 1
        # Destination node
        adj[self.n_nodes - 1, path[-1]] = 1
        adj[self.n_nodes - 1, self.n_nodes - 1] = 1  # Sink node absorbs

        # Fill adjacency matrix along the path
        for i in range(len(path) - 1):
            adj[path[i + 1], path[i]] = 1

        # Normalize adjacency matrix
        col_sums = adj.sum(axis=0)
        col_sums[col_sums == 0] = 1e-6  # Avoid division by zero
        adj = adj / col_sums

        return adj

    def __getitem__(self, index):
        path = self.df.iloc[index]['path']
        adj = self.get_adj(path)
        return torch.tensor(adj, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

# ==========================
# Step 2: Define the Model
# ==========================
class MatrixEmbedder(nn.Module):
    def __init__(self, n_nodes, embed_dim):
        super().__init__()
        self.layer1 = nn.Linear(n_nodes ** 2, 500, bias=True)
        self.layer2 = nn.Linear(500, 250, bias=True)
        self.layer3 = nn.Linear(250, 100, bias=True)
        self.layer4 = nn.Linear(100, 50, bias=True)
        self.layer5 = nn.Linear(50, embed_dim, bias=True)

    def forward(self, x):
        x = x.flatten(start_dim=1, end_dim=-1)
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        x = torch.tanh(self.layer4(x))
        x = torch.tanh(self.layer5(x))  # Activation isn't necessary but helps
        return x

# ==========================
# Step 3: Distance Calculation (Markov-based)
# ==========================
def dme(mat1, mat2, vector):
    mat = np.matmul(mat1.T, mat2)
    v = np.matmul(mat, vector)
    out = np.matmul(vector.T, v)
    return out[0, 0]

def markov_distance(mata, matb):
    v1 = np.ones((mata.shape[0], 1))
    out = dme(mata, matb, v1) - 0.5 * dme(mata, mata, v1) - 0.5 * dme(matb, matb, v1)
    return out

# ==========================
# Step 4: Training Loop
# ==========================
def train_model(dataset, embedder, optimizer, epochs, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses = []
    distances_captured = []

    for ep in range(epochs):
        print(f"Epoch {ep+1}/{epochs}")
        for ib, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Generate pairs
            n_per_example = 2
            right = np.random.randint(0, batch.shape[0], n_per_example * batch.shape[0])
            left = np.repeat(np.arange(0, batch.shape[0]), n_per_example)
            index_pairs = np.stack([left, right]).T

            # Compute distances
            distances = []
            with torch.no_grad():
                for (l, r) in index_pairs:
                    d = markov_distance(batch[l].cpu().numpy(), batch[r].cpu().numpy())
                    distances.append(d)
                    distances_captured.append(d)

            distances = torch.tensor(distances, dtype=torch.float32)

            # Embeddings
            left_vecs = embedder(batch[left])
            right_vecs = embedder(batch[right])
            diff = left_vecs - right_vecs

            # Compute Loss
            loss = torch.pow(((diff ** 2).sum(axis=1) - distances ** 2), 2).sum()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {ep+1}/{epochs}, Loss: {loss.item()}")
    return losses, distances_captured

# ==========================
# Step 5: Load Data (df234)
# ==========================
# Sample structure for df234
data = {
    'path': [[1, 2, 3, 4], [2, 3, 5], [1, 4, 6], [3, 5, 7, 8]]
}
df234 = pd.DataFrame(data)

# Compute n_nodes dynamically
n_nodes = max([max(path) for path in df234['path']]) + 2  # Adding 2 for source and sink

# Initialize dataset and dataloader
dataset = MatrixDataset(df234, n_nodes=n_nodes)

# ==========================
# Step 6: Train the Model
# ==========================
embed_dim = 20
embedder = MatrixEmbedder(n_nodes=n_nodes, embed_dim=embed_dim)
optimizer = optim.Adam(embedder.parameters(), lr=0.001)

epochs = 100
batch_size = 16
losses, distances_captured = train_model(dataset, embedder, optimizer, epochs, batch_size)

# ==========================
# Step 7: Embedding & Clustering
# ==========================
# Convert dataset to embeddings
all_vecs = []
for ib, batch in tqdm(enumerate(DataLoader(dataset, batch_size=batch_size)), total=len(dataset) // batch_size):
    vecs = embedder(batch.to(torch.float32))
    all_vecs.append(vecs.detach().numpy())

all_vecs_array = np.concatenate(all_vecs, axis=0)

# Save & Load Embeddings
with open("all_vecs_array.pkl", "wb") as f:
    pickle.dump(all_vecs_array, f)

with open("all_vecs_array.pkl", "rb") as f:
    all_vecs_array = pickle.load(f)

# UMAP Reduction
mapper = umap.UMAP()
ava_mapped = mapper.fit_transform(all_vecs_array)

# K-Medoids Clustering
distance_matrix = np.zeros((len(ava_mapped), len(ava_mapped)))
for i in range(len(ava_mapped)):
    for j in range(i + 1, len(ava_mapped)):
        distance = dtw.distance(ava_mapped[i], ava_mapped[j])
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

kmedoids = KMedoids(n_clusters=5, metric='precomputed', random_state=42)
labels = kmedoids.fit_predict(distance_matrix)

# Silhouette Score Evaluation
silhouette_avg = silhouette_score(distance_matrix, labels, metric="precomputed")
print(f"Silhouette Score: {silhouette_avg}")

# ==========================
# Step 8: Visualization
# ==========================
plt.scatter(ava_mapped[:, 0], ava_mapped[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.title("Clustered Data")
plt.show()


pip install torch numpy pandas scikit-learn umap-learn dtaidistance tqdm







# ==========================
# Import Libraries
# ==========================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from dtaidistance import dtw
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# ==========================
# Step 1: Define the Dataset Class
# ==========================
class MatrixDataset(Dataset):
    def __init__(self, df, n_nodes):
        self.df = df
        self.n_nodes = n_nodes

    def get_adj(self, path):
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)

        # Source node
        adj[path[0], 0] = 1
        # Destination node
        adj[self.n_nodes - 1, path[-1]] = 1
        adj[self.n_nodes - 1, self.n_nodes - 1] = 1  # Sink node absorbs

        # Fill adjacency matrix along the path
        for i in range(len(path) - 1):
            adj[path[i + 1], path[i]] = 1

        # Normalize adjacency matrix
        col_sums = adj.sum(axis=0)
        col_sums[col_sums == 0] = 1e-6  # Avoid division by zero
        adj = adj / col_sums

        return adj

    def __getitem__(self, index):
        path = self.df.iloc[index]['path']
        adj = self.get_adj(path)
        return torch.tensor(adj, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

# ==========================
# Step 2: Define the Model
# ==========================
class MatrixEmbedder(nn.Module):
    def __init__(self, n_nodes, embed_dim):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(n_nodes ** 2, 500), nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Linear(500, 250), nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Linear(250, 100), nn.LeakyReLU())
        self.layer4 = nn.Sequential(nn.Linear(100, 50), nn.LeakyReLU())
        self.layer5 = nn.Linear(50, embed_dim)

    def forward(self, x):
        x = x.flatten(start_dim=1, end_dim=-1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

# ==========================
# Step 3: Distance Calculation (Markov-based)
# ==========================
def dme(mat1, mat2, vector):
    mat = np.matmul(mat1.T, mat2)
    v = np.matmul(mat, vector)
    out = np.matmul(vector.T, v)
    return out[0, 0]

def markov_distance(mata, matb):
    v1 = np.ones((mata.shape[0], 1))
    out = dme(mata, matb, v1) - 0.5 * dme(mata, mata, v1) - 0.5 * dme(matb, matb, v1)
    return out

# ==========================
# Step 4: Training Loop
# ==========================
def train_model(dataset, embedder, optimizer, scheduler, epochs, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses = []
    distances_captured = []

    for ep in range(epochs):
        print(f"Epoch {ep+1}/{epochs}")
        for ib, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Generate pairs
            n_per_example = 2
            right = np.random.randint(0, batch.shape[0], n_per_example * batch.shape[0])
            left = np.repeat(np.arange(0, batch.shape[0]), n_per_example)
            index_pairs = np.stack([left, right]).T

            # Compute distances
            distances = []
            with torch.no_grad():
                for (l, r) in index_pairs:
                    d = markov_distance(batch[l].cpu().numpy(), batch[r].cpu().numpy())
                    distances.append(d)
                    distances_captured.append(d)

            distances = torch.tensor(distances, dtype=torch.float32)
            distances = (distances - distances.mean()) / (distances.std() + 1e-6)  # Normalize

            # Embeddings
            left_vecs = embedder(batch[left])
            right_vecs = embedder(batch[right])
            diff = left_vecs - right_vecs

            # Compute Loss
            diff_norm = torch.nn.functional.normalize(diff, p=2, dim=1)
            loss = torch.pow(((diff_norm ** 2).sum(axis=1) - distances ** 2), 2).mean()

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder.parameters(), max_norm=5.0)  # Gradient Clipping
            optimizer.step()
            scheduler.step(loss)  # Learning Rate Scheduler

            losses.append(loss.item())

        print(f"Epoch {ep+1}/{epochs}, Loss: {loss.item()}")
    return losses, distances_captured

# ==========================
# Step 5: Load Data (df234)
# ==========================
data = {
    'path': [[1, 2, 3, 4], [2, 3, 5], [1, 4, 6], [3, 5, 7, 8], [1, 3, 6, 9], [4, 5, 8], [2, 4, 7], [5, 6, 9]]
}
df234 = pd.DataFrame(data)

n_nodes = max([max(path) for path in df234['path']]) + 2  # Adding 2 for source and sink
dataset = MatrixDataset(df234, n_nodes=n_nodes)

# ==========================
# Step 6: Train the Model
# ==========================
embed_dim = 20
embedder = MatrixEmbedder(n_nodes=n_nodes, embed_dim=embed_dim)
optimizer = optim.Adam(embedder.parameters(), lr=0.0003, weight_decay=1e-5)  # Lower LR and Regularization
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

epochs = 100
batch_size = 16  # Fixed batch size as requested
losses, distances_captured = train_model(dataset, embedder, optimizer, scheduler, epochs, batch_size)

# ==========================
# Step 7: Plot Smoothed Loss
# ==========================
window_size = 50
moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')

plt.plot(moving_avg)
plt.title("Smoothed Loss Over Epochs")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

# ==========================
# Step 8: Embedding & Clustering
# ==========================
all_vecs = []
for ib, batch in tqdm(enumerate(DataLoader(dataset, batch_size=batch_size)), total=len(dataset) // batch_size):
    vecs = embedder(batch.to(torch.float32))
    all_vecs.append(vecs.detach().numpy())

all_vecs_array = np.concatenate(all_vecs, axis=0)

# Save & Load Embeddings
with open("all_vecs_array.pkl", "wb") as f:
    pickle.dump(all_vecs_array, f)

with open("all_vecs_array.pkl", "rb") as f:
    all_vecs_array = pickle.load(f)

# UMAP Reduction
mapper = umap.UMAP()
ava_mapped = mapper.fit_transform(all_vecs_array)

# K-Medoids Clustering
distance_matrix = np.zeros((len(ava_mapped), len(ava_mapped)))
for i in range(len(ava_mapped)):
    for j in range(i + 1, len(ava_mapped)):
        distance = dtw.distance(ava_mapped[i], ava_mapped[j])
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

kmedoids = KMedoids(n_clusters=5, metric='precomputed', random_state=42)
labels = kmedoids.fit_predict(distance_matrix)

# Silhouette Score Evaluation
silhouette_avg = silhouette_score(distance_matrix, labels, metric="precomputed")
print(f"Silhouette Score: {silhouette_avg}")

# ==========================
# Step 9: Visualization
# ==========================
plt.scatter(ava_mapped[:, 0], ava_mapped[:, 1], c=labels, cmap='viridis



















import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from dtaidistance import dtw
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# ===========================================
# Step 1: Define the Dataset Class
# ===========================================
class MatrixDataset(Dataset):
    def __init__(self, df, n_nodes):
        """
        Initializes the dataset with a DataFrame containing paths and the number of nodes.
        :param df: DataFrame containing a 'path' column with sequences.
        :param n_nodes: Total number of unique nodes in the dataset.
        """
        self.df = df
        self.n_nodes = n_nodes

    def get_adj(self, path):
        """ Converts a given path into an adjacency matrix representation. """
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        
        # Source and Sink nodes
        adj[path[0], 0] = 1
        adj[self.n_nodes - 1, path[-1]] = 1
        adj[self.n_nodes - 1, self.n_nodes - 1] = 1  # Sink node absorbs

        # Fill adjacency matrix along the path
        for i in range(len(path) - 1):
            adj[path[i + 1], path[i]] = 1

        # Normalize adjacency matrix
        col_sums = adj.sum(axis=0)
        col_sums[col_sums == 0] = 1e-6  # Avoid division by zero
        adj = adj / col_sums

        return adj

    def get_path(self, index):
        """ Retrieves the path given an index. """
        return self.df.at[index, 'path']

    def __getitem__(self, index):
        path = self.df.iloc[index]['path']
        adj = self.get_adj(path)
        return torch.tensor(adj, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

# ===========================================
# Step 2: Define the Embedding Model
# ===========================================
class MatrixEmbedder(nn.Module):
    def __init__(self, n_nodes, embed_dim):
        super().__init__()
        self.layer1 = nn.Linear(n_nodes ** 2, 500, bias=True)
        self.layer2 = nn.Linear(500, 250, bias=True)
        self.layer3 = nn.Linear(250, 100, bias=True)
        self.layer4 = nn.Linear(100, 50, bias=True)
        self.layer5 = nn.Linear(50, embed_dim, bias=True)

    def forward(self, x):
        x = x.flatten(start_dim=1, end_dim=-1)
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        x = torch.tanh(self.layer4(x))
        x = torch.tanh(self.layer5(x))  # Activation isn't necessary but helps
        return x

# ===========================================
# Step 3: Define Markov Distance Functions
# ===========================================
def dme(mat1, mat2, vector):
    mat = np.matmul(mat1.T, mat2)
    v = np.matmul(mat, vector)
    out = np.matmul(vector.T, v)
    return out[0, 0]

def markov_distance(mata, matb):
    v1 = np.ones((mata.shape[0], 1))
    out = dme(mata, matb, v1) - 0.5 * dme(mata, mata, v1) - 0.5 * dme(matb, matb, v1)
    return out

# ===========================================
# Step 4: Training Loop
# ===========================================
def train_model(dataset, embedder, optimizer, epochs, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses = []

    for ep in range(epochs):
        print(f"Epoch {ep+1}/{epochs}")
        for ib, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Generate pairs
            right = np.random.randint(0, batch.shape[0], batch.shape[0])
            left = np.arange(0, batch.shape[0])
            index_pairs = np.stack([left, right]).T

            # Compute distances
            distances = []
            with torch.no_grad():
                for (l, r) in index_pairs:
                    d = markov_distance(batch[l].cpu().numpy(), batch[r].cpu().numpy())
                    distances.append(d)

            distances = torch.tensor(distances, dtype=torch.float32)

            # Embeddings
            left_vecs = embedder(batch[left])
            right_vecs = embedder(batch[right])
            diff = left_vecs - right_vecs

            # Compute Loss
            loss = torch.pow(((diff ** 2).sum(axis=1) - distances ** 2), 2).sum()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {ep+1}/{epochs}, Loss: {loss.item()}")
    return losses

# ===========================================
# Step 5: Load Data
# ===========================================
df234 = pd.DataFrame({'path': [[1, 2, 3, 4], [2, 3, 5], [1, 4, 6], [3, 5, 7, 8]]})
n_nodes = max([max(path) for path in df234['path']]) + 2  # Adding 2 for source and sink

dataset = MatrixDataset(df234, n_nodes=n_nodes)
embedder = MatrixEmbedder(n_nodes=n_nodes, embed_dim=20)
optimizer = optim.Adam(embedder.parameters(), lr=0.001)

epochs = 100
batch_size = 16  # Fixed batch size
losses = train_model(dataset, embedder, optimizer, epochs, batch_size)

# ===========================================
# Step 6: Embedding & Clustering
# ===========================================
all_vecs = []
for batch in DataLoader(dataset, batch_size=batch_size):
    vecs = embedder(batch.to(torch.float32))
    all_vecs.append(vecs.detach().numpy())

all_vecs_array = np.concatenate(all_vecs, axis=0)

# Save & Load Embeddings
with open("all_vecs_array.pkl", "wb") as f:
    pickle.dump(all_vecs_array, f)

with open("all_vecs_array.pkl", "rb") as f:
    all_vecs_array = pickle.load(f)

# UMAP Reduction
mapper = umap.UMAP()
ava_mapped = mapper.fit_transform(all_vecs_array)

# K-Medoids Clustering
distance_matrix = np.zeros((len(ava_mapped), len(ava_mapped)))
for i in range(len(ava_mapped)):
    for j in range(i + 1, len(ava_mapped)):
        distance = dtw.distance(ava_mapped[i], ava_mapped[j])
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

kmedoids = KMedoids(n_clusters=5, metric='precomputed', random_state=42)
labels = kmedoids.fit_predict(distance_matrix)

# Silhouette Score Evaluation
silhouette_avg = silhouette_score(distance_matrix, labels, metric="precomputed")
print(f"Silhouette Score: {silhouette_avg}")

# ===========================================
# Step 7: Visualization
# ===========================================
plt.scatter(ava_mapped[:, 0], ava_mapped[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.title("Clustered Data")
plt.show()









import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ===========================================
# Step 1: Define Function to Plot Paths for Each Cluster
# ===========================================
def plot_cluster_paths(df, labels, cluster_id):
    """
    Visualizes the user journey for a given cluster using NetworkX.

    :param df: DataFrame containing paths
    :param labels: Cluster labels assigned to each journey
    :param cluster_id: The cluster ID to visualize
    """
    G = nx.DiGraph()
    cluster_paths = df[labels == cluster_id]['path'].tolist()

    edge_weights = {}

    # Extract edges and count occurrences (to use as weights)
    for path in cluster_paths:
        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            if edge in edge_weights:
                edge_weights[edge] += 1
            else:
                edge_weights[edge] = 1

    # Add edges with weights
    for edge, weight in edge_weights.items():
        G.add_edge(edge[0], edge[1], weight=weight)

    # Positioning nodes using spring layout
    pos = nx.spring_layout(G, seed=42)

    # Drawing the graph
    plt.figure(figsize=(10, 7))
    edges = G.edges(data=True)

    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color="skyblue", edgecolors="black")
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in edges], width=[d['weight'] * 0.2 for u, v, d in edges], alpha=0.7, edge_color="blue", arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    # Add edge labels (weights)
    edge_labels = {(u, v): d['weight'] for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5)

    plt.title(f"Journey Visualization for Cluster {cluster_id}")
    plt.show()

# ===========================================
# Step 2: Load the Data (df234) and Cluster Labels
# ===========================================
# Sample Data (Replace this with actual clustered data)
df234 = pd.DataFrame({'path': [[1, 2, 3, 4], [2, 3, 5], [1, 4, 6], [3, 5, 7, 8], [2, 6, 7, 9, 10], [1, 3, 6, 9]]})
labels = np.array([0, 1, 0, 1, 2, 2])  # Sample cluster labels (Replace with actual clustering output)

# ===========================================
# Step 3: Visualize Each Cluster
# ===========================================
unique_clusters = np.unique(labels)
for cluster in unique_clusters:
    plot_cluster_paths(df234, labels, cluster)









import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import umap
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from dtaidistance import dtw
import matplotlib.pyplot as plt
import pickle

# ================================
# Step 1: Preprocess Data
# ================================

# Function to map pages to numbers and retain both versions
def map_pages_to_numbers(df, path_column='path'):
    """
    Maps unique page names to numerical IDs while retaining the original names.

    :param df: DataFrame containing journeys
    :param path_column: Column containing ordered paths
    :return: DataFrame with both mapped numbers and actual page names
    """
    unique_pages = pd.Series([page for path in df[path_column] for page in path]).unique()

    # Create a page-to-number mapping
    page_to_number = {page: idx + 1 for idx, page in enumerate(unique_pages)}
    number_to_page = {v: k for k, v in page_to_number.items()}  # Reverse mapping

    # Map each journey to numbers
    df['path_numeric'] = df[path_column].apply(lambda path: [page_to_number[page] for page in path])

    return df, page_to_number, number_to_page

# Sample Data (Modify this with actual data)
data = {
    'channel_visit_id': [111, 112, 113, 114],
    'path': [
        ['home', 'login', 'dashboard', 'settings'],
        ['home', 'products', 'cart', 'checkout'],
        ['home', 'about', 'contact'],
        ['login', 'dashboard', 'logout']
    ]
}

# Create DataFrame
journey_paths_df = pd.DataFrame(data)

# Apply the function
mapped_journeys_df, page_to_number, number_to_page = map_pages_to_numbers(journey_paths_df)

# Save df234 (Numerical representation for clustering)
df234 = mapped_journeys_df[['channel_visit_id', 'path_numeric']].rename(columns={'path_numeric': 'path'})
df234_page_mapping = number_to_page  # Mapping numbers back to original pages

# ================================
# Step 2: Define Dataset Class
# ================================

class MatrixDataset(Dataset):
    def __init__(self, df, n_nodes):
        self.df = df
        self.n_nodes = n_nodes

    def get_adj(self, path):
        """
        Converts a given path into an adjacency matrix representation.
        """
        adj = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)

        # Source node
        adj[path[0], 0] = 1
        # Destination node
        adj[self.n_nodes - 1, path[-1]] = 1
        adj[self.n_nodes - 1, self.n_nodes - 1] = 1  # Sink node absorbs

        # Fill adjacency matrix along the path
        for i in range(len(path) - 1):
            adj[path[i + 1], path[i]] = 1

        col_sums = adj.sum(axis=0)
        col_sums[col_sums == 0] = 1e-6  # Avoid division by zero
        adj = adj / col_sums

        return adj

    def __getitem__(self, index):
        path = self.df.iloc[index]['path']
        adj = self.get_adj(path)
        return torch.tensor(adj, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

# Compute n_nodes dynamically
n_nodes = max([max(path) for path in df234['path']]) + 2  # Adding 2 for source and sink

# Initialize dataset and dataloader
dataset = MatrixDataset(df234, n_nodes=n_nodes)

# ================================
# Step 3: Define Model
# ================================

class MatrixEmbedder(nn.Module):
    def __init__(self, n_nodes, embed_dim):
        super().__init__()
        self.layer1 = nn.Linear(n_nodes ** 2, 500, bias=True)
        self.layer2 = nn.Linear(500, 250, bias=True)
        self.layer3 = nn.Linear(250, 100, bias=True)
        self.layer4 = nn.Linear(100, 50, bias=True)
        self.layer5 = nn.Linear(50, embed_dim, bias=True)

    def forward(self, x):
        x = x.flatten(start_dim=1, end_dim=-1)
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        x = torch.tanh(self.layer4(x))
        x = torch.tanh(self.layer5(x))
        return x

embed_dim = 20
embedder = MatrixEmbedder(n_nodes=n_nodes, embed_dim=embed_dim)
optimizer = optim.Adam(embedder.parameters(), lr=0.001)

# ================================
# Step 4: Train Model
# ================================

def train_model(dataset, embedder, optimizer, epochs=100, batch_size=16):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses = []

    for ep in range(epochs):
        print(f"Epoch {ep+1}/{epochs}")
        for batch in dataloader:
            optimizer.zero_grad()

            embeddings = embedder(batch)
            loss = (embeddings ** 2).sum()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Loss: {loss.item()}")

    return losses

losses = train_model(dataset, embedder, optimizer)

# ================================
# Step 5: Embedding & Clustering
# ================================

# Convert dataset to embeddings
all_vecs = []
for batch in tqdm(DataLoader(dataset, batch_size=16), total=len(dataset) // 16):
    vecs = embedder(batch.to(torch.float32))
    all_vecs.append(vecs.detach().numpy())

all_vecs_array = np.concatenate(all_vecs, axis=0)

# Save & Load Embeddings
with open("all_vecs_array.pkl", "wb") as f:
    pickle.dump(all_vecs_array, f)

with open("all_vecs_array.pkl", "rb") as f:
    all_vecs_array = pickle.load(f)

# UMAP Reduction
mapper = umap.UMAP()
ava_mapped = mapper.fit_transform(all_vecs_array)

# K-Medoids Clustering
distance_matrix = np.zeros((len(ava_mapped), len(ava_mapped)))
for i in range(len(ava_mapped)):
    for j in range(i + 1, len(ava_mapped)):
        distance = dtw.distance(ava_mapped[i], ava_mapped[j])
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

kmedoids = KMedoids(n_clusters=5, metric='precomputed', random_state=42)
labels = kmedoids.fit_predict(distance_matrix)

# Evaluate Clustering
silhouette_avg = silhouette_score(distance_matrix, labels, metric="precomputed")
print(f"Silhouette Score: {silhouette_avg}")

# ================================
# Step 6: Plot Journey Paths per Cluster
# ================================

def plot_cluster_paths_with_names(df, labels, cluster_id, number_to_page):
    G = nx.DiGraph()
    cluster_paths = df[labels == cluster_id]['path'].tolist()

    edge_weights = {}

    for path in cluster_paths:
        for i in range(len(path) - 1):
            edge = (number_to_page[path[i]], number_to_page[path[i+1]])  
            edge_weights[edge] = edge_weights.get(edge, 0) + 1

    for edge, weight in edge_weights.items():
        G.add_edge(edge[0], edge[1], weight=weight)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='blue', width=[d['weight']*0.2 for u,v,d in G.edges(data=True)])
    plt.title(f"Cluster {cluster_id} Journey Paths")
    plt.show()

plot_cluster_paths_with_names(df234, labels, cluster_id=0, number_to_page=number_to_page)








import networkx as nx
import matplotlib.pyplot as plt

def plot_cluster_paths_with_names(df, labels, cluster_id, number_to_page):
    """
    Plots user journey paths for a specific cluster using actual page names.

    :param df: DataFrame containing the journey paths.
    :param labels: Cluster labels assigned to each journey.
    :param cluster_id: The cluster to visualize.
    :param number_to_page: Dictionary mapping numerical page IDs back to page names.
    """

    # Reverse mapping from numbers to page names
    rev_page_mapping = {v: k for k, v in number_to_page.items()}

    # Extract paths for the specified cluster
    cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
    paths_in_group = df.iloc[cluster_indices]['path'].tolist()

    # Convert numerical paths to actual page names
    transitions_filtered = []
    for nodes in paths_in_group:
        nodes_reversed = [rev_page_mapping[n] for n in nodes]  # Convert to page names
        transitions_filtered.extend(list(zip(nodes_reversed, nodes_reversed[1:])))  # Create edges

    # Create a directed graph
    G = nx.DiGraph()
    
    # Add edges based on transitions
    G.add_edges_from(transitions_filtered)

    # Draw the graph
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.5)  # Layout for better visualization
    nx.draw(G, pos, with_labels=True, node_color="#009BA5", edge_color="#666", width=1.5, node_size=2000, arrows=True)
    
    # Show plot
    plt.title(f"Cluster {cluster_id} Journey Paths")
    plt.show()
plot_cluster_paths_with_names(df234, labels, cluster_id=3, number_to_page=page_mapping)






import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

def plot_cluster_paths_with_names(df, labels, cluster_id, number_to_page):
    """
    Plots user journey paths for a specific cluster using actual page names and transition frequencies.

    :param df: DataFrame containing the journey paths.
    :param labels: Cluster labels assigned to each journey.
    :param cluster_id: The cluster to visualize.
    :param number_to_page: Dictionary mapping numerical page IDs back to page names.
    """

    # Reverse mapping from numbers to actual page names
    rev_page_mapping = {v: k for k, v in number_to_page.items()}

    # Extract paths for the specified cluster
    cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
    paths_in_group = df.iloc[cluster_indices]['path'].tolist()

    # Convert numerical paths to actual page names
    transitions_filtered = []
    for nodes in paths_in_group:
        nodes_reversed = [rev_page_mapping[n] for n in nodes]  # Convert to page names
        transitions_filtered.extend(list(zip(nodes_reversed, nodes_reversed[1:])))  # Create transitions

    # Count transition frequencies
    transition_counts = Counter(transitions_filtered)

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges based on transition frequency
    for (src, dst), freq in transition_counts.items():
        G.add_edge(src, dst, weight=freq)

    # Draw the graph
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.5)  # Layout for better visualization

    # Extract edge weights (frequencies)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Draw the nodes and edges
    nx.draw(G, pos, with_labels=True, node_color="#009BA5", edge_color="#666", width=[w / 3 for w in edge_weights], 
            node_size=2000, arrows=True, alpha=0.8)

    # Add frequency labels on edges
    edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color="red")

    # Show plot
    plt.title(f"Cluster {cluster_id} Journey Paths with Transition Frequencies")
    plt.show()





import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

def plot_cluster_paths_with_names(df, labels, cluster_id, number_to_page, dark_mode=False):
    """
    Plots user journey paths for a specific cluster with improved aesthetics.

    :param df: DataFrame containing journey paths.
    :param labels: Cluster labels assigned to each journey.
    :param cluster_id: The cluster ID to visualize.
    :param number_to_page: Dictionary mapping page numbers back to names.
    :param dark_mode: If True, enables dark mode visualization.
    """

    # Toggle dark mode settings
    if dark_mode:
        plt.style.use("dark_background")
        node_color = "#00C0FF"
        edge_color = "#AAAAAA"
        font_color = "white"
    else:
        plt.style.use("seaborn-white")
        node_color = "#009BA5"
        edge_color = "#666"
        font_color = "black"

    # Reverse mapping from numbers to page names
    rev_page_mapping = {v: k for k, v in number_to_page.items()}

    # Extract paths for the specified cluster
    cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
    paths_in_group = df.iloc[cluster_indices]['path'].tolist()

    # Convert numerical paths to actual page names
    transitions_filtered = []
    for nodes in paths_in_group:
        nodes_reversed = [rev_page_mapping.get(n, f"Page {n}") for n in nodes]  # Convert numbers to names
        transitions_filtered.extend(list(zip(nodes_reversed, nodes_reversed[1:])))  # Create transitions

    # Count transition frequencies
    transition_counts = Counter(transitions_filtered)

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges based on transition frequency
    for (src, dst), freq in transition_counts.items():
        G.add_edge(src, dst, weight=freq)

    # Define figure size
    plt.figure(figsize=(16, 12))
    
    # Use a better layout
    pos = nx.kamada_kawai_layout(G)  # Alternative: nx.spring_layout(G, k=0.5)

    # Extract edge weights for scaling
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    edge_widths = [max(1, w / max(edge_weights) * 5) for w in edge_weights]  # Normalize widths

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_color, alpha=0.85, edgecolors="black")

    # Draw the edges (scaled thickness)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_color, alpha=0.6, arrows=True, arrowsize=20)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color=font_color, font_weight="bold")

    # Add frequency labels on edges
    edge_labels = {(u, v): str(G[u][v]['weight']) for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color="red", alpha=0.9)

    # Show plot
    plt.title(f"Cluster {cluster_id} Journey Paths with Transition Frequencies", fontsize=14, fontweight="bold")
    plt.show()
