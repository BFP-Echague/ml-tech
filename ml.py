"""Machine learning functions."""


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def drop_id(df: pd.DataFrame):
    """Drops the id column."""
    return df.drop(columns=["id"])


def rescale_data(df: pd.DataFrame):
    """Rescales the data."""
    df_copy = drop_id(df)
    scaler = StandardScaler()

    df_scaled = pd.DataFrame(scaler.fit_transform(df_copy), columns=df_copy.columns)
    df_scaled["id"] = df["id"]
    return df_scaled


def pca(df: pd.DataFrame, n_components: int):
    """Performs PCA on the data."""
    pca_model = PCA(n_components=n_components)

    df_remove_id = drop_id(df)
    df_pca = pd.DataFrame(data=pca_model.fit_transform(df_remove_id), columns=[f"PC{i}" for i in range(1, n_components + 1)])
    df_pca["id"] = df["id"]

    return df_pca



class ClusterResult():
    """Represents one result of clustering."""
    def __init__(self, cluster_count: int, labels: list[int], score: float):
        self.cluster_count = cluster_count
        self.labels = labels
        self.score = score

    @classmethod
    def perform_cluster(cls, df: pd.DataFrame, cluster_count: int):
        """Clusters the data."""
        df_kmeans_input = drop_id(df)

        kmeans = KMeans(n_clusters=cluster_count)
        kmeans.fit(df_kmeans_input)


        labels_dict: dict[int, list[int]] = { i: [] for i in range(cluster_count) }
        for idx, label in enumerate(kmeans.labels_):
            labels_dict[label].append(df.iloc[[idx]]["id"].array[0].item())

        labels = [value for value in labels_dict.values()]

        return ClusterResult(cluster_count, labels, silhouette_score(df, kmeans.labels_).item())


    def to_json(self):
        """Converts the object to a JSON serializable format."""
        return {
            "clusterCount": self.cluster_count,
            "labels": self.labels,
            "score": self.score
        }



class ClusterSilhouetteResult():
    """Represents multiple results of clustering."""
    def __init__(self, clusters: list[ClusterResult], optimal_cluster_count: int):
        self.clusters = clusters
        self.optimal_cluster_count = optimal_cluster_count


    @classmethod
    def perform_cluster_with_silhouette(cls, df: pd.DataFrame, cluster_count_start: int, cluster_count_end: int):
        """Clusters the data then finds the optimal clustering through silhouette score."""
        clusters: list[ClusterResult] = [
            ClusterResult.perform_cluster(df, cluster_count)
            for cluster_count in range(cluster_count_start, cluster_count_end + 1)
        ]

        optimal_cluster_count = max(clusters, key=lambda cluster: cluster.score).cluster_count

        return ClusterSilhouetteResult(clusters, optimal_cluster_count)

    def to_json(self):
        """Converts the object to a JSON serializable format."""
        return {
            "clusterResults": [cluster.to_json() for cluster in self.clusters],
            "optimalClusterCount": self.optimal_cluster_count
        }


def perform_pipeline(df: pd.DataFrame, component_count: int, cluster_count_start: int, cluster_count_end: int):
    """Performs the entire pipeline."""
    rescaled_data = rescale_data(df)

    if component_count == (df.shape[1] - 1):
        pca_data = rescaled_data
    else:
        pca_data = pca(rescaled_data, component_count)

    return ClusterSilhouetteResult.perform_cluster_with_silhouette(pca_data, cluster_count_start, cluster_count_end)
