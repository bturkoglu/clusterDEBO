# clusterDEBO
A Cluster-Assisted Differential Evolution-Based Hybrid Oversampling Method for Imbalanced Datasets 


📌 ClusterDEBO: A Cluster-Assisted Differential Evolution-Based Hybrid Oversampling Algorithm

This Python implementation provides ClusterDEBO, a hybrid oversampling method designed to address class imbalance problems in machine learning datasets. The method combines K-Means clustering with Differential Evolution (DE) to generate synthetic samples in a more structured and adaptive way.

⸻

🛠️ 1. Overview of the Algorithm

📌 ClusterDEBO aims to generate synthetic samples that better represent the distribution of the minority class. It follows these key steps:
	1.	Identifies the minority class instances in the dataset.
	2.	Applies K-Means clustering to divide the minority class into subgroups.
	3.	Uses Differential Evolution (DE) operations within each cluster to generate synthetic samples.
	4.	Injects noise to prevent overfitting and improve diversity.
	5.	Balances the dataset by adding the synthetic minority samples.

⸻

📌 2. Class Components and Explanation

🔹 __init__ (Constructor Method)

    def __init__(
        self,
        proportion=1.0,
        n_neighbors=5,
        crossover_rate=0.6,
        F=0.3,
        n_clusters=10,
        n_jobs=1,
        noise_scale=0.01,
        random_state=None,
    ):

This method initializes the ClusterDEBO object with the following parameters:
	•	proportion: The ratio of generated samples to balance the dataset.
	•	n_neighbors: Number of neighbors used for k-Nearest Neighbors (k-NN).
	•	crossover_rate: Crossover probability for the Differential Evolution algorithm.
	•	F: Differential mutation factor.
	•	n_clusters: Number of clusters for the K-Means algorithm.
	•	n_jobs: Number of CPU cores for parallel processing.
	•	noise_scale: Scale of noise added to synthetic samples.
	•	random_state: Seed for random number generation.

⸻

🔹 parameter_combinations (Hyperparameter Tuning)

    @classmethod
    def parameter_combinations(cls):
        return [
            {'proportion': 1.0, 'n_neighbors': 5, 'crossover_rate': 0.6, 'F': 0.3, 'n_clusters': 10},
            {'proportion': 1.5, 'n_neighbors': 7, 'crossover_rate': 0.8, 'F': 0.5, 'n_clusters': 15},
            {'proportion': 2.0, 'n_neighbors': 3, 'crossover_rate': 0.4, 'F': 0.2, 'n_clusters': 5}
        ]

This method provides a list of predefined parameter combinations to facilitate hyperparameter tuning.

⸻

🔹 cluster_minority_samples (Clustering the Minority Class)

    def cluster_minority_samples(self, X_min):
        kmeans = KMeans(n_clusters=min(len(X_min), self.n_clusters), random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(X_min)
        return cluster_labels, kmeans.cluster_centers_

📌 What does it do?
	•	Uses K-Means clustering to partition the minority class into smaller groups.
	•	Returns cluster labels and cluster centroids.

⸻

🔹 generate_samples (Synthetic Sample Generation)

    def generate_samples(self, n_to_sample, X_min, indices, cluster_labels):

📌 What does it do?
	•	Identifies which cluster each minority class instance belongs to.
	•	Uses Differential Evolution (DE) mutation and crossover to create new synthetic samples.
	•	Injects small noise to increase variability and realism.

🔹 Synthetic samples are generated using the following equation:

donor = X_min[r1] + self.F * (X_min[r2] - X_min[r1])

This equation creates a new synthetic data point using the difference between two randomly selected samples within the same cluster.

⸻

🔹 sample (Main Method for Data Balancing)

    def sample(self, X, y):

📌 What does it do?
1️⃣ Determines the minority and majority class instances.
2️⃣ Computes the imbalance ratio.
3️⃣ Applies K-Means to cluster the minority samples.
4️⃣ Generates synthetic samples using Differential Evolution within each cluster.
5️⃣ Adds the synthetic samples to the dataset to balance class distribution.

⸻

📊 3. How the Algorithm Works

Step 1: Identify Class Distribution
	•	Finds the number of instances in each class.

Step 2: Cluster the Minority Class
	•	Uses K-Means to divide the minority class into clusters.

Step 3: Generate Synthetic Samples Using DE
	•	Selects random instances within the same cluster.
	•	Creates new samples using Differential Evolution.
	•	Adds random noise to prevent overfitting.

Step 4: Merge New Samples with Original Data
	•	Appends the generated samples to the dataset.
	•	Ensures a more balanced class distribution.

⸻

🔍 4. Advantages of the ClusterDEBO Approach

✅ More realistic synthetic samples: Clustering ensures that new samples follow the real data distribution.
✅ Prevents overfitting: Differential Evolution helps maintain data diversity.
✅ Better decision boundaries: Improved sample placement results in more robust classification models.

⸻

🚀 5. Conclusion
	•	ClusterDEBO is an advanced hybrid oversampling technique for class imbalance problems.
	•	It partitions minority class data into clusters and applies Differential Evolution to generate realistic synthetic samples.
	•	This leads to better model generalization and improved classification performance.

⸻

💡 You can fine-tune the parameters and test the algorithm on different datasets for further optimization! 🚀
