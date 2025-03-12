import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

class ClusterDEBO:
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
        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.crossover_rate = crossover_rate
        self.F = F
        self.n_clusters = n_clusters
        self.n_jobs = n_jobs
        self.noise_scale = noise_scale
        self.random_state = np.random.RandomState(random_state)

    @classmethod
    def parameter_combinations(cls):
        return [
            {'proportion': 1.0, 'n_neighbors': 5, 'crossover_rate': 0.6, 'F': 0.3, 'n_clusters': 10},
            {'proportion': 1.5, 'n_neighbors': 7, 'crossover_rate': 0.8, 'F': 0.5, 'n_clusters': 15},
            {'proportion': 2.0, 'n_neighbors': 3, 'crossover_rate': 0.4, 'F': 0.2, 'n_clusters': 5}
        ]

    def cluster_minority_samples(self, X_min):
        kmeans = KMeans(n_clusters=min(len(X_min), self.n_clusters), random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(X_min)
        return cluster_labels, kmeans.cluster_centers_

    def generate_samples(self, n_to_sample, X_min, indices, cluster_labels):
        print(f"generate_samples fonksiyonuna gelen n_to_sample: {n_to_sample}")
        samples = []
        for _ in range(n_to_sample):
            try:
                idx = self.random_state.randint(len(X_min))
                x_selected = X_min[idx]
                cluster_id = cluster_labels[idx]
                cluster_neighbors = np.where(cluster_labels == cluster_id)[0]
                
                if len(cluster_neighbors) > 1:
                    r1, r2 = self.random_state.choice(cluster_neighbors, 2, replace=False)
                    donor = X_min[r1] + self.F * (X_min[r2] - X_min[r1])
                else:
                    r1 = self.random_state.choice(len(X_min))
                    donor = X_min[r1]
                
                trial = np.where(
                    self.random_state.random_sample(donor.shape) < self.crossover_rate,
                    donor,
                    x_selected,
                )
                noise = self.random_state.normal(loc=0, scale=self.noise_scale, size=trial.shape)
                sample = trial + noise
                samples.append(sample)
            except Exception as e:
                print(f"Sample generation error: {e}")
        print(f"Üretilen toplam örnek sayısı: {len(samples)}")
        return np.array(samples)

    def sample(self, X, y):
        try:
            labels, counts = np.unique(y, return_counts=True)
            min_label = labels[np.argmin(counts)]
            majority_label = labels[np.argmax(counts)]
            X_min = X[y == min_label]
            n_min = len(X_min)
            n_maj = len(X[y == majority_label])
            n_to_sample = n_maj - n_min
            
            print(f"Çoğunluk sınıfı örnek sayısı: {n_maj}")
            print(f"Azınlık sınıfı örnek sayısı: {n_min}")
            print(f"Üretilmesi gereken örnek sayısı: {n_to_sample}")
            
            if n_min <= 2:
                print("Azınlık sınıfı çok küçük! Örnekleme yapılmayacak.")
                return X, y
            
            self.n_neighbors = min(self.n_neighbors, n_min - 1)
            print(f"Kullanılan n_neighbors: {self.n_neighbors}")
            
            if n_to_sample <= 0:
                return X, y

            cluster_labels, cluster_centers = self.cluster_minority_samples(X_min)
            nn_min = NearestNeighbors(n_neighbors=self.n_neighbors + 1, n_jobs=self.n_jobs)
            nn_min.fit(X_min)
            indices = nn_min.kneighbors(X_min, return_distance=False)

            samples = self.generate_samples(n_to_sample, X_min, indices, cluster_labels)
            X_resampled = np.vstack([X, samples])
            y_resampled = np.hstack([y, [min_label] * len(samples)])
            
            print(f"Yeni eğitim seti boyutu: {X_resampled.shape}")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"Sampling error: {e}")
            return X, y

    def get_params(self):
        return {
            'proportion': self.proportion,
            'n_neighbors': self.n_neighbors,
            'crossover_rate': self.crossover_rate,
            'F': self.F,
            'n_clusters': self.n_clusters,
            'n_jobs': self.n_jobs,
            'noise_scale': self.noise_scale
        }
