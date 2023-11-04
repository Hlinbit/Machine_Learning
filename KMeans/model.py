import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def nearest_poins(x, centrodis):
    dists = np.array([euclidean_distance(x, cend) for cend in centrodis])
    idx = np.argmin(dists)
    return idx

np.random.seed(1996)

class KMeans:
    def __init__(self, K = 5, m_iters = 100, plot_steps = False):
        self.K = K
        self.m_iters = m_iters
        self.plot_steps = plot_steps

        self.clusters = [[] for _ in range(self.K)]
        self.centrodis = []

    def predict(self, X):
        self.sample = X
        self.n_samples, self.n_features = X.shape

        random_cen_idx = np.random.choice(self.n_samples, size=self.K, replace=False)
        self.centrodis = self.sample[random_cen_idx]

        for _ in range(self.m_iters):
            idxs = self._find_nearest_points(self.centrodis)
           
            old_centrodist = self.centrodis

            self._update_cluster(idxs)
            if self.plot_steps:
                self.plot_step()

            self.centrodis = self._update_cendrodis(idxs)

            if self._is_converged(old_centrodist, self.centrodis):
                break
        return self._get_cluster_lable()

    def _get_cluster_lable(self):
        labels = np.empty(self.n_samples)
        for label, cluster_idx in enumerate(self.clusters):
            for sample_idx in cluster_idx:
                labels[sample_idx] = label
        return labels

    def _find_nearest_points(self, centrodis):
        return np.array([nearest_poins(x, centrodis) for x in self.sample])

    def _update_cendrodis(self, idxs):
        centrodis = []
        for ith, _ in enumerate(self.centrodis):
            i_idxs = np.argwhere(idxs == ith)[:, 0]
            samples = self.sample[i_idxs]
            centrodis.append(np.mean(samples, axis=0))

        return np.array(centrodis)

    def _update_cluster(self, idxs):
        for ith, _ in enumerate(self.centrodis):
            i_idxs = np.argwhere(idxs == ith)[:, 0]
            self.clusters[ith] = i_idxs

    def _is_converged(self, old_cendrodis, cendrodis):
        dist = euclidean_distance(old_cendrodis, cendrodis)
        return dist == 0
    
    def plot_step(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for lable, idxs in enumerate(self.clusters):
            point = self.sample[idxs].T
            ax.scatter(*point)

        for point in self.centrodis:
            ax.scatter(*point, marker='x', color="black", linewidth=2)

        plt.show()