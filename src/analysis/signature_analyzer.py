import torch
import numpy as np
from sklearn.cluster import KMeans
from src.models.lstm import AttentionLSTM

class SignatureAnalyzer:
    """
    Pillar 4: Psychological Signatures (Attention & Clustering)
    Identifies physiological profiles associated with loss of agency
    and identity instability by clustering high-attention window features.
    """
    def __init__(self, model_path, input_dim, hidden_dim, num_layers, output_dim, device='cpu'):
        self.device = device
        self.model = AttentionLSTM(input_dim, hidden_dim, num_layers, output_dim).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def cluster_signatures(self, data_loader, n_clusters=3):
        """
        Extracts attention-weighted profiles and groups them into signature clusters.
        """
        all_profiles = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_x in data_loader:
                x = batch_x.to(self.device)
                _, attn = self.model(x) # attn shape (B, S, 1)
                
                # Weighted average of features at high-attention timestamps
                # Profile represents the 'biological state' most relevant to the model's decision
                weighted_x = torch.sum(attn * x, dim=1).cpu().numpy()
                all_profiles.append(weighted_x)
        
        profiles = np.concatenate(all_profiles, axis=0)
        
        # Unsupervised discovery of recurring states
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(profiles)
        
        return clusters, profiles, kmeans.cluster_centers_

    def map_to_psychological_states(self, centers, feature_names):
        """
        Scale-aware heuristic mapping of cluster centroids to psychological states.

        After Z-score normalisation all feature values are in standard-deviation
        units — absolute cuts like `EDA > 0.6` lose their meaning.  Instead we
        rank clusters by their centroid values for each key indicator and use the
        *relative ordering* to assign states.

        Mapping rules
        ──────────────
        • Cluster with the **highest EDA + lowest HRV rank sum** → highest
          sympathetic activation  → "Loss of Agency / Cognitive Overload"
        • Cluster with the **highest ACC magnitude** among the remaining two
          → "Identity Instability / High Agitation" (physical restlessness proxy)
        • Remaining cluster → "Stable Algorithmic Pacing"
        """
        n = len(centers)
        f_idx = {name: i for i, name in enumerate(feature_names)}

        # Extract key indicator vectors across clusters  (shape: n_clusters,)
        eda_vals = np.array([c[f_idx['EDA']] for c in centers]) if 'EDA' in f_idx else np.zeros(n)
        hrv_vals = np.array([c[f_idx['HRV']] for c in centers]) if 'HRV' in f_idx else np.zeros(n)
        acc_mag  = np.zeros(n)
        for ax in ('ACC_x', 'ACC_y', 'ACC_z'):
            if ax in f_idx:
                acc_mag += np.array([c[f_idx[ax]] for c in centers]) ** 2
        acc_mag = np.sqrt(acc_mag)

        # Rank clusters (higher rank = larger value)
        eda_rank =  np.argsort(np.argsort(eda_vals))   # ascending → rank 0 = lowest
        hrv_rank = -np.argsort(np.argsort(hrv_vals))   # we want LOW hrv → high score
        stress_score = eda_rank + hrv_rank              # composite stress indicator

        # Assign states based on relative ranking
        assigned = {}
        remaining = list(range(n))

        # State 1: highest combined stress score → Loss of Agency
        stress_cluster = int(np.argmax(stress_score))
        assigned[stress_cluster] = "Loss of Agency / Cognitive Overload"
        remaining = [i for i in remaining if i != stress_cluster]

        # State 2: highest ACC magnitude among remaining → High Agitation
        if remaining:
            acc_remaining = [(c, acc_mag[c]) for c in remaining]
            agitation_cluster = max(acc_remaining, key=lambda x: x[1])[0]
            assigned[agitation_cluster] = "Identity Instability / High Agitation"
            remaining = [i for i in remaining if i != agitation_cluster]

        # State 3: remaining cluster(s) → Stable
        for i in remaining:
            assigned[i] = "Stable Algorithmic Pacing"

        mappings = []
        for i, center in enumerate(centers):
            f_profile = dict(zip(feature_names, center))
            mappings.append({
                "cluster_id": i,
                "profile":    f_profile,
                "state":      assigned.get(i, "Stable Algorithmic Pacing"),
            })
        return mappings

