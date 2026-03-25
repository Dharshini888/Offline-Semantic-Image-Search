import numpy as np
from sklearn.cluster import DBSCAN
import datetime
import logging

logger = logging.getLogger("ClusteringEngine")

class ClusteringEngine:
    def detect_events(self, images_metadata):
        """
        Cluster images into events (albums) by time only.

        Previous version fed [hours, lat, lon] into DBSCAN with a single eps.
        That was broken in two ways:
          1. GPS degrees and hours are in completely different scales — for
             images without GPS (lat=0, lon=0 default) the GPS columns are
             useless noise; for images WITH GPS the large degree values
             dominate and blow up distances.
          2. min_samples=2 discarded every photo that was taken alone on a
             day, so most images became noise and got no album.

        Fix: cluster by time only (hours since first photo), min_samples=1
        so every photo belongs to an album, and eps=12 so photos within 12
        hours of each other are grouped into the same event.

        Returns list of integer cluster labels (len == len(images_metadata)).
        All values are >= 0 (no noise).
        """
        if not images_metadata:
            return []

        if len(images_metadata) == 1:
            return [0]

        start_time = min(m['timestamp'] for m in images_metadata)

        # Build 1-D feature: hours since first photo
        hours = np.array([
            [(m['timestamp'] - start_time).total_seconds() / 3600.0]
            for m in images_metadata
        ], dtype=np.float32)

        # eps=12 → photos within 12 hours are the same event
        # min_samples=1 → every photo belongs to some album (no noise)
        model = DBSCAN(eps=12.0, min_samples=1, metric="euclidean")
        labels = model.fit_predict(hours)
        return labels.tolist()

clustering_engine = ClusteringEngine()

