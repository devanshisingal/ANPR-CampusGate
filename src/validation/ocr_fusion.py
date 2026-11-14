"""src.validation.ocr_fusion

Multi-frame OCR fusion for the ALPR University Gate system.

This module collects validated plate strings across a sliding window of
frames for each tracked vehicle and fuses them into a single final plate.

Key improvements vs majority voting:
- Confidence-weighted scoring (sum of confidences per cluster) rather than
  count-based majority.
- Similarity-aware merging using edit-distance clustering so common OCR
  confusions (Q↔O, 0↔O, I↔1, etc.) can converge to a single canonical plate.

The caller is responsible for:
- validating plate strings (PlateValidator)
- passing track_id, plate, and confidence
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple


def _levenshtein(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(
                min(
                    prev[j + 1] + 1,  # deletion
                    curr[j] + 1,  # insertion
                    prev[j] + (0 if c1 == c2 else 1),  # substitution
                )
            )
        prev = curr
    return prev[-1]


@dataclass(frozen=True)
class _Cluster:
    plates: List[str]
    total_conf: float
    best_plate: str


class OCRFusion:
    """Fuse OCR results across multiple frames for each track_id."""

    def __init__(
        self,
        window_size: int = 7,
        *,
        max_edit_distance: int = 2,
        min_confidence: float = 0.0,
        min_exact_votes: int = 1,
    ) -> None:
        """Args:
        window_size: Number of frames to keep per track.
        max_edit_distance: Max edit distance for merging plate variants.
        min_confidence: Optional floor to ignore very low-confidence OCR hits.
        """
        self.window_size = window_size
        self.max_edit_distance = max_edit_distance
        self.min_confidence = min_confidence
        self.min_exact_votes = min_exact_votes

        # track_id -> deque[(plate_string, confidence)]
        self._buffers: Dict[int, Deque[Tuple[str, float]]] = {}

    def add_result(
        self,
        track_id: int,
        plate: str,
        confidence: float,
        crop_area: Optional[float] = None,
    ) -> None:
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self.window_size)
        if confidence < self.min_confidence:
            return
        conf = float(confidence)
        weight = conf
        if crop_area is not None and crop_area > 0:
            scale = math.sqrt(max(float(crop_area), 100.0) / 2500.0)
            weight = conf * float(max(0.5, min(2.5, scale)))
        self._buffers[track_id].append((plate, conf, weight))

    def get_result(self, track_id: int) -> Tuple[str, float] | None:
        buf = self._buffers.get(track_id)
        if not buf:
            return None
        return self._fuse(list(buf))

    def flush(self, track_id: int) -> Tuple[str, float] | None:
        buf = self._buffers.pop(track_id, None)
        if not buf:
            return None
        return self._fuse(list(buf))

    def has_consensus(
        self,
        track_id: int,
        *,
        early_exact_votes: int = 2,
        min_consensus_conf: float = 0.75,
        direction_confirmed: Optional[bool] = None,
    ) -> bool:
        """Return True when buffered reads already justify flushing this track.

        Triggers either when the sliding window is full (with at least ``min_exact_votes``
        for the dominant plate) or when *early_exact_votes* identical reads form an
        uncontested, high-confidence majority.

        If *direction_confirmed* is explicitly False, premature early exit is
        suppressed until the sliding window is full so active vehicles have time
        to confirm travel direction before committing.
        """
        buf = self._buffers.get(track_id)
        if not buf:
            return False

        plate_confs: Dict[str, List[float]] = {}
        for entry in buf:
            plate = entry[0]
            conf = float(entry[1])
            plate_confs.setdefault(plate, []).append(conf)

        best_plate, confs = max(
            plate_confs.items(),
            key=lambda item: (len(item[1]), sum(item[1])),
        )
        if len(confs) < self.min_exact_votes:
            return False

        if direction_confirmed is False:
            return False

        if len(buf) >= self.window_size:
            avg_conf = sum(confs) / len(confs)
            return bool(avg_conf >= self.min_confidence or max(confs) >= min_consensus_conf)

        required_votes = max(self.min_exact_votes, early_exact_votes)
        if len(confs) < required_votes:
            return False

        avg_conf = sum(confs) / len(confs)
        if max(confs) < min_consensus_conf and avg_conf < min_consensus_conf:
            return False

        for other_plate, other_confs in plate_confs.items():
            if other_plate != best_plate and len(other_confs) >= len(confs):
                return False
        return True

    def active_track_ids(self) -> List[int]:
        return [tid for tid, buf in self._buffers.items() if buf]

    def flush_all(self) -> Dict[int, Tuple[str, float]]:
        results: Dict[int, Tuple[str, float]] = {}
        for track_id in list(self._buffers.keys()):
            fused = self.flush(track_id)
            if fused is not None:
                results[track_id] = fused
        return results

    # ------------------------------------------------------------------
    # Internal fusion
    # ------------------------------------------------------------------

    def _fuse(self, entries: List[Tuple]) -> Tuple[str, float]:
        """Fuse entries into (plate, fused_confidence).
        
        Uses a hybrid scoring system that balances:
        1. Frequency (how many times a plate appears)
        2. Confidence quality (average confidence per plate)
        3. Area-weighted scoring where available
        4. Confidence-weighted character-level positional voting across candidate variants
        
        This prevents high-confidence wrong plates from beating lower-confidence
        correct plates when the wrong plate appears more frequently, and resolves
        individual character confusions (e.g. T vs I) using positional character consensus.
        """
        if not entries:
            return ("", 0.0)
        if len(entries) == 1:
            return (str(entries[0][0]), float(entries[0][1]))

        # Normalize entries to (plate, conf, weight)
        norm_entries: List[Tuple[str, float, float]] = []
        for e in entries:
            p = str(e[0])
            c = float(e[1])
            w = float(e[2]) if len(e) >= 3 else c
            norm_entries.append((p, c, w))

        # Build clusters over DISTINCT plate strings.
        plate_to_confs: Dict[str, List[float]] = {}
        plate_to_weights: Dict[str, List[float]] = {}
        for plate, conf, weight in norm_entries:
            plate_to_confs.setdefault(plate, []).append(conf)
            plate_to_weights.setdefault(plate, []).append(weight)

        distinct_plates: List[str] = list(plate_to_confs.keys())

        clusters: List[List[str]] = []
        used = [False] * len(distinct_plates)

        # Greedy clustering with seeds ordered by average confidence.
        indexed = list(range(len(distinct_plates)))
        indexed.sort(
            key=lambda i: (sum(plate_to_confs[distinct_plates[i]]) / len(plate_to_confs[distinct_plates[i]])),
            reverse=True,
        )

        for i in indexed:
            if used[i]:
                continue
            seed = distinct_plates[i]
            used[i] = True
            cluster = [seed]

            for j in indexed:
                if used[j]:
                    continue
                cand = distinct_plates[j]
                if abs(len(seed) - len(cand)) > self.max_edit_distance:
                    continue
                if _levenshtein(seed, cand) <= self.max_edit_distance:
                    used[j] = True
                    cluster.append(cand)

            clusters.append(cluster)

        best_cluster: Optional[_Cluster] = None
        best_score = -1.0

        for cluster_plates in clusters:
            cluster_confs: List[float] = []
            cluster_weights: List[float] = []
            for p in cluster_plates:
                cluster_confs.extend(plate_to_confs[p])
                cluster_weights.extend(plate_to_weights[p])

            total_conf = sum(cluster_confs)
            cluster_count = len(cluster_confs)
            avg_confidence = total_conf / cluster_count if cluster_count > 0 else 0.0

            # Hybrid score: count * average_confidence
            hybrid_score = cluster_count * avg_confidence

            # Canonical plate: pick plate with highest total weighted confidence within cluster
            per_plate_sum: Dict[str, float] = {
                p: sum(plate_to_weights[p]) for p in cluster_plates
            }
            per_plate_max: Dict[str, float] = {
                p: max(plate_to_confs[p]) for p in cluster_plates
            }

            best_plate_sum = max(per_plate_sum.keys(), key=lambda p: per_plate_sum[p])
            best_plate_max = max(per_plate_max.keys(), key=lambda p: per_plate_max[p])

            candidates = list(per_plate_sum.keys())
            max_sum = per_plate_sum[best_plate_sum]
            close = [p for p in candidates if abs(per_plate_sum[p] - max_sum) <= 1e-6]
            best_plate = best_plate_max if len(close) > 1 else best_plate_sum

            if best_cluster is None or hybrid_score > best_score:
                best_cluster = _Cluster(
                    plates=cluster_plates,
                    total_conf=total_conf,
                    best_plate=best_plate,
                )
                best_score = hybrid_score
            elif abs(hybrid_score - best_score) <= 1e-6:
                if per_plate_sum[best_plate] > per_plate_sum.get(best_cluster.best_plate, 0.0):
                    best_cluster = _Cluster(
                        plates=cluster_plates,
                        total_conf=total_conf,
                        best_plate=best_plate,
                    )
                    best_score = hybrid_score

        assert best_cluster is not None

        # Confidence-weighted character-level voting across equal-length variants in cluster
        canonical_plate = best_cluster.best_plate
        same_len_plates = [p for p in best_cluster.plates if len(p) == len(canonical_plate)]
        if len(same_len_plates) > 1 and len(canonical_plate) > 0:
            voted_chars = []
            for pos in range(len(canonical_plate)):
                char_weights: Dict[str, float] = {}
                for p in same_len_plates:
                    w_sum = sum(plate_to_weights.get(p, plate_to_confs.get(p, [1.0])))
                    ch = p[pos]
                    char_weights[ch] = char_weights.get(ch, 0.0) + w_sum
                top_char = max(char_weights.keys(), key=lambda ch: char_weights[ch])
                voted_chars.append(top_char)
            consensus_str = "".join(voted_chars)
            if consensus_str in plate_to_confs or consensus_str in best_cluster.plates:
                canonical_plate = consensus_str

        # Safe gate: require at least min_exact_votes for the canonical plate or cluster seed
        votes_count = len(plate_to_confs.get(canonical_plate, []))
        if votes_count < self.min_exact_votes:
            if len(plate_to_confs.get(best_cluster.best_plate, [])) < self.min_exact_votes:
                return ("", 0.0)
            canonical_plate = best_cluster.best_plate

        best_single = max(norm_entries, key=lambda e: e[1])
        return (canonical_plate, best_single[1])

