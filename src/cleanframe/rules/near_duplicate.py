import hashlib
import random
from typing import Any

import narwhals as nw

from ..base import BaseRule
from ..types import Decision


def _find_lsh_params(num_perm: int, threshold: float) -> tuple[int, int]:
    # Find b and r to maximize recall for the target threshold.
    # We choose b and r such that the theoretical LSH threshold (1/b)**(1/r)
    # is less than or equal to the target threshold, minimizing the difference.
    best_b, best_r = 1, num_perm
    best_diff = 1.0
    for r in range(1, num_perm + 1):
        b = num_perm // r
        if b == 0:
            continue
        lsh_thresh = (1.0 / b) ** (1.0 / r)
        if lsh_thresh <= threshold:
            diff = threshold - lsh_thresh
            if diff < best_diff:
                best_diff = diff
                best_b, best_r = b, r
    return best_b, best_r


class NearDuplicateDetector(BaseRule):
    """
    Locality-Sensitive Hashing (LSH) and MinHash algorithm to detect near-duplicate rows.
    """

    def __init__(self, num_perm: int = 32, threshold: float = 0.85) -> None:
        self.num_perm = num_perm
        self.threshold = threshold

    def detect(self, df: Any, params: dict[str, Any]) -> list[Decision]:
        num_perm = params.get("num_perm", self.num_perm)
        threshold = params.get("threshold", self.threshold)

        ndf = nw.from_native(df)
        nrows = ndf.shape[0]
        if nrows <= 1:
            return []

        # Convert rows into concatenated string documents using Narwhals.
        col_exprs = [nw.col(c) for c in ndf.columns]
        concatenated_df = ndf.select(
            nw.concat_str(col_exprs, separator=" ", ignore_nulls=True).alias("doc")
        )
        docs = concatenated_df["doc"].to_list()

        # Tokenize documents using character 3-grams
        docs_tokens: list[set[str]] = []
        for doc in docs:
            if doc is None:
                docs_tokens.append(set())
            else:
                doc_lower = doc.lower()
                shingles = {doc_lower[i : i + 3] for i in range(len(doc_lower) - 2)}
                docs_tokens.append(shingles)

        # Compute MinHash signatures
        PRIME = 4294967311
        rng = random.Random(42)
        a_coeffs = [rng.randint(1, PRIME - 1) for _ in range(num_perm)]
        b_coeffs = [rng.randint(0, PRIME - 1) for _ in range(num_perm)]

        unique_tokens = set()
        for tokens in docs_tokens:
            unique_tokens.update(tokens)

        token_hashes: dict[str, int] = {}
        for token in unique_tokens:
            token_hashes[token] = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)

        signatures: list[list[int]] = []
        for tokens in docs_tokens:
            if not tokens:
                signatures.append([PRIME] * num_perm)
                continue

            doc_hashes = [token_hashes[t] for t in tokens]
            sig = []
            for i in range(num_perm):
                a = a_coeffs[i]
                b = b_coeffs[i]
                min_val = min((a * h + b) % PRIME for h in doc_hashes)
                sig.append(min_val)
            signatures.append(sig)

        # LSH band bucketing
        b, r = _find_lsh_params(num_perm, threshold)
        buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
        for row_idx, sig in enumerate(signatures):
            for band_idx in range(b):
                band_val = tuple(sig[band_idx * r : (band_idx + 1) * r])
                bucket_key = (band_idx, band_val)
                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                buckets[bucket_key].append(row_idx)

        # Generate candidates from buckets
        candidates = set()
        max_bucket_size = 500  # Safety threshold to prevent pair explosions
        for row_indices in buckets.values():
            if 1 < len(row_indices) <= max_bucket_size:
                for i in range(len(row_indices)):
                    for j in range(i + 1, len(row_indices)):
                        idx1 = row_indices[i]
                        idx2 = row_indices[j]
                        candidates.add((min(idx1, idx2), max(idx1, idx2)))

        # Exact comparison ONLY on candidate pairs
        duplicate_pairs = set()
        try:
            from rapidfuzz.fuzz import token_sort_ratio
            has_rapidfuzz = True
        except ImportError:
            has_rapidfuzz = False

        for idx1, idx2 in candidates:
            tokens1 = docs_tokens[idx1]
            tokens2 = docs_tokens[idx2]

            # Jaccard similarity of token sets
            if not tokens1 and not tokens2:
                jaccard = 1.0
            elif not tokens1 or not tokens2:
                jaccard = 0.0
            else:
                jaccard = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

            is_dup = (jaccard >= threshold)

            # Fallback to RapidFuzz for fuzzy matching if Jaccard falls slightly short
            if not is_dup and has_rapidfuzz:
                d1 = docs[idx1] or ""
                d2 = docs[idx2] or ""
                score = token_sort_ratio(d1, d2) / 100.0
                if score >= threshold:
                    is_dup = True

            if is_dup:
                duplicate_pairs.add((idx1, idx2))

        if not duplicate_pairs:
            return []

        # Find connected components (clusters) using BFS
        from collections import defaultdict
        adj = defaultdict(list)
        nodes = set()
        for u, v in duplicate_pairs:
            adj[u].append(v)
            adj[v].append(u)
            nodes.add(u)
            nodes.add(v)

        visited = set()
        clusters = []
        for node in sorted(nodes):
            if node not in visited:
                cluster = []
                queue = [node]
                visited.add(node)
                while queue:
                    curr = queue.pop(0)
                    cluster.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                clusters.append(sorted(cluster))

        sorted_clusters = sorted(clusters, key=lambda c: c[0])
        total_dups = sum(len(c) for c in sorted_clusters)

        return [
            Decision(
                rule_name=self.name,
                column="all",
                action="flag_duplicates",
                parameters={
                    "clusters": sorted_clusters,
                    "num_duplicates": total_dups,
                    "threshold": threshold,
                    "num_perm": num_perm,
                },
                signal_strength=total_dups / nrows,
                rationale=(
                    f"Detected {len(sorted_clusters)} near-duplicate cluster(s) "
                    f"containing {total_dups} total rows at threshold {threshold}."
                ),
            )
        ]

    def transform(self, df: Any, decisions: list[Decision]) -> Any:
        # NearDuplicateDetector only flags duplicate clusters; it does not mutate the dataframe.
        return df

    def explain(self, decisions: list[Decision]) -> str:
        if not decisions:
            return "No near-duplicate rows detected."
        lines = []
        for d in decisions:
            clusters = d.parameters.get("clusters", [])
            num_duplicates = d.parameters.get("num_duplicates", 0)
            threshold = d.parameters.get("threshold", self.threshold)
            lines.append(
                f"Detected {len(clusters)} near-duplicate cluster(s) containing "
                f"{num_duplicates} rows with similarity >= {threshold}."
            )
        return "\n".join(lines)
