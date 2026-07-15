"""
academic_system/knowledge_graph.py
====================================
Builds a concept knowledge graph from lecture analysis results.

Nodes  = unique concepts / terms extracted across all frames
Edges  = co-occurrence on the same slide (weight = count)
       + definition relationships (term → definition node)

Outputs
-------
  • NetworkX DiGraph  — server-side analysis
  • D3.js JSON        — {"nodes": [...], "links": [...]}   for the frontend
  • API JSON          — full graph + centrality stats       for the endpoint

Usage
-----
    builder = KnowledgeGraphBuilder()
    graph   = builder.build(frame_analyses, audio_topics, lecture_summary)
    d3_data = builder.to_d3_json(graph)     # → GET /results/graph/{stem}
    builder.save(graph, "graphs/lecture.json")
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False


class KnowledgeGraphBuilder:
    """
    Builds and exports a concept-relationship graph from pipeline outputs.

    Parameters
    ----------
    min_edge_weight : int
        Minimum co-occurrence count to include an edge in the graph.
    max_nodes : int
        Prune lowest-degree nodes to keep the graph readable.
    """

    def __init__(self, min_edge_weight: int = 1, max_nodes: int = 80) -> None:
        self.min_edge_weight = min_edge_weight
        self.max_nodes       = max_nodes

    # ── public API ────────────────────────────────────────────────────────────

    def build(
        self,
        frame_analyses:  List[Dict],
        audio_topics:    Dict,
        lecture_summary: Dict,
    ) -> Any:
        """Build the graph. Returns nx.DiGraph or a plain dict fallback."""
        freq:    Dict[str, int]                = defaultdict(int)
        cooccur: Dict[Tuple[str, str], int]    = defaultdict(int)
        all_defs: List[Tuple[str, str]]        = []

        concepts_per_frame: List[List[str]] = []

        for frame in frame_analyses:
            ac = frame.get("academic_content", {})
            if not isinstance(ac, dict):
                continue

            concepts = [c.strip().lower() for c in ac.get("key_concepts", []) if c.strip()]
            if concepts:
                concepts_per_frame.append(concepts)
                for c in concepts:
                    freq[c] += 1

            for d in ac.get("definitions", []):
                term = d.get("term", "").strip()
                defn = d.get("definition", "").strip()
                if term and defn:
                    all_defs.append((term.lower(), defn))
                    freq[term.lower()] += 1

        # Audio concepts — audio_topics["key_concepts"] can be dicts or strings
        for c in audio_topics.get("key_concepts", []):
            if isinstance(c, dict):
                name = (c.get("concept") or c.get("name") or c.get("term") or "").strip().lower()
            else:
                name = str(c).strip().lower()
            if name:
                freq[name] += 1
                concepts_per_frame.append([name])

        # Summary topics — skip overly verbose strings (> 5 words = not a concept)
        for t in lecture_summary.get("main_topics", []):
            if isinstance(t, str) and t.strip():
                words = t.strip().split()
                if len(words) <= 5:   # keep short topic labels only
                    freq[t.strip().lower()] += 1

        # Co-occurrence edges
        for frame_concepts in concepts_per_frame:
            uniq = list(dict.fromkeys(frame_concepts))   # preserve order, dedupe
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    a, b = sorted([uniq[i], uniq[j]])
                    cooccur[(a, b)] += 1

        if NX_AVAILABLE:
            return self._build_nx(freq, cooccur, all_defs, lecture_summary)
        return self._build_dict(freq, cooccur, all_defs)

    def to_d3_json(self, graph: Any) -> Dict:
        """D3.js force-directed format: {"nodes": [...], "links": [...]}."""
        if NX_AVAILABLE and isinstance(graph, (nx.Graph, nx.DiGraph)):
            return self._nx_to_d3(graph)
        return self._dict_to_d3(graph)

    def to_api_json(self, graph: Any) -> Dict:
        """Full graph data with centrality metrics for the REST endpoint."""
        if NX_AVAILABLE and isinstance(graph, (nx.Graph, nx.DiGraph)):
            return self._nx_to_api(graph)
        raw = graph if isinstance(graph, dict) else {}
        return {
            "num_nodes": len(raw.get("nodes", [])),
            "num_edges": len(raw.get("edges", [])),
            **raw,
            "central_concepts": self._central_dict(raw),
        }

    def save(self, graph: Any, path: str) -> str:
        """Save D3-ready JSON to disk. Returns the path."""
        data = self.to_d3_json(graph)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    # ── NetworkX path ─────────────────────────────────────────────────────────

    def _build_nx(
        self,
        freq:    Dict[str, int],
        cooccur: Dict[Tuple[str, str], int],
        defs:    List[Tuple[str, str]],
        summary: Dict,
    ) -> "nx.DiGraph":
        G = nx.DiGraph()

        for concept, f in freq.items():
            G.add_node(concept, type="concept", freq=f, label=concept.title())

        for term, defn in defs:
            d_id = f"def:{term}"
            if term not in G:
                G.add_node(term, type="concept", freq=1, label=term.title())
            G.add_node(d_id, type="definition", label=defn[:80], full=defn)
            G.add_edge(term, d_id, relation="defines", weight=1)

        for (a, b), w in cooccur.items():
            if w >= self.min_edge_weight and a in G and b in G:
                G.add_edge(a, b, relation="co_occurs", weight=w)

        title = summary.get("lecture_title", "")
        if title:
            G.add_node("__lecture__", type="lecture", label=title)
            for t in summary.get("main_topics", []):
                if isinstance(t, str) and t.strip().lower() in G:
                    G.add_edge("__lecture__", t.strip().lower(), relation="covers", weight=1)

        # Prune to max_nodes by degree
        if G.number_of_nodes() > self.max_nodes:
            deg    = dict(G.degree())
            keep   = set(sorted(deg, key=lambda x: -deg[x])[: self.max_nodes])
            remove = [n for n in list(G.nodes) if n not in keep]
            G.remove_nodes_from(remove)

        return G

    def _nx_to_d3(self, G: "nx.DiGraph") -> Dict:
        node_idx = {n: i for i, n in enumerate(G.nodes)}
        return {
            "nodes": [
                {
                    "id":    n,
                    "index": node_idx[n],
                    "label": G.nodes[n].get("label", n),
                    "type":  G.nodes[n].get("type", "concept"),
                    "freq":  G.nodes[n].get("freq", 1),
                }
                for n in G.nodes
            ],
            "links": [
                {
                    "source":   node_idx[u],
                    "target":   node_idx[v],
                    "relation": G.edges[u, v].get("relation", "related"),
                    "weight":   G.edges[u, v].get("weight", 1),
                }
                for u, v in G.edges
            ],
        }

    def _nx_to_api(self, G: "nx.DiGraph") -> Dict:
        centrality = nx.degree_centrality(G)
        central = [
            n for n, _ in sorted(centrality.items(), key=lambda x: -x[1])
            if G.nodes[n].get("type") == "concept"
        ][:10]
        return {
            "num_nodes":        G.number_of_nodes(),
            "num_edges":        G.number_of_edges(),
            "central_concepts": central,
            "nodes": [{"id": n, **G.nodes[n]} for n in G.nodes],
            "edges": [
                {"source": u, "target": v, **G.edges[u, v]}
                for u, v in G.edges
            ],
        }

    # ── dict fallback (no networkx) ───────────────────────────────────────────

    def _build_dict(
        self,
        freq:    Dict[str, int],
        cooccur: Dict[Tuple[str, str], int],
        defs:    List[Tuple[str, str]],
    ) -> Dict:
        return {
            "nodes": [
                {"id": c, "type": "concept", "freq": f, "label": c.title()}
                for c, f in freq.items()
            ],
            "edges": [
                {"source": a, "target": b, "relation": "co_occurs", "weight": w}
                for (a, b), w in cooccur.items()
                if w >= self.min_edge_weight
            ],
        }

    def _dict_to_d3(self, g: Dict) -> Dict:
        nodes    = g.get("nodes", [])
        node_idx = {n["id"]: i for i, n in enumerate(nodes)}
        return {
            "nodes": nodes,
            "links": [
                {
                    "source":   node_idx.get(e["source"], e["source"]),
                    "target":   node_idx.get(e["target"], e["target"]),
                    "relation": e.get("relation", "related"),
                    "weight":   e.get("weight", 1),
                }
                for e in g.get("edges", [])
            ],
        }

    @staticmethod
    def _central_dict(g: Dict) -> List[str]:
        nodes = g.get("nodes", [])
        return [
            n["id"] for n in sorted(nodes, key=lambda x: -x.get("freq", 0))
            if n.get("type") == "concept"
        ][:10]