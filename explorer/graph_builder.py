# explorer/graph_builder.py
import networkx as nx

from .models import SAEFeature, SAERun


def build_knowledge_graph(run_id, threshold=0.1):
    """
    Costruisce il Knowledge Graph (MST) dalle feature interpretate.
    Ritorna un dizionario JSON-ready per Vis.js/D3.
    """
    run = SAERun.objects.get(pk=run_id)
    features = SAEFeature.objects.filter(run=run)

    # 1. Costruisci il grafo completo dalle co-occorrenze salvate
    G = nx.Graph()

    # Dizionario rapido per densità: {id: density}
    densities = {}
    labels = {}

    for f in features:
        # Aggiungi nodo
        # Usiamo l'ID feature come identificativo, ma salviamo label per display
        G.add_node(f.feature_index, label=f.label or f"Feat {f.feature_index}",
                   density=f.density or 0, group='feature')
        densities[f.feature_index] = f.density or 0
        labels[f.feature_index] = f.label or f"Feat {f.feature_index}"

        # Aggiungi archi (basati su co_occurring_features salvati da statistics.py)
        if f.co_occurring_features:
            for co in f.co_occurring_features:
                target_idx = co['index']
                weight = co['score'] # Probabilità condizionata

                # Filtro soglia (Thresholding)
                if weight > threshold:
                    # Nota: NetworkX MST minimizza, quindi usiamo weight negativo o 'maximum' dopo
                    G.add_edge(f.feature_index, target_idx, weight=weight)

    # 2. Calcola Maximum Spanning Tree (MST)
    # Il paper usa MST per eliminare cicli e tenere solo le connessioni più forti ("scheletro")
    mst = nx.maximum_spanning_tree(G, weight='weight')

    # 3. Orienta il grafo (Gerarchia)
    # Archi diretti da Alta Densità (Generale) -> Bassa Densità (Specifico)
    directed_G = nx.DiGraph()

    for u, v, data in mst.edges(data=True):
        # Aggiungi nodi se mancano (es. feature non interpretate ma citate nelle co-occorrenze)
        if u not in densities: densities[u] = 0
        if v not in densities: densities[v] = 0

        # Direzione: Parent (più denso) -> Child (meno denso)
        if densities[u] >= densities[v]:
            source, target = u, v
        else:
            source, target = v, u

        directed_G.add_edge(source, target, weight=data['weight'])

        # Copia attributi nodi
        for node in [source, target]:
            if node not in directed_G.nodes:
                directed_G.add_node(node, label=labels.get(node, f"Feat {node}"),
                                    density=densities.get(node, 0))

    # 4. Formatta per Vis.js
    nodes = []
    edges = []

    for n in directed_G.nodes(data=True):
        # Dimensione nodo basata sulla densità (log scale fake per visibilità)
        size = 10 + (n[1].get('density', 0) * 1000)
        nodes.append({
            'id': n[0],
            'label': n[1].get('label', str(n[0])),
            'value': size,
            'title': f"Density: {n[1].get('density', 0):.4f}", # Tooltip
            'group': 'feature'
        })

    for u, v, data in directed_G.edges(data=True):
        edges.append({
            'from': u,
            'to': v,
            'value': data['weight'], # Spessore arco
            'arrows': 'to',
            'color': {'opacity': 0.6}
        })

    return {'nodes': nodes, 'edges': edges}
