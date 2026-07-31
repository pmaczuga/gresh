from src.gresh import Gresh, NodeType


def export_avsucd(g: Gresh, filename: str):
    """
    Export Gresh as AVS UCD (Unstructured Cell Data).

    The standard extension is `.inp`.
    """
    with open(filename, "w") as f:
        n_map = dict()
        i_map = dict()
        list_nodes = list(g.nodes_except_type(NodeType.INTERIOR))
        list_interiors = list(g.interiors())

        # Number of nodes, number of cells, node data length, cell data length, model data length
        f.write(f"{len(list_nodes)} {len(list_interiors)} 4 1 0\n")

        counter = 0
        for n in list_nodes:
            x, y, z = g.xyz(n)
            counter = counter + 1
            # Node ID, coords
            f.write(f"{counter} {x} {y} {z}\n")
            n_map[n] = counter

        counter = 0
        for i in list_interiors:
            n1, n2, n3 = g.interior_connectivity(i)
            counter = counter + 1
            # Cell ID, material, cell type, nodes
            f.write(f"{counter} 0 tri {n_map[n1]} {n_map[n2]} {n_map[n3]}\n")
            i_map[i] = counter

        # 2 properties: first ones has 1 value, second one has 3
        f.write("2 1 3\n")
        # name of the first property: "vertex_id", and it doesn't have a unit ("nunits")
        f.write("vertex_id,nunits\n")
        # name of the second property: "uve", units: "degree"
        f.write("uve,degree\n")
        counter = 0
        for n in list_nodes:
            counter += 1
            u, v, e = g.uve(n)
            # ID, Vertex ID, coords in uve
            f.write(f"{counter} {n_map[n]} {u} {v} {e}\n")

        # 1 property with single value
        f.write("1 1\n")
        # name of the property and its unit (no unit: "nunits")
        f.write("interior_id,nunits\n")
        counter = 0
        for i in list_interiors:
            counter += 1
            # ID, Cell ID
            f.write(f"{counter} {i_map[i]}\n")
