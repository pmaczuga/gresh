from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable, Self, cast, override

import numpy as np
import numpy.typing as npt
import rustworkx as rx

from src.rectangle_mesh import rectangle_triangular_mesh


class NodeType(Enum):
    VERTEX = 1
    HANGING = 2
    INTERIOR = 3


class NodeData(ABC):
    @abstractmethod
    def type(self) -> NodeType:
        pass


class VertexData(NodeData):
    def __init__(self, xyz: npt.NDArray[np.float64], uve: npt.NDArray[np.float64]):
        self._xyz = xyz
        self._uve = uve

    @override
    def type(self) -> NodeType:
        return NodeType.VERTEX

    def xyz(self) -> npt.NDArray[np.float64]:
        return self._xyz

    def uve(self) -> npt.NDArray[np.float64]:
        return self._uve


class HangingData(VertexData):
    def __init__(
        self,
        xyz: npt.NDArray[np.float64],
        uve: npt.NDArray[np.float64],
        v1: int,
        v2: int,
    ):
        self._xyz = xyz
        self._uve = uve
        self._v1 = v1
        self._v2 = v2

    @override
    def type(self) -> NodeType:
        return NodeType.HANGING

    def to_vertex_data(self):
        return VertexData(self.xyz(), self.uve())

    @classmethod
    def from_vertex_data(cls, vertex_data: VertexData, v1: int, v2: int) -> Self:
        return cls(vertex_data.xyz(), vertex_data.uve(), v1, v2)


class InteriorData(NodeData):
    def __init__(self, v1: int, v2: int, v3: int, refine: bool):
        self._v1 = v1
        self._v2 = v2
        self._v3 = v3
        self._refine = refine

    def refine(self):
        return self._refine

    @override
    def type(self) -> NodeType:
        return NodeType.INTERIOR


class EdgeData:
    def __init__(self, boundary: bool = False):
        self._boundary = boundary

    def boundary(self) -> bool:
        return self._boundary


class AddVertexStrategy(Enum):
    USE_UVE = 1
    USE_XYZ = 2


class Gresh:
    """
    Gresh is a graph representing triangle mesh and has three types of
    nodes:
    - `VERTEX` - normal vertex of graph
    - `HANGING` - whether node is a hanging node - used during refinement, final graph will not have those
    - `INTERIOR` - node representing inside of triangle.

    Note: the following notation is used for the whole library:
    - "node" is a node of a graph
    - "vertex" is a vertex of a mesh
    So when I say "node", it can mean any of the types: `VERTEX`, `HANGING`, `INTERIOR`.
    But "vertex" can only be `VERTEX` or `HANGING` (since `INTERIOR` is not a vertex of a mesh, but
    rather a "meta" representation of an element.)

    Nodes are indexed by integers starting at 0.

    Node Properties
    ----------
    - `VERTEX`
        - `xyz` - carstesian coordinates of vertex
        - `elevation`- elevation of point above sea level (or below when negative)
        - `uv` - mapping of vertex to flat surface (e.g. latitude, longitude)
        - `uve` - `uv` + `elevation`
    - `HANGING`
        - all properties of `VERTEX` plus
        - `v1`, `v2` - hanging node lies between vertices `v1` and `v2`
    - `INTERIOR`:
        - `refine: bool`: whether this triangle should be refined

    Edge Properties
    ---------------
    - `boundary:bool` - whether this edge lies on boundary of mesh

    Simple usage
    ------------
    1. Create `Gresh()` graph
    2. Mark triangles that needs to be refined
    3. Use `refine` to break all the marked triangles
    4. Go to (2.)

    Refiner properties
    ------------------
    Properties of refiner that can be adjusted:
    - Which coordinates are used during refinement: `USE_UVE` or `USE_XYZ`
    - How to convert from `uve` to `xyz` (or the other way around)
    - How to calculate coordinates of new vertex based on its neighbors
    - How to calculate distance between two vertices - longest edge will be broken

    All of those can be controlled by implementing methods for following functions:
    - `convert`
    - `distance`
    - `new_vertex_coords`
    `USE_UVE` or `USE_XYZ` is passed as an argument in class constructor.

    Custom types
    ------------
    To easiest way to create custom graph type using `Gresh` is the following:

    - Create a subclass of Gresh (with all the custom fields you need)
    - Overwrite any of the following functions, whose behavior you want to adjust:
        - `convert`
        - `distance`
        - `new_vertex_coords`
        For example implementation see the default methods in this class.
    - Have fun

    See also
    --------
    `refine_xyz`, `refine_uve!`
    """

    graph: rx.PyGraph[NodeData, EdgeData]
    _vertex_count: int
    _hanging_count: int
    _interior_count: int
    _add_vertex_strategy: AddVertexStrategy

    def add_vertex_strategy(self) -> AddVertexStrategy:
        """What coordinates are used when new vertex is added.

        Can be either `AddVertexStrategy.USE_XYZ` or `AddVertexStrategy.USE_UVE`

        Notes
        -----
        Shifts behavior of ``add_vertex`` and `refine` methods.
        """
        return self._add_vertex_strategy

    def convert(self, coords: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert 3-element vector coords` from `uve` to `xyz` (or reverse).

        - If `add_vertex_strategy` returns `USE_UVE` new vertex will be created
        using `uve` coordinates and then converted to `xyz` using this function.
        - If `USE_XYZ` is returned, it will go the other way around.

        Defaults to identity.

        # Note
        Shifts behavior of `add_vertex` and `refine`

        Method taking parameterized `MeshGraph` can be created to adjust it to your
        needs.
        """
        return coords

    def distance(self, v1: int, v2: int) -> float:
        """Calculate distance between vertices `v1` and `v2` in graph `g`.

        Longest edge according to that distance will always be broken.
        Defaults to Euclidean distance of `xyz` or `uve` coordinates
        (depending on `add_vertex_strategy`.

        # Note
        Shifts behavior of `refine`

        Method taking parameterized `MeshGraph` can be created to adjust it to your
        needs.
        """
        if self.add_vertex_strategy() == AddVertexStrategy.USE_XYZ:
            res = np.linalg.norm(self.xyz(v1) - self.xyz(v2))
        else:
            res = np.linalg.norm(self.uve(v1) - self.uve(v2))
        return cast(float, res)

    def new_vertex_coords(self, v1: int, v2: int) -> npt.NDArray[np.float64]:
        """
        Calculate coordinates of vertex created when breaking edge based on neighbors
        (two vertices previously connected with now broken edge). Return
        coords of new vertex (as 3-element Vector).
        Defaults to average of `v1` and `v2`, coordinates.

        # Note
        Shifts behavior of `refine`

        Method taking parameterized `MeshGraph` can be created to adjust it to your
        needs.
        """
        coords_fun = (
            self.xyz
            if self.add_vertex_strategy() == AddVertexStrategy.USE_UVE
            else self.uve
        )
        return np.mean([coords_fun(v1), coords_fun(v2)], axis=0)

    def __init__(
        self, add_vertex_strategy: AddVertexStrategy = AddVertexStrategy.USE_XYZ
    ):
        self.graph = rx.PyGraph[NodeData, EdgeData](multigraph=False)
        self._vertex_count = 0
        self._hanging_count = 0
        self._interior_count = 0
        self._add_vertex_strategy = add_vertex_strategy

    def __str__(self):
        vs = self.vertex_count()
        ins = self.interior_count()
        es = self.graph.num_edges() - ins * 3  # only real edges
        hs = self.hanging_count()
        g_type = type(self).__name__
        string = f"{g_type} with ({vs} vertices), ({ins} interiors) and ({es} edges)"
        if hs:
            string += f" that has !{hs} hanging nodes!"
        return string

    # -----------------------------------------------------------------------------
    # ------ Methods for adding and removing vertices and edges -------------------
    # -----------------------------------------------------------------------------

    def add_vertex(self, coords: npt.NDArray[np.float64]) -> int:
        """
        Add new vertex to graph. Coords used (`xyz` or `uve`) depend on
        `Gresh.add_vertex_strategy`. Conversion from one to another is done using
        `Gresh.convert`.
        """
        coords = coords.copy()
        converted_coords = self.convert(coords)
        use_uve = self.add_vertex_strategy() == AddVertexStrategy.USE_UVE
        xyz_coords, uve_coords = (
            (converted_coords, coords) if use_uve else (coords, converted_coords)
        )
        node_data = VertexData(xyz_coords, uve_coords)
        node = self.graph.add_node(node_data)
        self._vertex_count += 1
        return node

    def add_hanging(self, coords: npt.NDArray[np.float64], v1: int, v2: int) -> int:
        """
        Add hanging node between vertices `v1` and `v2`. Return its `id`. Other arguments
        are similar to `Gresh.add_vertex!`.

        # Note
        Only add new vertex with type `HANGING`. **No** other changes will be made
        (specifically no edges will be added or removed).

        See Also
        --------
        Gresh.add_vertex : Add normal vertex.
        """
        v = self.add_vertex(coords)
        self.set_hanging(v, v1, v2)
        return v

    def add_pure_interior(self, v1: int, v2: int, v3: int, refine: bool = False) -> int:
        "Add interior without edges connecting vertices `v1`, `v2`, `v3` with one another."
        data = InteriorData(v1, v2, v3, refine)
        id = self.graph.add_node(data)
        self.graph.add_edge(v1, id, EdgeData())
        self.graph.add_edge(v2, id, EdgeData())
        self.graph.add_edge(v3, id, EdgeData())
        self._interior_count += 1
        return id

    def add_interior(self, v1: int, v2: int, v3: int, refine: bool = False):
        """
        Add interior to graph `g` that represents triangle with vertices `v1`, `v2` and
        `v3`. Return its `id`.

        # Note
        This **will** create edges between those vertices, as well as edges
        between new `INTERIOR` vertex and each of the three.
        """
        id = self.add_pure_interior(v1, v2, v3, refine)
        self.add_edge(v1, v2)
        self.add_edge(v2, v3)
        self.add_edge(v3, v1)
        return id

    def add_edge(self, n1: int, n2: int, boundary: bool = False):
        """
        Add edge between nodes `n1` and `n2`. Set `boundary` flag if delivered.
        If edge already exists the flag will be updated.
        """
        data = EdgeData(boundary)
        e = self.graph.add_edge(n1, n2, data)
        return e

    def remove_node(self, n: int):
        "Remove node `n` of any type from graph."
        data = self.graph.get_node_data(n)
        if data.type() == NodeType.VERTEX:
            self._vertex_count -= 1
        elif data.type() == NodeType.HANGING:
            self._hanging_count -= 1
        else:
            self._interior_count -= 1
        self.graph.remove_node(n)

    def remove_edge(self, n1: int, n2: int):
        "Remove edge from `n1` to `n2` from graph."
        self.graph.remove_edge(n1, n2)

    # -----------------------------------------------------------------------------
    # ------ Methods counting elements fo graph  ----------------------------------
    # -----------------------------------------------------------------------------

    def node_count(self) -> int:
        "Number of nodes of any type in graph."
        return self.graph.num_nodes()

    def vertex_count(self) -> int:
        "Number of normal vertices in graph."
        return self._vertex_count

    def hanging_count(self) -> int:
        "Number of hanging nodes in graph."
        return self._hanging_count

    def interior_count(self) -> int:
        "Number of interiors in graph."
        return self._interior_count

    # -----------------------------------------------------------------------------
    # ------ Iterators over vertices ----------------------------------------------
    # -----------------------------------------------------------------------------

    def nodes_with_type(self, type: NodeType) -> Iterable[int]:
        "Return all vertices with type `type`"
        return self.graph.filter_nodes(lambda data: data.type() == type)

    def nodes_except_type(self, type: NodeType) -> Iterable[int]:
        "Return all vertices with type different from `type`"
        return self.graph.filter_nodes(lambda data: data.type() != type)

    def vertices(self) -> Iterable[int]:
        "Return all vertices with type `VERTEX`"
        return self.nodes_with_type(NodeType.VERTEX)

    def hanging_nodes(self) -> Iterable[int]:
        "Return all vertices with type `HANGING`"
        return self.nodes_with_type(NodeType.HANGING)

    def interiors(self) -> Iterable[int]:
        "Return all vertices with type `INTERIOR`"
        return self.nodes_with_type(NodeType.INTERIOR)

    def neighbors(self, n: int) -> Iterable[int]:
        "Return neighbors with all types of vertex `n`"
        ids = self.graph.neighbors(n)
        return np.array(ids, dtype=np.int32)

    def neighbors_with_type(self, n: int, type: NodeType) -> Iterable[int]:
        "Return neighbors of node `n` with type equal to `type`."

        def type_filter(nn):
            return self.graph.get_node_data(nn).type() == type

        return filter(type_filter, self.neighbors(n))

    def neighbors_except_type(self, n: int, type: NodeType) -> Iterable[int]:
        "Return neighbors of node `n` with type different than `type`."

        def type_filter(nn):
            return self.graph.get_node_data(nn).type() != type

        return filter(type_filter, self.neighbors(n))

    def vertex_neighbors(self, n: int) -> Iterable[int]:
        "Return neighbors with type `VERTEX` of node `n`"
        return self.neighbors_with_type(n, NodeType.VERTEX)

    def hanging_neighbors(self, n: int) -> Iterable[int]:
        "Return neighbors with type `HANGING` of node `n`"
        return self.neighbors_with_type(n, NodeType.HANGING)

    def interior_neighbors(self, n: int) -> Iterable[int]:
        "Return neighbors with type `INTERIOR` of node `n`"
        return self.neighbors_with_type(n, NodeType.INTERIOR)

    def is_ordinary_edge(self, n1: int, n2: int) -> bool:
        """Check if edge between `n1` `n2` is ordinary, that is if it doesn't connect
        `INTERIOR` to its vertices."""
        return not self.is_interior(n1) and not self.is_interior(n2)

    def all_edges(self) -> Iterable[tuple[int, int]]:
        """Return *all* edges in graph `g` (including possibly edges between interiors
        and) its vertices. To get ordinary edges use `Gresh.edges`."""
        return self.graph.edge_list()

    def edges(self) -> Iterable[tuple[int, int]]:
        """Return ordinary edges in graph. To get all edges use
        `Gresh.all_edges`"""
        return filter(lambda e: self.is_ordinary_edge(e[0], e[1]), self.all_edges())

    def interior_connectivity(self, i: int) -> Iterable[int]:
        return self.neighbors(i)

    # -----------------------------------------------------------------------------
    # ------ Functions handling vertex properties  --------------------------------
    # -----------------------------------------------------------------------------

    def get_vertex_data(self, v: int) -> VertexData:
        node_data = self.graph.get_node_data(v)
        if not isinstance(node_data, VertexData):
            raise TypeError(
                f"Trying to get VertexData from node of type {node_data.type().name}"
            )
        return node_data

    def get_hanging_data(self, v: int) -> HangingData:
        node_data = self.graph.get_node_data(v)
        if not isinstance(node_data, HangingData):
            raise TypeError(
                f"Trying to get HangingData from node of type {node_data.type().name}"
            )
        return node_data

    def get_interior_data(self, v: int) -> InteriorData:
        node_data = self.graph.get_node_data(v)
        if not isinstance(node_data, InteriorData):
            raise TypeError(
                f"Trying to get InteriorData from node of type {node_data.type().name}"
            )
        return node_data

    def set_hanging(self, v: int, v1: int, v2: int):
        """
        Change type of vertex `v` to `hanging` from `vertex` and set its 'parents' to
        `v1` and `v2`.
        """
        if not self.is_hanging(v):
            self._hanging_count += 1
            self._vertex_count -= 1
        vertex_data = self.get_vertex_data(v)
        hanging_data = HangingData.from_vertex_data(vertex_data, v1, v2)
        self.graph[v] = hanging_data

    def unset_hanging(self, v: int):
        "Change type of vertex to `vertex` from `hanging`."
        if not self.is_hanging(v):
            return
        hanging_data = self.get_hanging_data(v)
        self.graph[v] = hanging_data.to_vertex_data()
        self._hanging_count -= 1
        self._vertex_count += 1

    def xyz(self, v: int) -> npt.NDArray[np.float64]:
        "Return vector with `xyz` coordinates of vertex `v`"
        return self.get_vertex_data(v).xyz()

    def uv(self, v: int) -> npt.NDArray[np.float64]:
        "Return vector with `uv` coordinates of vertex `v`"
        return self.uve(v)[:2]

    def uve(self, v: int) -> npt.NDArray[np.float64]:
        "Return array `[u, v, elevation]`"
        return self.get_vertex_data(v).uve()

    def get_elevation(self, n: int) -> np.float64:
        return self.uve(n)[2]

    def get_type(self, n: int) -> NodeType:
        return self.graph.get_node_data(n).type()

    def is_hanging(self, v: int) -> bool:
        return self.graph.get_node_data(v).type() == NodeType.HANGING

    def is_vertex(self, v: int) -> bool:
        return self.graph.get_node_data(v).type() == NodeType.VERTEX

    def is_interior(self, v: int) -> bool:
        return self.graph.get_node_data(v).type() == NodeType.INTERIOR

    def set_elevation(self, n: int, elevation: float):
        vertex_data = self.get_vertex_data(n)
        vertex_data._uve[2] = elevation

        if self.add_vertex_strategy() == AddVertexStrategy.USE_UVE:
            vertex_data._xyz = self.convert(vertex_data._xyz)
        else:
            vertex_data._xyz[2] = elevation
        self.graph[n] = vertex_data

    def should_refine(self, i: int) -> bool:
        return self.get_interior_data(i).refine()

    def set_refine(self, i: int):
        data = self.get_interior_data(i)
        data._refine = True
        self.graph[i] = data

    def unset_refine(self, i: int):
        data = self.get_interior_data(i)
        data._refine = False
        self.graph[i] = data

    # -----------------------------------------------------------------------------
    # ------ Functions handling edge properties -----------------------------------
    # -----------------------------------------------------------------------------

    def is_on_boundary(self, n1: int, n2: int) -> bool:
        return self.graph.get_edge_data(n1, n2).boundary()

    def set_boundary(self, n1: int, n2: int):
        data = self.graph.get_edge_data(n1, n2)
        data._boundary = True
        self.graph.update_edge(n1, n2, data)

    def unset_boundary(self, n1: int, n2: int):
        data = self.graph.get_edge_data(n1, n2)
        data._boundary = False
        self.graph.update_edge(n1, n2, data)

    def has_edge(self, n1: int, n2: int) -> bool:
        return self.graph.has_edge(n1, n2)

    # -----------------------------------------------------------------------------
    # ------ Other functions ------------------------------------------------------
    # -----------------------------------------------------------------------------

    def has_hanging_nodes(self) -> bool:
        "Whether graph has any hanging nodes"
        return self.hanging_count() != 0

    def get_hanging_node_between(self, v1: int, v2: int) -> int | None:
        "Get hanging node between vertices `v1` and `v2` in graph."
        if self.has_edge(v1, v2):
            return None
        hnodes1 = set(self.hanging_neighbors(v1))
        hnodes2 = set(self.hanging_neighbors(v2))
        hnodes_all = hnodes1.intersection(hnodes2)

        for h in hnodes_all:
            data = self.get_hanging_data(h)
            h_is_between = [data._v1, data._v2]
            if v1 in h_is_between and v2 in h_is_between:
                return h

        return None

    def vertex_map(self) -> dict:
        """
        Return dictionary that maps id's of all vertices with type `vertex` or `hanging`
        to number starting at 1.

        Note
        ----
        Removing vertices from graph **will** make previously generated mapping
        deprecated.
        """
        return dict(
            [(v, i) for (i, v) in enumerate(self.nodes_except_type(NodeType.INTERIOR))]
        )

    def update_boundaries(self):
        for v1, v2 in self.edges():
            interiors1 = self.interior_neighbors(v1)
            interiors2 = self.interior_neighbors(v2)
            interiors = set(interiors1).intersection(set(interiors2))
            if len(interiors) == 1:
                self.set_boundary(v1, v2)
            else:
                self.unset_boundary(v1, v2)

    @classmethod
    def on_rectangle(
        cls,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        n_elem_x: int,
        n_elem_y: int,
    ):
        r"""
        Create instance on a rectangular domain.

        For `n_elem_x = 3` and `n_elem_y = 2` we get:
        ```
        8-----9-----10---11
        |\    |\    |\    |
        | \   | \   | \   |
        |  \  |  \  |  \  |
        |   \ |   \ |   \ |
        |    \|    \|    \|
        4-----5-----6-----7
        |\    |\    |\    |
        | \   | \   | \   |
        |  \  |  \  |  \  |
        |   \ |   \ |   \ |
        |    \|    \|    \|
        0-----1-----2-----3
        ```
        with coordinates of vertex 0: `(x_min, y_min)`
        and of vertex 11: `(x_max, y_max)`.
        """
        coords, conec = rectangle_triangular_mesh(
            x_min, x_max, y_min, y_max, n_elem_x, n_elem_y
        )
        return cls.from_connectivity(
            coords,
            conec,
        )

    @classmethod
    def from_connectivity(
        cls,
        coords: npt.NDArray[np.float64],
        conec: npt.NDArray[np.int32],
    ):
        """
        Create instance from 2D vertex coordinates and their connectivity.

        - `coords` should be a matrix, where each column is a vector of point coordinates.
        - `conec` is a connectivity in form of a matrix, where each column
          is a vector of vertex ID's (starting from 0).
        """
        g = cls(AddVertexStrategy.USE_UVE)

        for uv in coords.T:
            g.add_vertex(np.append(uv, 0))

        for vs in conec.T:
            g.add_interior(*vs)
            g.add_edge(vs[0], vs[1])
            g.add_edge(vs[1], vs[2])
            g.add_edge(vs[2], vs[0])

        g.update_boundaries()
        return g
