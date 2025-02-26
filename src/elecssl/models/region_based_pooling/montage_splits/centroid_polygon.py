import copy
import dataclasses
import warnings
from typing import NamedTuple, List, Tuple, Dict, Optional

import matplotlib
import numpy
from matplotlib import pyplot
from shapely import Polygon, LineString, Point

from elecssl.data.datasets.getter import get_channel_system
from elecssl.models.region_based_pooling.montage_splits.montage_split_base import MontageSplitBase
from elecssl.models.region_based_pooling.utils import RegionID, project_to_2d


# -----------------
# Small convenient classes
# -----------------
class CPRegionID(RegionID):

    id: str

    def __lt__(self, other):
        # Type check
        if not isinstance(other, type(self)):
            raise TypeError(f"Cannot compute '<' for other types than {type(self)}, but found {type(other)}")

        # If they are equal, return False
        if self == other:
            return False

        # Loop through all polygon IDs
        for self_polygon_id, other_polygon_id in zip(self.id.split(sep="|"), other.id.split(sep="|")):
            if int(self_polygon_id) < int(other_polygon_id):
                return True
            elif int(self_polygon_id) > int(other_polygon_id):
                return False

        # Since we have already checked if the two objects are equal, this code should not be reachable
        raise ValueError(f"Something wrong happened. LHS: {self}, RHS: {other}")


class Point2D(NamedTuple):
    x: float
    y: float


class Edge:
    """
    Class for defining an Edge between two nodes of type Point2D
    """

    __slots__ = "_node_1", "_node_2"

    def __init__(self, node_1, node_2):
        # Input check
        if not isinstance(node_1, Point2D) or not isinstance(node_2, Point2D):
            raise TypeError(f"Expected node 1 and node 2 to be of type {Point2D.__name__}, but found {type(node_1)} "
                            f"and {type(node_2)}")
        if node_1 == node_2:
            raise ValueError("The nodes passed were identical. An edge defined by two equal nodes is likely wrong")

        # -------------------
        # Set the nodes of the edge
        # -------------------
        self._node_1 = node_1
        self._node_2 = node_2

    def __repr__(self):
        """
        Method for printing

        Returns
        -------
        String for printing

        Examples
        --------
        >>> Edge(Point2D(1, 4), Point2D(5, 3))
        Edge(node_1=Point2D(x=1, y=4), node_2=Point2D(x=5, y=3))
        """
        return f"{type(self).__name__}(node_1={self._node_1}, node_2={self._node_2})"

    def line_intersect(self, point, angle):
        """
        Get the intersection point where the self-edge intersects with the ray defined by the point and angle. If there
        is no intersection, None will be returned

        Parameters
        ----------
        point : Point2D
        angle : float

        Returns
        -------
        Point2D | None

        Examples
        --------
        >>> Edge(Point2D(0, 0), Point2D(2, 2)).line_intersect(Point2D(2, 0), angle=3/4*numpy.pi)  # doctest: +ELLIPSIS
        Point2D(x=1.0, y=1.0...)

        If no intersection is found, None is returned

        >>> type(Edge(Point2D(0, 0), Point2D(2, 2)).line_intersect(Point2D(0, 2), angle=3/4*numpy.pi))
        <class 'NoneType'>
        """
        # -------------------
        # Model the ray
        # -------------------
        # The ray is represented by a line segment. If the length of this line segment is equal or greater than the
        # maximum distance from the point to the line segment, this sufficiently models the ray.
        max_distance = max((_euclidean_distance(self._node_1, point), _euclidean_distance(self._node_2, point)))
        ray = LineString(((point.x, point.y), (point.x + (max_distance + 1e-2) * numpy.cos(angle),
                                               point.y + (max_distance + 1e-2) * numpy.sin(angle))))

        # -------------------
        # Compute the intersection
        # -------------------
        intersection = LineString((self._node_1, self._node_2)).intersection(ray)
        if intersection:
            # Convert to Point2D
            return Point2D(intersection.x, intersection.y)
        else:
            # If no intersection is found, return None instead
            return None

    # ---------------
    # Properties
    # ---------------
    @property
    def node_1(self):
        return self._node_1

    @property
    def node_2(self):
        return self._node_2


# -----------------
# Larger convenient classes
# -----------------
class PolygonGraph:
    """
    Class for defining polygons. They contain nodes in an ordered tuple.

    Examples
    --------
    >>> _ = PolygonGraph(nodes=(Point2D(-4, 1), Point2D(-1, -2), Point2D(4, -4), Point2D(7, -1), Point2D(2, 3)))

    The polygon cannot itersect itself

    >>> _ = PolygonGraph(nodes=(Point2D(-4, 1), Point2D(-1, -2), Point2D(4, -4), Point2D(2, 3), Point2D(7, -1)))
    Traceback (most recent call last):
    ...
    ValueError: The polygon contains intersections
    """

    __slots__ = "_nodes"

    def __init__(self, nodes):
        """
        Initialise

        Parameters
        ----------
        nodes: tuple[Point2D, ...]
            The nodes of the Polygon. Note that the tuple is ordered, and that the (i+1)-th element will be the 'next'
            element of the i-th element. The 'next' element of the final element is the first element (cyclic)
        """
        # -----------
        # Input checks
        # -----------
        # Length check
        if len(nodes) < 3:
            raise ValueError(f"The number of nodes to define a polygon must be greater than 3, but found {len(nodes)}")

        # Check if the polygon intersects itself
        if not Polygon(tuple((node.x, node.y) for node in nodes)).is_simple:
            raise ValueError("The polygon contains intersections")

        # -----------
        # Store attribute
        # -----------
        self._nodes: Tuple[Point2D, ...] = nodes  # mypy is forcing me to have in-line type hinting here

    def line_intersection(self, point, angle):
        """
        Given a ray defined by a point and angle, this method returns the edge which it first intersects with, as well
        as the coordinates of intersection. If there is no intersection, None will be returned.

        Parameters
        ----------
        point : Point2D
        angle : float

        Returns
        -------
        tuple[Edge, Point2D] | None
            The closest edge of intersection and its coordinates

        Examples
        --------
        >>> my_poly = PolygonGraph((Point2D(-4, -3), Point2D(4, -1), Point2D(-2, 2)))
        >>> my_poly.line_intersection(point=Point2D(-1, -1), angle=numpy.pi / 4)  # doctest: +ELLIPSIS
        (Edge(node_1=Point2D(x=4, y=-1), node_2=Point2D(x=-2, y=2)), Point2D(x=0.66..., y=0.66...))

        If there is no intersect, None is returned

        >>> type(my_poly.line_intersection(Point2D(4, 4), angle=numpy.pi / 4))
        <class 'NoneType'>

        If there are multiple intersects, the closest one to the point is selected
        >>> my_poly = PolygonGraph((Point2D(-4, -2), Point2D(2, -3), Point2D(2, 4), Point2D(-4, 2), Point2D(-1, 1)))
        >>> my_poly.line_intersection(Point2D(-2, -1.5), angle=numpy.pi / 2)  # doctest: +ELLIPSIS
        (Edge(node_1=Point2D(x=-1, y=1), node_2=Point2D(x=-4, y=-2)), Point2D(x=-1.99..., y=1.2...e-16))
        """
        # Initialise lists
        intersections: List[Point2D] = []  # Will contain coordinates
        edge_intersections: List[Edge] = []  # Will contain Edge objects

        # ---------------
        # Checking all edges for intersection
        # ---------------
        for edge in self.edges:
            # Get the intersection of the line
            intersection = edge.line_intersect(point=point, angle=angle)

            # If it is not None, store it
            if intersection is not None:
                intersections.append(intersection)
                edge_intersections.append(edge)

        # If no intersections are found, return None. For the class' intended and internal use cases, this should not
        # happen. A warning is therefore raised
        if not intersections:
            warnings.warn("No intersections were found, which is likely due to an error")
            return None

        # If the polygon is non-convex, too many intersections may have been found
        if len(intersections) != 1:
            # Select the edge closest to the point
            idx = numpy.argmin([_euclidean_distance(point, intersection) for intersection in intersections])
            return edge_intersections[idx], intersections[idx]

        # Otherwise, there is only one element in the list
        return edge_intersections[0], intersections[0]

    def split(self, point, angles):
        """
        Method for splitting a PolygonGraph into multiple PolygonGraphs, given a set of rays

        todo: I basically copied this from RBP

        Parameters
        ----------
        point : Point2D
        angles : tuple[float, ...]

        Returns
        -------
        tuple[typing.Self, ...]
            Multiple PolygonGraphs, which are made by splitting the original PolygonGraph
        """
        # Loop through all angle pairs (defining the region)
        polygon_graphs: List['PolygonGraph'] = list()  # not using type Self for backwards compatibility
        for i, (angle1, angle2) in enumerate(zip(angles, angles[1:] + (angles[0],))):
            # Get the edges and position of intersection
            edge_1, pos_1 = self.line_intersection(point=point, angle=angle1)
            edge_2, pos_2 = self.line_intersection(point=point, angle=angle2)

            # Get the node of the edges intersected, in order (determined by the ordering of the nodes)
            first_split_node, second_split_node = edge_1.node_1, edge_2.node_2

            # Make a copy of the node set, and convert it to a list
            nodes = list(copy.deepcopy(self._nodes))

            # Insert the new nodes
            if edge_1 == edge_2 and \
                    (numpy.linalg.norm(numpy.array(pos_2) - numpy.array(first_split_node.position))
                     < numpy.linalg.norm(numpy.array(pos_1) - numpy.array(first_split_node.position))):
                nodes.insert(nodes.index(first_split_node) + 1, pos_1)
                nodes.insert(nodes.index(first_split_node) + 1, pos_2)
                nodes.insert(nodes.index(first_split_node) + 2, point)
            else:
                nodes.insert(nodes.index(first_split_node) + 1, pos_1)
                nodes.insert(nodes.index(second_split_node), pos_2)
                nodes.insert(nodes.index(second_split_node), point)

            # --------------------
            # Store split
            # --------------------
            i0 = nodes.index(pos_1)
            i1 = nodes.index(point)

            if i0 < i1:
                polygon_nodes = nodes[i0:(i1 + 1)]
            else:
                polygon_nodes = nodes[i0:] + nodes[:(i1 + 1)]
            polygon_graphs.append(PolygonGraph(tuple(polygon_nodes)))

        return tuple(polygon_graphs)

    def plot(self, face_color="random", edge_color="black", line_width=2):
        """
        Plot the polygon

        Parameters
        ----------
        face_color : str
        edge_color : str
        line_width : int

        Returns
        -------
        None
        """
        # -------------
        # Get positions
        # -------------
        node_positions = tuple((node.x, node.y) for node in self._nodes)

        # Get as x and y values for compatibility with pyplot.fill()
        x, y = zip(*node_positions)

        # -------------
        # Plotting
        # -------------
        if face_color == "random":
            # Sample color from colormap
            cmap = matplotlib.colormaps['YlOrBr']

            # Get face color
            face_color = cmap(numpy.random.randint(low=0, high=cmap.N // 2))

        pyplot.fill(x, y, linewidth=line_width, facecolor=face_color, edgecolor=edge_color)

    # ---------------
    # Properties
    # ---------------
    @property
    def nodes(self) -> Tuple[Point2D, ...]:
        return self._nodes

    @property
    def edges(self) -> Tuple[Edge, ...]:
        """Generate edges from the nodes"""
        # Initialise edge list. All edges will be appended to this list
        edge_list: List[Edge] = []

        # Loop through nodes, and get both the i-th and the i+1-th element (not using itertools.pairwise for backward
        # compatibility)
        for node_1, node_2 in zip(self._nodes[:-1], self._nodes[1:]):
            # Append edge between the i-th and the i+1-th element to the set of edges
            edge_list.append(Edge(node_1=node_1, node_2=node_2))

        # Add a connection from the last element to the first element, and return as a tuple
        edge_list.append(Edge(node_1=self._nodes[-1], node_2=self._nodes[0]))
        return tuple(edge_list)


# -----------------
# Main class
# -----------------
class CentroidPolygons(MontageSplitBase):
    """
    The algorithm used in the original RBP paper (Tveitstøl et al., submitted)

    However, I am struggling to make this work for multiple channel systems...
    """

    __slots__ = "_polygon", "_centroid", "_angles", "_child_polygons", "_children_split"

    def __init__(self, channel_positions, k, min_nodes, add_node_noise=True, _polygon="auto"):
        """
        Initialise

        Parameters
        ----------
        channel_positions: dict[str, tuple[Point2D, ...]] | tuple[str, ...]
            Keys are channel system/dataset names, values are 2D projected electrode positions. Or just dataset names
        k: tuple[int, ...]
            Split vector
        min_nodes: int
            Stopping criteria
        add_node_noise: bool
        _polygon: PolygonGraph | str
        """
        # -----------------------
        # Maybe load the channel systems
        # -----------------------
        if isinstance(channel_positions, (tuple, list)) and all(isinstance(channel_system, str)
                                                                for channel_system in channel_positions):
            # If multiple dataset names are provided, we will fit on the intersection of channel systems
            channel_positions = _get_channel_positions_intersection(channel_positions)

        # -----------------------
        # Maybe add a small noise to the channel positions.
        # This is to prevent aligned electrodes
        # -----------------------
        if add_node_noise:
            channel_positions = {name: tuple(Point2D(pos.x + numpy.random.uniform(low=-1e-8, high=1e-8),
                                                     pos.y + numpy.random.uniform(low=-1e-8, high=1e-8))
                                             for pos in positions) for name, positions in channel_positions.items()}

        # -----------------------
        # Store the polygon defining the region
        # -----------------------
        # Get or define the polygon graph of the entire region, before splitting
        if _polygon == "auto":
            # Design a box which usually capture the MNE default channel systems
            _x_min = -0.17
            _x_max = 0.17
            _y_min = -0.17
            _y_max = 0.17

            node_a = Point2D(_x_min, _y_min)
            node_b = Point2D(_x_max, _y_min)
            node_c = Point2D(_x_max, _y_max)
            node_d = Point2D(_x_min, _y_max)

            # Set the polygon
            self._polygon = PolygonGraph(nodes=(node_a, node_b, node_c, node_d))

        elif isinstance(_polygon, PolygonGraph):
            self._polygon = _polygon

        else:
            raise ValueError(f"Expected the polygon to be 'auto' of type {PolygonGraph.__name__}, but found type "
                             f"{type(_polygon)}")

        # -----------------------
        # Compute the centroid and angles of
        # separating lines todo: is this really the best approach?
        # -----------------------
        # Compute separately for the channel systems. Then, use the midpoint of midpoints (to weight the channel systems
        # equally)
        _midpoints = tuple(_compute_centroid(nodes=positions) for positions in channel_positions.values())
        midpoint = _compute_centroid(_midpoints)

        # Check if the centroid is contained in the Polygon. If this is not the case, a 'representative point' which is
        # guaranteed to be contained in the polygon is used instead
        if not Polygon(tuple((node.x, node.y) for node in self._polygon.nodes)).contains(Point((midpoint.x,
                                                                                                midpoint.y))):
            midpoint = Polygon(tuple((node.x, node.y) for node in self._polygon.nodes)).representative_point()
            midpoint = Point2D(midpoint.x, midpoint.y)

        # Check if the polygon intersects itself. Should never happen, but it was a part of the RBP code, and there
        # is no reason to remove it
        if not Polygon(tuple((node.x, node.y) for node in self._polygon.nodes)).is_simple:
            raise ValueError("Polygon contains intersection, this should never happen")

        # Selecting separating angles by averaging
        _seed = numpy.random.randint(1_000_000_000)
        angles = tuple(_compute_separating_angles(nodes=positions, centroid=midpoint, k=k[0], seed=_seed)
                       for positions in channel_positions.values())
        angles = tuple(numpy.mean(numpy.array(angles), axis=0))

        # Store attributes
        self._centroid = midpoint
        self._angles = angles

        # -----------------------
        # Compute new polygons (split the current one
        # into regions)
        # -----------------------
        self._child_polygons = self._polygon.split(point=self._centroid, angles=self._angles)

        # -----------------------
        # Place the points of the channel systems in
        # the newly created polygons (needed for further computations)
        # -----------------------
        placed_nodes = _place_nodes_in_polygons(nodes=channel_positions, polygons=self._child_polygons)

        # Maybe regret recursion (this step was part of RBP)
        if not (len(placed_nodes) == len(self._child_polygons) == k[0]):
            raise RegretRecursion(f"Inconsistencies in calculating the nuber of regions: "
                                  f"{len(placed_nodes), len(self._child_polygons), k[0]}")

        # -----------------------
        # (Maybe) split children polygons
        # -----------------------
        self._children_split: Dict[int, Optional['CentroidPolygons']] = dict()
        try:
            for region, polygon_nodes in placed_nodes.items():
                # Check if the split vector is finished or the stopping criteria is reached for any of the provided
                # channel systems for any of the child regions
                if len(k) == 1 or any(len(ch_nodes) // k[1] < min_nodes for region_nodes in placed_nodes.values()
                                      for ch_nodes in region_nodes.values()):
                    self._children_split[region] = None
                else:
                    self._children_split[region] = CentroidPolygons(channel_positions=polygon_nodes, k=k[1:],
                                                                    min_nodes=min_nodes, add_node_noise=False,
                                                                    _polygon=self._child_polygons[region])
        except RegretRecursion:
            for region in placed_nodes:
                self._children_split[region] = None

    # ---------------
    # Methods for placing points in regions
    # ---------------
    def _place_single_node(self, node, _color_sequence=()):
        """
        Method for placing a single node in a region

        Parameters
        ----------
        node : Point2D
        _color_sequence : tuple[int, ...]

        Returns
        -------
        CPRegionID
        """
        # Place the point and add it to the sequence (here, we imagine that the node is 'coloured' with an integer)
        node_color = _place_single_node_in_polygon(node=node, polygons=self._child_polygons)
        color_sequence = _color_sequence + (node_color,)

        # Maybe initiate child split
        region = None
        for polygon_id, child_split in self._children_split.items():
            if node_color == polygon_id and child_split is not None:
                region = child_split._place_single_node(node=node, _color_sequence=color_sequence)
                break

        # Return CPRegionID if and only if the node split is the final one
        if all(child_split is None for child_split in self._children_split.values()):
            # While the original implementation of RBP used a clever trick with prime numbers to map the color sequence
            # to a unique integer, here we just convert it to a string
            return CPRegionID("|".join(str(color) for color in color_sequence))

        return region

    def place_in_regions(self, electrodes_3d):
        """
        Place electrodes in regions

        Parameters
        ----------
        electrodes_3d : cdl_eeg.models.region_based_pooling.utils.ELECTRODES_3D

        Returns
        -------
        cdl_eeg.models.region_based_pooling.utils.CHANNELS_IN_MONTAGE_SPLIT
        """
        # Make a 2D-projection
        electrodes_2d = project_to_2d(electrode_positions=electrodes_3d)

        # Place all electrode using their 2D projections
        channels_in_regions: Dict[CPRegionID, List[str]] = dict()
        for electrode_name, position in electrodes_2d.items():
            # Place in region and store in dict
            region = self._place_single_node(node=Point2D(*position))

            if region not in channels_in_regions:
                channels_in_regions[region] = []

            channels_in_regions[region].append(electrode_name)

        # Order the dict correctly
        ordered_regions = _order_region_ids(tuple(channels_in_regions.keys()))
        channels_in_regions = {reg: channels_in_regions[reg] for reg in ordered_regions}

        # Return with correct type
        return {id_: tuple(ch_names) for id_, ch_names in channels_in_regions.items()}

    # ---------------
    # Methods for plotting
    # ---------------
    def plot(self, face_color="random", edge_color="black", line_width=2):
        """
        Method for plotting the polygons

        Parameters
        ----------
        face_color : str
        edge_color : str
        line_width : int

        Returns
        -------
        None
        """
        # ------------------
        # Initiate plot children polygons,
        # if they exist
        # ------------------
        for child_split in self._children_split.values():
            if child_split is not None:
                child_split.plot(face_color=face_color, edge_color=edge_color, line_width=line_width)

        # ------------------
        # Plot if and only if there are no children
        # nodes (the current node is a terminal node)
        # ------------------
        if all(child_split is None for child_split in self._children_split.values()):
            for child_polygon in self._child_polygons:
                child_polygon.plot(face_color=face_color, edge_color=edge_color, line_width=line_width)

    # ---------------
    # Properties
    # ---------------
    def _get_num_regions(self, _sum: int = 0) -> int:
        """
        Method for computing the number of regions

        Parameters
        ----------
        _sum : int
            Internal use only!

        Returns
        -------
        int
            Number of regions
        """
        # ------------------
        # Go to children splits if they exist
        # ------------------
        for child_split in self._children_split.values():
            if child_split is not None:
                _sum = child_split._get_num_regions(_sum=_sum)

        # ------------------
        # Return if there are no children nodes
        # ------------------
        if all(child_split is None for child_split in self._children_split.values()):
            return len(self._children_split) + _sum

        return _sum

    @property
    def num_regions(self) -> int:
        return self._get_num_regions()


# -----------------
# Handling multiple names for the same electrodes
# -----------------
_EQUIVALENT_CHANNELS = frozenset({frozenset({"T4", "T8"}), frozenset({"T3", "T7"})})


def _equivalent_channels(ch_1, ch_2):
    for equivalent_channel_set in _EQUIVALENT_CHANNELS:
        if ch_1 in equivalent_channel_set and ch_2 in equivalent_channel_set:
            return True
    return False


@dataclasses.dataclass(frozen=True)
class _ChannelName:
    """
    This class handles the problem of channels having multiple names in EEG

    Examples
    --------
    >>> _ChannelName("T3") == _ChannelName("T7")
    True
    >>> _ChannelName("T4") == _ChannelName("T8")
    True
    >>> _ChannelName("P1") == _ChannelName("P1")
    True
    >>> _ChannelName("NotReallyAChannelName") == _ChannelName("NotReallyAChannelName")  # But True is still returned
    True
    >>> _ChannelName("P1") == _ChannelName("P3")
    False
    """

    name: str

    def __eq__(self, other):
        # Type check
        if not isinstance(other, self.__class__):
            return False

        # If the names are the same, True should be returned
        if self.name == other.name:
            return True

        # Check if the 'other' channel name is equivalent to the self channel name, just with different as per EEG
        # convention
        return _equivalent_channels(self.name, other.name)


# -----------------
# Functions
# -----------------
def _get_channel_positions_intersection(dataset_names):
    """
    Function for getting the channel positions which are present in all requested channel systems

    Returns
    -------

    """
    # Loop through all datasets
    merged_channel_system = {}
    for i, dataset_name in enumerate(dataset_names):
        # If it is the first
        channel_system = get_channel_system(dataset_name=dataset_name).electrode_positions
        if i == 0:
            merged_channel_system = get_channel_system(dataset_name=dataset_name).electrode_positions
            continue

        # Find all channels which are not present in the current channel system
        to_delete = set()
        for channel_name in merged_channel_system:
            if _ChannelName(channel_name) not in (_ChannelName(ch_name) for ch_name in channel_system):
                to_delete.add(channel_name)

        # Remove all channels which are not present in the current channel system
        for channel_name in to_delete:
            del merged_channel_system[channel_name]

    # Get it on the correct form
    return {"MergeDataset": _electrode_2d_to_point_tuple(project_to_2d(merged_channel_system))}


def _order_region_ids(region_ids):
    """
    Function for ordering region IDs

    Parameters
    ----------
    region_ids : tuple[CPRegionID, ...]

    Returns
    -------
    tuple[RegionID, ...]

    Examples
    --------
    >>> _order_region_ids((CPRegionID(id='0|1'), CPRegionID(id='1|0'), CPRegionID(id='0|0'), CPRegionID(id='1|1')))
    (CPRegionID(id='0|0'), CPRegionID(id='0|1'), CPRegionID(id='1|0'), CPRegionID(id='1|1'))
    >>> _order_region_ids((CPRegionID(id='10|0|20'), CPRegionID(id='0|10'), CPRegionID(id='0|11|0'),
    ...                    CPRegionID(id='11|1|11')))
    (CPRegionID(id='0|10'), CPRegionID(id='0|11|0'), CPRegionID(id='10|0|20'), CPRegionID(id='11|1|11'))
    """
    return tuple(sorted(region_ids))


def _electrode_2d_to_point_tuple(electrodes_2d):
    """
    Convert from Electrodes2D to a tuple of Point2D (channel names are omitted)

    Parameters
    ----------
    electrodes_2d : cdl_eeg.models.region_based_pooling.utils.ELECTRODES_2D

    Returns
    -------
    tuple[Point2D, ...]
    """
    return tuple(Point2D(*pos) for pos in electrodes_2d.values())


def _euclidean_distance(node_1, node_2):
    """
    Compute the Euclidean distance between two nodes

    Parameters
    ----------
    node_1 : Point2D
    node_2 : Point2D

    Returns
    -------
    float

    Examples
    --------
    >>> _euclidean_distance(Point2D(0, 0), Point2D(1, 1))  # doctest: +ELLIPSIS
    np.float64(1.414...)
    """
    return numpy.linalg.norm((node_1.x - node_2.x, node_1.y - node_2.y))


def _compute_centroid(nodes):
    """
    Function for computing the centroid 2D position of a tuple of nodes

    Parameters
    ----------
    nodes : tuple[Point2D, ...]

    Returns
    -------
    Point2D

    Examples
    --------
    >>> _compute_centroid((Point2D(0.2, 0.1), Point2D(0.3, -50)))
    Point2D(x=np.float64(0.25), y=np.float64(-24.95))
    """
    return Point2D(*numpy.mean(numpy.array([(node.x, node.y) for node in nodes]), axis=0))


def _compute_separating_angles(nodes, centroid, k, seed=None):
    """
    Get the angles for splitting into k regions

    Parameters
    ----------
    nodes: tuple[Point2D, ...]
    centroid: Point2D
    k: int
    seed : int | None

    Returns
    -------
    tuple[float, ...]
        The angles of the separating lines, with length k.

    Examples
    --------
    >>> numpy.random.seed(2)
    >>> _compute_separating_angles((Point2D(-4, 1), Point2D(2, 3), Point2D(4, -4)), Point2D(1, 1), 3)
    ... # doctest: +ELLIPSIS
    (2.73..., np.float64(4.19...), np.float64(0.038...))

    The number of points must be equal or greater than the number of separating lines

    >>> _compute_separating_angles((Point2D(-4, 1), Point2D(2, 3), Point2D(4, -4)), Point2D(1, 1), 4)
    Traceback (most recent call last):
    ...
    AssertionError: The number of points must be equal or greater than the number of separating lines, but found 3 and 4
    """
    if seed is not None:
        numpy.random.seed(seed)

    # Input check
    assert len(nodes) >= k, (f"The number of points must be equal or greater than the number of separating lines, but "
                             f"found {len(nodes)} and {k}")

    # Get the positions only, as numpy arrays
    node_positions = numpy.array([(point.x, point.y) for point in nodes])

    # Set origin to centroid
    node_positions -= numpy.array([centroid.x, centroid.y])

    # Randomly select a starting angle
    start_angle = numpy.random.uniform(0, 2 * numpy.pi)

    # Compute angles with respect to start angle
    angles = numpy.mod(numpy.mod(numpy.arctan2(node_positions[:, 1], node_positions[:, 0]), 2 * numpy.pi) - start_angle,
                       2 * numpy.pi)

    # Sort angles (not the best implementation. But it is probably not significant anyway)
    sorted_angles = numpy.sort(numpy.insert(angles, 0, 0))

    # Make k partitions
    partitions = numpy.array_split(sorted_angles, indices_or_sections=k)

    # Compute separating angles
    separating_angles = [numpy.mod((p0[-1] + p1[0]) / 2 + start_angle, 2 * numpy.pi) for p0, p1 in zip(partitions[:-1],
                                                                                                       partitions[1:])]

    # Add the starting angle as separating angle
    separating_angles.insert(0, start_angle)

    return tuple(separating_angles)


def _place_single_node_in_polygon(node, polygons):
    """
    Place a single node in a polygon. The index of the first polygon the node is in will be returned.

    Parameters
    ----------
    node : Point2D
    polygons : tuple[PolygonGraph, ...]

    Returns
    -------
    int
    """
    # Try all polygons
    for i, polygon in enumerate(polygons):
        # If the node is contained in the current polygon, return it
        if Polygon(tuple((node.x, node.y) for node in polygon.nodes)).contains(Point(node.x, node.y)):
            return i

    # Could not be placed in a polygon
    return -1


def _place_nodes_in_polygons(nodes, polygons):
    """
    Function for placing nodes in polygons. The nodes will be placed in the first polygon it is in. If a node is not in
    any of the polygons, it will not be added to the output dict

    Parameters
    ----------
    nodes : dict[str, tuple[Point2D, ...]]
    polygons : tuple[PolygonGraph, ...]

    Returns
    -------
    dict[int, dict[str, tuple[Point2D, ...]]]
    """
    placed_nodes: Dict[int, Dict[str, List[Point2D]]] = dict()
    # Loop through all channel systems
    for ch_system_name, node_positions in nodes.items():
        # Loop through all nodes/electrodes
        for node in node_positions:
            # Try all polygons
            for i, polygon in enumerate(polygons):
                # If the node is contained in the current polygon, store it
                if Polygon(tuple((vertex.x, vertex.y) for vertex in polygon.nodes)).contains(Point((node.x, node.y))):
                    # Maybe add the group as a key
                    if i not in placed_nodes:
                        placed_nodes[i] = dict()

                    # Maybe add the channel system to the polygon
                    if ch_system_name not in placed_nodes[i]:
                        placed_nodes[i][ch_system_name] = []

                    # Place node
                    placed_nodes[i][ch_system_name].append(node)

                    # No need to check the other polygons. TODO: what to do if a node is not contained in any polygon?
                    break
    return {polygon_number: {ch_system_name: tuple(positions)
                             for ch_system_name, positions in polygon_contained.items()}
            for polygon_number, polygon_contained in placed_nodes.items()}


# -----------------
# Errors
# -----------------
class RegretRecursion(Exception):
    ...


if __name__ == "__main__":
    # Just a note to self: with some variations of hyperparameters it looks like the splitting is wrong. This, however,
    # occurred because T4 and T8 has the same position. So no worries
    import random

    import mne

    numpy.random.seed(7)
    random.seed(7)

    # ------------------
    # Get/compute the electrode positions
    # ------------------
    # 3D
    montage = mne.channels.make_standard_montage("standard_1020")
    my_nodes_3d = montage.get_positions()["ch_pos"]
    my_nodes_3d = {key: (value[0], value[1], value[2]) for key, value in my_nodes_3d.items()}

    # 2D and numpy arrays
    my_nodes_2d = project_to_2d(my_nodes_3d)
    my_points_ = tuple(Point2D(*node_) for node_ in my_nodes_2d.values())

    # ------------------
    # Generate split and plot it
    # ------------------
    # my_split_ = CentroidPolygons({"my_dataset": my_points_}, min_nodes=1, k=(5, 5, 3))
    my_split_ = CentroidPolygons(channel_positions=("LEMON", "Wang"), min_nodes=6, k=[3, 3, 3, 3, 3, 3, 3, 3])

    print(f"Number of regions: {my_split_.num_regions}")
    pyplot.figure()
    my_split_.plot(edge_color="black", line_width=2)

    # ------------------
    # Plot the electrodes
    # ------------------
    # Place in regions
    my_placed_electrodes = my_split_.place_in_regions(my_nodes_3d)

    # Loop through to plot all regions separately
    for region_, electrodes in my_placed_electrodes.items():
        # Get all channels and their positions
        positions_ = tuple(my_nodes_2d[ch_name] for ch_name in electrodes)
        x_, y_ = zip(*positions_)
        pyplot.scatter(x_, y_)

    pyplot.show()
