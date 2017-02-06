import numpy as np

class Cell:

    def __init__(self, inp_dimension, inp_corner=None, inp_width=None):
        self.dimension = inp_dimension
        self.corner = np.empty((self.dimension))
        self.width = np.empty((self.dimension))
        if inp_corner is not None:
            self.corner = inp_corner
            self.width = inp_width

    def get_corner(self):
        return self.corner

    def get_width(self):
        return self.width

    def contains_point(self, point):
        if np.any(self.corner - self.width > point):
            return False
        elif np.any(self.corner + self.width < point):
            return False
        else:
            return True


class SPTree:
    def __init__(self, inp_dimension, inp_data, inp_num_of_points=None,
                 inp_parent=None, inp_corner=None, inp_width=None):

        # Fixed constant (max number of objects in a node)
        self.qt_node_capacity = 1

        # Properties of the node in the tree
        self.parent = None
        self.dimension = inp_dimension
        self.is_leaf = True
        self.size = 0
        self.cum_size = 0

        # Axis-aligned bounding box stored as a center with half-dimensions to represent the boundaries of this quad tree
        self.boundary = Cell(self.dimension)

        # Indices in this space-partitioning tree node, corresponding center-of-mass, and list of all children
        self.data = inp_data
        self.index = np.empty((self.qt_node_capacity))
        self.center_of_mass = np.zeros(self.dimension)

        # Children
        self.num_children = 2 ** self.dimension
        self.children = np.empty(self.num_children)

        # A buffer we use when doing force computations
        self.buffer = np.empty(self.dimension)

        # Build the tree starting from a node based on the data (default use)
        if inp_parent is None and inp_corner is None and inp_width is None and inp_num_of_points is not None:
            min_of_data = np.min(inp_data, axis=0)
            max_of_data = np.max(inp_data, axis=0)
            mean_of_data = np.mean(inp_data, axis=0)
            width = np.max(np.array([max_of_data - mean_of_data, mean_of_data - min_of_data]), axis=1)

            self.boundary.corner = mean_of_data
            self.boundary.width = width
            self.fill(inp_num_of_points)

        # Build the tree starting from a node of given size and position
        if inp_parent is None and inp_num_of_points is not None and inp_corner is not None and inp_width is not None:
            self.boundary.corner = inp_corner
            self.boundary.width = inp_width
            self.fill(inp_num_of_points)

        # Generate a base node of particular size and position but do not fill the tree
        if inp_parent is None and inp_num_of_points is None and inp_corner is not None and inp_width is not None:
            self.boundary.corner = inp_corner
            self.boundary.width = inp_width

        # Generate a tree starting from a node of a particular size, position and parent but do not fill the tree
        if inp_parent is not None and inp_num_of_points is None and inp_corner is not None and inp_width is not None:
            self.parent = inp_parent
            self.boundary.corner = inp_corner
            self.boundary.width = inp_width

        # Generate a tree starting from a node of a particular size, position and parent and fill the tree
        if inp_parent is not None and inp_num_of_points is not None and inp_corner is not None and inp_width is not None:
            self.parent = inp_parent
            self.boundary.corner = inp_corner
            self.boundary.width = inp_width
            self.fill(inp_num_of_points)

    def insert(self, new_index):
        point = self.data[new_index, :]
        if not self.boundary.contains_point(point):
            return False

        self.cum_size += 1
        mult1 = (self.cum_size - 1) / self.cum_size
        mult2 = 1.0 / self.cum_size
        self.center_of_mass *= mult1
        self.center_of_mass += (mult2 * point)

        if self.is_leaf and self.size < self.qt_node_capacity:
            self.index[self.size] = new_index
            self.size += 1
            return True

        any_dublicate = False
        for i in np.arange(self.size):
            if np.any(point != self.data[self.index[i]]):
                duplicate = False




        return False

    def fill(self, num_of_points):
        for i in np.arange(num_of_points):
            self.insert(i)