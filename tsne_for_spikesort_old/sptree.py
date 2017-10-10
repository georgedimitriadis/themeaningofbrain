import numpy as np

class Cell:

    def __init__(self, inp_dimension, inp_corner=None, inp_width=None):

        self.dimension = inp_dimension
        self.corner = np.empty(self.dimension)
        self.width = np.empty(self.dimension)
        if inp_corner is not None:
            self.corner = inp_corner
            self.width = inp_width

    def get_corner(self):
        return self.corner

    def get_width(self):
        return self.width

    def contains_point(self, point: np.array):
        assert np.any(point.shape == np.array(self.dimension))
        if np.any((self.corner - self.width) >= point):
            return False
        elif np.any((self.corner + self.width) <= point):
            return False
        else:
            return True


class SPTree:
    def __init__(self, inp_dimension, inp_data, inp_num_of_points=None,
                 inp_parent=None, inp_corner=None, inp_width=None):

        # Make sure the data are a two dimensional array with the 2nd dimension equal to the tree's one
        assert np.any(np.shape(inp_data.shape) == np.array(2))
        assert np.any(inp_data.shape[1] == np.array(inp_dimension))

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
        self.index = np.empty(self.qt_node_capacity)
        self.center_of_mass = np.zeros(self.dimension)

        # Children
        self.num_children = 2 ** self.dimension
        self.children = []

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

        # Do not add any duplicate points
        any_duplicate = False
        for i in np.arange(self.size):
            duplicate = True
            if np.any(point != self.data[self.index[i]]):
                duplicate = False
            any_duplicate = any_duplicate or duplicate
        if any_duplicate:
            return True

        # Then subdivide the current cell
        if self.is_leaf:
            self.subdivide()

        # Find where the point can be inserted
        for i in np.arange(self.num_children):
            if self.children[i].insert(new_index):
                return True

        # If none of the above has happened the point cannot be inserted. This should never happen
        print('The point indexed: ' + str(new_index) + ' was not inserted!')
        return False

    def subdivide(self):
        for child in np.arange(self.num_children):
            new_corner = np.empty(self.dimension)
            new_width = 0.5 * self.boundary.get_width()
            div = 1
            for dim in np.arange(self.dimension):
                if (child / div) % 2 >= 1:
                    new_corner[dim] = self.boundary.get_corner()[dim] - new_width[dim]
                else:
                    new_corner[dim] = self.boundary.get_corner()[dim] + new_width[dim]
                div *= 2
            self.children.append(SPTree(inp_dimension=self.dimension, inp_data=self.data,
                                        inp_parent=self, inp_corner=new_corner,
                                        inp_width=new_width))

        for i in np.arange(self.size):
            success = False
            for j in np.arange(self.num_children):
                if not success:
                    success = self.children[j].insert(self.index[i])
            self.index[i] = -1

        self.size = 0
        self.is_leaf = False

    def fill(self, num_of_points):
        for i in np.arange(num_of_points):
            self.insert(i)

    def is_correct(self):
        for n in np.arange(self.size):
            point = self.data[self.index[n], :]
            if not self.boundary.contains_point(point):
                return False

        if not self.is_leaf:
            correct = True
            for i in np.arange(self.num_children):
                correct = correct and self.children[i].is_correct()
            return correct
        else:
            return True

    def get_all_indices(self, indices, loc=0):
        for i in np.arange(self.size):
            indices[loc + i] = self.index[i]
            loc += self.size

        if not self.is_leaf:
            for i in np.arange(self.num_children):
                loc = self.children[i].get_all_indices(indices, loc)

        return loc

    def get_depth(self):
        if self.is_leaf:
            return 1
        depth = 0
        for i in np.arange(self.num_children):
            depth = np.max(depth, self.children[i].get_depth())

        return depth + 1

    def compute_non_edge_forces(self, point_index, theta, neg_force, sum_q):
        # Make sure that we spend no time on empty nodes or self-interactions
        if self.cum_size == 0 or (self.is_leaf and self.size == 1 and self.index[0] == point_index):
            return

        # Compute distance between point and center-of-mass
        buffer = self.data[point_index, :] - self.center_of_mass
        distance = np.sum(buffer * buffer)

        # Check whether we can use this node as a "summary"
        max_width = np.max(self.boundary.get_width())
        if self.is_leaf or max_width / np.sqrt(distance) < theta:
            # Compute and add t-SNE force between point and current node
            distance = 1 / (1 + distance)
            mult = self.cum_size * distance
            sum_q[0] += mult
            mult *= distance
            neg_force[point_index, :] += mult * buffer
        else:
            # Recursively apply Barnes-Hut to children
            for i in np.arange(self.num_children):
                self.children[i].compute_non_edge_forces(point_index, theta, neg_force, sum_q)

    def compute_edge_forces(self, indices_p, values_p, N):
        pos_force = np.zeros((N, self.dimension))

        for n in np.arange(N):
            distances = np.ones(indices_p.shape[1])
            a = self.data[indices_p[n, :], :]  # shape = (per*3 , dimension)
            b = self.data[n, :]
            buffer = a - b # shape = (per*3, dimension)
            distances += np.sum(buffer**2, axis=1)  # shape = (per*3)
            distances = values_p[n, :] / distances
            pos_force[n, :] += np.sum(np.tile(distances, (self.dimension, 1)).T * buffer, axis=0)  # shape = (dimension)

        return pos_force

