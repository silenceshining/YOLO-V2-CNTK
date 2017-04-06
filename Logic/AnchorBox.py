from Bbox import Bbox
from Grid import Grid

import math
import numpy as np


class AnchorBox:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.x_pos = 0
        self.y_pos = 0

    @staticmethod
    def get_anchor_boxes_for_centroids(n_grid, anchor_scale_list):
        centroid_pos = Grid.get_grid_positions(n_grid)
        anchor_box_number = len(anchor_scale_list)

        grid_width = 1. / n_grid
        anchor_box_grid = np.empty([n_grid,n_grid, anchor_box_number], dtype=Bbox)

        for y in range(n_grid):
            for x in range(n_grid):
                for b in range(anchor_box_number):

                    x_pos = centroid_pos[x, y, 0]
                    y_pos = centroid_pos[x, y, 1]

                    width = grid_width * anchor_scale_list[b].width_scale
                    height = grid_width * anchor_scale_list[b].height_scale

                    anchor_box_grid[x, y, b] = Bbox(0,x_pos, y_pos, width, height)

        return anchor_box_grid

    def jaccard_overlap(self, x_pos: float, y_pos: float, box: Bbox):
        anchorbox = Bbox(0, x_pos, y_pos, self.width, self.height)

        return Bbox.intersection_over_union(anchorbox, box)


class AnchorBoxScale:

    def __init__(self, width_scale: float, height_scale: float):
        self.width_scale = width_scale
        self.height_scale = height_scale

