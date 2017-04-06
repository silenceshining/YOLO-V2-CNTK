import numpy as np

import Bbox


class Grid:
    GridPositions = None

    # get grid tiles center
    @staticmethod
    def get_grid_positions(n_grid):
        pos = np.zeros((n_grid, n_grid, 2))

        step = 1 / n_grid

        for x in range(0, n_grid):
            for y in range(0, n_grid):
                pos[x, y, 0] = x * step + step / 2
                pos[x, y, 1] = y * step + step / 2
        return pos

    # gets grid tiles left upper corner (see yolo v2 paper)
    @staticmethod
    def get_grid_positions_v2(n_grid):
        pos = np.zeros((n_grid, n_grid, 2))

        step = 1 / n_grid

        for x in range(0, n_grid):
            for y in range(0, n_grid):
                pos[x, y, 0] = x * step
                pos[x, y, 1] = y * step
        return pos


    # decides if a point(x,y) is in the grid tile (x,y)
    @staticmethod
    def in_grid_field(grid_x, grid_y, n_grid, x, y):
        size = 1 / n_grid
        return (grid_x - size / 2 < x < grid_x + size / 2) and (grid_y - size / 2 < y < grid_y + size / 2)


    @staticmethod
    def find_largest_IOU(centroid_box_map, box, ignorex, ignorey):
        n_grid = centroid_box_map.shape[0]

        max_iou = 0

        for x in range(n_grid):
            for y in range(n_grid):
                if not (ignorex == x and ignorey == y):
                    max_iou = max(max_iou, Bbox.Bbox.intersection_over_union(centroid_box_map[x, y], box))

        return max_iou
