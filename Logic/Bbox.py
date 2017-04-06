import math
import numpy as np

from Logic import Grid


class Bbox:

    XPos = 0  # type: int

    def __init__(self, bclass, xpos: float, ypos: float, width: float, height:float, objectness:float = 0):
        self.ClassNumbers = bclass
        self.XPos = xpos
        self.YPos = ypos
        self.Width = width
        self.Height = height
        self.XMin = self.XPos - self.Width / 2
        self.YMin = self.YPos - self.Height / 2
        self.XMax = self.XPos + self.Width / 2
        self.YMax = self.YPos + self.Height / 2
        self.Objectness = objectness


    def distance_to_point(self, x, y):
        return math.sqrt(pow(self.XPos - x, 2) + pow(self.YPos - y, 2))

    # reads ground truth boxes from the output of the minibatch source
    @staticmethod
    def generate_boxes(cntk_input):
        minibatch_size = cntk_input.shape[0]
        input_size = cntk_input.shape[2]
        box_input_number = int(input_size / 5)

        box_lists = []

        for i in range(minibatch_size):
            box_list = []
            flat = cntk_input[i, 0, :]
            for x in range(0, box_input_number, 5):
                if flat[x] != -1:
                    box = Bbox(flat[x], flat[x + 1], flat[x + 2], flat[x + 3], flat[x + 4])
                    box_list.append(box)
            box_lists.append(box_list)

        return box_lists

    # gets boxes with the smallest euklidian distance to the grid tile centers
    @staticmethod
    def get_nearest_boxes(box_lists, n_grid):
        minibatch_size = len(box_lists)

        centroid_box_list = []

        for i in range(minibatch_size):
            centroid_boxes = np.empty((n_grid, n_grid), dtype=Bbox)
            minibatch_boxes = box_lists[i]
            grid_positions = Grid.Grid.get_grid_positions(n_grid)
            for x in range(n_grid):
                for y in range(n_grid):
                    min_distance_box = min(minibatch_boxes, key=lambda l: l.distance_to_point(grid_positions[x, y, 0],
                                                                                              grid_positions[x, y, 1]))
                    if Grid.Grid.in_grid_field(grid_positions[x, y, 0], grid_positions[x, y, 1], n_grid,
                                          min_distance_box.XPos, min_distance_box.YPos):
                        centroid_boxes[x, y] = min_distance_box
            centroid_box_list.append(centroid_boxes)
        return centroid_box_list




    @staticmethod
    def get_highest_iou(box, centroid_box_map):
        n_grid = centroid_box_map.shape[0]

        highest_iou = 0
        for x in range(n_grid):
            for y in range(n_grid):
                if centroid_box_map[x, y] != None:
                    iou = Bbox.intersection_over_union(centroid_box_map[x,y], box)
                    if iou > highest_iou:
                        highest_iou = iou

        return highest_iou

    @staticmethod
    def intersection_over_union(box1, box2):

        xA = max(box1.XMin, box2.XMin)
        yA = max(box1.YMin, box2.YMin)
        xB = min(box1.XMax, box2.XMax)
        yB = min(box1.YMax, box2.YMax)

        a_xmin = min(box1.XMin, box2.XMin)
        a_xmax = max(box1.XMax, box2.XMax)

        a_ymin = min(box1.YMin, box2.YMin)
        a_ymax = max(box1.YMax, box2.YMax)


        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (box1.XMax - box1.XMin) * (box1.YMax - box1.YMin)
        boxBArea = (box2.XMax - box2.XMin) * (box2.YMax - box2.YMin)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return max(iou, 0)
