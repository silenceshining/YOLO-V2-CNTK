import numpy as np
import math
from scipy import misc

import Logic
from Bbox import Bbox
from Grid import Grid
from AnchorBox import AnchorBox
from matplotlib import *




class YOLOError:
    def __init__(self, n_bounding_boxes, n_classes, n_grid, alpha_coord, alpha_noobj, anchor_box_scales):

        # number of anchor boxes
        self.n_bounding_boxes = n_bounding_boxes

        # number of classes
        self.n_classes = n_classes

        # number of grid tiles (in this case 13 -> 13 * 13 grid)
        self.n_grid = n_grid

        # error scaling of x, y, width, height of a box
        self.alpha_coord = alpha_coord

        # error scaling when there is no object
        self.alpha_no_obj = alpha_noobj

        # size of all values of a tile in the 13 * 13 grid
        # each tile has a center (centroid) and has n_bounding_boxes anchored to it
        # each bounding box predicts "n_classes" classes + (x, y, width, height) = 5 an there are "n_bouding_boxes" boxes
        self.box_entry_size = (self.n_classes + 5) * self.n_bounding_boxes

        # contains default anchor box scalings
        # contains "n_bounding_boxes" scalings
        self.anchor_box_scales = anchor_box_scales

        # computes the width and height of a tile
        # all coordinates widths and heights are relative and not absolute
        self.grid_width = 1. / self.n_grid

        # could also be named "get_anchor_boxes_for_grid_tiles"
        # creates a 3D array (13 * 13 * "n_bounding_boxes") for the 13 * 13 grid width default anchor boxes
        # this array is used to find the best fitting anchor boxes for the ground truth
        self.anchor_boxes = AnchorBox.get_anchor_boxes_for_centroids(self.n_grid, self.anchor_box_scales)

    # This function creates error vector and the error scaling vector
    def create_scale_and_roi_error_vector(self, centroid_box_maps, eval_results):

        # compute size of one tile in the 13 * 13 grid
        vector_size = self.box_entry_size * self.n_grid * self.n_grid

        roi_error_vectors_list = []
        error_scale_vectors_list = []

        # iterate over each minibatch and calculate the vectors
        for i in range(len(centroid_box_maps)):
            error_scale_vector = [0] * vector_size
            roi_error_vector = [0] * vector_size

            self.process_one_vector(roi_error_vector, error_scale_vector, centroid_box_maps[i], eval_results[i][0])

            roi_error_vectors_list.append([roi_error_vector])
            error_scale_vectors_list.append([error_scale_vector])

        return np.array(error_scale_vectors_list, dtype=np.float32), np.array(roi_error_vectors_list, dtype=np.float32)

    # computes the roi_error_vector and the error_scale_vector of one input of the minibatch
    def process_one_vector(self, roi_error_vector, error_scale_vector, centroid_map, eval_result):

        # get the boxes the network computed
        eval_boxes = self.eval_results_to_boxes(eval_result)
        sum_error = 0
        sum_iou = 0
        sum_counter = 0

        # iterate over all grid positions and anchor boxes anchored in them
        for y in range(self.n_grid):
            for x in range(self.n_grid):

                # check if there is an object on the position (x,y) in the ground truth 13 * 13 map
                # if that is the case the grid position (x, y) contains an object
                # if it  is not the case it doesn't
                if centroid_map[x, y] is not None:

                    # there are "n_bounding_boxes" anchor boxes but which one is the best
                    # -> returns an index in the range of [0, n_bounding_boxes)
                    anchor_index = self.find_best_overlapping_index(centroid_map[x, y], x, y)

                    # iterate over all anchor boxes
                    for b in range(self.n_bounding_boxes):

                        # get the specific computed box
                        eval_box = eval_boxes[x, y, b]

                        # compute the iou of a default anchor box with the ground truth
                        iou = Bbox.intersection_over_union(self.anchor_boxes[x, y, b], centroid_map[x, y])

                        # compute the iou of the anchor box the network computed with the ground truth
                        eiou = Bbox.intersection_over_union(eval_box, centroid_map[x, y])

                        # if this is the best fitting anchor box
                        #
                        # always corrects objectness and class probs
                        # if the (x, y) contains an object
                        #       if b is the best fitting box of the tile (x, y)
                        #           -> objectness is set to 1
                        #           -> x, y, w, h are set to the correct values
                        #       else
                        #           -> objectness is set to "iou" (iou between default anchor box and ground truth)
                        # else
                        #       -> objectness is set to 0

                        if b == anchor_index:

                            # set the specific positions in the output vector to the right values
                            #
                            # the use of "centroid_map[x, y].ClassNumbers" is sloppy here
                            # that's because for the ground truth map "centroid_map" "ClassNumbers" actually contains
                            # just one class number -> the one of the ground truth
                            # BUT for teh "eval_boxes" map it contains all computed probabilities for all classes
                            #
                            self.set_ground_truth_vector(x, y, b, 1, roi_error_vector, centroid_map[x, y])
                            self.set_error_scale_vector(x, y, b, error_scale_vector, True, centroid_map[x, y].ClassNumbers )

                            # some logging
                            sum_error += eval_box.Objectness
                            sum_iou += eiou
                            sum_counter += 1

                        else:

                            # same as if true
                            # in "set_error_scale_vector" teh parameter "ground
                            self.set_ground_truth_vector(x, y, b, iou, roi_error_vector, None, None)
                            self.set_error_scale_vector(x, y, b, error_scale_vector, False,  centroid_map[x, y].ClassNumbers)
                else:
                    # nearly looks the same as if there was an object beside that the iou is set 0
                    #
                    for b in range(self.n_bounding_boxes):

                        eval_box = eval_boxes[x, y, b]
                        iou = 0 #self.find_largest_iou(self.anchor_boxes[x, y, b], centroid_map)
                        self.set_ground_truth_vector(x, y, b, iou, roi_error_vector)
                        self.set_error_scale_vector(x, y, b, error_scale_vector, False)


        if sum_counter != 0:
            sum_error /= sum_counter
            sum_iou /= sum_counter

        #print("Average IOU: " + str(sum_iou))
        print("Average Objectness: " + str(sum_error))

    # returns the map teh network computed
    def eval_results_to_boxes(self, eval_results):

        # eval results is the output of one network forward pass for a minibatch

        # create 13 * 13 * "bounding_boxes" map for the computed boxes
        eval_box_list = np.empty([self.n_grid, self.n_grid, self.n_bounding_boxes], dtype=Bbox)
        grid_positions = Grid.get_grid_positions_v2(self.n_grid)

        for y in range(self.n_grid):
            for x in range(self.n_grid):
                for b in range(self.n_bounding_boxes):

                    class_props = [0] * self.n_classes
                    index = self.get_index(x, y, b, 0)
                    for i in range(self.n_classes):
                        class_props[i] = eval_results[index + 5 + i]

                    x_pos = eval_results[index]
                    y_pos = eval_results[index + 1]

                    width = eval_results[index + 2]
                    height = eval_results[index + 3]

                    objectness = eval_results[index + 4]

                    eval_box_list[x, y, b] = Bbox(class_props, x_pos, y_pos, width, height, objectness)

        return eval_box_list

    # iterates over all default anchor boxes in range [0, n_bounding_boxes) and return the index of best iou
    # with a ground truth box
    def find_best_overlapping_index(self, ground_truth_box, x: int, y: int) ->  int:
        max_i = 0
        max_overlap = -1
        for i in range(self.n_bounding_boxes):
            overlap = Bbox.intersection_over_union(self.anchor_boxes[x, y, i], ground_truth_box)

            if overlap > max_overlap:
                max_overlap = overlap
                max_i = i
        if max_i == 4:
            return max_i
        return max_i

    # sets the correct values for x, y, w, h, objectness, and classes for each map position (x, y, b)
    # in the teaching vector
    def set_ground_truth_vector(self, x, y, b, iou, error_vector, ground_truth_box=None, cclass = None):

        index = self.get_index(x, y, b, 0)
        if ground_truth_box is not None:
            error_vector[index] = ground_truth_box.XPos
            error_vector[index + 1] = ground_truth_box.YPos
            error_vector[index + 2] = ground_truth_box.Width
            error_vector[index + 3] = ground_truth_box.Height
            error_vector[index + 5 + int(ground_truth_box.ClassNumbers)] = 1
            error_vector[index + 4] = 1
        else:
            error_vector[index + 4] = iou
            if cclass is not None:
                error_vector[index + 5 + int(cclass)] = 1

    # sets the scalings for the different squared errors (see yolo v1 paper)
    def set_error_scale_vector(self, x, y, b, error_scale_vector, ground_truth=False, class_number=None):

        index = self.get_index(x, y, b, 0)
        if ground_truth:
            #error_scale_vector[index] = self.alpha_coord
            #error_scale_vector[index + 1] = self.alpha_coord
            #error_scale_vector[index + 2] = self.alpha_coord
            #error_scale_vector[index + 3] = self.alpha_coord
            error_scale_vector[index + 4] = 5
            for i in range(self.n_classes):
                error_scale_vector[self.get_index(x, y, b, 5 + i)] = 5
        else:
            error_scale_vector[index + 4] = 1/5
            for i in range(self.n_classes):
                error_scale_vector[self.get_index(x, y, b, 5 + i)] = 1./5

    # maps a position in the shape ( 13, 13, "n_bounding_boxes" ) output tensor of the network to the flat error/teaching
    # vector shape (13 * 13 * "n_bounding_boxes")
    def get_index(self, x, y, b, pos):
        ind = x * self.box_entry_size + y * self.box_entry_size * self.n_grid + b * (self.n_classes + 5) + pos
        return ind

    def find_largest_iou(self, box, centroid_map):
        largest_iou = 0
        for x in range(self.n_grid):
            for y in range(self.n_grid):
                if centroid_map[x, y] is not None:
                    iou = Bbox.intersection_over_union(centroid_map[x, y], box)
                    if iou > largest_iou:
                        largest_iou = iou
        return largest_iou

