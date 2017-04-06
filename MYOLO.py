from __future__ import print_function
from cntk.layers import Placeholder, Constant, Activation, Sequential, Dense, Convolution, MaxPooling
from cntk.io import ImageDeserializer, CTFDeserializer
import cntk

import os, sys
from cntk.utils import *
from cntk.ops import *
import cntk.io.transforms as xforms

from cntk.io import ImageDeserializer, MinibatchSource, StreamDef, StreamDefs, FULL_DATA_SWEEP
from cntk.layers import Placeholder,  BatchNormalization,Convolution2D, Activation, MaxPooling, Dense, Dropout, default_options, \
    Sequential


from Logic.Bbox import Bbox
from Logic.YOLOError import YOLOError
from Logic.AnchorBox import AnchorBoxScale
from cntk.debugging import debug


def create_mb_source(img_height, img_width, img_channels, output_size, image_file, roi_file):
    transforms = []

    transforms += [
        xforms.scale(width=img_width, height=img_height, channels=img_channels, interpolations='linear',
                     scale_mode='fill'),
    ]

    image_source = ImageDeserializer(image_file, cntk.io.StreamDefs(features=cntk.io.StreamDef(field='image', transforms=transforms)))


    # read rois and labels
    roi_source = CTFDeserializer(roi_file, cntk.io.StreamDefs(label=cntk.io.StreamDef(field='rois', shape=output_size)))

    rc = MinibatchSource([image_source, roi_source], epoch_size=sys.maxsize, randomize=True)
    return rc


def create_network(input, n_bounding_boxes, n_classes):
    with default_options(activation=leaky_relu, pad=True):
        net1 = Sequential([
            #Feature Extraction

            Convolution((3, 3), 32, strides=(1, 1), pad=True), BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=True), MaxPooling((2, 2), strides=(2, 2)),

            Convolution((3, 3), 64, pad=True),BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=True) ,MaxPooling((2, 2), strides=(2, 2)),

            Convolution((3, 3), 128, pad=True), Convolution((1, 1), 64, pad=True), Convolution((3, 3), 128, pad=True),
            MaxPooling((2, 2), (2, 2), pad=True),

            Convolution((3, 3), 256, pad=True), Convolution((1, 1), 128, pad=True),
            Convolution((3, 3), 256, pad=True),
            MaxPooling((2, 2), (2, 2), pad=True),

            Convolution((3, 3), 512, pad=True), Convolution((1, 1), 256, pad=True),
            Convolution((3, 3), 512, pad=True), Convolution((1, 1), 256, pad=True),
            Convolution((3, 3), 512, pad=True)
        ]
        )(input)



        net2 = Sequential([

            MaxPooling((2, 2), strides=(2, 2)),

            Convolution((3, 3), 1024, pad=True), Convolution((1, 1), 512, pad=True),
            BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=True),
            Convolution((3, 3), 1024, pad=True), Convolution((1, 1), 512, pad=True),
            BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=True),
            Convolution((3, 3), 1024, pad=True)

        ])(net1)

        netpass = cntk.ops.reshape(net1, (2048, 13, 13))

        # here i was trying to do the pass through described in the yolo v2 paper
        # how do i concat the tensors? always get an error if i comment it in
        netreorg = net2 #cntk.ops.splice(net2, netpass, axis=0)

        net3 = Sequential([
            # Prediction

            Convolution((3, 3), 1024, pad=True), Convolution((3, 3), 1024, pad=True),
            Convolution((1, 1), ((n_classes + 5) * n_bounding_boxes), activation=leaky_relu)

        ])(netreorg)

    # net = cntk.ops.reshape(net, (7, 7, 30))

    return net3


# the idea of this method is to map the (100, 13, 13) output tensor of the "create_network" method to a flat
# (100 * 13 * 13) vector while applying things like sigmoid
def create_last_layer(network, n_grid, n_bounding_boxes, anchor_box_scales):

    box_list = []
    centroid_value_number = (n_classes + 5) * n_bounding_boxes
    grid_width = 1./n_grid
    counter = 0

    # better way of doing it -> current way is inefficient due to many small computations instead of a few large
    # -> to many computation nodes
    # not finished yet but a start
    #
    # boxes = cntk.ops.reshape(network, (n_bounding_boxes, n_classes + 5, n_grid * n_grid))
    # coords = boxes[:, 0:2, :]
    # scales = boxes[:, 2:4, :]
    # confs = boxes[:, 4:5, :]
    # classes  = boxes[:, 5:5+n_classes, :]

    for y in range(n_grid):
        for x in range(n_grid):
            centroid_values = network[:,
                               y:y + 1, x: x + 1]
            centroid_values = cntk.ops.reshape(centroid_values, (centroid_value_number,))

            for i in range(n_bounding_boxes):
                box_index = i * (n_classes + 5)

                #coords (x,y)
                # the offset is the top left corner of each grid tile
                # the degrees of freedem should it just make possible to move the anchor box in the tile it belongs to
                coords = cntk.ops.slice(centroid_values, 0, box_index, box_index + 2)
                offset = cntk.ops.constant(np.array([x * grid_width, y * grid_width], dtype=np.float32))
                scale = cntk.ops.constant(np.array([grid_width, grid_width], dtype=np.float32))
                coords = cntk.ops.plus(cntk.ops.element_times(cntk.ops.sigmoid(coords), scale), offset)


                #(width, height)
                # must be scaled according to the tile width
                scales = cntk.ops.slice(centroid_values, 0, box_index + 2, box_index + 4)
                scale = cntk.ops.constant(np.array([grid_width*anchor_box_scales[i].width_scale, grid_width*anchor_box_scales[i].height_scale], dtype=np.float32))

                scales = cntk.ops.element_times(cntk.ops.exp(scales), scale)

                #confs/objectness

                confs = cntk.ops.slice(centroid_values, 0, box_index + 4, box_index + 5)
                confs = cntk.ops.sigmoid(confs)


                #classes

                classes = cntk.ops.slice(centroid_values, 0, box_index + 5, box_index + 5 + n_classes)
                classes = cntk.ops.softmax(classes)


                #box

                # append all cntk.ops.blocks belonging to one box oen vector
                box = cntk.ops.splice(coords, scales, confs, classes)

                # append the vectors of all boxes to one list
                box_list.append(box)

    # make a large vector out of the small ones
    result = cntk.ops.splice(*box_list)

    # FUNFACT
    # even though this whole code as it is right now doesn't converge you can do the following:
    # 1) disable (set it to 0) the scaling for all errors but the objectness error
    # 2) delete the line "confs = cntk.ops.sigmoid(confs)" out of this method
    #
    # obviously your boxes won't have the right (x, y, w, h) values and the class props are wrong
    # since the are not part of the error and gradient computation
    # even though the default anchor boxes should be on the right positions (in the right tiles)
    # this really works quite good

    return result

if __name__ == '__main__':

    image_file = "images_map_voc.txt"
    roi_file = "rois_map_voc.txt"
    model_path = "./model"
    model_name = "yolo_voc_{}.model"

    cntk.set_default_device(cntk.gpu(0))

    img_height = 416
    img_width = 416
    img_channel = 3

    n_grid = 13
    n_classes = 20
    n_bounding_boxes = 4

    grid_cell_size = (5 * n_bounding_boxes + n_classes)
    output_dim = (grid_cell_size, n_grid, n_grid)
    output_size = (n_classes + 5) * n_bounding_boxes *  n_grid * n_grid
    roi_input_size = (5 * 20)

    image_input = input_variable((img_channel, img_height, img_width), name='halli')
    label_input = input_variable(n_classes)
    roi_input = input_variable(roi_input_size)
    scale_input = input_variable(output_size)
    nroi_input = input_variable(output_size)
    error_tensor = input_variable(((n_classes + 5) * n_bounding_boxes, 13, 13))
    scale_tensor = input_variable(((n_classes + 5) * n_bounding_boxes, 13, 13))

    minibatch_source = create_mb_source(img_height, img_width, img_channel, roi_input_size, image_file, roi_file)

    # define mapping from reader streams to network inputs
    input_map = {
        image_input: minibatch_source["features"],
        roi_input: minibatch_source["label"]
    }

    # create base network
    network = create_network(image_input, n_bounding_boxes, n_classes)

    # define default anchor box scales
    anchor_box_scales = [AnchorBoxScale(1.08, 1.19), AnchorBoxScale(3.42, 4.41),
                         AnchorBoxScale(6.63, 11.38), AnchorBoxScale(9.42, 5.11),
                         AnchorBoxScale(16.62, 10.52)]

    # map output to vector
    network = create_last_layer(network, n_grid, n_bounding_boxes, anchor_box_scales)

    # define error function
    # "n_roi_input" is the ground truth
    # "scale_input" scales the errors
    # look yolo paper
    err = cntk.ops.minus(network, nroi_input)
    sq_err = cntk.ops.element_times(err, err)
    sc_err = cntk.ops.element_times(sq_err, scale_input)
    mse = cntk.ops.reduce_sum(sc_err)

    # positions = get_grid_positions(n_grid)

    max_epochs = 50
    epoch_size = 10000

    minibatch_size = 5

    # cntk.element_times([5., 10., 15., 30.], [2., 2]).eval()

    lr_schedule = cntk.learning_rate_schedule([0.00005], cntk.learner.UnitType.sample, epoch_size)
    mm_schedule = cntk.learner.momentum_as_time_constant_schedule([600], epoch_size)

    # Instantiate the trainer object to drive the model training
    learner = cntk.learner.momentum_sgd(network.parameters, lr_schedule, mm_schedule, unit_gain=True)
    trainer = cntk.Trainer(network, (mse, mse), [learner])

    debug.set_computation_network_trace_level(1)

    # error computation object
    error = YOLOError(n_bounding_boxes, n_classes, n_grid, 5, 0.5, anchor_box_scales)

    progress_printer = cntk.utils.ProgressPrinter(tag='Training')

    # Get minibatches of images to train with and perform model training
    for epoch in range(max_epochs):  # loop over epochs
        sample_count = 0
        while sample_count < epoch_size - minibatch_size:  # loop over minibatches in the epoch

            # get next minibatch data
            data = minibatch_source.next_minibatch(min(minibatch_size, epoch_size - sample_count),
                                                   input_map=input_map)  # fetch minibatch.
            # get the roi data of the current minibatch
            roii = data[roi_input]

            # convert the "roii" data to te ground truth boxes
            box_lists = Bbox.generate_boxes(roii.value)

            # create a (13, 13, 1) ground truth mapping of those boxes to the grid tiles
            centroid_box_map = Bbox.get_nearest_boxes(box_lists, n_grid)

            # do a forward pass
            # this is really inefficient and a waste but i don't know how to do it better
            # i need the results of a network computation to decide which boxes have a bad iou and to set their
            # objectness 0 based on that (see yolo v2 paper)
            #
            # problem:
            # i do a forward pass here but basically i also do a forward pass for the training
            # -> i do it two times when only one time is needed
            # -> i still must do it because i don't get access to the forward pass in the training step
            #   BEFORE the backward pass is starting
            # How can i make it better? this nearly doubles training time
            eval_result = network.eval({image_input: data[image_input].value})

            # compute the error/teaching and the scaling vector
            scale_vectors_list, roi_vectors_list = error.create_scale_and_roi_error_vector(centroid_box_map, eval_result)

            # train width teaching and sclaing vector
            trainer.train_minibatch({image_input: data[image_input].value, scale_input: scale_vectors_list,
                                     nroi_input: roi_vectors_list}, device=cntk.gpu(0))  # update model with it
            sample_count += data[image_input].num_samples  # count samples processed so far
            progress_printer.update_with_trainer(trainer, with_metric=True)  # log progress

        progress_printer.epoch_summary(with_metric=True)
        network.save_model(os.path.join(model_path, model_name.format(epoch)))

    print("defined")
