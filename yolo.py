"""
YOLOv3 Library
YOLOv3 Public site : https://pjreddie.com/
use cpu:0 when test, predict
:author Chanwoo Gwon
:reference https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/18f825d730786f3be1c83cd6a0bf411e94526e00/4-Object_Detection/YOLOV3
"""
import cv2
import os
import shutil
import tensorflow as tf
from core.config import cfg

from core.dataset import Dataset
import core.utils as utils
import core.common as common
import numpy as np

from lib.GPUSetting import initialize_gpu
from lib.utill import get_all_file


class YOLO:
    def __init__(self, model_path, class_path):
        self.__class_path = class_path
        self.__NUM_CLASS = len(utils.read_class_names(self.__class_path))
        self.__CLASSES = utils.read_class_names(self.__class_path)
        self.__ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.__STRIDES = np.array(cfg.YOLO.STRIDES)
        self.__IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
        self.__img_size = 416
        self.__img_channels = 3
        self.__model_path = model_path

    def darknet53(self, input_data):
        """
        build darknet (backbone)
        :param input_data input image
        :return darknet network
        """
        input_data = common.convolutional(input_data, (3, 3,  3,  32))
        input_data = common.convolutional(input_data, (3, 3, 32,  64), downsample=True)

        for i in range(1):
            input_data = common.residual_block(input_data,  64,  32, 64)

        input_data = common.convolutional(input_data, (3, 3,  64, 128), downsample=True)

        for i in range(2):
            input_data = common.residual_block(input_data, 128,  64, 128)

        input_data = common.convolutional(input_data, (3, 3, 128, 256), downsample=True)

        for i in range(8):
            input_data = common.residual_block(input_data, 256, 128, 256)

        route_1 = input_data
        input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True)

        for i in range(8):
            input_data = common.residual_block(input_data, 512, 256, 512)

        route_2 = input_data
        input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True)

        for i in range(4):
            input_data = common.residual_block(input_data, 1024, 512, 1024)

        return route_1, route_2, input_data

    def build_network(self, input_layer):
        """
        :param input_layer image input layer [image_size, image_size, chnnals]
        :return yolo network
        """
        route_1, route_2, conv = self.darknet53(input_layer) # use backbone (darknet53)

        conv = common.convolutional(conv, (1, 1, 1024, 512))
        conv = common.convolutional(conv, (3, 3, 512, 1024))
        conv = common.convolutional(conv, (1, 1, 1024, 512))
        conv = common.convolutional(conv, (3, 3, 512, 1024))
        conv = common.convolutional(conv, (1, 1, 1024, 512))

        conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024))
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (self.__NUM_CLASS + 5)), activate=False, bn=False)

        conv = common.convolutional(conv, (1, 1, 512, 256))
        conv = common.upsample(conv)

        conv = tf.concat([conv, route_2], axis=-1)

        conv = common.convolutional(conv, (1, 1, 768, 256))
        conv = common.convolutional(conv, (3, 3, 256, 512))
        conv = common.convolutional(conv, (1, 1, 512, 256))
        conv = common.convolutional(conv, (3, 3, 256, 512))
        conv = common.convolutional(conv, (1, 1, 512, 256))

        conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512))
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (self.__NUM_CLASS + 5)), activate=False, bn=False)

        conv = common.convolutional(conv, (1, 1, 256, 128))
        conv = common.upsample(conv)

        conv = tf.concat([conv, route_1], axis=-1)

        conv = common.convolutional(conv, (1, 1, 384, 128))
        conv = common.convolutional(conv, (3, 3, 128, 256))
        conv = common.convolutional(conv, (1, 1, 256, 128))
        conv = common.convolutional(conv, (3, 3, 128, 256))
        conv = common.convolutional(conv, (1, 1, 256, 128))

        conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256))
        conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (self.__NUM_CLASS + 5)), activate=False, bn=False)

        return [conv_sbbox, conv_mbbox, conv_lbbox]

    def decode(self, conv_output, i=0):
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
                contains (x, y, w, h, score, probability)
        """
        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + self.__NUM_CLASS))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * self.__STRIDES[i]
        pred_wh = (tf.exp(conv_raw_dwdh) * self.__ANCHORS[i]) * self.__STRIDES[i]
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def compute_iou(self, boxes1, boxes2):
        """
        compute iou between two boxes
        :param boxes1 box set one
        :param boxes2 box set two
        """
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = (boxes1[2] - boxes1[0]) * (boxes1[3] - boxes1[1])
        boxes2_area = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

        left_up       = np.maximum(boxes1[:2], boxes2[:2])
        right_down    = np.minimum(boxes1[2:], boxes2[2:])

        inter_section = np.maximum(right_down - left_up, 0.0)  
        inter_area    = inter_section[0] * inter_section[1]
        union_area    = boxes1_area + boxes2_area - inter_area
        ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

        return ious

    def bbox_iou(self, boxes1, boxes2):
        """
        get iou between two box sets
        :param boxes1 box set one
        :param boxes2 box set two
        """
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return 1.0 * inter_area / union_area

    def bbox_giou(self, boxes1, boxes2):
        """
        get generalized iou between two box set
        :param boxes1 box set one
        :param boxes2 box set two
        """
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def compute_loss(self, pred, conv, label, bboxes, i=0):
        """
        compute loss
        :param pred predict result
        :param conv output conv result
        :param label image label
        :param bboxes recommended bounding box
        """
        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = self.__STRIDES[i] * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + self.__NUM_CLASS))

        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.__IOU_LOSS_THRESH, tf.float32)

        conf_focal = tf.pow(respond_bbox - pred_conf, 2)

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return giou_loss, conf_loss, prob_loss

    def check_data(self, anno_path, res_path, cnt=-1):
        """
        check reformed data
        :param anno_path annotation file path
        :param res_path path to save result
        :param cnt counting number of checking image
        """
        with open(anno_path) as fp:
            lines = fp.readlines()
            for i, line in zip(range(len(lines)), lines):
                splited = line.split(' ')

                print(splited)
                image_path = splited[0]
                image = cv2.imread(image_path)
                for i in range(1, len(splited)):
                    image_box_info = splited[i]
                    bbox_and_tag = image_box_info.split(',')
                    bbox = [int(bbox_and_tag[i]) for i in range(len(bbox_and_tag) - 1)]
                    cv2.rectangle(image, pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]), color=(255, 0, 0), thickness=2)

                cv2.imwrite(res_path, image)
                if cnt == -1 or cnt > len(lines):
                    pass
                elif cnt == i:
                    break

                i += 1

    def train(self, main_dir, anno_path, epoch=cfg.TRAIN.EPOCHS):
        """
        start train
        :param main_dir path to save logs, etc.
        :param anno_path path of annotation file
        :param epoch the number of epoch (default=30)
        """
        initialize_gpu(0, 4096)  # allocate 1 GB to index 0 gpu
        trainset = Dataset('train', anno_path, self.__class_path)

        logdir = main_dir + "/log"

        steps_per_epoch = len(trainset)
        global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
        total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch

        input_tensor = tf.keras.layers.Input([self.__img_size, self.__img_size, self.__img_channels])
        conv_tensors = self.build_network(input_tensor)

        output_tensors = []
        for i, conv_tensor in enumerate(conv_tensors):
            pred_tensor = self.decode(conv_tensor, i)
            output_tensors.append(conv_tensor)
            output_tensors.append(pred_tensor)

        model = tf.keras.Model(input_tensor, output_tensors)
        optimizer = tf.keras.optimizers.Adam()
        if os.path.exists(logdir): shutil.rmtree(logdir)
        writer = tf.summary.create_file_writer(logdir)

        def train_step(image_data, target):
            with tf.GradientTape() as tape:
                pred_result = model(image_data, training=True)
                giou_loss = conf_loss = prob_loss = 0

                # optimizing process
                for i in range(3):
                    conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                    loss_items = self.compute_loss(pred, conv, *target[i], i)
                    giou_loss += loss_items[0]
                    conf_loss += loss_items[1]
                    prob_loss += loss_items[2]

                total_loss = giou_loss + conf_loss + prob_loss

                gradients = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                      "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, optimizer.lr.numpy(),
                                                                giou_loss, conf_loss,
                                                                prob_loss, total_loss))
                # update learning rate
                global_steps.assign_add(1)
                if global_steps < warmup_steps:
                    lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
                else:
                    lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                        (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                    )
                optimizer.lr.assign(lr.numpy())

                # writing summary data
                with writer.as_default():
                    tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                    tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                    tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                    tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                    tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
                writer.flush()

        for epoch in range(epoch):
            for image_data, target in trainset:
                train_step(image_data, target)
            model.save_weights(self.__model_path)

    def resize_image(self, origin, width, height):
        """
        resize image
        :param height: height of origin image
        :param width: width of origin image
        :param origin origin image (numpy)
        """
        resized = None
        if height > self.__img_size or width > self.__img_size:
            resized = cv2.resize(origin, dsize=(self.__img_size, self.__img_size), interpolation=cv2.INTER_LINEAR)
        else:
            resized = cv2.resize(origin, dsize=(self.__img_size, self.__img_size), interpolation=cv2.INTER_AREA)

        return resized

    def test(self, predicted_dir_path, ground_truth_dir_path, anno_path):
        """
        test dataset
        :param predicted_dir_path path to save predicted result
        :param ground_truth_dir_path path to save ground truth information
        :param anno_path annotation file path for test
        """
        # clear subfile in pred, gb path
        if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
        if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
        if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): shutil.rmtree(cfg.TEST.DECTECTED_IMAGE_PATH)

        # create paths if do not exist.
        os.mkdir(predicted_dir_path)
        os.mkdir(ground_truth_dir_path)
        os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH)

        with tf.device('/CPU:0'):

            input_tensor = tf.keras.layers.Input([self.__img_size, self.__img_size, self.__img_channels])
            conv_tensors = self.build_network(input_tensor)

            output_tensors = []
            for i, conv_tensor in enumerate(conv_tensors):
                pred_tensor = self.decode(conv_tensor, i)
                conv_tensor.append(pred_tensor)

            model = tf.keras.Model(input_tensor, output_tensors)
            model.load_weights(self.__model_path)

            with open(anno_path, 'r') as annotation_file:
                for num, line in enumerate(annotation_file):
                    annotation = line.strip().split()
                    image_path = annotation[0]
                    image_name = image_path.split('/')[-1]
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

                    if len(bbox_data_gt) == 0:
                        bboxes_gt = []
                        classes_gt = []
                    else:
                        bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                    ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

                    num_bbox_gt = len(bboxes_gt)
                    with open(ground_truth_path, 'w') as f:
                        for i in range(num_bbox_gt):
                            class_name = self.__CLASSES[classes_gt[i]]
                            xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                            bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                            f.write(bbox_mess)

                    predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
                    # Predict Process
                    image_size = image.shape[:2]
                    image_data = utils.image_preporcess(np.copy(image), [self.__img_size, self.__img_size])
                    image_data = image_data[np.newaxis, ...].astype(np.float32)

                    pred_bbox = model.predict(image_data)
                    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
                    pred_bbox = tf.concat(pred_bbox, axis=0)
                    bboxes = utils.postprocess_boxes(pred_bbox, image_size, self.__img_size, cfg.TEST.SCORE_THRESHOLD)
                    bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')

                    if cfg.TEST.DECTECTED_IMAGE_PATH is not None:
                        image = utils.draw_bbox(image, bboxes)
                        cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH + image_name, image)

                    with open(predict_result_path, 'w') as f:
                        for bbox in bboxes:
                            coor = np.array(bbox[:4], dtype=np.int32)
                            score = bbox[4]
                            class_ind = int(bbox[5])
                            class_name = self.__CLASSES[class_ind]
                            score = '%.4f' % score
                            xmin, ymin, xmax, ymax = list(map(str, coor))
                            bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                            f.write(bbox_mess)

            self.check_score(predicted_dir_path, ground_truth_dir_path)

    def check_score(self, predicted_dir_path, ground_truth_dir_path, iou_threshold=0.7):
        """
        check score
        :param predicted_dir_path path to saved predicted result from test dataset
        :param ground_truth_dir_path path to ground truth information from test dataset
        :param iou_threshold iou threshold value to calculate object detection accuracy (default=0.7)
        :return (avg_bbox_rate, avg_class_rate) accuracy result
        """
        # check score
        list_gb = get_all_file(ground_truth_dir_path, None)
        list_pred = get_all_file(predicted_dir_path, None)

        if len(list_gb) == len(list_pred):
            total_bbox_rate_list = []
            total_class_rate_list = []
            total_bbox_rate = 0.0
            total_class_rate = 0.0
            text_count = 0
            total_text_count = 0

            for i in range(len(list_gb)):
                gb_file = list_gb[i]
                pred_file = gb_file.replace("ground-truth", "predicted")
                gb_list = []
                pred_list = []
                with open(gb_file) as fp:
                    lines = fp.readlines()
                    for line in lines:
                        # gb_file format
                        # [tag] [x0] [y0] [x1] [y1]
                        info = line.replace("\n", "").split(" ")
                        gb_list.append({
                            "tag": info[0],
                            "bbox": [float(item) for item in info[1:]]
                        })

                with open(pred_file) as fp:
                    lines = fp.readlines()
                    for line in lines:
                        # pred_file format
                        # [tag] [pred_score] [x0] [y0] [x1] [y1]
                        info = line.replace("\n", "").split(" ")

                        if float(info[1]) > 0.6:
                            pred_list.append({
                                "tag": info[0],
                                "bbox": [float(item) for item in info[2:]],
                                "predict_score": info[1]
                            })

                bbox_correct = 0
                class_correct = 0
                # calculate iou
                for i in range(len(gb_list)):
                    gb_bbox = gb_list[i]["bbox"]

                    for j in range(len(pred_list)):
                        pred_bbox = pred_list[j]["bbox"]
                        iou = self.compute_iou(gb_bbox, pred_bbox)
                        if iou > iou_threshold:
                            bbox_correct += 1
                            if pred_list[j]["tag"] == gb_list[i]["tag"]:
                                class_correct += 1
                            break

                bbox_rate = bbox_correct / float(len(gb_list))
                classification_rate = class_correct / float(len(gb_list))

                total_bbox_rate += bbox_rate
                total_class_rate += classification_rate

                total_bbox_rate_list.append(bbox_rate)
                total_class_rate_list.append(classification_rate)

            avg_bbox_rate = total_bbox_rate / float(len(list_gb))
            avg_class_rate = total_class_rate / float(len(list_gb))

            return avg_bbox_rate, avg_class_rate
        else:
            print("file list size is different")

    def predict(self, image_path):
        """
        predict image
        :param image_path original image path
        :return found image
        """
        with tf.device('/CPU:0'):
            input_tensor = tf.keras.layers.Input([self.__img_size, self.__img_size, 3])
            conv_tensors = self.build_network(input_tensor)

            output_tensors = []
            for i, conv_tensor in enumerate(conv_tensors):
                pred_tensor = self.decode(conv_tensor, i)
                output_tensors.append(pred_tensor)

            model = tf.keras.Model(input_tensor, output_tensors)
            model.load_weights(self.__model_path)

            original_image = cv2.imread(image_path)
            height, width, channels = original_image.shape

            resized = self.resize_image(original_image, width, height)
            # resized = 255 - resized
            image_data = utils.image_preporcess(np.copy(resized), [self.__img_size, self.__img_size])
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            pred_bbox = model.predict(image_data)
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)
            bboxes = utils.postprocess_boxes(pred_bbox, (self.__img_size, self.__img_size), self.__img_size, 0.6)
            bboxes = utils.nms(bboxes, 0.45, method='nms')

            resize_bboxes = []

            capture_image = []

            for i in range(len(bboxes)):
                bbox = bboxes[i]
                resize_bbox = [bbox[0] * width / self.__img_size, bbox[1] * height / self.__img_size,
                               bbox[2] * width / self.__img_size, bbox[3] * height / self.__img_size, bbox[4], bbox[5]]
                resize_bboxes.append(resize_bbox)
                image = original_image[int(resize_bbox[1]):int(resize_bbox[3]), int(resize_bbox[0]):int(resize_bbox[2])]
                capture_image.append(image)

            return capture_image
