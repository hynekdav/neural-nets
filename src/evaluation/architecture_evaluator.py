import numpy as np


class ArchitectureEvaluator(object):
    def __init__(self, model, num_objects):
        self.model = model
        self.num_objects = num_objects

    def _iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
        x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

        w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
        h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
        if w_I <= 0 or h_I <= 0:
            return 0.

        I = w_I * h_I
        U = w1 * h1 + w2 * h2 - I

        return I / U

    def evaluate(self, images, boxes):
        preds = self.model.predict(images)
        split = lambda A, n=4: [A[i:i + n] for i in range(len(A) // n)]

        overall_ious = []

        for sample, (pred_bboxes, exp_bboxes) in enumerate(zip(preds, boxes)):
            pred_bboxes = pred_bboxes.reshape(self.num_objects, -1)
            exp_bboxes = exp_bboxes.reshape(self.num_objects, -1)

            ious = np.zeros((self.num_objects, self.num_objects))
            for i, exp_bbox in enumerate(exp_bboxes):
                split_exp = split(exp_bbox)
                for j, pred_bbox in enumerate(pred_bboxes):
                    split_pred = split(pred_bbox)
                    ious[i, j] = sum(self._iou(split_pred[i], split_exp[i]) for i in range(len(split_pred)))

            overall_ious.append(max(ious.flatten()))

        overall_ious = np.asarray(overall_ious) / self.num_objects
        return overall_ious
