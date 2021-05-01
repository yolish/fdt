import cv2


def plot_bboxes(img, gt_boxes, prediced_boxes=None):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i in range(gt_boxes.shape[0]):
        cv2.rectangle(img, (gt_boxes[i, 0], gt_boxes[i, 1]), (gt_boxes[i, 2], gt_boxes[i, 3]), (0, 0, 255))
    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()