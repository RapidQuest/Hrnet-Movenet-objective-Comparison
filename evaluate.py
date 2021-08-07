import numpy as np

from inference import get_max_preds


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, thr=0.5,keypoint_thr = 0.5,model_type = 'movenet',frame_batch= None,hrnet_post_process_attributes = None,config = None):
    norm = 1.0
    if model_type=='movenet':
      idx = list(range(output.shape[2]))
      pred = get_processed_predictions(frame_batch,output = output,keypoint_thr = keypoint_thr)
      h = movenet_input_size[0]
      w = movenet_input_size[1]
      norm = np.ones((pred.shape[0], 2)) * frame_batch / 10
    else:
        idx = list(range(output.shape[1]))
        pred, _ = get_final_preds(config,output,hrnet_post_process_attributes['center'],hrnet_post_process_attributes['scale'])
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([w, h]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return avg_acc
