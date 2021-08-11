import argparse
from mir_eval import transcription, io, util
import json
import pandas as pd
import numpy as np
import time

def get_match_time_shift(ref_intervals, est_intervals):
    diff_points = []
    ref_onset = [x[0] for x in ref_intervals]
    est_onset = [x[0] for x in est_intervals]
    diff_points = [j - i for i in est_onset for j in ref_onset]
    return diff_points

def count_onset(diff_points, onset_tolerance):
    left = 0
    ans = 0
    ans_timespan = [0.0, 0.0]
    onset_span = onset_tolerance * 2
    for right in range(len(diff_points)):
        dr = diff_points[right] - onset_span
        while diff_points[left] < dr:
            left += 1
        if ans < right - left:
            ans = right - left
            ans_timespan = [diff_points[right-1]-onset_tolerance, diff_points[right]-onset_tolerance]
    return ans + 1, ans_timespan

def prepare_data(answer_true, answer_pred, time_shift):
    ref_pitches = []
    est_pitches = []
    ref_intervals = []
    est_intervals = []

    if time_shift >= 0.0:
        for i in range(len(answer_true)):
            if (answer_true[i] is not None and float(answer_true[i][1]) - float(answer_true[i][0]) > 0
                and answer_true[i][0] >= 0.0):
                ref_intervals.append([answer_true[i][0], answer_true[i][1]])
                ref_pitches.append(answer_true[i][2])

        for i in range(len(answer_pred)):
            if (answer_pred[i] is not None and float(answer_pred[i][1]) - float(answer_pred[i][0]) > 0 
                and answer_pred[i][0]+time_shift >= 0.0):
                est_intervals.append([answer_pred[i][0]+time_shift, answer_pred[i][1]+time_shift])
                est_pitches.append(answer_pred[i][2])

    else:
        for i in range(len(answer_true)):
            if (answer_true[i] is not None and float(answer_true[i][1]) - float(answer_true[i][0]) > 0
                and answer_true[i][0]-time_shift >= 0.0):
                ref_intervals.append([answer_true[i][0]-time_shift, answer_true[i][1]-time_shift])
                ref_pitches.append(answer_true[i][2])

        for i in range(len(answer_pred)):
            if (answer_pred[i] is not None and float(answer_pred[i][1]) - float(answer_pred[i][0]) > 0
                and answer_pred[i][0] >= 0.0):
                est_intervals.append([answer_pred[i][0], answer_pred[i][1]])
                est_pitches.append(answer_pred[i][2])

    ref_intervals = np.array(ref_intervals)
    est_intervals = np.array(est_intervals)

    return ref_intervals, est_intervals, ref_pitches, est_pitches



def eval_one_data(answer_true, answer_pred, onset_tolerance=0.05, shifting=0, gt_pitch_shift=0):
    
    ref_intervals, est_intervals, ref_pitches, est_pitches = prepare_data(answer_true, answer_pred, time_shift=shifting)

    ref_pitches = np.array([float(ref_pitches[i])+gt_pitch_shift for i in range(len(ref_pitches))])
    est_pitches = np.array([float(est_pitches[i]) for i in range(len(est_pitches))])

    ref_pitches = util.midi_to_hz(ref_pitches)
    est_pitches = util.midi_to_hz(est_pitches)

    if len(est_intervals) == 0:
        ret = np.zeros(14)
        ret[9] = len(ref_pitches)
        return ret

    raw_data = transcription.evaluate(ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=onset_tolerance, pitch_tolerance=50)

    ret = np.zeros(14)
    ret[0] = raw_data['Precision']
    ret[1] = raw_data['Recall']
    ret[2] = raw_data['F-measure']
    ret[3] = raw_data['Precision_no_offset']
    ret[4] = raw_data['Recall_no_offset']
    ret[5] = raw_data['F-measure_no_offset']
    ret[6] = raw_data['Onset_Precision']
    ret[7] = raw_data['Onset_Recall']
    ret[8] = raw_data['Onset_F-measure']
    ret[9] = len(ref_pitches)
    ret[10] = len(est_pitches)
    ret[11] = int(round(ret[1] * ret[9]))
    ret[12] = int(round(ret[4] * ret[9]))
    ret[13] = int(round(ret[7] * ret[9]))

    # print (ret[13], ret[8])
    return ret

def find_bestshift(answer_true, answer_pred, onset_tolerance):
    ref_intervals, est_intervals, _, _ = prepare_data(answer_true, answer_pred, time_shift=0.0)
    diff_points = np.array(get_match_time_shift(ref_intervals, est_intervals))
    diff_points.sort()
    best_onset, best_shift = count_onset(diff_points, onset_tolerance=onset_tolerance)
    p = best_onset / max(1, float(len(ref_intervals)))
    r = best_onset / max(1, float(len(est_intervals)))
    f1_score = (2*p*r) / (p+r)

    # print (best_onset, f1_score, best_shift)
    return best_onset, f1_score, best_shift


def find_allshift(answer_true, answer_pred, onset_tolerance):
    diff_points = []
    ref_length = 0
    est_length = 0
    for i in range(len(answer_true)):
        ref_intervals, est_intervals, _, _ = prepare_data(answer_true[i], answer_pred[i], time_shift=0.0)
        temp = get_match_time_shift(ref_intervals, est_intervals)
        diff_points.extend(temp)
        ref_length = ref_length + len(ref_intervals)
        est_length = est_length + len(est_intervals)

    diff_points = np.array(diff_points)
    diff_points.sort()
    best_onset, best_shift = count_onset(diff_points, onset_tolerance=onset_tolerance)
    print ("best global shift:", (best_shift[0]+best_shift[1])/2.0)
    p = best_onset / max(1, float(ref_length))
    r = best_onset / max(1, float(est_length))
    f1_score = (2*p*r) / (p+r)

    # print (best_onset, f1_score, best_shift)
    return best_onset, f1_score, best_shift

def eval_all(answer_true, answer_pred, onset_tolerance=0.05, method='traditional', shifting=0, print_result=True, id_list=None):

    avg = np.zeros(14)

    if method == 'traditional':
        for i in range(len(answer_true)):
            # ret = eval_one_data(answer_true[i], answer_pred[i], onset_tolerance=onset_tolerance, shifting=shifting)

            # trial = [-24.0, -12.0, 0.0, 12.0, 24.0]
            trial = [0.0,]
            cur_best = np.zeros(14)
            for j in trial:
                ret = eval_one_data(answer_true[i], answer_pred[i], onset_tolerance=onset_tolerance
                    , shifting=shifting, gt_pitch_shift=j)
                # print (ret)
                if ret[5] >= cur_best[5] or ret[8] >= cur_best[8]:
                    for k in range(14):
                        cur_best[k] = ret[k]

            out_message = f"song id {id_list[i]} COnPOff {cur_best[2]:.6f} COnP {cur_best[5]:.6f} COn {cur_best[8]:.6f}"
            # print (out_message)

            for k in range(14):
                avg[k] = avg[k] + cur_best[k]

        for j in range(9):
            avg[j] = avg[j] / len(answer_true)

    elif method == 'best_shift':
        for i in range(len(answer_true)):
            # ret = eval_one_data(answer_true[i], answer_pred[i], onset_tolerance=onset_tolerance, shifting=shifting)

            # trial = [-24.0, -12.0, 0.0, 12.0, 24.0]
            # trial = [0.0,]
            # cur_best = np.zeros(14)
            # for j in trial:
            #     ret = eval_one_data(answer_true[i], answer_pred[i], onset_tolerance=onset_tolerance
            #         , shifting=shifting, gt_pitch_shift=j)
            #     # print (ret)
            #     if ret[5] > cur_best[5] or ret[8] > cur_best[8]:
            #         for k in range(14):
            #             cur_best[k] = ret[k]

            best_onset, f1_score, best_shift = find_bestshift(answer_true[i], answer_pred[i], onset_tolerance=onset_tolerance)
            # out_message = f"song id {id_list[i]} COnPOff {cur_best[2]:.6f} COnP {cur_best[5]:.6f} COn {cur_best[8]:.6f} Best_Onset {f1_score:.6f}"
            out_message = f"song id {id_list[i]} Best_Shift {best_shift[0]:.6f} ~ {best_shift[1]:.6f}"

            print (out_message)

            sub_opt_shift = (best_shift[0] + best_shift[1]) / 2
            # print (sub_opt_shift)
            ret = eval_one_data(answer_true[i], answer_pred[i], onset_tolerance=onset_tolerance
                    , shifting=sub_opt_shift, gt_pitch_shift=0.0)

            out_message = f"song id {id_list[i]} sub_opt shift {sub_opt_shift:.6f}"
            # out_message = out_message + f" COnPOff {ret[2]:.6f} COnP {ret[5]:.6f} COn {ret[8]:.6f}"
            # print (out_message)

            # for k in range(14):
            #     avg[k] = avg[k] + cur_best[k]

            for j in range(14):
                avg[j] = avg[j] + ret[j]

        for j in range(9):
            avg[j] = avg[j] / len(answer_true)

    elif method == 'all_shift':

        best_onset, f1_score, best_shift = find_allshift(answer_true, answer_pred, onset_tolerance=onset_tolerance)
        sub_opt_shift = (best_shift[0] + best_shift[1]) / 2

        for i in range(len(answer_true)):
            # out_message = f"song id {id_list[i]} COnPOff {cur_best[2]:.6f} COnP {cur_best[5]:.6f} COn {cur_best[8]:.6f} Best_Onset {f1_score:.6f}"
            # out_message = out_message + f" Best_Shift {best_shift[0]:.6f} ~ {best_shift[1]:.6f}"

            # print (out_message)
            # print (sub_opt_shift)
            ret = eval_one_data(answer_true[i], answer_pred[i], onset_tolerance=onset_tolerance
                    , shifting=sub_opt_shift, gt_pitch_shift=0.0)

            out_message = f"song id {id_list[i]} sub_opt shift COnPOff {ret[2]:.6f} COnP {ret[5]:.6f} COn {ret[8]:.6f}"
            print (out_message)

            # for k in range(14):
            #     avg[k] = avg[k] + cur_best[k]

            for j in range(14):
                avg[j] = avg[j] + ret[j]

        for j in range(9):
            avg[j] = avg[j] / len(answer_true)



    if print_result:
        print("         Precision Recall F1-score")
        print("COnPOff  %f %f %f" % (avg[0], avg[1], avg[2]))
        print("COnP     %f %f %f" % (avg[3], avg[4], avg[5]))
        print("COn      %f %f %f" % (avg[6], avg[7], avg[8]))
        print ("gt note num:", avg[9], "tr note num:", avg[10])
        print ("song number:", len(answer_true))

    return avg


class MirEval():
    def __init__(self):
        self.gt = None
        self.tr = None
        self.gt_raw = None

    def add_gt(self, gt_path):
        with open(gt_path) as json_data:
            self.gt_raw = json.load(json_data)

    def add_tr_tuple_and_prepare(self, tr):
        length = len(tr)
        gt_data = []
        tr_data = []
        id_list = []
        for i in tr.keys():
            if i in self.gt_raw.keys():
                gt_data.append(self.gt_raw[i])
                tr_data.append(tr[i])
                id_list.append(i)

        self.gt = gt_data
        self.tr = tr_data
        self.id_list = id_list

    def prepare_data(self, gt_path, tr_path):
        with open(tr_path) as json_data:
            tr = json.load(json_data)

        with open(gt_path) as json_data:
            gt = json.load(json_data)

        length = len(tr)
        gt_data = []
        tr_data = []
        id_list = []
        for i in tr.keys():
            if i in gt.keys():
                gt_data.append(gt[i])
                tr_data.append(tr[i])
                id_list.append(i)

        self.gt = gt_data
        self.tr = tr_data
        self.id_list = id_list

    def accuracy(self, onset_tolerance, method, print_result=True):
        return eval_all(self.gt, self.tr, onset_tolerance=onset_tolerance, method=method, print_result=print_result, id_list=self.id_list)


def main(args):
    my_eval = MirEval()
    my_eval.prepare_data(args.gt_file, args.predicted_file)
    print (time.time())
    my_eval.accuracy(onset_tolerance=float(args.tol), method=args.method)
    print (time.time())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_file')
    parser.add_argument('predicted_file')
    parser.add_argument('tol')
    parser.add_argument('-m', '--method', default='traditional')

    args = parser.parse_args()

    main(args)
