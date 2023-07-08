import pandas as pd
import numpy as np
import cv2
import os
import json
import argparse
import sys


user_x, user_y = 0, 0
user_click = False
user_op = 0

COMMAND_OR = 0
COMMAND_AND = 1

PATH_IMAGE = "frame000003.png"
PATH_FRAME = "frame000003"
PATH_METADATA = "metadata.csv"
PATH_OUTPUT = "output/"


class DataEntry:
    def __init__(self, mask, bx, by, bw, bh, label) -> None:
        self.mask = mask
        self.bx = bx
        self.by = by
        self.bw = bw
        self.bh = bh
        self.label = label

    def __str__(self):
        return "<" + self.label + ">"

    def __repr__(self) -> str:
        return str(self)


def save_dataset(data_dict, output_dir: str):
    path_source = os.path.join(output_dir, 'rgb')
    path_semantic = os.path.join(output_dir, 'semantic')
    if not os.path.exists(path_source):
        os.mkdir(path_source)
    if not os.path.exists(path_semantic):
        os.mkdir(path_semantic)

    # Check which images to skip because empty
    skip_lst = []
    for k in data_dict.keys():
        num_labels = 0
        for label in data_dict[k]['Entries'].keys():
            num_labels += len(data_dict[k]['Entries'][label])
        if num_labels == 0:
            skip_lst.append(k)

    # Dump source images
    for k in data_dict.keys():
        if k in skip_lst:
            print('skipping', k)
            continue
        cv2.imwrite(os.path.join(path_source, os.path.basename(k)),
                    data_dict[k]['Source'])

    output_dict = dict()

    output_dict['RGB'] = [os.path.join(
        path_source, os.path.basename(k)) for k in data_dict.keys()]

    output_dict['BB'] = list()
    output_dict['Mask'] = list()
    output_dict['Labels'] = list()
    cnt = 0
    for k in data_dict.keys():
        if k in skip_lst:
            print('skipping', k)
            continue
        list_k_bb = list()
        list_k_mask = list()
        list_k_labels = list()
        for lab in data_dict[k]['Entries'].keys():
            for j in range(len(data_dict[k]['Entries'][lab])):
                entry = data_dict[k]['Entries'][lab][j]
                list_k_bb.append([entry.bx, entry.by, entry.bw, entry.bh])
                list_k_labels.append(entry.label)
                smask = entry.mask
                path_smask = os.path.join(path_semantic, "seg_" + str(cnt) + os.path.basename(
                    k))
                cnt += 1
                cv2.imwrite(path_smask, smask)
                list_k_mask.append(path_smask)
        output_dict['BB'].append(list_k_bb)
        output_dict['Mask'].append(list_k_mask)
        output_dict['Labels'].append(list_k_labels)

    print(json.dumps(output_dict))
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        f.write(json.dumps(output_dict))


def mouse_cb(event, x, y, flags, param):
    global user_x, user_y, user_click, user_op

    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
        user_x = x
        user_y = y
        user_click = True
        if event == cv2.EVENT_LBUTTONDOWN:
            user_op = COMMAND_OR
        elif event == cv2.EVENT_RBUTTONDOWN:
            user_op = COMMAND_AND
        print("User has clicked")


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Porocodiio')
    parser.add_argument('--images', dest='path_image',
                        required=True, type=str)
    parser.add_argument('--seg', dest='path_segment', required=True, type=str)
    parser.add_argument('--output', dest='path_output',
                        required=True, type=str)

    args = parser.parse_args()

    PATH_OUTPUT = args.path_output
    if not os.path.exists(PATH_OUTPUT):
        os.mkdir(PATH_OUTPUT)

    dict_results = {}
    dict_source_images = {}
    flag_stop = False
    num_read_frames = 0
    num_total_frames = len(os.listdir(args.path_segment))
    
    for sdir in os.listdir(args.path_segment):
        num_read_frames += 1
        print(f"frame {num_read_frames}/{num_total_frames}")
        PATH_FRAME = os.path.join(args.path_segment, sdir)
        PATH_IMAGE = os.path.join(args.path_image, sdir + ".png")

        image = cv2.imread(PATH_IMAGE)
        dict_results[PATH_IMAGE] = dict()
        dict_results[PATH_IMAGE]['Source'] = image
        dict_results[PATH_IMAGE]['Entries'] = {
            'can': [],
            'bottle': [],
            'pouch': []
        }

        metadata = pd.read_csv(os.path.join(PATH_FRAME, "metadata.csv"))

        cv2.namedWindow("Masks")
        cv2.setMouseCallback("Masks", mouse_cb)

        lut = np.zeros(image.shape[:2], dtype=np.int64)
        mask_frame = np.zeros(image.shape, dtype=np.uint8)
        mask_dict = {}
        mask_size_list = []

        for index, row in metadata.iterrows():
            mask_id = int(row["id"])
            mask_area = int(row["area"])
            filename_mask = os.path.join(
                PATH_FRAME, str(int(row["id"])) + ".png")
            mask_cv = cv2.imread(filename_mask)
            mask_cv = cv2.cvtColor(mask_cv, cv2.COLOR_BGR2GRAY)
            mask_dict[mask_id] = mask_cv
            mask_size_list.append((mask_area, mask_id))

        mask_size_list.sort(key = lambda x: x[0], reverse=True)
        for _, mask_id in mask_size_list:
            lut[mask_dict[mask_id] == 255] = mask_id
            mask_frame[mask_dict[mask_id] == 255] = np.random.random_integers(
                0, 255, (3,))
            print('.', end='')
            sys.stdout.flush()

        print()

        frame_mask_colors = mask_frame
        mask_frame = np.zeros_like(image)
        mask_frame = image
        mask_frame = cv2.addWeighted(image, 0.8, frame_mask_colors, 0.2, 0)

        is_ok = True

        resulting_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        frame_bbox = np.zeros_like(resulting_mask)
        resulting_bbox = None
        last_detection_label = None
        last_mask_show = None

        while is_ok:
            cv2.imshow("Masks", mask_frame)
            frame_mask_show = resulting_mask.copy()
            cv2.imshow("query_mask", cv2.resize(
                frame_mask_show, (int(frame_mask_show.shape[1]*0.5), int(frame_mask_show.shape[0]*0.5))))
            frame_bb_show = frame_bbox.copy()
            cv2.imshow("query_bb", cv2.resize(
                frame_bb_show, (int(frame_bb_show.shape[1]*0.5), int(frame_bb_show.shape[0]*0.5))))
            ret = cv2.waitKey(1)

            if user_click:
                query_id = lut[user_y, user_x]
                print("Query_ID=", query_id)
                query_mask = mask_dict[query_id]
                if user_op == COMMAND_OR:
                    resulting_mask |= query_mask
                if user_op == COMMAND_AND:
                    resulting_mask &= ~query_mask

                resulting_bbox = cv2.boundingRect(resulting_mask)
                x, y, w, h = resulting_bbox
                frame_bbox = cv2.rectangle(
                    resulting_mask.copy(), (x, y), (x+w, y+h), 128, 2)
                user_click = False

            if ret == ord('n'):
                is_ok = False

            if ret == ord('o'):
                if resulting_bbox is None:
                    print('Skipping labelling due to empty selection')
                    continue
                last_mask_show = mask_frame.copy()
                # Optimize mask removing noisy shit
                print("Removing spurious shit")
                se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
                se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
                resulting_mask = cv2.morphologyEx(
                    resulting_mask, cv2.MORPH_CLOSE, se1)
                resulting_mask = cv2.morphologyEx(
                    resulting_mask, cv2.MORPH_OPEN, se2)
                resulting_bbox = cv2.boundingRect(resulting_mask)
                x, y, w, h = resulting_bbox
                frame_bbox = cv2.rectangle(
                    resulting_mask.copy(), (x, y), (x+w, y+h), 128, 2)

            if ret == ord('c'):
                if resulting_bbox is None:
                    print('Skipping labelling due to empty selection')
                    continue
                last_mask_show = mask_frame.copy()
                print("Saving as DIO-CAN")
                dict_results[PATH_IMAGE]['Entries']['can'].append(
                    DataEntry(resulting_mask, *resulting_bbox, 'can'))
                mask_frame[resulting_mask == 255] = 0
                resulting_bbox = None
                frame_bbox *= 0
                resulting_mask *= 0
                last_detection_label = 'can'
                # TODO

            if ret == ord('b'):
                if resulting_bbox is None:
                    print('Skipping labelling due to empty selection')
                    continue
                last_mask_show = mask_frame.copy()
                print("Saving as BOTTLE")
                dict_results[PATH_IMAGE]['Entries']['bottle'].append(
                    DataEntry(resulting_mask, *resulting_bbox, 'bottle'))
                mask_frame[resulting_mask == 255] = 0
                resulting_bbox = None
                frame_bbox *= 0
                resulting_mask *= 0
                last_detection_label = 'bottle'
                # TODO

            if ret == ord('p'):
                if resulting_bbox is None:
                    print('Skipping labelling due to empty selection')
                    continue
                last_mask_show = mask_frame.copy()
                print("Saving as POUCH")
                dict_results[PATH_IMAGE]['Entries']['pouch'].append(
                    DataEntry(resulting_mask, *resulting_bbox, 'pouch'))
                mask_frame[resulting_mask == 255] = 0
                resulting_bbox = None
                frame_bbox *= 0
                resulting_mask *= 0
                last_detection_label = 'pouch'
                # TODO

            if ret == ord('q'):
                print('Quitting')
                flag_stop = True
                break

            if ret == ord('d'):
                print('Explode last mask')
                if last_detection_label is None or len(dict_results[PATH_IMAGE]['Entries'][last_detection_label]) == 0:
                    print('Skipping explosion of last entry since it doesnt exists')
                    continue
                dict_results[PATH_IMAGE]['Entries'][last_detection_label].pop()
                mask_frame = last_mask_show.copy()

        if flag_stop:
            break
    save_dataset(dict_results, PATH_OUTPUT)

    exit(0)
