import logging

import dataset.format_dataset
import metrics_utils.mAP
import detect.multi_gen_project_detect
import detect.self_mvpf_detect

import detect.illustrate.experiment_


def index_experiment_wo_fusion(index: int):
    format_dataset = dataset.format_dataset.IndexFormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
        index_size=index,
    )

    mAP_1 = metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="",
        detect_func=detect.illustrate.experiment_.weight_proj_detect(
            detect.multi_gen_project_detect.multi_gen_project_detect,
            pano_weight_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/equi_1n_exp13_best.pt',
            proj_weight_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt',
        ),
        detect_save_dir=f"/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/multi_gen_project_detect",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir=f'/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/experiments/tmp',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},

    )

    mAP_2 = metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="",
        detect_func=detect.illustrate.experiment_.weight_proj_detect(
            detect.self_mvpf_detect.self_mvpf_detect,
            pano_weight_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/equi_1n_exp13_best.pt',
            proj_weight_path='/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/weights/stereo_1n_d_exp113_best.pt',
        ),
        detect_save_dir=f"/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/self_mvpf_detect",
        cache=False,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir=f'/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/experiments/tmp',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},

    )

    logging.error(f'{i}: {mAP_2} - {mAP_1} = {mAP_2 - mAP_1} ')


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%d-%m-%Y:%H:%M:%S',
        level=logging.ERROR,
    )

    for i in range(100):
        index_experiment_wo_fusion(i)
