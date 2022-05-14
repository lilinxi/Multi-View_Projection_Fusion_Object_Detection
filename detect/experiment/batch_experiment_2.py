import logging

import dataset.format_dataset
import metrics_utils.mAP

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.ERROR,
)


def batch_experiment(mini_size: int, weight_detect_func: callable, exp_name: str):
    if mini_size < 1:
        format_dataset = dataset.format_dataset.FormatDataset(
            dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
            train=False,
            image_width=1024,
            image_height=512,
            class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                          'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
        )
    else:
        format_dataset = dataset.format_dataset.MiniFormatDataset(
            dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
            train=False,
            image_width=1024,
            image_height=512,
            class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                          'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
            mini_size=mini_size,
        )

    metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="",
        detect_func=weight_detect_func,
        detect_save_dir=f"/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/mvpf_detect",
        cache=True,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir=f'/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/experiments/{exp_name}',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},

    )


def sub_experiment(weight_detect_func: callable, exp_name: str):
    format_dataset = dataset.format_dataset.SubFormatDataset(
        dataset_root="/Users/bytedance/Dataset/sun360_extended_dataset_format",
        train=False,
        image_width=1024,
        image_height=512,
        class_labels=['bed', 'painting', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa',
                      'door', 'cabinet', 'bedside', 'tv', 'computer', "glass", "rug", "shelf"],
    )

    mAP = metrics_utils.mAP.compute_mAP(
        format_dataset=format_dataset,
        weights_path="",
        detect_func=weight_detect_func,
        detect_save_dir=f"/Users/bytedance/Dataset/sun360_extended_dataset_format/labels/detect/mvpf_detect",
        cache=True,
        names_dict={0: 'bed', 1: 'painting', 2: 'table', 3: 'mirror', 4: 'window', 5: 'curtain', 6: 'chair', 7: 'light',
                    8: 'sofa',
                    9: 'door', 10: 'cabinet', 11: 'bedside', 12: 'tv', 13: 'computer', 14: 'glass', 15: 'rug',
                    16: 'shelf'},
        plot_save_dir=f'/Users/bytedance/PycharmProjects/Multi-View_Projection_Fusion_Object_Detection/experiments/{exp_name}',
        # plot_names_dict={0: True, 1: True, 2: True, 3: False, 4: False, 5: True, 6: False, 7: False, 8: False,
        #                  9: True, 10: False, 11: True, 12: True, 13: False, 14: False, 15: False, 16: False},

    )


if __name__ == '__main__':
    import detect.mvpf_detect

    sub_experiment(detect.mvpf_detect.weighted_detect, f"sub_experiment_mvpf_detect")
