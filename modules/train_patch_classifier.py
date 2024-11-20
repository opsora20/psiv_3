# -*- coding: utf-8 -*- noqa
"""
Created on Wed Nov 20 23:06:19 2024

@author: JoelT
"""


def train_patch_classifier(model, dataloader):
    target_labels = []
    fred_list = []

    for batch_id, (inputs, labels) in enumerate(dataloader):
        if batch_id % 100 == 0:
            print(batch_id)
        outputs = model.encode(inputs)
        for input_image, output_image, label in zip(inputs, outputs, labels):

            fred_result = model.calculate_fred(
                input_image,
                output_image,
                plot=False,
            )

            fred_list.append(fred_result)

            target_labels.append(label)

    return fred_list, target_labels
