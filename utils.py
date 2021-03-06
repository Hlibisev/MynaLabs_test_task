import os

import numpy as np
from PIL import Image
from tqdm import tqdm
import face_recognition as fr
import torch


def to_numpy(x):
    return x.detach().cpu().numpy().reshape(-1)


def make_batch(face_loc, indices, reprocess, numpy_images, device='cuda') -> torch.Tensor:
    """ Takes face location and connect for one torch.Tensor  """
    batch_list = []

    for index in indices:
        x0, x1, y1, y0 = face_loc[index][0]

        cropped_image = numpy_images[x0: x1, y0: y1]
        tensor_image = reprocess(cropped_image)
        batch_list.append(tensor_image.unsqueeze(0))

    batch = torch.cat(batch_list, axis=0).to(device)
    return batch


def delete_small_faces(face_locs, indices, resol=30):
    """ Funny name """
    face_locs1 = np.array([face_locs[index][0] for index in indices])
    indices1 = face_locs1[:, 1] - face_locs1[:, 0] > resol
    indices2 = face_locs1[:, 2] - face_locs1[:, 3] > resol
    indices12 = indices1 & indices2

    return indices[indices12]


def find_pictures_with_glasses(path_to_pictures, glasses_model, transform_picture, resol=256, n_samples=1000):
    """
    Main function of task
    :param path_to_pictures: path to file with pictures
    :param glasses_model: torch model_weights which works with (120, 120) input size
    :param transform_picture: torchvision.transform applying on numpy array
    :param resol: threshold when pictures is small
    :return:
    """
    file_names = list(filter(lambda x: x.endswith('.jpg'), os.listdir(path_to_pictures)))

    # list of paths for good pictures
    files_with_glasses = []

    # collect batch_size == 128 and use model_weights
    batch_images = []
    batch_paths = []

    for i, file_name in tqdm(enumerate(file_names[::-1])):

        if len(files_with_glasses) > n_samples:
            return files_with_glasses

        path = os.path.join(path_to_pictures, file_name)
        image = Image.open(path)

        # pass if small picture
        x, y = image.size
        if x < 400 or y < 400:
            continue

        # reduce picture
        resized_image = image.resize((128, 128), Image.ANTIALIAS)
        numpy_image = np.array(resized_image)

        # fill in batch
        batch_images.append(numpy_image)
        batch_paths.append(file_name)

        if len(batch_images) >= 128 or i == len(file_names) - 1:
            # find faces
            face_locs = fr.batch_face_locations(batch_images)

            # pass pictures with many people
            face_count = np.array(list(map(len, face_locs)))

            indices = np.where(face_count == 1)[0]
            indices = delete_small_faces(face_locs, indices, resol)

            if len(indices) == 0:
                batch_images, batch_paths = [], []
                continue

            # find glasses
            batch = make_batch(face_locs, indices, transform_picture, batch_images)
            predict = to_numpy(glasses_model(batch)) > 0.5

            files_with_glasses += list(np.array(batch_paths)[indices][predict])
            batch_images, batch_paths = [], []

    return files_with_glasses
