from model import get_model, get_transform
from utils import find_pictures_with_glasses


if __name__ == '__main__':
    path_to_folder = '/Users/antonfonin/Downloads/MeGlass_120x120'
    model = get_model('model_weights')
    transform = get_transform()

    good_pictures = find_pictures_with_glasses(path_to_folder, model, transform)
