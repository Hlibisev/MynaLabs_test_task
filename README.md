# MynaLabs_test_task

Task: preparing a dataset with people wearing glasses

1) I gave data on https://github.com/cleardusk/MeGlass, but this dataset contains low-resolution pictures too. In many photos, there are several people therefore I was inspired to make a pipeline.
2) I downloaded cropped (120 * 120) faces with glasses and without and trained model for classification task (model.py).
3) With face_recognition library I found faces and skip picture if it has more than one face, if face without glasses (my model was used), with face occupies a small part
4) Create main function find_pictures_with_glasses which takes path_to_file and return a list of good photos
5) Apply on original MeGlass dataset
6) This pipline can be used for other unmarked datasets.



# Dataset
https://drive.google.com/drive/u/3/folders/14fBKO8JNZs3eb07uuBHpYmJnIo359WOD

# Colab
https://colab.research.google.com/drive/199e185g6FNLtsAPWaGamKjVlRa5l0iE7?usp=sharing
