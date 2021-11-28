# Классификация степени надетости маски

## Запуск
```bash
$ pip install requirements.txt
```
Так же скачайте модели с [google drive](https://drive.google.com/drive/folders/1kLAh4-si-_8TU7b_P-rTtzlqTyB2CjqD)\
(shape_predictor_68_face_landmarks.dat - landmarks и model_6.pt - detection)

Затем запустите python-скрипт
```bash
$ python3 run.py -img=path/to/image -md=path/to/model_detection -ml=path/to/model_landmarks -out=out_image_path
```
## Полезное
так же в папке doc есть документация по проекту\
в папке examples есть примеры распознавания

## Результат работы
![image](https://drive.google.com/uc?export=view&id=1-6WZzo9WdeLc1cq9BMUClt2BWq4wNEZO)
