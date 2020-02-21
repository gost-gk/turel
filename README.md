# ТУРЕЛЬ: 1-1 генератор сорцовой психозы
Вореции: [https://github.com/1024--/voretions](https://github.com/1024--/voretions).

## Установка
* `pip install numpy`
* `pip install Pillow`
* `pip install tensorflow` или `pip install tensorflow-gpu`
* `pip install keras`
* `git clone --recurse-submodules https://github.com/gost-gk/turel.git`
* `cd turel`
* В папке с turel.py создать файл `input.txt` со входной кобенадой.

## Использование
### Обычное использование для генерации психозы (модель-образец есть в репозитории)
`python turel.py load-model --use-substitution`

Вход — input.txt, выход — output.txt (настраивается аргументами `--input`, `--output`).

### Генерация тренировочных наборов и тренировка модели
`python turel.py train-model --sets gen --save-sets --save-model`

### Обучение и сохранение новой модели по старым наборам
`python turel.py train-model --sets load --save-model`

Остальные аргументы в коде.