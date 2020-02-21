from argparse import ArgumentParser
import random
import string
import sys

from PIL import Image, ImageDraw, ImageFont
import keras
from keras.layers import Dense
from keras.models import Sequential
import numpy as np


CHARS_ENG  = string.ascii_letters

CHARS_RUS = 'йцукенгшщзхъфывапролджэячсмитьбюё'
CHARS_RUS += CHARS_RUS.upper()

CHARS_SPECIAL = string.punctuation
CHARS_SPECIAL += '«»'
CHARS_SPECIAL += string.digits

CHARS_ALL = CHARS_ENG + CHARS_RUS + CHARS_SPECIAL

CHARS_NUM = len(CHARS_ALL)

_CHAR_TO_ONEHOT = dict((
        (char, np.array([0] * idx + [1] + [0] * (max(CHARS_NUM - idx - 1, 0))))
    for
        idx, char
    in
        enumerate(CHARS_ALL)
))

_GLYPH_OFFSET_X = 2
_GLYPH_OFFSET_Y = 2

def char_to_onehot(char):
    return np.copy(_CHAR_TO_ONEHOT[char])


def onehot_to_char(onehot):
    idx = max(enumerate(onehot), key=lambda x: x[1])[0]
    return CHARS_ALL[idx]


def onehot_to_char_gauss(onehot, sigma=1.5):
    indexes_desc = sorted(((idx, weight) for idx, weight in enumerate(onehot)), key=lambda x: -x[1])
    indexes_idx = min(abs(int(random.normalvariate(0, sigma))), len(indexes_desc) - 1)
    return CHARS_ALL[indexes_desc[indexes_idx][0]]


def calculate_max_size(chars, font):
    sizes = []
    for ch in chars:
        sizes.append(font.font.getsize(ch)[0])
    return (max(sizes, key=lambda x: x[0])[0] + _GLYPH_OFFSET_X * 2, max(sizes, key=lambda x: x[1])[1] + _GLYPH_OFFSET_Y * 2)


def generate_base_images(chars, font, max_size):
    img_dict = dict()
    for char in chars:
        img = Image.new('LA', max_size, (0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((_GLYPH_OFFSET_X, _GLYPH_OFFSET_Y), char, (0,), font=font)
        img_dict[char] = img
    return img_dict


# Copyright https://github.com/kuszaj/claptcha
def random_transform_image(image):
    w, h = image.size

    dx = w * random.uniform(0.1, 0.3)
    dy = h * random.uniform(0.1, 0.3)

    x1, y1 = _random_point_disposition(dx, dy)
    x2, y2 = _random_point_disposition(dx, dy)

    w += abs(x1) + abs(x2)
    h += abs(x1) + abs(x2)

    quad = _quad_points((w, h), (x1, y1), (x2, y2))

    return image.transform(image.size, Image.QUAD,
                            data=quad, resample=Image.BILINEAR)


def _random_point_disposition(dx, dy):
    x = int(random.uniform(-dx, dx))
    y = int(random.uniform(-dy, dy))
    return (x, y)


def _quad_points(size, disp1, disp2):
    w, h = size
    x1, y1 = disp1
    x2, y2 = disp2

    return (
        x1,     -y1,
        -x1,    h + y2,
        w + x2, h - y2,
        w - x2, y1
    )


def generate_distorted_sample(img_LA):
    img_arr = np.array(random_transform_image(img_LA))
    shape = img_arr.shape
    img_arr_solid = np.zeros((shape[0], shape[1]), dtype='float')
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img_arr[i, j][1] == 255:
                    img_arr_solid[i, j] = 1
            else:
                    img_arr_solid[i, j] = 0
    return img_arr_solid.reshape(img_arr.shape[0] * img_arr.shape[1])


def shuffle_sets_equally(x_set, y_set):
    # LOL
    rng_state = np.random.get_state()
    np.random.shuffle(x_set)
    np.random.set_state(rng_state)
    np.random.shuffle(y_set)


def generate_set(max_size, base_images, samples_per_char):
    input_vec_len = max_size[0] * max_size[1]
    output_vec_len = CHARS_NUM
    set_size = samples_per_char * len(base_images)

    x_values = []
    y_values = []
    for c, img in base_images.items():
        for _ in range(samples_per_char):
            x_values.append(generate_distorted_sample(img))
            y_values.append(char_to_onehot(c))

    x_set = np.concatenate(x_values).reshape((set_size, input_vec_len))
    y_set = np.concatenate(y_values).reshape((set_size, output_vec_len))
    shuffle_sets_equally(x_set, y_set)

    return x_set, y_set


def generate_set_biased(max_size, base_images_list, base_images_samples):
    assert len(base_images_list) == len(base_images_samples)
    res_x = []
    res_y = []
    for base_images, samples_per_image in zip(base_images_list, base_images_samples):
        x_set, y_set = generate_set(max_size, base_images, samples_per_image)
        res_x.append(x_set)
        res_y.append(y_set)
    x_set = np.concatenate(res_x)
    y_set = np.concatenate(res_y)

    shuffle_sets_equally(x_set, y_set)

    return x_set, y_set


def generate_batch_from_string(max_size, base_images, target_string):
    input_vec_len = max_size[0] * max_size[1]
    batch_size = len(target_string)
    x_set = np.empty(shape=(batch_size, input_vec_len))

    sample_num = 0
    for c in target_string:
        x_set[sample_num] = generate_distorted_sample(base_images[c])
        sample_num += 1
    return x_set


def generate_sets(max_size,
                 base_images_rus,
                 base_images_eng,
                 base_images_special):
    base_images_list = (base_images_rus, base_images_special, base_images_eng)
    x_train, y_train = generate_set_biased(max_size, base_images_list, (85, 10, 5))
    x_valid, y_valid = generate_set_biased(max_size, base_images_list, (20, 3, 2))
    x_test, y_test = generate_set_biased(max_size, base_images_list, (7, 2, 1))
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def save_sets(file_prefix, x_train, y_train, x_valid, y_valid, x_test, y_test):
    np.save(file_prefix + '_x_train', x_train)
    np.save(file_prefix + '_y_train', y_train)
    np.save(file_prefix + '_x_valid', x_valid)
    np.save(file_prefix + '_y_valid', y_valid)
    np.save(file_prefix + '_x_test', x_test)
    np.save(file_prefix + '_y_test', y_test)


def load_sets(file_prefix):
    x_train, y_train = np.load(file_prefix + '_x_train.npy'), np.load(file_prefix + '_y_train.npy')
    x_valid, y_valid = np.load(file_prefix + '_x_valid.npy'), np.load(file_prefix + '_y_valid.npy')
    x_test, y_test = np.load(file_prefix + '_x_test.npy'), np.load(file_prefix + '_y_test.npy')
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def filter_out_chars(string, whitelist):
    whitelist = set(whitelist)
    res = []
    filtered_chars = []
    for idx, c in enumerate(string):
        if c in whitelist:
            res.append(c)
        else:
            filtered_chars.append((idx, c))
    return ''.join(res), filtered_chars[::-1]


def restore_chars(string, filtered_chars):
    idx = 0
    res = []
    for c in string:
        while len(filtered_chars) > 0:
            if filtered_chars[-1][0] == idx:
                res.append(filtered_chars[-1][1])
                filtered_chars.pop()
                idx += 1
            else:
                break
        res.append(c)
        idx += 1
    return ''.join(res)


def main():
    parser = ArgumentParser()
    parser.add_argument('model', choices=('load-model', 'train-model'))
    parser.add_argument('--sets', choices=('load', 'gen'), default=None)
    parser.add_argument('--sets-prefix', type=str, dest='sets_prefix', default='turel')
    parser.add_argument('--model-filename', type=str, dest='model_filename', default='turel-model.keras')
    parser.add_argument('--save-model', action='store_true', dest='save_model', default=False)
    parser.add_argument('--train-epochs', type=int, dest='train_epochs', default=10)
    parser.add_argument('--save-sets', action='store_true', dest='save_sets', default=False)
    parser.add_argument('--font-filename', type=str, dest='font_filename', default='DroidSansMono.ttf')
    parser.add_argument('--font-size', type=int, dest='font_size', default=64)
    parser.add_argument('--use-gauss', action='store_true', dest='use_gauss', default=False, help='Manually emulate model errors')
    parser.add_argument('--gauss-sigma', type=float, dest='gauss_sigma', default=0.8)
    parser.add_argument('--input', type=str, dest='input_filename', default='input.txt', help='Input filename')
    parser.add_argument('--output', type=str, dest='output_filename', default='output.txt', help='Output filename')
    args = parser.parse_args()

    args.load_model = args.model == 'load-model'
    args.train_model = args.model == 'train-model'

    FONT_SIZE = args.font_size
    FONT_FILENAME = args.font_filename
    
    font = ImageFont.truetype(FONT_FILENAME, FONT_SIZE)
    max_size = calculate_max_size(CHARS_ALL, font)
    
    base_images_rus = generate_base_images(CHARS_RUS, font, max_size)
    base_images_eng = generate_base_images(CHARS_ENG, font, max_size)
    base_images_special = generate_base_images(CHARS_SPECIAL, font, max_size)
    base_images_all = generate_base_images(CHARS_ALL, font, max_size)

    input_vec_len = max_size[0] * max_size[1]

    if args.train_model and args.sets is None:
        print('To train the model you must specify either "--sets load" or "--sets gen".')
        sys.exit(1)
    
    if args.save_sets and args.sets != 'gen':
        print('To save sets you must generate them with "--sets gen".')
        sys.exit(1)

    x_train, y_train, x_valid, y_valid, x_test, y_test = (None,) * 6
    if args.sets == 'gen':
        print('Generating sets, this may take several minutes...')
        x_train, y_train, x_valid, y_valid, x_test, y_test = generate_sets(max_size,
                                                                           base_images_rus,
                                                                           base_images_eng,
                                                                           base_images_special)
        print('Done!')
    elif args.sets == 'load':
        print('Loading sets...')
        x_train, y_train, x_valid, y_valid, x_test, y_test = load_sets(args.sets_prefix)
        print('Done!')

    if args.save_sets:
        print('Saving sets...')
        save_sets(args.sets_prefix, x_train, y_train, x_valid, y_valid, x_test, y_test)
        print('Done!')

    print('Creating the model...')
    optimizer = keras.optimizers.Adagrad(learning_rate=0.02)
    model = Sequential()
    model.add(Dense(units=CHARS_NUM * 2, activation='relu', input_dim=input_vec_len))
    model.add(Dense(units=CHARS_NUM, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print('Done!')

    if args.load_model:
        print('Loading the model...')
        model.load_weights(args.model_filename)
        print('Done!')

    if args.train_model:
        print('Training the model...')
        model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=args.train_epochs, batch_size=32)
        loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
        print(model.metrics_names)
        print(loss_and_metrics)
        print('Done!')

    if args.save_model:
        print('Saving the model to', args.model_filename)
        model.save(args.model_filename)
        print('Done!')

    with open(args.input_filename, 'r', encoding='utf-8') as f:
        text = f.read()

    string_filtered, filtered_chars = filter_out_chars(text, CHARS_ALL)
    x_manual = generate_batch_from_string(max_size, base_images_all, string_filtered)

    print(f'Cobenizing {args.input_filename}...')
    model_prediction = model.predict(x_manual)
    predicted_chars = []
    for predicted_one_hot in model_prediction:
        if args.use_gauss:
            predicted_chars.append(onehot_to_char_gauss(predicted_one_hot, args.gauss_sigma))
        else:
            predicted_chars.append(onehot_to_char(predicted_one_hot))
    predict = ''.join(predicted_chars)
    
    restored_text = restore_chars(predict, filtered_chars)
    print(restored_text)

    with open(args.output_filename, 'w', encoding='utf-8') as f:
        f.write(restored_text)

    # show_batch(x_manual, max_size)


def show_batch(x_values, max_size, max_count=-1):
    imgs = []
    for i in range(x_values.shape[0] if max_count < 0 else max_count):
        img_arr = x_values[i].reshape((max_size[1], max_size[0])) * 255
        imgs.append(Image.fromarray(img_arr))
    show_combined_image(imgs, max_size)


def show_combined_image(images, max_size):
    img_combined = Image.new('L', (max_size[0] * len(images), max_size[1]), (255,))
    offset_x = 0
    for img in images:
        img_combined.paste(img, (offset_x, 0))
        offset_x += max_size[0]
    img_combined.show()


if __name__ == '__main__':
    main()
