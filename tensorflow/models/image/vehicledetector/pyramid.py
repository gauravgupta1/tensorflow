import cv2

def resize(image, scale_factor):
    return cv2.resize(image, (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)

def pyramid(image, scale=1.5, max_size=(960, 540)):
    last_scale = 1
    yield image.copy(), last_scale

    while True:
        print("pyramid", scale)
        image = resize(image, scale)
        last_scale = last_scale * scale
        if image.shape[0] > max_size[1] or image.shape[1] > max_size[0]:
            break

        yield image.copy(), last_scale
