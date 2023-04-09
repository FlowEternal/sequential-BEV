import imageio


def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def main():
    image_list = ['cat/1.png', 'cat/2.png', 'cat/3.png', 'cat/4.png', 'cat/5.png', 'cat/6.png']
    gif_name = 'cat/cat.gif'
    duration = 0.35
    create_gif(image_list, gif_name, duration)


if __name__ == '__main__':
    main()