from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import RegularPolygon
from PIL import Image
import imageio
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import os
import math

IM_SCALE = 0.25

def animate_images(file_name, images, format='GIF'):
    if format.lower() not in file_name[file_name.rfind("."):]:
        if '.' in format: file_name = file_name[:file_name.rfind(".")]
        file_name += '.'+ format.lower()

    if format=='MP4':
        imageio.mimsave(file_name, images)
    elif format=='GIF':
        img, *imgs = images
        img.save(fp=file_name, format=format, append_images=imgs,
                 save_all=True, duration=200, loop=0)


def get_asset_path(asset_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    asset_dir_path = os.path.join(dir_path, 'assets')
    return os.path.join(asset_dir_path, asset_name)


def fig2data(fig, dpi=150):
    fig.set_dpi(dpi)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    data[..., [0, 1, 2, 3]] = data[..., [1, 2, 3, 0]]
    return data


def initialize_figure(height, width, fig_scale=1.):
    fig = plt.figure(figsize=((width + 2) * fig_scale, (height + 2) * fig_scale))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0),
                      aspect='equal', frameon=False,
                      xlim=(-0.05, width + 0.05),
                      ylim=(-0.05, height + 0.05))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.NullFormatter())
        axis.set_major_locator(plt.NullLocator())

    # Draw a grid in the background
    for r in range(height):
        for c in range(width):
            edge_color = '#888888'
            face_color = 'white'

            drawing = RegularPolygon((c + 0.5, (height - 1 - r) + 0.5),
                                     numVertices=4,
                                     radius=0.5 * np.sqrt(2),
                                     orientation=np.pi / 4,
                                     ec=edge_color,
                                     fc=face_color)
            ax.add_patch(drawing)

    return fig, ax


def render_from_layout(layout, get_token_images, dpi=150): #, draw_trace=False
    height, width = layout.shape[:2]

    fig, ax = initialize_figure(height, width)

    for r in range(height):
        for c in range(width):
            token_images = get_token_images(layout[r, c])
            for im in token_images:
                draw_token(im, r, c, ax, height, width)

    im = fig2data(fig, dpi=dpi)
    plt.close(fig) # if not draw_trace:

    im = Image.fromarray(im)
    new_width, new_height = (int(im.size[0] * IM_SCALE), int(im.size[1] * IM_SCALE))
    im = im.resize((new_width, new_height), Image.ANTIALIAS)
    # im = np.array(im)

    return im


def draw_token(token_image, r, c, ax, height, width, token_scale=1.0, fig_scale=1.0):
    oi = OffsetImage(token_image, zoom=fig_scale * (token_scale / max(height, width) ** 0.5))
    box = AnnotationBbox(oi, (c + 0.5, (height - 1 - r) + 0.5), frameon=False)
    ax.add_artist(box)
    return box

## ------------------------------------------------------------------------

##  for visualizing planned trace

## ------------------------------------------------------------------------

def display_image(img, title=None):
    """Render a figure inline
    """
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(img)
    _ = plt.axis('off')
    # plt.show()

def get_robot_pos(pos):
    return ((pos[0]+0.5)*56, (pos[1]+0.5)*56)

def draw_line(pos1, pos2, linewidth=5, color='#e74c3c'):
    point1 = get_robot_pos(pos1)
    point2 = get_robot_pos(pos2)
    plt.plot([point1[1], point2[1]], [point1[0], point2[0]], linewidth=linewidth, color=color)

def generate_color_wheel(original_color, size):

    def hex2int(hex1):
        return int('0x'+str(hex1),0)

    def int2hex(int1):
        hex1 = str(hex(int1)).replace('0x','')
        if len(hex1) == 1:
            hex1 = '0'+hex1
        return hex1

    def hex2ints(original_color):
        R_hex = original_color[0:2]
        G_hex = original_color[2:4]
        B_hex = original_color[4:6]
        R_int = hex2int(R_hex)
        G_int = hex2int(G_hex)
        B_int = hex2int(B_hex)
        return R_int, G_int, B_int

    def ints2hex(R_int, G_int, B_int):
        return '#'+int2hex(R_int)+int2hex(G_int)+int2hex(B_int)

    def portion(total, size, index):
        return total + round((225-total) / size * index)

    def gradients(start, end, size, index):
        return start + round((end-start) / size * index)

    color_wheel = []

    ## for experience replay, find all the colors between two colors
    if len(original_color) == 2:

        color1, color2 = original_color
        R1_int, G1_int, B1_int = hex2ints(color1.replace('#',''))
        R2_int, G2_int, B2_int = hex2ints(color2.replace('#',''))
        for index in range(size):
            color_wheel.append(ints2hex(
                gradients(R1_int, R2_int, size, index),
                gradients(G1_int, G2_int, size, index),
                gradients(B1_int, B2_int, size, index)
            ))

    ## for RL, the color of different shades symbolizes frequency
    else:

        R_int, G_int, B_int = hex2ints(original_color.replace('#',''))

        seq = list(range(size))
        seq.reverse()
        for index in seq:
            color_wheel.append(ints2hex(
                portion(R_int, size, index),
                portion(G_int, size, index),
                portion(B_int, size, index)
            ))

    return color_wheel

def initializee_color_wheel(color_density=None):

    COLOR_WHEEL = []

    ## rainbow color of the material UI style
    colors = ['#F44336','#E91E63','#9C27B0','#673AB7', '#3F51B5', '#2196F3', '#03A9F4', '#00BCD4', '#009688', '#4CAF50', '#8BC34A', '#CDDC39', '#FFEB3B', '#FFC107', '#FF9800', '#FF5722']
    COLOR_DENSITY = math.ceil(color_density/(len(colors)-1))
    for i in range(1,len(colors)):
        COLOR_WHEEL += generate_color_wheel((colors[i-1],colors[i]), COLOR_DENSITY)

    return COLOR_WHEEL

def draw_trace(trace, env=None):
    """ plot the path on map from trace, which is list of robot positions """
    length = len(trace)
    color_wheel = initializee_color_wheel(length)
    for index in range(1, length):
        if env:
            draw_line(env.get_robot_pos(trace[index]), env.get_robot_pos(trace[index-1]),
                      linewidth=int(10 - (10 - 2) * index / length),
                    color=color_wheel[index])
        else: ## randomly drawing action
            draw_line(trace[index], trace[index - 1],
                      linewidth=int(10 - (10 - 2) * index / length),
                      color=color_wheel[index])
    # plt.show()

def record_trace(trace, env, name, dpi=300):
    """ plot the path on map from trace, which is list of robot positions """
    images = []
    for state in trace:
        env.set_state(state)
        images.append(env.render(dpi=dpi))
    outfile = join('tests', name+'.mp4')
    imageio.mimsave(outfile, images)
