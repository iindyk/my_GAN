from PIL import Image

im_dir = '/home/iindyk/PycharmProjects/my_GAN/images/for_graphs/1vs0/10.0/'
images = map(Image.open, [im_dir+'1_1.jpeg', im_dir+'1_2.jpeg', im_dir+'1_3.jpeg', im_dir+'1_4.jpeg'])
im_list = list(images)

new_im = Image.new('L', (55, 55))

new_im.paste(im_list[0], (0, 0))
new_im.paste(im_list[1], (29, 0))
new_im.paste(im_list[2], (0, 29))
new_im.paste(im_list[3], (29, 29))

new_im.save(im_dir+'10.0_1.jpg')
