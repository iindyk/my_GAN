from PIL import Image

alpha = '10.0'
label = '0'
im_dir = '/home/iindyk/PycharmProjects/my_GAN/images/for_graphs/1vs0/'+alpha+'/'

images = map(Image.open, [im_dir+label+'_1.jpeg', im_dir+label+'_2.jpeg', im_dir+label+'_3.jpeg', im_dir+label+'_4.jpeg'])
im_list = list(images)

new_im = Image.new('L', (55, 55))

new_im.paste(im_list[0], (0, 0))
new_im.paste(im_list[1], (29, 0))
new_im.paste(im_list[2], (0, 29))
new_im.paste(im_list[3], (29, 29))

new_im.save(im_dir+alpha+'_'+label+'.jpg')
