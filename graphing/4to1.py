from PIL import Image

alpha = '10.'
label = '7'
im_dir = '/home/iindyk/PycharmProjects/my_GAN/images/for_graphs/7vs8vs9/'+alpha+'/'

images = map(Image.open, [im_dir+label+'_1.png', im_dir+label+'_2.png', im_dir+label+'_3.png', im_dir+label+'_4.png'])
im_list = list(images)

new_im = Image.new('L', (57, 57))

new_im.paste(im_list[0], (0, 0))
new_im.paste(im_list[1], (29, 0))
new_im.paste(im_list[2], (0, 29))
new_im.paste(im_list[3], (29, 29))

new_im.save(im_dir+alpha+'_'+label+'.jpg')
