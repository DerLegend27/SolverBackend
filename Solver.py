import cv2
import numpy as np
import numpy
from PIL import Image
from latex2sympy2 import latex2sympy, latex2latex
from sympy import *


import torch
from torch.autograd import Variable
from Attention_RNN import AttnDecoderRNN
from Densenet_torchvision import densenet121

gpu = [0]
dictionaries = ['dictionary.txt']
hidden_size = 256
batch_size_t = 1
maxlen = 100

BLOCK_SIZE = 50
THRESHOLD = 25

def preprocess(image):
    image = cv2.medianBlur(image, 3)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return 255 - image

def postprocess(image):
    image = cv2.medianBlur(image, 5)
    #kernel = numpy.ones((3,3), numpy.uint8)
    #image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image

def get_block_index(image_shape, yx, block_size): 
    y = numpy.arange(max(0, yx[0]-block_size), min(image_shape[0], yx[0]+block_size))
    x = numpy.arange(max(0, yx[1]-block_size), min(image_shape[1], yx[1]+block_size))
    return numpy.meshgrid(y, x)

def adaptive_median_threshold(img_in):
    med = numpy.median(img_in)
    img_out = numpy.zeros_like(img_in)
    img_out[img_in - med < THRESHOLD] = 255
    return img_out

def block_image_process(image, block_size):
    out_image = numpy.zeros_like(image)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            idx = (row, col)
            block_idx = get_block_index(image.shape, idx, block_size)
            out_image[block_idx] = adaptive_median_threshold(image[block_idx])

    return out_image

def processing_image():
    image_in = cv2.cvtColor(cv2.imread("images/math-equation.png"), cv2.COLOR_BGR2GRAY)

    image_in = preprocess(image_in)
    image_out = block_image_process(image_in, BLOCK_SIZE)
    image_out = postprocess(image_out)

    binaryImage = cv2.adaptiveThreshold(image_out, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 85, 10)

    # For debug purpose only
    cv2.imshow("processed-image", binaryImage)
    cv2.waitKey(0)
    cv2.imwrite("images/processed-image.bmp", binaryImage)

    img_open = Image.open("images/processed-image.bmp").convert("L")
    img_open2 = torch.from_numpy(np.array(img_open)).type(torch.FloatTensor)
    img_open2 = img_open2/255.0
    img_open2 = img_open2.unsqueeze(0)
    img_open2 = img_open2.unsqueeze(0)

    attention, prediction = solver(img_open2, load_dict())
    prediction_string = ''

    for i in range(attention.shape[0]):
        if prediction[i] == '<eol>':
            continue
        else:
            prediction_string = prediction_string + prediction[i]

    return prediction_string

def load_dict():
    fp=open(dictionaries[0])
    stuff=fp.readlines()
    fp.close()
    lexicon={}
    
    for l in stuff:
        w=l.strip().split()
        lexicon[w[0]]=int(w[1])
    
    worddicts = lexicon
    worddicts_r = [None] * len(worddicts)
    
    for kk, vv in worddicts.items():
        worddicts_r[vv] = kk
        
    print("! Dictionary sucessfully iniated !")
    return worddicts_r

def solver(x_t, worddicts_r):

	h_mask_t = []
	w_mask_t = []
	encoder = densenet121()
	attn_decoder1 = AttnDecoderRNN(hidden_size,112,dropout_p=0.5)

	encoder = torch.nn.DataParallel(encoder, device_ids=gpu)
	attn_decoder1 = torch.nn.DataParallel(attn_decoder1, device_ids=gpu)
	encoder = encoder.cuda()
	attn_decoder1 = attn_decoder1.cuda()

	encoder.load_state_dict(torch.load('models/encoder_lr0.00001_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl'))
	attn_decoder1.load_state_dict(torch.load('models/attn_decoder_lr0.00001_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl'))

	encoder.eval()
	attn_decoder1.eval()

	x_t = Variable(x_t.cuda())
	x_mask = torch.ones(x_t.size()[0],x_t.size()[1],x_t.size()[2],x_t.size()[3]).cuda()
	x_t = torch.cat((x_t,x_mask),dim=1)
	x_real_high = x_t.size()[2]
	x_real_width = x_t.size()[3]
	h_mask_t.append(int(x_real_high))
	w_mask_t.append(int(x_real_width))
	x_real = x_t[0][0].view(x_real_high,x_real_width)
	output_highfeature_t = encoder(x_t)

	x_mean_t = torch.mean(output_highfeature_t)
	x_mean_t = float(x_mean_t)
	output_area_t1 = output_highfeature_t.size()
	output_area_t = output_area_t1[3]
	dense_input = output_area_t1[2]

	decoder_input_t = torch.LongTensor([111]*batch_size_t)
	decoder_input_t = decoder_input_t.cuda()

	decoder_hidden_t = torch.randn(batch_size_t, 1, hidden_size).cuda()
	# nn.init.xavier_uniform_(decoder_hidden_t)
	decoder_hidden_t = decoder_hidden_t * x_mean_t
	decoder_hidden_t = torch.tanh(decoder_hidden_t)

	prediction = torch.zeros(batch_size_t,maxlen)
	#label = torch.zeros(batch_size_t,maxlen)
	prediction_sub = []
	label_sub = []
	decoder_attention_t = torch.zeros(batch_size_t,1,dense_input,output_area_t).cuda()
	attention_sum_t = torch.zeros(batch_size_t,1,dense_input,output_area_t).cuda()
	decoder_attention_t_cat = []


	for i in range(maxlen):
	    decoder_output, decoder_hidden_t, decoder_attention_t, attention_sum_t = attn_decoder1(decoder_input_t,
	                                                                                     decoder_hidden_t,
	                                                                                     output_highfeature_t,
	                                                                                     output_area_t,
	                                                                                     attention_sum_t,
	                                                                                     decoder_attention_t,dense_input,batch_size_t,h_mask_t,w_mask_t,gpu)

	    
	    decoder_attention_t_cat.append(decoder_attention_t[0].data.cpu().numpy())
	    topv,topi = torch.max(decoder_output,2)
	    if torch.sum(topi)==0:
	        break
	    decoder_input_t = topi
	    decoder_input_t = decoder_input_t.view(batch_size_t)

	    # prediction
	    prediction[:,i] = decoder_input_t


	k = numpy.array(decoder_attention_t_cat)
	x_real = numpy.array(x_real.cpu().data)

	prediction = prediction[0]

	prediction_real = []
	for ir in range(len(prediction)):
		if int(prediction[ir]) ==0:
			break
		prediction_real.append(worddicts_r[int(prediction[ir])])
	prediction_real.append('<eol>')


	prediction_real_show = numpy.array(prediction_real)

	return k,prediction_real_show

def calculate(formel, operation):
	
	sym = latex2sympy(formel)

	print(sym)

if __name__ == "__main__":
    print(processing_image())