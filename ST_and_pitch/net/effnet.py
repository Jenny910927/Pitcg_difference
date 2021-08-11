import torch.nn as nn
import torch
import torch.nn.functional as F
import time

class down_layer(nn.Module):
    def __init__(self, in_channel, out_channel, down_scale=2):
        super(down_layer, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(5,5), stride=(2,2), padding=(2,2))
        # self.down1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=(1,down_scale), padding=(1,1))

        # self.bn1 = nn.BatchNorm2d(num_features=in_channel)
        self.do = nn.Dropout2d(p=0.2)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
    
    def forward(self, x):
        out = F.leaky_relu(self.do(self.bn2(self.conv1(x))), 0.2)
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = F.relu(self.bn2(self.down1(out)))
        return out

class up_layer(nn.Module):
    def __init__(self, in_channel, out_channel, up_scale=2):
        super(up_layer, self).__init__()
        
        self.conv1 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(5,5), stride=(2,2), padding=(2,2), output_padding=(1,1))
        # if up_scale == 2:
        #     self.up1 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(3,4), stride=(1,up_scale), padding=(1,1))
        # elif up_scale == 3:
        #     self.up1 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(3,3), stride=(1,up_scale), padding=(1,0))

        # self.bn1 = nn.BatchNorm2d(num_features=in_channel)
        self.do = nn.Dropout2d(p=0.2)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
    
    def forward(self, x):
        
        out = self.do(self.bn2(F.relu(self.conv1(x))))
        # out = F.relu(self.bn2(self.up1(out)))
        return out



class Mysvs(nn.Module):
    def __init__(self):
        super(Mysvs, self).__init__()

        self.conv1 = down_layer(3, 8)
        self.conv2 = down_layer(8, 16)
        self.conv3 = down_layer(16, 32)
        self.conv4 = down_layer(32, 64)
        self.conv5 = down_layer(64, 128)
        self.conv6 = down_layer(128, 128)
        
        self.deconv1 = up_layer(128, 128)
        self.deconv2 = up_layer(256, 64)
        self.deconv3 = up_layer(128, 32)
        self.deconv4 = up_layer(64, 16)
        self.deconv5 = up_layer(32, 8)
        self.deconv6 = up_layer(16, 8)

        self.conv7 = nn.Conv2d(8, 3, kernel_size=(3, 3), padding=(1, 1))
        
    def forward(self, x):
        # print (x.shape)
        out = self.conv1(x)
        out2 = self.conv2(out)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)
        # print (out6.shape)
        # print (out5.shape)
        out7 = torch.cat((self.deconv1(out6), out5), dim=1)
        out8 = torch.cat((self.deconv2(out7), out4), dim=1)
        out9 = torch.cat((self.deconv3(out8), out3), dim=1)
        out10 = torch.cat((self.deconv4(out9), out2), dim=1)
        out11 = torch.cat((self.deconv5(out10), out), dim=1)
        final_out = self.deconv6(out11)
        final_out = torch.sigmoid(self.conv7(final_out))

        voc_out = x * final_out
        # acc_out = x - voc_out

        return voc_out
        # return voc_out, acc_out


# class MyAst(nn.Module):
#     def __init__(self, pitch_class, pitch_octave):
#         super(MyAst, self).__init__()
#         self.pitch_octave = pitch_octave
#         self.conv1 = nn.Conv2d(2, 16, kernel_size=(9, 9), stride=(1, 8), padding=(4, 4))
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
#         self.conv4 = nn.Conv2d(64, 64, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
#         self.conv5 = nn.Conv2d(64, 1, kernel_size=(9, 9), padding=(4, 4))
#         # self.conv5 = nn.Conv2d(32, 32, kernel_size=(1, 1), padding=(0, 0))
#         # self.conv6 = nn.Conv2d(32, 2, kernel_size=(5, 5), padding=(2, 2))
#         # self.global_conv = nn.Conv2d(32, 2, kernel_size=(127, 1), padding=(63, 0))

#         # self.global_conv1 = nn.Conv2d(64, 8, kernel_size=(1, 1), padding=(0, 0))
#         # self.global_conv2 = nn.Conv2d(8, 1, kernel_size=(127, 1), padding=(63, 0))
#         self.get_value = nn.Linear(96, 96)
#         self.query = nn.Linear(96, 10)
#         self.key = nn.Linear(96, 10)

#         self.fc1   = nn.Linear((96*8*2)//8, 64)
#         self.fc2   = nn.Linear(64, 32)
#         # self.fc3   = nn.Linear(128, 2+pitch_class+pitch_octave+2)
#         self.fc3   = nn.Linear(32, 2+pitch_class+pitch_octave+2+1)
        
#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         # # out = F.max_pool2d(out, 2)
#         out = F.relu(self.conv2(out))
#         # # out = F.max_pool2d(out, 2)
#         out = F.relu(self.conv3(out))
#         out = F.relu(self.conv4(out))
#         # out = F.relu(self.conv5(out))
#         # out1 = F.relu(self.conv6(out))
#         out1 = F.relu(self.conv5(out))

#         out_value = self.get_value(out1)
#         out_query = self.query(out1)
#         out_key = torch.transpose(self.key(out1), 2, 3)
#         out_weight = torch.matmul(out_query, out_key) / torch.sqrt(torch.tensor(10.0))
#         out_weight = F.softmax(out_weight, dim=3)

#         # out_glob = F.relu(self.global_conv1(out))
#         # out_glob = F.relu(self.global_conv2(out_glob))
#         out_glob = torch.matmul(out_weight, out_value)
#         # print (out1.shape)
#         # print (out_glob.shape)
#         out = torch.cat((out1, out_glob), dim=3)

#         # pitch_in = out[:, 0, :, :]
#         # on_off_in = out[:, 1, :, :]

#         # print (pitch_in.shape)

#         # out = out.view(out.size(0), -1)
#         # print (out.shape)

#         # out = out.squeeze(1)
#         # pitch_out = F.relu(self.p_fc1(pitch_in))
#         # pitch_out = F.relu(self.p_fc2(pitch_out))
#         # pitch_out = self.p_fc3(pitch_out)

#         # # out = out.squeeze(1)
#         # on_off_out = F.relu(self.o_fc1(on_off_in))
#         # on_off_out = F.relu(self.o_fc2(on_off_out))
#         # on_off_out = self.o_fc3(on_off_out)

        
#         # out = self.effnet(x)
#         out = out.squeeze(1)
#         out = F.relu(self.fc1(out))
#         out = F.relu(self.fc2(out))
#         out = self.fc3(out)

#         onset_logits = out[:, :, 0]
#         offset_logits = out[:, :, 1]
#         # onset_d_logits = out[:, :, 2]
#         pitch_out = out[:, :, 2:-1]
#         onset_d_logits = out[:, :,-1]


#         # onset_logits = on_off_out[:, :, 0]
#         # offset_logits = on_off_out[:, :, 1]
#         # onset_d_logits = on_off_out[:, :, 2]
#         # pitch_out = out[:, :, 3:]
        
#         # print (onset_logits.shape)
#         # print (out[:, 2])
#         # pitch_out = out[:, 2] + x2

#         # pitch_out = self.effnet(x)

#         pitch_octave_logits = pitch_out[:, :, 0:(self.pitch_octave+1)]
#         pitch_class_logits = pitch_out[:, :, (self.pitch_octave+1):]

#         # pitch_out = self.Linear1(feature)
#         # pitch_octave_logits = pitch_out[:, 0:self.pitch_octave]
#         # pitch_class_logits = pitch_out[:, self.pitch_octave:]
#         return onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits, onset_d_logits

class Onset_cnn(nn.Module):
    def __init__(self):
        super(Onset_cnn, self).__init__()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
        self.conv5 = nn.Conv2d(64, 1, kernel_size=(9, 9), padding=(4, 4))
        self.fc1   = nn.Linear((96*8)//8, 64)
        self.fc2   = nn.Linear(64, 32)
        self.fc3   = nn.Linear(32, 4)

    def forward(self, x):
        # out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(x))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out1 = F.relu(self.conv5(out))

        out = out1.squeeze(1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        on_off_logits = self.fc3(out)

        return on_off_logits, out1

class Pitch_cnn(nn.Module):
    def __init__(self, pitch_class=12, pitch_octave=4):
        super(Pitch_cnn, self).__init__()
        self.pitch_octave = pitch_octave
        self.conv2 = nn.Conv2d(17, 32, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
        self.conv5 = nn.Conv2d(64, 1, kernel_size=(9, 9), padding=(4, 4))
        self.fc1   = nn.Linear((96*8)//8, 64)
        self.fc2   = nn.Linear(64, 32)
        self.fc3   = nn.Linear(32, pitch_class+pitch_octave+2)

        self.fc3.bias.data[self.pitch_octave] = -1.0
        self.fc3.bias.data[-1] = -1.0

    def forward(self, x):
        # out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(x))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out1 = F.relu(self.conv5(out))

        out = out1.squeeze(1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        pitch_out = self.fc3(out)

        pitch_octave_logits = pitch_out[:, :, 0:(self.pitch_octave+1)]
        pitch_class_logits = pitch_out[:, :, (self.pitch_octave+1):]

        return pitch_octave_logits, pitch_class_logits



class EffNetb0(nn.Module):
    def __init__(self, pitch_class=12, pitch_octave=4, svs_path=None):
        super(EffNetb0, self).__init__()

        # self.svs = Mysvs()
        # if svs_path is not None:
        #     self.svs.load_state_dict(torch.load(svs_path, map_location= "cpu"))
        #     print('svs model read from {}.'.format(svs_path))

        # self.ast = MyAst(pitch_class, pitch_octave)

        # ast_path = "models/1207_e2_200"
        # if ast_path is not None:
        #     self.ast.load_state_dict(torch.load(ast_path, map_location= "cpu"))
        #     print('ast model read from {}.'.format(ast_path))

        self.pitch_octave = pitch_octave

        self.conv1 = nn.Conv2d(6, 16, kernel_size=(9, 9), stride=(1, 4), padding=(4, 4))

        # self.conv2 = nn.Conv2d(16, 32, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
        # self.conv4 = nn.Conv2d(32, 64, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
        # self.conv5 = nn.Conv2d(64, 1, kernel_size=(9, 9), padding=(4, 4))
        # self.fc1   = nn.Linear((96*8)//8, 64)
        # self.fc2   = nn.Linear(64, 32)
        # self.fc3   = nn.Linear(32, 4+pitch_class+pitch_octave+2)

        # self.fc3.bias.data[4+self.pitch_octave] = -1.0
        # self.fc3.bias.data[-1] = -1.0

        self.onset_cnn = Onset_cnn()
        self.pitch_cnn = Pitch_cnn(pitch_class=pitch_class, pitch_octave=pitch_octave)
        
    def forward(self, x):
        # print(out.shape)
        # [batch, output_size]
        # x = x.permute(0, 1, 3, 2)
        # after_svs, _ = self.svs(x[:, -1:, :, :])
        # print (x.shape)

        # after_svs = self.svs(x[:, :3, :, :])
        # new_x = torch.cat((after_svs, x[:, :3, :, :]), dim=1)
        # new_x = x

        # print (x.shape)

        features = F.relu(self.conv1(x))
        # print ("===")
        # print (time.time())
        # out = F.relu(self.conv1(x))
        # out = F.relu(self.conv2(out))
        # out = F.relu(self.conv3(out))
        # out = F.relu(self.conv4(out))
        # out = F.relu(self.conv5(out))
        # out = out.squeeze(1)
        # out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        # out = self.fc3(out)

        on_off_logits, conv5_features = self.onset_cnn(features)
        pitch_features = torch.cat((features, conv5_features), dim=1)
        pitch_octave_logits, pitch_class_logits = self.pitch_cnn(pitch_features)

        # print (time.time())
        # 0: blank, 1:onset, 2:offset, 3:have pitch
        # on_off_logits = out[:, :, :4]
        on_off_logits_sm = F.softmax(on_off_logits, dim=2)

        # pitch_out = out[:, :, 4:]
        # pitch_octave_logits = pitch_out[:, :, 0:(self.pitch_octave+1)]
        # pitch_class_logits = pitch_out[:, :, (self.pitch_octave+1):]

        pitch_octave_sm = F.log_softmax(pitch_octave_logits[:,:,:self.pitch_octave], dim=2)
        pitch_class_sm = F.log_softmax(pitch_class_logits[:,:,:12], dim=2)

        all_result = torch.zeros((pitch_class_logits.shape[0], pitch_class_logits.shape[1], 3+4*12))
        all_result[:,:,0:2] = torch.log(on_off_logits_sm[:,:,0:2])
        for i in range(4):
            for j in range(12):
                index_num = i*12+j+3
                all_result[:,:,index_num] = pitch_octave_sm[:,:,i] + pitch_class_sm[:,:,j] + torch.log(on_off_logits_sm[:,:,3])

        # print (time.time())
        # onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits, onset_d_logits = self.ast(new_x)
        return on_off_logits, on_off_logits_sm, pitch_octave_logits, pitch_class_logits, all_result
        # return onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits, onset_d_logits
        # return onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits, onset_d_logits, after_svs
        # return onset_logits, offset_logits, pitch_out





# class EffNetb0_extralayer(nn.Module):
#     def __init__(self, pitch_class=12, pitch_octave=4):
#         super(EffNetb0_extralayer, self).__init__()
#         self.model_name = 'effnet'
#         self.pitch_octave = pitch_octave
#         self.pitch_class = pitch_class
#         # Create model
#         torch.hub.list('rwightman/gen-efficientnet-pytorch')
#         self.effnet = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=False)
        
#         self.effnet.conv_stem = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         # Modify last linear layer
#         num_ftrs = self.effnet.classifier.in_features
#         self.effnet.classifier= nn.Linear(num_ftrs, 50)

#         self.Linear1 = nn.Linear(50-2, 50)
#         self.Linear2 = nn.Linear(50, pitch_class+pitch_octave)
#         # print (self.effnet)
        
#     def forward(self, x):
#         out = self.effnet(x)
#         # print(out.shape)
#         # [batch, output_size]

#         onset_logits = out[:, 0]
#         offset_logits = out[:, 1]

#         feature = F.relu(out[:, 2:])
#         pitch_out = F.relu(self.Linear1(feature))
#         pitch_out = self.Linear2(pitch_out)
#         pitch_octave_logits = pitch_out[:, 0:self.pitch_octave]
#         pitch_class_logits = pitch_out[:, self.pitch_octave:]

#         return onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits

# class EffNetb0_withrnn(nn.Module):
#     def __init__(self, pitch_class=12, pitch_octave=4):
#         super(EffNetb0_withrnn, self).__init__()
#         self.model_name = 'effnet'
#         self.pitch_octave = pitch_octave
#         self.pitch_class = pitch_class
#         # Create model
#         torch.hub.list('rwightman/gen-efficientnet-pytorch')
#         self.effnet = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=False)
        
#         self.effnet.conv_stem = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         # Modify last linear layer
#         num_ftrs = self.effnet.classifier.in_features
#         self.effnet.classifier= nn.Linear(num_ftrs, 50)

#         self.Linear1 = nn.Linear(50-2, 50)
#         self.rnn1 = nn.GRU(50, pitch_class+pitch_octave, num_layers=3, bidirectional=False)
#         # print (self.effnet)
        
#     def forward(self, x, device, state, start_flag):
#         out = self.effnet(x)
#         # print(out.shape)
#         # [batch, output_size]

#         onset_logits = out[:, 0]
#         offset_logits = out[:, 1]

#         feature = F.relu(out[:, 2:])
#         pitch_out = self.Linear1(feature)
#         # if start_flag == 0:
#         #     state = self.initHidden(3, 1, self.pitch_octave+self.pitch_class, device)
#         # state = state.detach()
#         # pitch_out, new_state = self.rnn1(pitch_out.unsqueeze(1), state)
#         # pitch_octave_logits = pitch_out[:, 0, 0:self.pitch_octave]
#         # pitch_class_logits = pitch_out[:, 0, self.pitch_octave:]

#         pitch_octave_logits = pitch_out[:, 0:self.pitch_octave]
#         pitch_class_logits = pitch_out[:, self.pitch_octave:self.pitch_octave+self.pitch_class]

#         # return onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits, new_state
#         return onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits, state

#     def initHidden(self, size1, size2, size3, device):
#         return torch.zeros((size1, size2, size3), dtype=torch.float, requires_grad=True, device=device)


# class EffNet(nn.Module):
#     def __init__(self, output_size=52):
#         super(EffNet, self).__init__()
#         self.model_name = 'effnet'

#         # Create model
#         torch.hub.list('rwightman/gen-efficientnet-pytorch')
#         self.effnet = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b3', pretrained=False)
        
#         self.effnet.conv_stem = nn.Conv2d(1, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         # Modify last linear layer
#         num_ftrs = self.effnet.classifier.in_features
#         self.effnet.classifier= nn.Linear(num_ftrs, output_size)
#         #print (self.effnet)

#     def forward(self, x):
#         out = self.effnet(x)
#         # print(out.shape)
#         # [batch, output_size]

#         onset_logits = out[:, 0]
#         offset_logits = out[:, 1]
#         pitch_logits = out[:, 2:]

#         return onset_logits, offset_logits, pitch_logits


if __name__ == '__main__':
    from torchsummary import summary
    model = EffNetb0().cuda()
    # model.load_state_dict(torch.load("../models/b0_c2_e_30000", map_location= "cuda:0"))
    # summary(model, input_size=(1, 512, 11))

    # model = EffNet().cuda()
    summary(model, input_size=(2, 11, 84))
