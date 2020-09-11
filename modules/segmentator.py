#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
import __init__ as booger
from postproc.CRF import CRF
from common.debugger import breakpoint
import global_info

setting = global_info.setting()


class Segmentator(nn.Module):
  def __init__(self, ARCH, nclasses, path=None, path_append="", strict=False):
    super().__init__()
    self.ARCH = ARCH
    self.nclasses = nclasses
    self.path = path
    self.path_append = path_append
    self.strict = False
    self.use_rv = ARCH["dataset"]["sensor"]["use_rv"]
    self.use_bev= ARCH["dataset"]["sensor"]["use_bev"]
    self.use_head = True
    # >>>>>>>>>>>>>>>>>>> we eithe train the model on RV data
    # if self.use_rv:
    bboneModule = imp.load_source("bboneModule",
                                  booger.TRAIN_PATH + '/backbones/' +
                                  self.ARCH["backbone"]["name"] + '.py')
    self.backbone = bboneModule.Backbone(params=self.ARCH["backbone"])
    #
    # do a pass of the backbone to initialize the skip connections
    stub = torch.zeros((1,
                        self.backbone.get_input_depth(),
                        self.ARCH["dataset"]["sensor"]["img_prop"]["height"],
                        self.ARCH["dataset"]["sensor"]["img_prop"]["width"]))

    if torch.cuda.is_available():
      stub = stub.cuda()
      self.backbone.cuda()
    _, stub_skips = self.backbone(stub)

    decoderModule = imp.load_source("decoderModule",
                                    booger.TRAIN_PATH + '/decoders/' +
                                    self.ARCH["decoder"]["name"] + '.py')
    self.decoder = decoderModule.Decoder(params=self.ARCH["decoder"],
                                         stub_skips=stub_skips,
                                         OS=self.ARCH["backbone"]["OS"],
                                         feature_depth=self.backbone.get_last_depth())
    self.head = nn.Sequential(nn.Dropout2d(p=ARCH["head"]["dropout"]),
                              nn.Conv2d(self.decoder.get_last_depth(),
                                        self.nclasses, kernel_size=3,
                                        stride=1, padding=1))

    if self.ARCH["post"]["CRF"]["use"]:
      self.CRF = CRF(self.ARCH["post"]["CRF"]["params"], self.nclasses)
    else:
      self.CRF = None

    # train backbone?
    if not self.ARCH["backbone"]["train"]:
      for w in self.backbone.parameters():
        w.requires_grad = False

    # train decoder?
    if not self.ARCH["decoder"]["train"]:
      for w in self.decoder.parameters():
        w.requires_grad = False

    # train head?
    if not self.ARCH["head"]["train"]:
      for w in self.head.parameters():
        w.requires_grad = False

    # train CRF?
    if self.CRF and not self.ARCH["post"]["CRF"]["train"]:
      for w in self.CRF.parameters():
        w.requires_grad = False

    weights_total = sum(p.numel() for p in self.parameters())
    weights_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print("Total number of parameters: ", weights_total)
    print("Total number of parameters requires_grad: ", weights_grad)

    # breakdown by layer
    weights_enc = sum(p.numel() for p in self.backbone.parameters())
    weights_dec = sum(p.numel() for p in self.decoder.parameters())
    weights_head = sum(p.numel() for p in self.head.parameters())
    print("Param encoder ", weights_enc)
    print("Param decoder ", weights_dec)
    print("Param head ", weights_head)
    if self.CRF:
      weights_crf = sum(p.numel() for p in self.CRF.parameters())
      print("Param CRF ", weights_crf)

    # get weights
    if path is not None:
      # try backbone
      try:
        w_dict = torch.load(path + "/backbone" + path_append,
                            map_location=lambda storage, loc: storage)
        self.backbone.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model backbone weights")
      except Exception as e:
        print()
        print("Couldn't load backbone, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      # try decoder
      try:
        w_dict = torch.load(path + "/segmentation_decoder" + path_append,
                            map_location=lambda storage, loc: storage)
        self.decoder.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model decoder weights")
      except Exception as e:
        print("Couldn't load decoder, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      # try head
      try:
        w_dict = torch.load(path + "/segmentation_head" + path_append,
                            map_location=lambda storage, loc: storage)
        self.head.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model head weights")
      except Exception as e:
        print("Couldn't load head, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      # try CRF
      if self.CRF:
        try:
          w_dict = torch.load(path + "/segmentation_CRF" + path_append,
                              map_location=lambda storage, loc: storage)
          self.CRF.load_state_dict(w_dict, strict=True)
          print("Successfully loaded model CRF weights")
        except Exception as e:
          print("Couldn't load CRF, using random weights. Error: ", e)
          if strict:
            print("I'm in strict mode and failure to load weights blows me up :)")
            raise e
    else:
      print("No path to pretrained, using random init.")

  def forward(self, x, gpu=0, mask=None, return_feature_maps=False):
    y, skips = self.backbone(x)
    y = self.decoder(y, skips)
    # breakpoint()
    last_feat = y
    y = self.head(y)
    y = F.softmax(y, dim=1)
    if self.CRF:
      assert(mask is not None)
      y = self.CRF(x, y, mask)
    if setting.USE_RV_PRED:
      last_feat = y
    if return_feature_maps:
      return y, last_feat
    else:
      return y

  def save_checkpoint(self, logdir, suffix=""):
    # Save the weights
    torch.save(self.backbone.state_dict(), logdir +
               "/backbone" + suffix)
    torch.save(self.decoder.state_dict(), logdir +
               "/segmentation_decoder" + suffix)
    torch.save(self.head.state_dict(), logdir +
               "/segmentation_head" + suffix)
    if self.CRF:
      torch.save(self.CRF.state_dict(), logdir +
                 "/segmentation_CRF" + suffix)

if __name__ == '__main__':
    gpu = 0
    num_classes = 20
    # test.cuda(gpu)
    deploy_device   = torch.device('cuda:{}'.format(gpu))
    x = torch.randn(2, 5, 64, 2048, requires_grad=True, device=deploy_device)
    y = torch.empty(2, 64, 2048, dtype=torch.long, device=deploy_device).random_(0, 20)
    RV_CFG = yaml.safe_load(open('../config/arch/darknet53-crf.yaml', 'r'))
    model = Segmentator(RV_CFG, num_classes)
    model.cuda(gpu)

    pred, blocks, feat_dec = model.forward(x, return_feature_maps=True, return_feature_banks=True)
    print(pred.shape)
    for i, feat_layer in enumerate(blocks):
        print('Encoder {} has shape: {}'.format(i, feat_layer.size()))
    for i, feat_layer in enumerate(feat_dec):
        print('Decoder {} has shape: {}'.format(i, feat_layer.size()))
