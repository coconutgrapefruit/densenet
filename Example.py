"""
@article{huang2019convolutional,
 title={Convolutional Networks with Dense Connectivity},
 author={Huang, Gao and Liu, Zhuang and Pleiss, Geoff and Van Der Maaten, Laurens and Weinberger, Kilian},
 journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
 year={2019}
 }

@inproceedings{huang2017densely,
  title={Densely Connected Convolutional Networks},
  author={Huang, Gao and Liu, Zhuang and van der Maaten, Laurens and Weinberger, Kilian Q },
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
"""

"""
This implementation allows more control over the architecture of DenseNet
For each dense block:
    bottleneck size, growth rate, dropout rate of every layer must be given. 
For each transition block:
    output channel, pooling kernel size must be given. 
"""

Config = [
    ['Dense',
        [[8, 4, .1],
        [8, 4, .1],
        [8, 4, .1],
        [8, 4, .1],
        [8, 4, .1],
        [8, 4, .1]]],
    ['Transition',
        [32, 2]],
    ['Dense',
        [[8, 4, 0],
        [8, 4, 0],
        [8, 4, 0],
        [8, 4, 0],
        [8, 4, 0],
        [8, 4, 0]]],
    ['Transition',
        [16, 2]],
    ['Dense',
        [[8, 4, 0],
        [8, 4, 0],
        [8, 4, 0],
        [8, 4, 0],
        [8, 4, 0],
        [8, 4, 0]]]
]

