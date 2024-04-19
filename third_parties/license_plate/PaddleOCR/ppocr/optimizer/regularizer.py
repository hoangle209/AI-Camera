# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import paddle


class L1Decay:
    """
    L1 Weight Decay Regularization, which encourages the weights to be sparse.
    Args:
        factor(float): regularization coeff. Default:0.0.
    """

    def __init__(self, factor=0.0):
        super().__init__()
        self.coeff = factor

    def __call__(self):
        reg = paddle.regularizer.L1Decay(self.coeff)
        return reg


class L2Decay:
    """
    L2 Weight Decay Regularization, which helps to prevent the model over-fitting.
    Args:
        factor(float): regularization coeff. Default:0.0.
    """

    def __init__(self, factor=0.0):
        super().__init__()
        self.coeff = float(factor)

    def __call__(self):
        return self.coeff