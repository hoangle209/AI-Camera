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


class ClsPostProcess:
    """Convert between text-label and text-index"""

    def __init__(self, label_list=None, key=None, **kwargs):
        super().__init__()
        self.label_list = label_list
        self.key = key

    def __call__(self, preds, label=None, *args, **kwargs):
        if self.key is not None:
            preds = preds[self.key]

        label_list = self.label_list
        if label_list is None:
            label_list = {idx: idx for idx in range(preds.shape[-1])}

        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()

        pred_idxs = preds.argmax(axis=1)
        decode_out = [
            (label_list[idx], preds[i, idx]) for i, idx in enumerate(pred_idxs)
        ]
        if label is None:
            return decode_out
        label = [(label_list[idx], 1.0) for idx in label]
        return decode_out, label
