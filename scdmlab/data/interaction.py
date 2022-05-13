import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils


def _convert_to_tensor(data):
    elem = data[0]
    if isinstance(elem, (float, int, np.float, np.int64)):
        new_data = torch.as_tensor(data)
    elif isinstance(elem, (list, tuple, pd.Series, np.ndarray, torch.Tensor)):
        seq_data = [torch.as_tensor(d) for d in data]
        new_data = rnn_utils.pad_sequence(seq_data, batch_first=True)
    else:
        raise ValueError(f'[{type(elem)}] is not supported')

    if new_data.dtype == torch.float64:
        new_data = new_data.float()
    return new_data


class Interaction(object):
    def __init__(self, interaction):
        self.interaction = dict()
        if isinstance(interaction, dict):
            for key, value in interaction.items():
                if isinstance(value, (list, np.ndarray)):
                    self.interaction[key] = _convert_to_tensor(value)
                elif isinstance(value, torch.Tensor):
                    self.interaction[key] = value
                else:
                    raise ValueError(f'The type of {key}[{type(value)}] is not supported')
        elif isinstance(interaction, pd.DataFrame):
            for key in interaction:
                value = interaction[key].values
                self.interaction[key] = _convert_to_tensor(value)
        else:
            raise ValueError(f'[{type(interaction)}] is not supported')

        self.length = -1
        for k in self.interaction:
            self.length = max(self.length, self.interaction[k].unsqueeze(-1).shape[0])

    @property
    def columns(self):
        """
        Returns:
            list of str: The columns of interaction.
        """
        return list(self.interaction.keys())

    def to(self, device, selected_field=None):
        """Transfer Tensors in this Interaction object to the specified device.

        Args:
            device (torch.device): target device.
            selected_field (str or iterable object, optional): if specified, only Tensors
            with keys in selected_field will be sent to device.

        Returns:
            Interaction: a coped Interaction object with Tensors which are sent to
            the specified device.
        """
        ret = {}
        if isinstance(selected_field, str):
            selected_field = [selected_field]

        if selected_field is not None:
            selected_field = set(selected_field)
            for k in self.interaction:
                if k in selected_field:
                    ret[k] = self.interaction[k].to(device)
                else:
                    ret[k] = self.interaction[k]
        else:
            for k in self.interaction:
                ret[k] = self.interaction[k].to(device)
        return Interaction(ret)

    def __iter__(self):
        return self.interaction.__iter__()

    def __getattr__(self, item):
        if 'interaction' not in self.__dict__:
            raise AttributeError(f"'Interaction' object has no attribute 'interaction'")
        if item in self.interaction:
            return self.interaction[item]
        raise AttributeError(f"'Interaction' object has no attribute '{item}'")

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.interaction[index]
        else:
            ret = {}
            for k in self.interaction:
                ret[k] = self.interaction[k][index]
            return Interaction(ret)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise KeyError(f'{type(key)} object does not support item assigment')
        self.interaction[key] = value

    def __delitem__(self, key):
        if key not in self.interaction:
            raise KeyError(f'{type(key)} object does not in this interaction')
        del self.interaction[key]

    def __contains__(self, item):
        return item in self.interaction

    def __len__(self):
        return self.length

    def __str__(self):
        info = [f'The batch_size of interaction: {self.length}']
        for k in self.interaction:
            inter = self.interaction[k]
            temp_str = f"    {k}, {inter.shape}, {inter.device.type}, {inter.dtype}"
            info.append(temp_str)
        info.append('\n')
        return '\n'.join(info)

    def __repr__(self):
        return self.__str__()
