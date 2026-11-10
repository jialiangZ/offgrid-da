"""Full-weather context with a one-channel T2M analysis increment."""
import torch
from torch import nn
from .analysis import CrossAttentionAnalysisModel


class FullStateT2MAnalysis(CrossAttentionAnalysisModel):
    def __init__(self, input_channels=82, **kwargs):
        super().__init__(channels=1, **kwargs)
        self.input_channels = input_channels
        self.input = nn.Conv2d(input_channels, self.input.out_channels, 3, padding=1)

    def forward(self, background, observations, grid_state=None):
        state = observations.get('grid_state') if grid_state is None else grid_state
        if state is None or state.shape != (background.shape[0], self.input_channels, *background.shape[-2:]):
            raise ValueError('Explicit normalized grid_state with the configured channels is required')
        if background.shape[1] != 1:
            raise ValueError('Residual background must contain T2M only')
        encoded = self.observation_encoder(observations)
        tokens, mask = encoded['tokens'], encoded['observation_mask']
        locations = observations['obs_locations']
        valid = mask & (locations[:, :, 0].abs() <= 90)
        x = self.input(state)
        for i, block in enumerate(self.blocks):
            x = x + block(x)
            if self.injection == 'layered' or i == 0:
                b, d, h, w = x.shape
                update = self._local(x.flatten(2).transpose(1, 2), tokens, valid, locations, h, w, i)
                x = x + update.transpose(1, 2).reshape(b, d, h, w)
        increment = self.output(x).masked_fill(~valid.any(dim=1)[:, None, None, None], 0)
        return {'analysis': background + increment, **encoded}

