from torch import nn
import torch


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, input_ids, attention_mask):
        outputs = []
        for model in self.models:
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs.append(output.logits)

        ensemble_output = torch.mean(torch.stack(outputs), dim=0)
        return ensemble_output
