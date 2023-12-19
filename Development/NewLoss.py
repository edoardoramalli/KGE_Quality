import torch
from pykeen.losses import BCEWithLogitsLoss

def clip_before_exp(value):
    """Clip the value for stability of exponential."""
    return torch.clamp(
        value,
        min=-75,
        max=75,
    )


class EdoLoss(PairwiseLoss):

    def forward(
            self,
            scores: torch.FloatTensor,
            labels: torch.FloatTensor,
    ) -> torch.FloatTensor:  # noqa: D102

        # print(labels)

        positive_scores = scores[labels == 1]
        negative_scores = scores[labels == 0]

        # print(positive_scores.shape)
        # print(negative_scores.shape)
        # print(scores.shape)
        #
        # print('carlo', scores.shape[0] == (positive_scores.shape[0] + negative_scores.shape[0]))
        #
        # print(scores)
        #
        # print(labels)

        positive_scores = clip_before_exp(positive_scores)
        negative_scores = clip_before_exp(negative_scores)

        neg_exp = torch.exp(negative_scores)
        pos_exp = torch.exp(positive_scores)
        softmax_score = pos_exp / (torch.mean(neg_exp, dim=0) + pos_exp)
        loss = torch.mean(-torch.log(softmax_score), dim=0)

        # print(loss)

        # return torch.sum(scores)
        return loss

        # return functional.binary_cross_entropy_with_logits(scores, labels, reduction=self.reduction)
