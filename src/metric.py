from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.add_state('TP', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('FP', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('FN', default=torch.zeros(num_classes), dist_reduce_fx='sum')

    def update(self, preds, target):
        classes_of_preds = torch.argmax(preds, dim=1)

        for class_index in range(self.num_classes):
            pred_equal_class = (classes_of_preds == class_index)
            target_equal_class = (target == class_index)

            self.TP[class_index] += torch.sum(pred_equal_class & target_equal_class)
            self.FP[class_index] += torch.sum(pred_equal_class & ~target_equal_class)
            self.FN[class_index] += torch.sum(~pred_equal_class & target_equal_class)

    def compute(self):
        eps = 1e-8 
        precision = self.TP / (self.TP + self.FP + eps)
        recall = self.TP / (self.TP + self.FN + eps)
        f1score = 2 * precision * recall / (precision + recall + eps)
        return f1score
    


class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        classes_of_preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        if classes_of_preds.shape != target.shape:
            print("Does not have equal shape")

        # [TODO] Count the number of correct prediction
        Correct_prediction = (classes_of_preds == target).sum()

        # Accumulate to self.correct
        self.correct += Correct_prediction

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
