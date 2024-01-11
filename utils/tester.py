import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm

class Tester:
    def __init__(self, args, params):
        self.args = args
        self.params = params

    def test(self):
        filenames = self.get_validation_filenames()
        loader = self.prepare_data_loader(filenames)
        model = self.load_model()

        device = torch.device('cpu')
        model.to(device)
        model.eval()

        iou_v = torch.linspace(0.5, 0.95, 10, device=device)
        n_iou = iou_v.numel()

        m_pre, m_rec, map50, mean_ap, metrics = self.evaluate(loader, model, device, iou_v, n_iou)

        self.print_results(m_pre, m_rec, map50, mean_ap)

        model.float()

        return mean_ap, map50, m_rec, m_pre

    def get_validation_filenames(self):
        filenames = []
        with open('../Dataset/COCO/val2017.txt') as reader:
            for filename in reader.readlines():
                filename = filename.rstrip().split('/')[-1]
                filenames.append('../Dataset/COCO/images/val2017/' + filename)
        return filenames

    def prepare_data_loader(self, filenames):
        dataset = Dataset(filenames, self.args.input_size, self.params, False)
        loader = DataLoader(dataset, self.args.batch_size // 2, False, num_workers=8,
                             pin_memory=True, collate_fn=Dataset.collate_fn)
        return loader

    def load_model(self):
        model = torch.jit.load(f='./weights/best.ts')
        return model

    def evaluate(self, loader, model, device, iou_v, n_iou):
        m_pre = 0.
        m_rec = 0.
        map50 = 0.
        mean_ap = 0.
        metrics = []

        p_bar = tqdm.tqdm(loader, desc=('%10s' * 4) % ('precision', 'recall', 'mAP50', 'mAP'))
        for samples, targets in p_bar:
            samples = samples.to(device)
            samples = samples.float()
            samples = samples / 255.0
            _, _, h, w = samples.shape
            scale = torch.tensor((w, h, w, h), device=device)

            outputs = model(samples)

            outputs = non_max_suppression(outputs, 0.001, 0.7, model.nc)

            for i, output in enumerate(outputs):
                idx = targets['idx'] == i
                cls = targets['cls'][idx]
                box = targets['box'][idx]

                cls = cls.to(device)
                box = box.to(device)

                metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool, device=device)

                if output.shape[0] == 0:
                    if cls.shape[0]:
                        metrics.append((metric, *torch.zeros((2, 0), device=device), cls.squeeze(-1)))
                    continue

                if cls.shape[0]:
                    target = torch.cat((cls, wh2xy(box) * scale), 1)
                    metric = compute_metric(output[:, :6], target, iou_v)

                metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

        metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]
        if len(metrics) and metrics[0].any():
            tp, fp, m_pre, m_rec, map50, mean_ap = compute_ap(*metrics)

        print('%10.3g' * 4 % (m_pre, m_rec, map50, mean_ap))

        model.float()

        return m_pre, m_rec, map50, mean_ap

    def print_results(self, m_pre, m_rec, map50, mean_ap):
        print(f"Precision: {m_pre:.4f}, Recall: {m_rec:.4f}, mAP@0.5: {map50:.4f}, mAP: {mean_ap:.4f}")

if __name__ == "__main__":
    tester = Tester(args, params)
    mean_ap, map50, m_rec, m_pre = tester.test()
    print(f"mAP: {mean_ap:.4f}, mAP@0.5: {map50:.4f}, Recall: {m_rec:.4f}, Precision: {m_pre:.4f}")
