# Environments/Sandbox_Signal/snn_signal_env.py
import torch
import snntorch.spikegen as spikegen


class SNNSignalEnv:
    def __init__(self, num_steps=50, num_sensors=20):
        self.num_steps = num_steps      # Dòng thời gian (50 nhịp)
        self.num_sensors = num_sensors  # Số lượng đầu thu tín hiệu (20 tần số)

    def get_sample(self, is_danger=False):
        """Sinh ra tín hiệu thô và chuyển thành Spike Train (0 và 1)"""
        # Trạng thái an toàn: Các cảm biến chỉ có nhiễu nhẹ (Rate thấp)
        signal = torch.rand(self.num_sensors) * 0.2
        target = 0

        if is_danger:
            # Nguy hiểm (Tiếng hét/Còi báo động): Tần số từ 10->15 tăng vọt!
            signal[10:15] = 0.95
            target = 1

        # spikegen.rate biến số thực (0.95) thành một dải các số 0,1 theo thời gian
        # Kích thước output: (num_steps, num_sensors)
        spike_train = spikegen.rate(signal, num_steps=self.num_steps)

        return spike_train, target